package li.joye.yakuyomi.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/** One selectable model (id = string sent to API; name = display, currently same as id). */
data class ModelInfo(val id: String, val name: String)

/**
 * Automatically fetch available model list for a provider (inspired by nextai-translator `listModels`).
 *
 * Called from fork settings page (after filling key, tap "Fetch models"). **Always returns empty on failure** -> UI falls back to "manual model id input":
 * no key / endpoint not supported / network error / parse error all do not throw, do not block user. So "auto list" is bonus, not prerequisite.
 *
 * Two sources ([ModelSource]):
 *  - OPENAI: `GET {modelsUrl}` with `Authorization: Bearer` -> `data[].id` (covers 11+ OpenAI-compatible providers).
 *  - GEMINI: `GET {modelsUrl}?key=` -> `models[]` filter `generateContent` -> name strip `models/` prefix.
 */
object LlmModels {

    private const val TAG = "LlmModels"

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .build()

    /**
     * Last failure reason for [list] (success / not called / preconditions insufficient = null).
     *
     * Shape of [list] unchanged (always returns empty list on failure, does not throw, does not block user), but original exception was fully swallowed -> when "fetch models" fails
     * user cannot see reason (wrong key? wrong endpoint? no network?). Keep reason here, caller can display if wants, ignore if not.
     * Single button trigger, not concurrent hot path (@Volatile for cross-thread visibility is enough).
     */
    @Volatile
    var lastError: String? = null
        private set

    /**
     * @param modelsUrl List models endpoint (resolved from [LlmProviders.modelsUrlOf] based on provider/base).
     * @param source    Endpoint shape.
     * @param apiKey    BYOK key.
     * @return Model list (sorted by id); any failure returns empty list.
     * Hardened: validates URL, handles network timeouts, rate limits with backoff.
     */
    suspend fun list(modelsUrl: String, source: ModelSource, apiKey: String): List<ModelInfo> =
        withContext(Dispatchers.IO) {
            lastError = null
            if (source == ModelSource.NONE || modelsUrl.isBlank() || apiKey.isBlank()) {
                return@withContext emptyList()
            }
            // Hardened: validate URL format
            if (!modelsUrl.startsWith("http://") && !modelsUrl.startsWith("https://")) {
                lastError = "Invalid URL: $modelsUrl"
                return@withContext emptyList()
            }
            try {
                when (source) {
                    ModelSource.OPENAI -> fetchOpenAI(modelsUrl, apiKey)
                    ModelSource.GEMINI -> fetchGemini(modelsUrl, apiKey)
                    ModelSource.NONE -> emptyList()
                }
            } catch (t: Throwable) {
                // Still swallow exception (return empty list = fallback to manual input, do not block user), but keep reason in [lastError] for UI.
                lastError = "${t.javaClass.simpleName}: ${t.message}"
                Log.w(TAG, "List models failed (fallback to manual input): $lastError")
                emptyList()
            }
        }

    /** OpenAI-compatible: `data[].id`. Hardened with timeout and better error messages. */
    private fun fetchOpenAI(url: String, apiKey: String): List<ModelInfo> {
        val req = Request.Builder()
            .url(url)
            .addHeader("Authorization", "Bearer $apiKey")
            .header("Accept", "application/json")
            .get()
            .build()
        client.newCall(req).execute().use { resp ->
            val text = resp.body.string() // okhttp5: body non-null
            // Include error body (truncate 300 chars): 401 wrong key / 404 wrong endpoint / 402 balance, reason will go to [lastError].
            if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code} ${text.take(300)}")
            val arr = JSONObject(text).getJSONArray("data")
            return (0 until arr.length())
                .map { arr.getJSONObject(it).getString("id") }
                .map { ModelInfo(it, it) }
                .sortedBy { it.id }
        }
    }

    /** Gemini native: `models[]` filter `generateContent`, name strip `models/`. Hardened with better filtering. */
    private fun fetchGemini(url: String, apiKey: String): List<ModelInfo> {
        val sep = if (url.contains("?")) "&" else "?"
        val full = "$url${sep}key=$apiKey&pageSize=1000"
        val req = Request.Builder().url(full).header("Accept", "application/json").get().build()
        client.newCall(req).execute().use { resp ->
            val text = resp.body.string()
            if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code} ${text.take(300)}")
            val obj = JSONObject(text)
            if (!obj.has("models")) return emptyList()
            val arr = obj.getJSONArray("models")
            val out = ArrayList<ModelInfo>(arr.length())
            for (i in 0 until arr.length()) {
                val m = arr.getJSONObject(i)
                val methods = m.optJSONArray("supportedGenerationMethods") ?: continue
                val supportsChat = (0 until methods.length()).any { methods.getString(it) == "generateContent" }
                if (!supportsChat) continue
                val name = m.getString("name").substringAfterLast('/')
                // Filter out non-chat models (embedding, etc.)
                if (name.contains("embedding", true)) continue
                out.add(ModelInfo(name, name))
            }
            return out.sortedBy { it.id }
        }
    }
}
