package li.joye.yakuyomi.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/** 一個可選模型（id＝送給 API 的字串；name＝顯示用，目前同 id）。 */
data class ModelInfo(val id: String, val name: String)

/**
 * 自動撈 provider 的可用模型清單（借鏡 nextai-translator `listModels`）。
 *
 * 由 fork 設定頁呼叫（填好 key 後按「抓取模型」）。**撈不到一律回空** → UI 退回「手動輸入 model id」：
 * 無 key / 端點不支援 / 網路錯 / 解析錯 都不拋、不擋使用者。所以「自動清單」是加分、不是前置條件。
 *
 * 兩種來源（[ModelSource]）：
 *  - OPENAI：`GET {modelsUrl}` 帶 `Authorization: Bearer` → `data[].id`（涵蓋 11+ OpenAI 相容 provider）。
 *  - GEMINI：`GET {modelsUrl}?key=` → `models[]` 濾 `generateContent` → name 去 `models/` 前綴。
 */
object LlmModels {

    private const val TAG = "LlmModels"

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .build()

    /**
     * 最近一次 [list] 的失敗原因（成功 / 未呼叫 / 前置條件不足＝null）。
     *
     * [list] 的形狀不變（失敗一律回空清單、不拋、不擋使用者），但原本例外被整個吞掉 → 「抓取模型」失敗時
     * 使用者看不到原因（key 錯？端點不對？沒網路？）。把原因留在這，呼叫端要顯示就讀得到、不想理就無視。
     * 單一按鈕觸發、非併發熱路徑（@Volatile 保跨執行緒可見即可）。
     */
    @Volatile
    var lastError: String? = null
        private set

    /**
     * @param modelsUrl 列模型端點（由 [LlmProviders.modelsUrlOf] 依 provider/base 解出）。
     * @param source    端點形狀。
     * @param apiKey    BYOK 金鑰。
     * @return 模型清單（已依 id 排序）；任何失敗回空清單。
     */
    suspend fun list(modelsUrl: String, source: ModelSource, apiKey: String): List<ModelInfo> =
        withContext(Dispatchers.IO) {
            lastError = null
            if (source == ModelSource.NONE || modelsUrl.isBlank() || apiKey.isBlank()) {
                return@withContext emptyList()
            }
            try {
                when (source) {
                    ModelSource.OPENAI -> fetchOpenAI(modelsUrl, apiKey)
                    ModelSource.GEMINI -> fetchGemini(modelsUrl, apiKey)
                    ModelSource.NONE -> emptyList()
                }
            } catch (t: Throwable) {
                // 例外照舊吞掉（回空清單＝退回手動輸入、不擋使用者），但原因留在 [lastError] 供 UI 顯示。
                lastError = "${t.javaClass.simpleName}: ${t.message}"
                Log.w(TAG, "列模型失敗（退回手動輸入）：$lastError")
                emptyList()
            }
        }

    /** OpenAI 相容：`data[].id`。 */
    private fun fetchOpenAI(url: String, apiKey: String): List<ModelInfo> {
        val req = Request.Builder()
            .url(url)
            .addHeader("Authorization", "Bearer $apiKey")
            .get()
            .build()
        client.newCall(req).execute().use { resp ->
            val text = resp.body.string() // okhttp5：body 非空
            // 帶上 error body（截 300 字）：401 key 錯 / 404 端點不對 / 402 餘額，原因會傳到 [lastError]。
            if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code} ${text.take(300)}")
            val arr = JSONObject(text).getJSONArray("data")
            return (0 until arr.length())
                .map { arr.getJSONObject(it).getString("id") }
                .map { ModelInfo(it, it) }
                .sortedBy { it.id }
        }
    }

    /** Gemini native：`models[]` 濾 `generateContent`、name 去 `models/`。 */
    private fun fetchGemini(url: String, apiKey: String): List<ModelInfo> {
        val sep = if (url.contains("?")) "&" else "?"
        val full = "$url${sep}key=$apiKey&pageSize=1000"
        val req = Request.Builder().url(full).get().build()
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
                out.add(ModelInfo(name, name))
            }
            return out.sortedBy { it.id }
        }
    }
}
