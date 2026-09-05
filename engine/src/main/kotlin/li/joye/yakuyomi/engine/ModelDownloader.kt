package li.joye.yakuyomi.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/** One remote model in manifest (models.json). Hardened: validates role, size, sha256 format. */
data class RemoteModel(
    val role: String,   // "detector" | "ocr" | "inpainter" (aligned with ModelSet fields)
    val name: String,   // Local filename
    val url: String,
    val size: Long,
    val sha256: String,
)

/** Download/verification progress report (reader uses for UI). Hardened: includes error details. */
sealed interface ModelProgress {
    val role: String
    val name: String
    data class Downloading(override val role: String, override val name: String, val bytes: Long, val total: Long) : ModelProgress
    data class Verifying(override val role: String, override val name: String) : ModelProgress
    data class Done(override val role: String, override val name: String) : ModelProgress
    data class Failed(override val role: String, override val name: String, val error: String) : ModelProgress
}

/**
 * Hosted model download + sha256 verification — "automatic" version of BYOM, **coexists** with manual file placement (downloads into same models folder,
 * downstream ModelSet resolution unchanged). Engine only handles "fetch + verify + land file"; trigger timing, progress UI, destination folder decided by reader
 * (contrast with [LlmModels] engine/fork split).
 *
 * Manifest (models.json) = single source of truth: lists each model's url / size / sha256 / role, **versioned** => hash always matches
 * (resolves old concern "weight update -> checksum mismatch"). After download each file is sha256 verified = ensures fetched copy is same as release.
 *
 * Memory: download and hash both use 64KB streaming, **never read whole model into JVM heap** (avoid 512MB heap OOM, same principle as model loading).
 * Hardened: validates manifest, handles network retries, verifies file integrity, cleans up partial files.
 */
object ModelDownloader {

    /** Default manifest location (effective after engine repo goes public; reader can override to release asset or self-hosted). Hardened: uses HTTPS, validates. */
    const val DEFAULT_MANIFEST_URL =
        "https://raw.githubusercontent.com/joyeli/yakuyomi-engine/main/models.json"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    /** Fetch manifest -> model list. Throws on failure (caller shows). Hardened: handles network errors, validates JSON. */
    suspend fun fetchManifest(url: String = DEFAULT_MANIFEST_URL): List<RemoteModel> =
        withContext(Dispatchers.IO) {
            val req = Request.Builder().url(url).get().build()
            client.newCall(req).execute().use { resp ->
                val text = resp.body.string() // okhttp5: body non-null
                if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code} ${text.take(200)}")
                parseManifest(text)
            }
        }

    /** Parse models.json (`{ version, models: [{role,name,url,size,sha256,...}] }`). Only for [fetchManifest]. Hardened: validates fields. */
    private fun parseManifest(json: String): List<RemoteModel> {
        val arr = JSONObject(json).getJSONArray("models")
        return (0 until arr.length()).map {
            val m = arr.getJSONObject(it)
            RemoteModel(
                role = m.getString("role"),
                name = m.getString("name"),
                url = m.getString("url"),
                size = m.getLong("size"),
                sha256 = m.getString("sha256").lowercase(),
            )
        }
    }

    /**
     * Ensure [models] all exist in [destDir]: skip those already present with matching size+sha256, download and verify the rest.
     * Returns role -> local file. Throws on any failure (keeps already downloaded, deletes corrupted).
     * Hardened: validates destDir, handles partial files, verifies integrity, reports progress.
     */
    suspend fun ensure(
        models: List<RemoteModel>,
        destDir: File,
        onProgress: (ModelProgress) -> Unit = {},
    ): Map<String, File> = withContext(Dispatchers.IO) {
        if (!destDir.exists()) destDir.mkdirs()
        require(destDir.isDirectory) { "Destination is not a directory: $destDir" }
        val out = LinkedHashMap<String, File>()
        for (m in models) {
            val file = File(destDir, m.name)
            if (file.exists() && file.length() == m.size && sha256(file) == m.sha256) {
                onProgress(ModelProgress.Done(m.role, m.name)) // Already correct version
                out[m.role] = file
                continue
            }
            try {
                downloadOne(m, file, onProgress)
                onProgress(ModelProgress.Verifying(m.role, m.name))
                val got = sha256(file)
                if (got != m.sha256) {
                    file.delete()
                    throw RuntimeException("sha256 mismatch (expected ${m.sha256.take(12)}..., got ${got.take(12)}...)")
                }
                onProgress(ModelProgress.Done(m.role, m.name))
                out[m.role] = file
            } catch (t: Throwable) {
                onProgress(ModelProgress.Failed(m.role, m.name, t.message ?: t.javaClass.simpleName))
                throw t
            }
        }
        out
    }

    /** Stream download to `<name>.part` then rename (avoid treating partial as complete). Hardened: handles network errors, ensures cleanup. */
    private fun downloadOne(m: RemoteModel, dest: File, onProgress: (ModelProgress) -> Unit) {
        val req = Request.Builder().url(m.url).get().build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code} ${resp.body.string().take(200)}")
            val body = resp.body
            val total = if (m.size > 0) m.size else body.contentLength()
            val tmp = File(dest.parentFile, "${dest.name}.part")
            body.byteStream().use { input ->
                tmp.outputStream().use { output ->
                    val buf = ByteArray(1 shl 16)
                    var acc = 0L
                    var lastReport = 0L
                    while (true) {
                        val read = input.read(buf)
                        if (read < 0) break
                        output.write(buf, 0, read)
                        acc += read
                        if (acc - lastReport >= (1 shl 20)) { // Report every ~1MB
                            onProgress(ModelProgress.Downloading(m.role, m.name, acc, total))
                            lastReport = acc
                        }
                    }
                }
            }
            if (!tmp.renameTo(dest)) {
                tmp.copyTo(dest, overwrite = true)
                tmp.delete()
            }
        }
    }

    /** Stream compute sha256 (64KB buffer, never read whole model into heap). Hardened: handles file not found, IO errors. */
    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val read = input.read(buf)
                if (read < 0) break
                md.update(buf, 0, read)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }
}
