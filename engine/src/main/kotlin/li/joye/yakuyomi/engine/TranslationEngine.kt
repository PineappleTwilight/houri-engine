package li.joye.yakuyomi.engine

import android.graphics.Bitmap

/**
 * Main engine external interface (CLAUDE.md §6 "engine decoupling", §13). **Use [Yakuyomi.create] to get instance**; sole implementation = [Pipeline].
 *
 * Returns [PageResult] instead of bare Bitmap: needs to express §11's "Skipped (do not overwrite) / Failed (do not overwrite, retryable) / Translated (can overwrite + marker)",
 * bare Bitmap cannot distinguish these three states. Overwriting original file / writing marker / page-level resume / cross-page batch concurrency = caller responsibility (§3).
 *
 * **Lifecycle**: holds native ONNX sessions, must [close] to release native memory. Recommended with `use { }`:
 * ```
 * Yakuyomi.create(models, alphabet, apiKey).use { engine ->
 *     when (val r = engine.translatePage(bitmap)) {
 *         is PageResult.Translated -> writeBack(r.page)
 *         is PageResult.Skipped    -> { /* keep original, do not overwrite */ }
 *         is PageResult.Failed     -> { /* keep original, retryable */ }
 *     }
 * }
 * ```
 * **Threading**: [translatePage] is suspend, handles one page at a time, call on background dispatcher; single instance is **not concurrent-safe** (do not translate multiple pages concurrently on same instance).
 * Hardened: validates input bitmap, handles recycled, ensures native resources are released even on failure.
 */
interface TranslationEngine : AutoCloseable {
    /**
     * Translate single page. Will not recycle input [page]; on success [PageResult.Translated.page] is **another new bitmap** (not original object).
     * Hardened: validates bitmap not recycled, size within bounds, handles OOM gracefully.
     * @param page Source page bitmap (ownership remains with caller).
     */
    suspend fun translatePage(page: Bitmap): PageResult

    /**
     * Single-threaded warmup: run one inference for each native session (detector / OCR / inpaint) to complete first lazy initialization (weight preprocessing / arena setup etc.).
     * **Call once before concurrent multi-page translation** (single-threaded) — otherwise multiple pages hitting "just loaded, never inferred" cold session simultaneously will race first initialization -> native crash.
     * Call once after construction (see fork `TranslationEngineService.ensureEngine`); boxfill inpaint does not run its session, so no need to warm it. Best-effort.
     */
    fun warmUp()

    /** Release native resources of underlying models (detector/ocr/inpainter ONNX sessions). After calling, this instance cannot be reused. Hardened: safe to call multiple times. */
    override fun close()
}
