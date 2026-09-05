package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Typeface
import android.util.Log
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.coroutineScope

/**
 * Single-page translation result (§11: Translated overwrites + marker, Skipped keeps original with marker, Failed keeps original without marker for retry).
 * Engine only returns result, never touches files — overwriting/marker/resume handled by caller (download worker) (§3, §12.6).
 */
sealed interface PageResult {
    /** Success: can overwrite original file + write "translated" marker. */
    data class Translated(val page: Bitmap, val stats: PageStats, val analysis: PageAnalysis? = null) : PageResult

    /** Nothing to translate (no text detected / OCR empty / all translations filtered): keep original, mark skipped, **do not overwrite**. */
    data class Skipped(val reason: String, val stats: PageStats) : PageResult

    /** Error (network/429 after retries/exception): keep original, **no marker**, retry later. */
    data class Failed(val reason: String) : PageResult
}

/**
 * Redraw material (for "lowest-cost inpaint method switch"): seg text mask + regions (with quad/angle/onArt/source/translated).
 * Original image held by caller (translatePage input), inpaint method decided by caller, not here.
 * Serialization/persistence (including mask-to-text JSON) handled by caller (reader).
 */
data class PageAnalysis(val mask: Bitmap, val regions: List<TextRegion>)

/** Per-stage timing and counts (for debugging / performance). */
data class PageStats(
    val lines: Int,
    val regions: Int,
    val kept: Int,
    val detectMs: Long,
    val ocrMs: Long,
    val translateMs: Long,
    val inpaintMs: Long,
    val renderMs: Long,
    val wallMs: Long = 0,   // Actual wall-clock time (inpaint || translate overlap => usually < totalMs; saved = overlapped)
    val promptTokens: Int = 0,      // Prompt tokens for this page's LLM request (0 if no LLM/proxy). For stats: counted, not billed.
    val completionTokens: Int = 0,  // Completion tokens for this page's LLM request.
) {
    /** Sum of pure compute time for each stage (without overlap correction); actual elapsed see [wallMs]. */
    val totalMs: Long get() = detectMs + ocrMs + translateMs + inpaintMs + renderMs
}

/**
 * Main engine pipeline: single page detection -> OCR -> grouping -> translation -> filtering -> inpainting -> typesetting.
 * Order aligns with manga_translator.py main flow (§5 order = first layer); orchestration = second layer.
 *
 * **§11 Invariant baked in here: never overwrite original with something worse.**
 *   - No text detected / OCR empty / all translations failed -> [PageResult.Skipped] (keep original, do not overwrite; discard inpaint if already running).
 *   - Single block translation failed -> **re-paste OCR source text (Japanese) after inpaint** — not "keep source image". Allows inpaint to run without waiting for translation => inpaint (CPU) and translation (network) overlap.
 *   - Any stage throws (network/429 after retries etc.) -> [PageResult.Failed] (keep original, do not overwrite, retry later; discard concurrent inpaint).
 *
 * **Inpaint || Translation overlap**: both depend only on OCR, no resource contention (network vs CPU) => run concurrently, [PageStats.wallMs] < [PageStats.totalMs] (saved = overlapped).
 *
 * Models are built and passed in by caller; this class never touches files nor handles cross-page batching or resume.
 * **Lifecycle**: [close] will release the native sessions of detector/ocr/inpainter passed in —
 * when going through [Yakuyomi.create] these three are built by factory and owned by this pipeline, `use { }` is enough.
 * Advanced: if you inject "reusable, shared" components, manage lifecycle yourself, do not call [close] here (otherwise shared components will be closed together).
 */
class Pipeline(
    private val detector: Detector,
    private val ocr: Ocr,
    private val translator: Translator?, // null = no translation (pure detection/OCR debugging)
    private val inpainter: Inpainter,
    private val cfg: EngineConfig = EngineConfig(),
    private val typeface: Typeface? = null,
) : TranslationEngine {

    override suspend fun translatePage(page: Bitmap): PageResult = coroutineScope {
        val tWall = System.currentTimeMillis()
        // Hardening: input validation - prevent destructive processing of empty/recycled bitmap
        if (page.isRecycled || page.width < 32 || page.height < 32 || page.width > 8000 || page.height > 8000) {
            return@coroutineScope PageResult.Failed("invalid page bitmap ${page.width}x${page.height} recycled=${page.isRecycled}")
        }
        // Pre-scale extremely large pages to protect memory and inpaint tiles
        val workPage = if (page.width > 4000 || page.height > 4000) {
            val scale = 4000f / maxOf(page.width, page.height)
            val nw = (page.width * scale).toInt().coerceAtLeast(32)
            val nh = (page.height * scale).toInt().coerceAtLeast(32)
            try { Bitmap.createScaledBitmap(page, nw, nh, true) } catch (_: Throwable) { page }
        } else page
        EngineTrace.log("pipe.page.enter ${workPage.width}x${workPage.height}")
        // Detection - hardened with timeout
        val tDet = System.currentTimeMillis()
        val detection = try {
            // 15s timeout to prevent Detector hang (NCNN occasional hang)
            kotlinx.coroutines.withTimeout(15000) { detector.detect(workPage) }
        } catch (t: Throwable) {
            Log.e(TAG, "Detection failed", t); return@coroutineScope PageResult.Failed("detect: ${t.message}")
        }
        val lines = detection.lines
        EngineTrace.log("pipe.detect.done lines=${lines.size}")
        val detectMs = System.currentTimeMillis() - tDet
        if (lines.isEmpty()) {
            return@coroutineScope PageResult.Skipped("No text detected", PageStats(0, 0, 0, detectMs, 0, 0, 0, 0))
        }

        // OCR + grouping - hardened with timeout and fallback
        val tOcr = System.currentTimeMillis()
        EngineTrace.log("pipe.ocr.enter lines=${lines.size}")
        try {
            // 12s timeout for OCR (ONNX occasional hang on tiny crops)
            kotlinx.coroutines.withTimeout(12000) { ocr.recognize(page, lines) }
        } catch (t: Throwable) {
            Log.e(TAG, "OCR failed", t); return@coroutineScope PageResult.Failed("ocr: ${t.message}")
        }
        EngineTrace.log("pipe.ocr.exit")
        val regions = try {
            Grouping.group(lines)
        } catch (t: Throwable) {
            Log.e(TAG, "Grouping failed", t); return@coroutineScope PageResult.Failed("group: ${t.message}")
        }
        val ocrMs = System.currentTimeMillis() - tOcr
        // Inpaint set = regions with non-blank OCR source text (blank = likely false detection, skip inpaint to preserve image). Determined before translation => inpaint can run concurrently with translation.
        val textRegions = regions.filter { it.sourceText.isNotBlank() }
        if (textRegions.isEmpty()) {
            return@coroutineScope PageResult.Skipped("OCR empty", PageStats(lines.size, regions.size, 0, detectMs, ocrMs, 0, 0, 0))
        }

        // Inpaint (CPU) || Translation (network) concurrent: both depend only on OCR, can run simultaneously (CPU inpaint while waiting for network).
        // §11 change: failed regions no longer "keep source image", but "re-paste OCR source text after inpaint" (user decision: re-pasting is cheap, no source image state needed) => inpaint and translation decoupled.
        // Any inpaint method (boxfill/lama) can overlap. boxfill (~0.5s) fully hidden under translation (~2.7s) = free.
        var inpaintMs = 0L
        val inpaintJob = async {
            val t0 = System.currentTimeMillis()
            EngineTrace.log("pipe.inpaint.enter regions=${textRegions.size}")
            val r = inpainter.inpaint(page, textRegions, detection.textMask, cfg.render)
            EngineTrace.log("pipe.inpaint.exit")
            inpaintMs = System.currentTimeMillis() - t0
            r
        }

        var translateMs = 0L
        var promptTok = 0
        var completionTok = 0
        // Per-call diagnostic metadata (local variables, not reading LlmTranslator's shared singleton fields -> safe for concurrent pages, each page gets its own, no race).
        var llmError: String? = null
        var llmRaw: String? = null
        if (translator != null) {
            val tTr = System.currentTimeMillis()
            EngineTrace.log("pipe.translate.enter n=${textRegions.size}")
            val cht = try {
                val llm = translator as? LlmTranslator
                if (llm != null) {
                    // Use translateDetailed -> per-call get translations + usage + error + raw (safe for concurrent pages).
                    // Hardened: retry once on transient 429/5xx with backoff
                    var lastErr: Throwable? = null
                    var result: LlmTranslator.TranslateResult? = null
                    repeat(2) { attempt ->
                        try {
                            val r = llm.translateDetailed(textRegions.map { it.sourceText })
                            result = r
                            return@repeat
                        } catch (t: Throwable) {
                            lastErr = t
                            val msg = t.message ?: ""
                            val isTransient = msg.contains("429") || msg.contains("503") || msg.contains("timeout", true)
                            if (isTransient && attempt == 0) {
                                kotlinx.coroutines.delay(800)
                            } else throw t
                        }
                    }
                    val r = result ?: throw lastErr ?: IllegalStateException("translation failed")
                    r.usage?.let { promptTok = it.promptTokens; completionTok = it.completionTokens }
                    llmError = r.error
                    llmRaw = r.raw
                    r.translations
                } else {
                    // Hardened: also retry for plain Translator
                    var lastErr: Throwable? = null
                    var res: List<String>? = null
                    repeat(2) { attempt ->
                        try {
                            res = translator.translate(textRegions.map { it.sourceText })
                            return@repeat
                        } catch (t: Throwable) {
                            lastErr = t
                            if (attempt == 0) kotlinx.coroutines.delay(500) else throw t
                        }
                    }
                    res ?: throw lastErr!!
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Translation failed", t)
                inpaintJob.cancelAndJoin() // Translation failed -> discard inpaint, keep original ( §11; native run cannot be interrupted, cancel actually waits then discards)
                return@coroutineScope PageResult.Failed("translate: ${t.message}")
            }
            textRegions.forEachIndexed { j, r -> r.translatedText = cht.getOrElse(j) { r.sourceText } }
            translateMs = System.currentTimeMillis() - tTr
            EngineTrace.log("pipe.translate.exit err=$llmError")
        } else {
            textRegions.forEach { it.translatedText = it.sourceText } // No key debug: typeset original text
        }

        // Determine per-region translation validity (blank/numeric/regex/translated==source = failure). Whole page failed -> keep original (Skipped, discard inpaint).
        val kept = if (translator != null) TextFilter.apply(textRegions, cfg.translator.filterText) else textRegions
        if (kept.isEmpty()) {
            val aligned = textRegions.count { it.translatedText.isNotBlank() && it.translatedText != it.sourceText }
            val dbg = textRegions.take(2).joinToString(" | ") { "${it.sourceText.take(8)}->${it.translatedText.take(8)}" }
            Log.w(TAG, "All filtered aligned=$aligned/${textRegions.size} err=$llmError raw=$llmRaw")
            inpaintJob.cancelAndJoin()
            // §11 blind spot fix: distinguish "network/format soft failure" vs "truly untranslatable".
            // LlmTranslator catches network/HTTP exceptions and "returns source text" (no throw) -> whole page translated==source -> lands here as all filtered.
            // If always return Skipped (marked as handled), network-failed pages would be considered "translated" and chapter would not turn red (exactly this blind spot). Use error split (per-call, no race):
            //  - error != null (exception [network/HTTP] or partial parse) -> Failed: no marker, retry later, whole chapter red (caller drain marks ERROR).
            //  - error == null (LLM normally parsed but all content filtered, e.g., whole page sound effects returned as translated==source) -> Skipped: skip, do not retry infinitely.
            return@coroutineScope if (llmError != null) {
                PageResult.Failed("All filtered (LLM failed $llmError) | raw=${llmRaw?.take(80)}")
            } else {
                PageResult.Skipped(
                    "All filtered aligned=$aligned/${textRegions.size} | raw=$llmRaw | $dbg",
                    PageStats(
                        lines.size, regions.size, 0, detectMs, ocrMs, translateMs, 0, 0,
                        promptTokens = promptTok, completionTokens = completionTok,
                    ),
                )
            }
        }
        // Failed regions (not in kept) -> revert translation to source = re-paste OCR Japanese after inpaint (TextRegion has no equals override -> HashSet uses identity).
        val keptSet = kept.toHashSet()
        textRegions.forEach { if (it !in keptSet) it.translatedText = it.sourceText }

        // Wait for inpaint to complete (mostly already overlapped with translation)
        EngineTrace.log("pipe.inpaint.await")
        val cleaned = try {
            inpaintJob.await()
        } catch (t: Throwable) {
            Log.e(TAG, "Inpaint failed", t); return@coroutineScope PageResult.Failed("inpaint: ${t.message}")
        }

        // Typesetting (all textRegions: kept with translated text, failed regions with original)
        val tRn = System.currentTimeMillis()
        EngineTrace.log("pipe.render.enter")
        val finalPage = try {
            Renderer.render(cleaned, textRegions, cfg.render, typeface)
        } catch (t: Throwable) {
            Log.e(TAG, "Render failed", t); return@coroutineScope PageResult.Failed("render: ${t.message}")
        }
        val renderMs = System.currentTimeMillis() - tRn
        EngineTrace.log("pipe.page.done")

        PageResult.Translated(
            finalPage,
            PageStats(
                lines.size, regions.size, kept.size, detectMs, ocrMs, translateMs, inpaintMs, renderMs,
                System.currentTimeMillis() - tWall, promptTok, completionTok,
            ),
            PageAnalysis(detection.textMask, textRegions),
        )
    }

    /**
     * Single-threaded warmup: run one inference for each native session of detector / OCR / inpainter to complete first lazy initialization.
     * Call once after construction, before allowing cross-page concurrency (see interface docs). Three in order (single thread), best-effort (each catch).
     */
    override fun warmUp() {
        EngineTrace.log("warmup.detector.enter")
        runCatching { detector.warmUp() }
        EngineTrace.log("warmup.ocr.enter")
        runCatching { ocr.warmUp() }
        EngineTrace.log("warmup.inpainter.enter")
        runCatching { inpainter.warmUp() }
        EngineTrace.log("warmup.done")
    }

    /** Release native ONNX sessions of detector/ocr/inpainter (see class lifecycle notes). */
    override fun close() {
        runCatching { detector.close() }
        runCatching { ocr.close() }
        runCatching { inpainter.close() }
    }

    companion object {
        private const val TAG = "Pipeline"
    }
}
