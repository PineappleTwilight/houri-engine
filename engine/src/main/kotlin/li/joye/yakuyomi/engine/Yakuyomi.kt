package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Typeface

/**
 * Engine entry factory: hides "build three model components + assemble [Pipeline]" into one line, returns [TranslationEngine] usable with `use { }`.
 *
 * Replaces repetitive code where each consumer manually assembles and remembers to `close()`:
 * ```
 * val models = ModelSet.resolve(localModelFiles) ?: return   // Models not ready -> skip
 * Yakuyomi.create(models, alphabet, apiKey).use { engine ->
 *     for (bmp in pages) when (val r = engine.translatePage(bmp)) {
 *         is PageResult.Translated -> writeBack(r.page)
 *         is PageResult.Skipped    -> { /* keep original */ }
 *         is PageResult.Failed     -> { /* keep original, retryable */ }
 *     }
 * }
 * ```
 * **Advanced** (per-component debugging, e.g., debug overlay): can directly new [Detector]/[Ocr]/[Inpainter]/[LlmTranslator] and assemble [Pipeline] yourself,
 * but lifecycle must be managed manually (only this factory path will close for you).
 * Hardened: validates models and alphabet, handles missing API key gracefully.
 */
object Yakuyomi {
    /**
     * Build a translation engine.
     *
     * @param models   Local paths of three models (see [ModelSet]; use [ModelSet.resolve] to match by filename).
     * @param alphabet OCR alphabet (for 48px CTC decoding; usually loaded from engine assets).
     * @param apiKey   Translation LLM API key; **null/blank = no translation** (only detection/OCR/inpaint, for debugging).
     * @param config   Engine config (all tunable, defaults see each `*Config`).
     * @param typeface Rendering typeface; null = system default CJK.
     * @return [TranslationEngine] usable with `use { }`; its [TranslationEngine.close] will release native sessions of the three models.
     * Hardened: validates inputs, checks NCNN availability before loading.
     */
    fun create(
        models: ModelSet,
        alphabet: List<String>,
        apiKey: String?,
        config: EngineConfig = EngineConfig(),
        typeface: Typeface? = null,
    ): TranslationEngine {
        require(alphabet.isNotEmpty()) { "Alphabet must not be empty" }
        // Detection + inpaint are pure NCNN (product arm64, NCNN required; ORT fallback and LaMa retired).
        check(NcnnBackend.available) { "NCNN native library not loaded (arm64 should be available)" }
        EngineTrace.log("create.detector")
        val detector = Detector(models.detectorNcnn ?: error("Requires NCNN detection model (.param)"), config.detector)
        EngineTrace.log("create.ocr")
        val ocr = Ocr(models.ocr, alphabet, config.ocr)
        // Both inpaint methods (boxfill/aot) use same NCNN AOT model (boxfill only flat-fills, does not run it, but still needs to be loadable).
        EngineTrace.log("create.inpainter")
        val inpainter = Inpainter(models.aotInpainterNcnn ?: error("Requires NCNN AOT inpaint model (.param)"), config.inpainter)
        val translator = apiKey?.takeIf { it.isNotBlank() }?.let { LlmTranslator(it, config.translator) }
        EngineTrace.log("create.done")
        return Pipeline(detector, ocr, translator, inpainter, config, typeface)
    }

    /**
     * Diagnostics (for sandbox): run detection on one page -> each line OCR with bilinear / bicubic crop, return per-line read comparison + each recognize timing.
     * Proves on device that "bicubic preprocessing restores small kana blurred by scaling (sentence-ending negation -> meaning inversion)" in quality and cost.
     * Both share same ORT session (only crop interpolation differs); warm up both paths before timing (session lazy init + JIT) for reliable numbers.
     */
    suspend fun ocrAbTest(
        models: ModelSet,
        alphabet: List<String>,
        page: Bitmap,
        config: EngineConfig = EngineConfig(),
    ): OcrAbResult {
        check(NcnnBackend.available) { "NCNN native library not loaded" }
        val detector = Detector(models.detectorNcnn ?: error("Requires NCNN detection model (.param)"), config.detector)
        val ocr = Ocr(models.ocr, alphabet, config.ocr)
        try {
            val tDet = System.nanoTime()
            val det = detector.detect(page)
            val detectMs = (System.nanoTime() - tDet) / 1e6
            val clone = { det.lines.map { TextLine(it.quad, it.score) } } // recognize writes text in place -> use fresh copy each run
            // Warm up both paths (not timed): session first run lazy init + bicubic loop JIT
            ocr.warmUp()
            ocr.recognize(page, clone(), bicubic = false)
            ocr.recognize(page, clone(), bicubic = true)
            // Formal timing
            val linesBil = clone()
            val t0 = System.nanoTime()
            ocr.recognize(page, linesBil, bicubic = false)
            val bilinearMs = (System.nanoTime() - t0) / 1e6
            val linesBic = clone()
            val t1 = System.nanoTime()
            ocr.recognize(page, linesBic, bicubic = true)
            val bicubicMs = (System.nanoTime() - t1) / 1e6
            val rows = linesBil.indices.map { OcrAbRow(linesBil[it].text, linesBic[it].text) }
            return OcrAbResult(rows, bilinearMs, bicubicMs, detectMs)
        } finally {
            runCatching { detector.close() }
            runCatching { ocr.close() }
        }
    }
}

/** Result of [Yakuyomi.ocrAbTest]: per-line bilinear vs bicubic OCR reads [rows] + each interpolation total recognize time (ms) + detection time. */
class OcrAbResult(
    val rows: List<OcrAbRow>,
    val bilinearMs: Double,
    val bicubicMs: Double,
    val detectMs: Double,
)

/** Single line comparison: same line box, [bilinear] and [bicubic] crops each OCR read text (empty = dropped below confidence threshold). */
class OcrAbRow(val bilinear: String, val bicubic: String)
