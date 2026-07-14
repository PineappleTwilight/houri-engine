package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Typeface

/**
 * 引擎入口工廠：把「建三顆模型元件 + 組 [Pipeline]」收成一行，回傳可 `use { }` 的 [TranslationEngine]。
 *
 * 取代各消費端各自手動拼裝 + 各自記得 `close()` 的重複碼：
 * ```
 * val models = ModelSet.resolve(localModelFiles) ?: return   // 模型沒備齊 → 略過
 * Yakuyomi.create(models, alphabet, apiKey).use { engine ->
 *     for (bmp in pages) when (val r = engine.translatePage(bmp)) {
 *         is PageResult.Translated -> writeBack(r.page)
 *         is PageResult.Skipped    -> { /* 保留原圖 */ }
 *         is PageResult.Failed     -> { /* 保留原圖、可重試 */ }
 *     }
 * }
 * ```
 * **進階**（逐元件除錯，如 debug overlay）：可直接 new [Detector]/[Ocr]/[Inpainter]/[LlmTranslator] 再自組 [Pipeline]，
 * 但生命週期得自己管（這條工廠路徑才會幫你 close）。
 */
object Yakuyomi {
    /**
     * 建一個翻譯引擎。
     *
     * @param models   三顆模型的本機路徑（見 [ModelSet]；用 [ModelSet.resolve] 從檔名比對）。
     * @param alphabet OCR 字元表（48px CTC 解碼用；通常由引擎 assets 載入後傳入）。
     * @param apiKey   翻譯 LLM 的 API key；**null/空白＝不翻譯**（只跑偵測/OCR/去字，純除錯）。
     * @param config   引擎設定（全可調，預設見各 `*Config`）。
     * @param typeface 算繪字型；null＝系統預設 CJK。
     * @return 可 `use { }` 的 [TranslationEngine]；其 [TranslationEngine.close] 會釋放三顆模型的 native session。
     */
    fun create(
        models: ModelSet,
        alphabet: List<String>,
        apiKey: String?,
        config: EngineConfig = EngineConfig(),
        typeface: Typeface? = null,
    ): TranslationEngine {
        // 偵測 + 去字皆純 NCNN（產品 arm64、NCNN 必在；ORT 備援與 LaMa 已退役移除）。
        check(NcnnBackend.available) { "NCNN 原生庫未載入（arm64 應可用）" }
        EngineTrace.log("create.detector")
        val detector = Detector(models.detectorNcnn ?: error("需 NCNN 偵測模型（.param）"), config.detector)
        EngineTrace.log("create.ocr")
        val ocr = Ocr(models.ocr, alphabet, config.ocr)
        // 去字兩門別（boxfill/aot）皆用同一顆 NCNN AOT 模型（boxfill 只平塗不跑它、但仍要載得起來）。
        EngineTrace.log("create.inpainter")
        val inpainter = Inpainter(models.aotInpainterNcnn ?: error("需 NCNN AOT 去字模型（.param）"), config.inpainter)
        val translator = apiKey?.takeIf { it.isNotBlank() }?.let { LlmTranslator(it, config.translator) }
        EngineTrace.log("create.done")
        return Pipeline(detector, ocr, translator, inpainter, config, typeface)
    }

    /**
     * 診斷（sandbox 用）：對一頁跑偵測 → 每行分別以 bilinear / bicubic 裁切做 OCR，回逐行讀取對照 + 各自 recognize 耗時。
     * 真機 A/B 驗證「bicubic 前處理救回被縮放糊掉的小假名（句尾否定→意思相反）」的品質提升與效能代價。
     * 兩者共用同一 ORT session（只差裁切內插法）；計時前先暖跑兩條路徑（session lazy init + JIT），數字才可靠。
     */
    suspend fun ocrAbTest(
        models: ModelSet,
        alphabet: List<String>,
        page: Bitmap,
        config: EngineConfig = EngineConfig(),
    ): OcrAbResult {
        check(NcnnBackend.available) { "NCNN 原生庫未載入" }
        val detector = Detector(models.detectorNcnn ?: error("需 NCNN 偵測模型（.param）"), config.detector)
        val ocr = Ocr(models.ocr, alphabet, config.ocr)
        try {
            val tDet = System.nanoTime()
            val det = detector.detect(page)
            val detectMs = (System.nanoTime() - tDet) / 1e6
            val clone = { det.lines.map { TextLine(it.quad, it.score) } } // recognize 就地寫 text → 每次跑用新副本
            // 暖跑兩條路徑（不計時）：session 首次 run 的 lazy init + bicubic 迴圈 JIT
            ocr.warmUp()
            ocr.recognize(page, clone(), bicubic = false)
            ocr.recognize(page, clone(), bicubic = true)
            // 正式計時
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

/** [Yakuyomi.ocrAbTest] 結果：逐行 bilinear vs bicubic OCR 讀取 [rows] + 各內插法 recognize 總耗時（ms）+ 偵測耗時。 */
class OcrAbResult(
    val rows: List<OcrAbRow>,
    val bilinearMs: Double,
    val bicubicMs: Double,
    val detectMs: Double,
)

/** 單行對照：同一行框，[bilinear] 與 [bicubic] 裁切各自 OCR 讀出的文字（空＝低於信心門檻被丟）。 */
class OcrAbRow(val bilinear: String, val bicubic: String)
