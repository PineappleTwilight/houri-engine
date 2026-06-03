package li.joye.yakuyomi.engine

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
        val detector = Detector(models.detector, config.detector)
        val ocr = Ocr(models.ocr, alphabet, config.ocr)
        val inpainter = Inpainter(models.inpainter, config.inpainter)
        val translator = apiKey?.takeIf { it.isNotBlank() }?.let { LlmTranslator(it, config.translator) }
        return Pipeline(detector, ocr, translator, inpainter, config, typeface)
    }
}
