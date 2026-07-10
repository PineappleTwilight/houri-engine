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
        // 偵測：NCNN 為主（手機 CPU ~3.7× ORT；固定 1024 輸入、無 ncnn 跨尺寸崩問題），ORT 僅在 NCNN 不可用時備援。
        val detectorModel = if (models.detectorNcnn != null && NcnnBackend.available) {
            models.detectorNcnn
        } else {
            models.detector ?: error("需偵測模型（NCNN .param 或 ORT .onnx）")
        }
        val detector = Detector(detectorModel, config.detector)
        val ocr = Ocr(models.ocr, alphabet, config.ocr)
        // method="aot"/"auto_aot" 用 AOT-GAN 模型（I/O 契約與 LaMa 不同、Inpainter 依 method 分流）；缺 aot 檔則退回 LaMa 模型。
        val aotMethod = config.inpainter.method == "aot" || config.inpainter.method == "auto_aot"
        // AOT 去字有 NCNN 版且**整頁模式**才走 NCNN（固定 512、同尺寸復用安全）；逐格(wholeImage=false)變動尺寸 → ncnn net 崩 → 留 ORT。
        val useNcnnAot = aotMethod && config.inpainter.wholeImage &&
            models.aotInpainterNcnn != null && NcnnBackend.available
        // 去字模型解析：AOT 為主、LaMa 退役後選配。缺對應模型時 loud-error（別靜默跑錯）。
        //   任一去字模型（NCNN AOT 優先）＝可載的 fallback（boxfill 不跑模型、只要載得起來；I/O 契約各 method 自行分流）。
        val anyInpaint = models.aotInpainterNcnn ?: models.aotInpainter ?: models.inpainter
        val inpainterModel = when {
            useNcnnAot -> models.aotInpainterNcnn!!
            aotMethod -> models.aotInpainter ?: models.aotInpainterNcnn ?: models.inpainter
                ?: error("去字方法 '${config.inpainter.method}' 需 AOT 模型（未備齊）")
            config.inpainter.method == "boxfill" -> anyInpaint // boxfill 只平塗、不跑去字模型，載任一在手的
                ?: error("需去字模型（未備齊）")
            else -> models.inpainter // lama/auto（已退役）：有 LaMa 才行，否則 loud-error
                ?: error("去字方法 '${config.inpainter.method}' 需 LaMa 模型，但 LaMa 已退役——請改用 aot/auto_aot/boxfill")
        }
        val inpainter = Inpainter(inpainterModel, config.inpainter)
        val translator = apiKey?.takeIf { it.isNotBlank() }?.let { LlmTranslator(it, config.translator) }
        return Pipeline(detector, ocr, translator, inpainter, config, typeface)
    }
}
