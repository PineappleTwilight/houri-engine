package li.joye.yakuyomi.engine

import android.graphics.Bitmap

/**
 * 引擎對外主介面（CLAUDE.md §6「引擎解耦」、§13）。**建議用 [Yakuyomi.create] 取得實例**；唯一實作＝[Pipeline]。
 *
 * 回傳 [PageResult] 而非裸 Bitmap：要能表達 §11 的「略過（不覆蓋）／失敗（不覆蓋、可重試）／成功（可覆蓋+marker）」，
 * 裸 Bitmap 無法區分這三態。覆蓋原檔／寫 marker／page-level resume／跨頁批次併發＝呼叫端職責（§3）。
 *
 * **生命週期**：持有原生 ONNX session，用完要 [close] 釋放 native 記憶體。建議搭 `use { }`：
 * ```
 * Yakuyomi.create(models, alphabet, apiKey).use { engine ->
 *     when (val r = engine.translatePage(bitmap)) {
 *         is PageResult.Translated -> writeBack(r.page)
 *         is PageResult.Skipped    -> { /* 保留原圖、不覆蓋 */ }
 *         is PageResult.Failed     -> { /* 保留原圖、可重試 */ }
 *     }
 * }
 * ```
 * **執行緒**：[translatePage] 是 suspend、一次處理一頁，請在背景 dispatcher 呼叫；單一實例**非並發安全**（勿同實例同時翻多頁）。
 */
interface TranslationEngine : AutoCloseable {
    /**
     * 翻譯單頁。不會 recycle 輸入 [page]；成功時 [PageResult.Translated.page] 是**另一個新 bitmap**（非原物件）。
     * @param page 來源頁點陣圖（所有權仍屬呼叫端）。
     */
    suspend fun translatePage(page: Bitmap): PageResult

    /** 釋放底層模型的原生資源（detector/ocr/inpainter 的 ONNX session）。呼叫後此實例不可再用。 */
    override fun close()
}
