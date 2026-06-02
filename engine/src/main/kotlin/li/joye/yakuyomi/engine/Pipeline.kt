package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Typeface
import android.util.Log

/**
 * 單頁翻譯結果（§11：成功才覆蓋+marker、略過不覆蓋、失敗不覆蓋待重試）。
 * 引擎只回結果，不碰檔案——覆蓋/marker/resume 由呼叫端（下載 worker）依此處理（§3、§12.6）。
 */
sealed interface PageResult {
    /** 成功：可覆蓋原檔 + 寫「已翻譯」marker。 */
    data class Translated(val page: Bitmap, val stats: PageStats) : PageResult

    /** 沒東西可翻（偵測不到字 / OCR 全空 / 譯文全被過濾）：保留原圖、標記略過、**不覆蓋**。 */
    data class Skipped(val reason: String, val stats: PageStats) : PageResult

    /** 出錯（網路/429 重試後仍失敗/例外）：保留原圖、**不標記**、之後可重試。 */
    data class Failed(val reason: String) : PageResult
}

/** 逐階段計時與計數（除錯/效能用）。 */
data class PageStats(
    val lines: Int,
    val regions: Int,
    val kept: Int,
    val detectMs: Long,
    val ocrMs: Long,
    val translateMs: Long,
    val inpaintMs: Long,
    val renderMs: Long,
) {
    val totalMs: Long get() = detectMs + ocrMs + translateMs + inpaintMs + renderMs
}

/**
 * 引擎主 pipeline：單頁 偵測→OCR→分群→翻譯→過濾→去字→排版。
 * 順序對齊 manga_translator.py 主流程（§5 順序＝第一層）；orchestration＝第二層。
 *
 * **§11 不變式焊進此處：永不用比原圖更糟的東西覆蓋。**
 *   - 偵測不到字 / OCR 全空 / 譯文全被過濾 → [PageResult.Skipped]（保留原圖、不覆蓋）。
 *   - 單 block 翻譯失敗 → 留原文（[Translator] 失敗項回原文 + [TextFilter] 丟「譯==原」的區 ⇒ 該區不去字、保留日文），其餘照翻。
 *   - 任一階段拋例外（網路/429 重試後仍失敗等）→ [PageResult.Failed]（保留原圖、不覆蓋、可重試）。
 *
 * 模型由呼叫端建好傳入（BYOM 路徑/下載與 close 都歸呼叫端）；本類不碰檔案、不管跨頁批次與 resume。
 */
class Pipeline(
    private val detector: Detector,
    private val ocr: Ocr,
    private val translator: Translator?, // null＝不翻譯（純偵測/OCR 除錯用）
    private val inpainter: Inpainter,
    private val cfg: EngineConfig = EngineConfig(),
    private val typeface: Typeface? = null,
) : TranslationEngine {

    override suspend fun translatePage(page: Bitmap): PageResult {
        // 偵測
        val tDet = System.currentTimeMillis()
        val detection = try {
            detector.detect(page)
        } catch (t: Throwable) {
            Log.e(TAG, "偵測失敗", t); return PageResult.Failed("detect: ${t.message}")
        }
        val lines = detection.lines
        val detectMs = System.currentTimeMillis() - tDet
        if (lines.isEmpty()) {
            return PageResult.Skipped("偵測不到文字", PageStats(0, 0, 0, detectMs, 0, 0, 0, 0))
        }

        // OCR + 分群
        val tOcr = System.currentTimeMillis()
        try {
            ocr.recognize(page, lines)
        } catch (t: Throwable) {
            Log.e(TAG, "OCR 失敗", t); return PageResult.Failed("ocr: ${t.message}")
        }
        val regions = Grouping.group(lines)
        val ocrMs = System.currentTimeMillis() - tOcr
        if (regions.isEmpty() || regions.all { it.sourceText.isBlank() }) {
            return PageResult.Skipped("OCR 全空", PageStats(lines.size, regions.size, 0, detectMs, ocrMs, 0, 0, 0))
        }

        // 翻譯（per-block fallback：失敗項留原文；整批例外＝網路/429 → Failed、不覆蓋）
        var translateMs = 0L
        if (translator != null) {
            val tTr = System.currentTimeMillis()
            val cht = try {
                translator.translate(regions.map { it.sourceText })
            } catch (t: Throwable) {
                Log.e(TAG, "翻譯失敗", t); return PageResult.Failed("translate: ${t.message}")
            }
            regions.forEachIndexed { j, r -> r.translatedText = cht.getOrElse(j) { r.sourceText } }
            translateMs = System.currentTimeMillis() - tTr
        }

        // 過濾：空白/數字/regex/譯==原 → 丟（未譯或誤判的區不去字、保留原圖該處）
        val kept = if (translator != null) TextFilter.apply(regions, cfg.translator.filterText) else regions
        if (kept.isEmpty()) {
            return PageResult.Skipped(
                "無有效譯文（全數過濾）",
                PageStats(lines.size, regions.size, 0, detectMs, ocrMs, translateMs, 0, 0),
            )
        }

        // 去字
        val tIn = System.currentTimeMillis()
        val cleaned = try {
            inpainter.inpaint(page, kept, detection.textMask)
        } catch (t: Throwable) {
            Log.e(TAG, "去字失敗", t); return PageResult.Failed("inpaint: ${t.message}")
        }
        val inpaintMs = System.currentTimeMillis() - tIn

        // 排版
        val tRn = System.currentTimeMillis()
        val finalPage = Renderer.render(cleaned, kept, cfg.render, typeface)
        val renderMs = System.currentTimeMillis() - tRn

        return PageResult.Translated(
            finalPage,
            PageStats(lines.size, regions.size, kept.size, detectMs, ocrMs, translateMs, inpaintMs, renderMs),
        )
    }

    companion object {
        private const val TAG = "Pipeline"
    }
}
