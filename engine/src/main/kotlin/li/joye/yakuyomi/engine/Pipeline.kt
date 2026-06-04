package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Typeface
import android.util.Log
import kotlinx.coroutines.async
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.coroutineScope

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
    val wallMs: Long = 0,   // 實際牆鐘時間（去字‖翻譯重疊 ⇒ 通常 < totalMs；省下的＝重疊掉的）
) {
    /** 各階段純計算時間之和（不含重疊修正）；實際耗時看 [wallMs]。 */
    val totalMs: Long get() = detectMs + ocrMs + translateMs + inpaintMs + renderMs
}

/**
 * 引擎主 pipeline：單頁 偵測→OCR→分群→翻譯→過濾→去字→排版。
 * 順序對齊 manga_translator.py 主流程（§5 順序＝第一層）；orchestration＝第二層。
 *
 * **§11 不變式焊進此處：永不用比原圖更糟的東西覆蓋。**
 *   - 偵測不到字 / OCR 全空 / 譯文全失敗 → [PageResult.Skipped]（保留原圖、不覆蓋；去字若已並發跑出來也丟棄）。
 *   - 單 block 翻譯失敗 → **去字後重貼 OCR 原文**（日文）——非「保留源圖」。讓去字不必等翻譯 ⇒ 去字(CPU)與翻譯(網路)並發重疊。
 *   - 任一階段拋例外（網路/429 重試後仍失敗等）→ [PageResult.Failed]（保留原圖、不覆蓋、可重試；丟棄已並發的去字）。
 *
 * **去字‖翻譯重疊**：兩者只依賴 OCR、互不爭資源（網路 vs CPU）⇒ 同時跑，[PageStats.wallMs] < [PageStats.totalMs]（省下重疊掉的）。
 *
 * 模型由呼叫端建好傳入；本類不碰檔案、不管跨頁批次與 resume。
 * **生命週期**：[close] 會收掉傳入的 detector/ocr/inpainter 的原生 session ——
 * 走 [Yakuyomi.create] 時這三顆由工廠建、歸本 pipeline 所有，`use { }` 即可。
 * 進階：若你注入「想重用、共享」的元件，請自己管生命週期、別呼叫本 [close]（否則會把共享元件一起關掉）。
 */
class Pipeline(
    private val detector: Detector,
    private val ocr: Ocr,
    private val translator: Translator?, // null＝不翻譯（純偵測/OCR 除錯用）
    private val inpainter: Inpainter,
    private val cfg: EngineConfig = EngineConfig(),
    private val typeface: Typeface? = null,
) : TranslationEngine {

    override suspend fun translatePage(page: Bitmap): PageResult = coroutineScope {
        val tWall = System.currentTimeMillis()
        // 偵測
        val tDet = System.currentTimeMillis()
        val detection = try {
            detector.detect(page)
        } catch (t: Throwable) {
            Log.e(TAG, "偵測失敗", t); return@coroutineScope PageResult.Failed("detect: ${t.message}")
        }
        val lines = detection.lines
        val detectMs = System.currentTimeMillis() - tDet
        if (lines.isEmpty()) {
            return@coroutineScope PageResult.Skipped("偵測不到文字", PageStats(0, 0, 0, detectMs, 0, 0, 0, 0))
        }

        // OCR + 分群
        val tOcr = System.currentTimeMillis()
        try {
            ocr.recognize(page, lines)
        } catch (t: Throwable) {
            Log.e(TAG, "OCR 失敗", t); return@coroutineScope PageResult.Failed("ocr: ${t.message}")
        }
        val regions = Grouping.group(lines)
        val ocrMs = System.currentTimeMillis() - tOcr
        // 去字集＝有 OCR 原文的區（空白＝疑似誤偵測，不去字、保畫面）。此集翻譯前就確定 ⇒ 去字可與翻譯並發。
        val textRegions = regions.filter { it.sourceText.isNotBlank() }
        if (textRegions.isEmpty()) {
            return@coroutineScope PageResult.Skipped("OCR 全空", PageStats(lines.size, regions.size, 0, detectMs, ocrMs, 0, 0, 0))
        }

        // ★ 去字（CPU）‖ 翻譯（網路）並發：兩者只依賴 OCR，可同時跑（網路等待時 CPU 去字、互不爭資源）。
        // §11 改：失敗區不再「保留源圖」，而是「去字後重貼 OCR 原文」(使用者拍板：重貼成本低、不需源圖狀態) ⇒ 去字與翻譯解耦。
        // 不限去字方法（boxfill/lama 皆可重疊）。boxfill(~0.5s)整段藏進翻譯(~2.7s)＝免費。
        var inpaintMs = 0L
        val inpaintJob = async {
            val t0 = System.currentTimeMillis()
            val r = inpainter.inpaint(page, textRegions, detection.textMask)
            inpaintMs = System.currentTimeMillis() - t0
            r
        }

        var translateMs = 0L
        if (translator != null) {
            val tTr = System.currentTimeMillis()
            val cht = try {
                translator.translate(textRegions.map { it.sourceText })
            } catch (t: Throwable) {
                Log.e(TAG, "翻譯失敗", t)
                inpaintJob.cancelAndJoin() // 翻譯掛 → 丟棄去字、留原圖（§11；native run 不可中斷，cancel 實為等它跑完再丟）
                return@coroutineScope PageResult.Failed("translate: ${t.message}")
            }
            textRegions.forEachIndexed { j, r -> r.translatedText = cht.getOrElse(j) { r.sourceText } }
            translateMs = System.currentTimeMillis() - tTr
        } else {
            textRegions.forEach { it.translatedText = it.sourceText } // 無 key debug：排版原文
        }

        // 判定每區譯文有效性（空白/數字/regex/譯==原＝失敗）。整頁全失敗 → 留原圖（Skipped、丟棄去字）。
        val kept = if (translator != null) TextFilter.apply(textRegions, cfg.translator.filterText) else textRegions
        if (kept.isEmpty()) {
            val aligned = textRegions.count { it.translatedText.isNotBlank() && it.translatedText != it.sourceText }
            val tr = translator as? LlmTranslator
            val dbg = textRegions.take(2).joinToString(" ‖ ") { "${it.sourceText.take(8)}→${it.translatedText.take(8)}" }
            Log.w(TAG, "全數過濾 對齊$aligned/${textRegions.size} err=${tr?.lastError} 回應=${tr?.lastRaw}")
            inpaintJob.cancelAndJoin()
            return@coroutineScope PageResult.Skipped(
                "全數過濾 對齊$aligned/${textRegions.size} err=${tr?.lastError}｜回應=${tr?.lastRaw}｜$dbg",
                PageStats(lines.size, regions.size, 0, detectMs, ocrMs, translateMs, 0, 0),
            )
        }
        // 失敗的區（不在 kept）→ 譯文改回原文＝去字後重貼 OCR 日文（TextRegion 無 equals override→HashSet 走 identity）。
        val keptSet = kept.toHashSet()
        textRegions.forEach { if (it !in keptSet) it.translatedText = it.sourceText }

        // 等去字完成（多半已與翻譯重疊跑完）
        val cleaned = try {
            inpaintJob.await()
        } catch (t: Throwable) {
            Log.e(TAG, "去字失敗", t); return@coroutineScope PageResult.Failed("inpaint: ${t.message}")
        }

        // 排版（全 textRegions：kept 貼譯文、失敗區貼原文）
        val tRn = System.currentTimeMillis()
        val finalPage = Renderer.render(cleaned, textRegions, cfg.render, typeface)
        val renderMs = System.currentTimeMillis() - tRn

        PageResult.Translated(
            finalPage,
            PageStats(lines.size, regions.size, kept.size, detectMs, ocrMs, translateMs, inpaintMs, renderMs, System.currentTimeMillis() - tWall),
        )
    }

    /** 釋放 detector/ocr/inpainter 的原生 ONNX session（見類別說明的生命週期注意事項）。 */
    override fun close() {
        runCatching { detector.close() }
        runCatching { ocr.close() }
        runCatching { inpainter.close() }
    }

    companion object {
        private const val TAG = "Pipeline"
    }
}
