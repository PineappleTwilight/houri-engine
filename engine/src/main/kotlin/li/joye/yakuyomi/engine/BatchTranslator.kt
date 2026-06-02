package li.joye.yakuyomi.engine

import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit

/**
 * 跨頁批次翻譯（效率核心）。對映 m-i-t `--batch-size` / `--batch-concurrent`（§2 翻譯批次策略、§10 並發旋鈕）。
 *
 *   batchConcurrent=true（推薦／預設）：每頁各自一個 request、批內並發，[TranslatorConfig.batchSize] 經 Semaphore 限同時在飛數。
 *     對齊 m-i-t concurrent 模式：逐頁分開可防 truncation/幻覺（args.py 註解）；單頁失敗只影響該頁（§11）。
 *   batchConcurrent=false：每 batchSize 頁的所有區併成一個大 prompt 送出（省 API call，但易 truncation、整塊一起失敗）。
 *
 * 包一個逐頁 [Translator]（如 [LlmTranslator]）。
 * @param pages 每頁的原文清單（已分區）；回傳對齊的譯文清單（pages[i] ↔ result[i]）。
 */
class BatchTranslator(
    private val translator: Translator,
    private val cfg: TranslatorConfig = TranslatorConfig(),
) {

    suspend fun translatePages(pages: List<List<String>>): List<List<String>> = coroutineScope {
        if (pages.isEmpty()) return@coroutineScope emptyList()
        val limit = cfg.batchSize.coerceAtLeast(1)
        if (cfg.batchConcurrent) {
            val sem = Semaphore(limit)
            pages.map { page -> async { sem.withPermit { translator.translate(page) } } }.awaitAll()
        } else {
            val out = ArrayList<List<String>>(pages.size)
            for (chunk in pages.chunked(limit)) out.addAll(translateMerged(chunk))
            out
        }
    }

    /** merged：整個 chunk 的所有頁區併一次翻譯，再依各頁區數切回；回傳不足（truncation）的缺額補原文（§11）。 */
    private suspend fun translateMerged(chunk: List<List<String>>): List<List<String>> {
        val flat = chunk.flatten()
        if (flat.isEmpty()) return chunk.map { emptyList() }
        val res = translator.translate(flat)
        var i = 0
        return chunk.map { page ->
            val end = (i + page.size).coerceIn(i, res.size)
            val slice = res.subList(i, end).toList()
            i += page.size
            if (slice.size < page.size) slice + page.subList(slice.size, page.size) else slice
        }
    }
}
