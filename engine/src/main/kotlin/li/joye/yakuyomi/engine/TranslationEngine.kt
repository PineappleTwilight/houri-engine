package li.joye.yakuyomi.engine

import android.graphics.Bitmap

/**
 * 引擎對外唯一介面（CLAUDE.md §6「引擎解耦」、§13）。實作＝[Pipeline]。
 *
 * 回傳 [PageResult] 而非裸 Bitmap：要能表達 §11 的「略過（不覆蓋）／失敗（不覆蓋、可重試）／成功（可覆蓋+marker）」，
 * 裸 Bitmap 無法區分。覆蓋原檔／寫 marker／page-level resume／跨頁批次併發＝呼叫端職責（§3）。
 */
interface TranslationEngine {
    suspend fun translatePage(page: Bitmap): PageResult
}
