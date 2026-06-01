package li.joye.yakuyomi.engine

import android.graphics.Bitmap

/**
 * 引擎對外唯一介面（CLAUDE.md §6「引擎解耦」、§13）。
 *
 * M0–M1 先不實作；M2 才把整條 pipeline（偵測→OCR→翻譯→渲染）串進來。
 */
interface TranslationEngine {
    suspend fun translatePage(page: Bitmap): Bitmap
}
