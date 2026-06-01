package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface

/**
 * M3 排版：把譯文排進（去字後的）氣泡。
 *
 * 第一版（陽春橫排 + 自動字級 + CJK 斷行 + 描邊），對齊 parity/typeset_parity.py。
 * ★ 後續精修（§9）：直排、貼合氣泡形狀、字級/行距更聰明、字型可選（Noto/Source Han）。
 * 此處用系統 Typeface（裝置上能算繪 CJK）。
 */
object Renderer {

    fun render(page: Bitmap, regions: List<TextRegion>): Bitmap {
        val out = page.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val fill = Paint().apply {
            color = Color.BLACK
            isAntiAlias = true
            typeface = Typeface.DEFAULT
        }
        val stroke = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
            typeface = Typeface.DEFAULT
        }

        for (region in regions) {
            val text = region.translatedText.ifBlank { region.sourceText }
            if (text.isBlank()) continue
            val regionW = region.x1 - region.x0
            val regionH = region.y1 - region.y0
            val bw = regionW * 1.1f
            val bh = regionH * 1.15f
            if (bw < 8f || bh < 8f) continue

            val (size, lines) = fit(text, bw, bh, fill)
            fill.textSize = size
            stroke.textSize = size
            val lh = size * 1.25f
            var ty = region.y0 + (regionH - lines.size * lh) / 2f + size
            for (ln in lines) {
                val tx = region.x0 + (regionW - fill.measureText(ln)) / 2f
                canvas.drawText(ln, tx, ty, stroke)
                canvas.drawText(ln, tx, ty, fill)
                ty += lh
            }
        }
        return out
    }

    /** 由大到小找最大、能讓斷行後文字塞進框的字級。 */
    private fun fit(text: String, bw: Float, bh: Float, paint: Paint): Pair<Float, List<String>> {
        var size = minOf(bh.toInt(), MAX_SIZE)
        while (size >= MIN_SIZE) {
            paint.textSize = size.toFloat()
            val lines = wrapCjk(text, paint, bw)
            val lh = size * 1.25f
            val maxW = lines.maxOfOrNull { paint.measureText(it) } ?: 0f
            if (lines.size * lh <= bh && maxW <= bw) return size.toFloat() to lines
            size--
        }
        paint.textSize = MIN_SIZE.toFloat()
        return MIN_SIZE.toFloat() to wrapCjk(text, paint, bw)
    }

    private fun wrapCjk(text: String, paint: Paint, maxW: Float): List<String> {
        val lines = ArrayList<String>()
        val cur = StringBuilder()
        for (ch in text) {
            if (ch == '\n') {
                lines.add(cur.toString()); cur.clear(); continue
            }
            if (cur.isNotEmpty() && paint.measureText(cur.toString() + ch) > maxW) {
                lines.add(cur.toString()); cur.clear()
            }
            cur.append(ch)
        }
        if (cur.isNotEmpty()) lines.add(cur.toString())
        return lines
    }

    private const val MAX_SIZE = 46
    private const val MIN_SIZE = 9
}
