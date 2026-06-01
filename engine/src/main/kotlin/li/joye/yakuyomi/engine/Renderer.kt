package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlin.math.ceil
import kotlin.math.min

/** 排版方向（使用者可切換設定）：繁中/日文用直排，英文等用橫排。 */
enum class TextOrientation { VERTICAL, HORIZONTAL }

/**
 * M3 排版：把譯文排進（去字後的）氣泡。對齊 parity/typeset_parity.py。
 *   直排：CJK 字由上而下、欄由右而左（漫畫原生）。
 *   橫排：左→右、上→下（英文等）。
 * 兩者皆自動字級 + 斷行 + 置中 + 白描邊。系統 Typeface 算繪 CJK。
 * ★ 後續精修（§9）：標點旋轉、貼合氣泡形狀、可選字型（Noto/Source Han）。
 */
object Renderer {

    fun render(
        page: Bitmap,
        regions: List<TextRegion>,
        orientation: TextOrientation = TextOrientation.VERTICAL,
    ): Bitmap {
        val out = page.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val fill = Paint().apply { color = Color.BLACK; isAntiAlias = true; typeface = Typeface.DEFAULT }
        val stroke = Paint().apply {
            color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 4f
            isAntiAlias = true; typeface = Typeface.DEFAULT
        }
        for (region in regions) {
            val text = region.translatedText.ifBlank { region.sourceText }
            if (text.isBlank()) continue
            val rw = region.x1 - region.x0
            val rh = region.y1 - region.y0
            if (rw < 8f || rh < 8f) continue
            when (orientation) {
                TextOrientation.HORIZONTAL -> drawHorizontal(canvas, region, rw, rh, text, fill, stroke)
                TextOrientation.VERTICAL -> drawVertical(canvas, region, rw, rh, text, fill, stroke)
            }
        }
        return out
    }

    private fun drawHorizontal(
        canvas: Canvas, region: TextRegion, rw: Float, rh: Float,
        text: String, fill: Paint, stroke: Paint,
    ) {
        val bw = rw * 1.1f
        val bh = rh * 1.15f
        var size = min(bh.toInt(), MAX)
        var lines = listOf(text)
        while (size >= MIN) {
            fill.textSize = size.toFloat()
            lines = wrapCjk(text, fill, bw)
            val lh = size * 1.25f
            val maxW = lines.maxOfOrNull { fill.measureText(it) } ?: 0f
            if (lines.size * lh <= bh && maxW <= bw) break
            size--
        }
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        val lh = size * 1.25f
        var baseline = region.y0 + (rh - lines.size * lh) / 2f + size * ASCENT
        for (ln in lines) {
            val tx = region.x0 + (rw - fill.measureText(ln)) / 2f
            canvas.drawText(ln, tx, baseline, stroke)
            canvas.drawText(ln, tx, baseline, fill)
            baseline += lh
        }
    }

    private fun drawVertical(
        canvas: Canvas, region: TextRegion, rw: Float, rh: Float,
        text: String, fill: Paint, stroke: Paint,
    ) {
        val bw = rw * 1.1f
        val bh = rh * 1.15f
        val chars = text.filter { it != '\n' }
        if (chars.isEmpty()) return
        var size = min(bh.toInt(), MAX)
        var cols = 1
        var cpc = 1
        var lh = MIN * 1.05f
        var cw = MIN * 1.18f
        while (size >= MIN) {
            lh = size * 1.05f
            cw = size * 1.18f
            cpc = maxOf(1, (bh / lh).toInt())
            cols = ceil(chars.length / cpc.toFloat()).toInt()
            if (cols * cw <= bw) break
            size--
        }
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        val ascent = size * ASCENT
        val totalW = cols * cw
        val rightCx = region.x0 + (rw - totalW) / 2f + totalW - cw / 2f // 最右欄中心
        for (col in 0 until cols) {
            val cx = rightCx - col * cw
            val start = col * cpc
            val end = min(start + cpc, chars.length)
            val colTop = region.y0 + (rh - (end - start) * lh) / 2f
            for (i in start until end) {
                val s = chars[i].toString()
                val w = fill.measureText(s)
                val baseline = colTop + (i - start) * lh + ascent
                canvas.drawText(s, cx - w / 2f, baseline, stroke)
                canvas.drawText(s, cx - w / 2f, baseline, fill)
            }
        }
    }

    private fun wrapCjk(text: String, paint: Paint, maxW: Float): List<String> {
        val lines = ArrayList<String>()
        val cur = StringBuilder()
        for (ch in text) {
            if (ch == '\n') { lines.add(cur.toString()); cur.clear(); continue }
            if (cur.isNotEmpty() && paint.measureText(cur.toString() + ch) > maxW) {
                lines.add(cur.toString()); cur.clear()
            }
            cur.append(ch)
        }
        if (cur.isNotEmpty()) lines.add(cur.toString())
        return lines
    }

    private const val MAX = 46
    private const val MIN = 9
    private const val ASCENT = 0.82f // drawText 的 y 是 baseline，約略加 ascent 把頂端對齊
}
