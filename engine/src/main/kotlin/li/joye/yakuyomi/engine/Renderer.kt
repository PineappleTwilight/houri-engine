package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlin.math.ceil
import kotlin.math.min

/** 排版方向。AUTO＝依內容（CJK 直排、純英數橫排），對應 m-i-t config 的 direction:auto。 */
enum class TextOrientation { VERTICAL, HORIZONTAL, AUTO }

/**
 * M3 排版（純文字框法，可靠）：定位 + 大小都用文字框，框適度放大（[RenderConfig.expandW]/[RenderConfig.expandH]）給呼吸空間。
 * 對齊 parity/typeset_parity.py（§4 第二層：同輸入近輸出）。
 *   不靠氣泡 flood-fill——相鄰氣泡會連通成一塊、整個算錯，已棄用。
 *   直排：CJK 上→下、欄右→左、向上對齊、每欄少 [RenderConfig.colTrim] 字（縮短欄長、減少凸出）、標點旋轉。
 *   橫排：左→右、上→下、向上對齊。
 *   文字色：[RenderConfig.colorMode]=auto 取去字後背景亮度判黑/白字（白底黑字、黑底白字）；OCR color head 色相太雜不採用。
 * ★ 後續：tate-chu-yoko、字型可選。
 */
object Renderer {

    fun render(
        page: Bitmap,
        regions: List<TextRegion>,
        cfg: RenderConfig = RenderConfig(),
        tf: Typeface? = null,
    ): Bitmap {
        val out = page.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val face = tf ?: Typeface.DEFAULT
        val fill = Paint().apply { color = Color.BLACK; isAntiAlias = true; typeface = face }
        val stroke = Paint().apply {
            color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 4f
            isAntiAlias = true; typeface = face
        }
        for (region in regions) {
            val text = region.translatedText.ifBlank { region.sourceText }
            if (text.isBlank()) continue
            if (region.x1 - region.x0 < 8f || region.y1 - region.y0 < 8f) continue
            val (fillColor, outlineColor) = textColors(page, region, cfg)
            fill.color = fillColor
            stroke.color = outlineColor
            val vertical = when (cfg.orientation) {
                TextOrientation.AUTO -> isCjk(text)
                TextOrientation.VERTICAL -> true
                TextOrientation.HORIZONTAL -> false
            }
            if (vertical) drawVertical(canvas, region, text, fill, stroke, cfg)
            else drawHorizontal(canvas, region, text, fill, stroke, cfg)
        }
        return out
    }

    /** 文字色 (fill, outline)：auto＝取去字後背景亮度（暗底白字、亮底黑字，對齊 parity auto_colors）；mono＝黑字白邊。 */
    private fun textColors(page: Bitmap, r: TextRegion, cfg: RenderConfig): Pair<Int, Int> {
        if (cfg.colorMode == "mono") return Color.BLACK to Color.WHITE
        val lum = bgLuminance(page, r.x0, r.y0, r.x1, r.y1)
        return if (lum < cfg.bgDark) Color.WHITE to Color.BLACK else Color.BLACK to Color.WHITE
    }

    /** 去字後背景在文字框內的平均亮度（Rec.601）。 */
    private fun bgLuminance(page: Bitmap, rx0: Float, ry0: Float, rx1: Float, ry1: Float): Float {
        val w = page.width
        val h = page.height
        val x0 = rx0.toInt().coerceIn(0, w - 1)
        val y0 = ry0.toInt().coerceIn(0, h - 1)
        val x1 = rx1.toInt().coerceIn(x0 + 1, w)
        val y1 = ry1.toInt().coerceIn(y0 + 1, h)
        val bw = x1 - x0
        val bh = y1 - y0
        val px = IntArray(bw * bh)
        page.getPixels(px, 0, bw, x0, y0, bw, bh)
        var sum = 0.0
        for (p in px) {
            sum += 0.299 * ((p shr 16) and 0xFF) + 0.587 * ((p shr 8) and 0xFF) + 0.114 * (p and 0xFF)
        }
        return (sum / px.size).toFloat()
    }

    private fun isCjk(text: String): Boolean = text.any {
        val o = it.code
        o in 0x3040..0x30FF || o in 0x4E00..0x9FFF || o in 0x3400..0x4DBF || o in 0xFF00..0xFFEF
    }

    /** 直排：欄右→左、字上→下、向上對齊；大小填滿放大後的文字框、每欄少 colTrim 字。 */
    private fun drawVertical(canvas: Canvas, r: TextRegion, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig) {
        val chars = text.filter { it != '\n' }
        if (chars.isEmpty()) return
        val bw = (r.x1 - r.x0) * cfg.expandW         // 寬：放大後的文字框寬
        val colRoom = (r.y1 - r.y0) * cfg.expandH    // 直欄可用高（從文字框頂往下）
        var size = cfg.fontSizeMin
        var s = min(colRoom.toInt(), cfg.fontSizeMax)
        while (s >= cfg.fontSizeMin) {
            val lh = s * 1.05f; val cw = s * 1.1f
            val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
            if (ceil(chars.length / cpc.toFloat()).toInt() * cw <= bw) { size = s; break }
            s--
        }
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        val lh = size * 1.05f; val cw = size * 1.1f
        val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
        val cols = ceil(chars.length / cpc.toFloat()).toInt()
        val tcx = (r.x0 + r.x1) / 2f                  // 定位：水平置中於文字框中心
        val rightCx = tcx + cols * cw / 2f - cw / 2f
        for (col in 0 until cols) {
            val cx = rightCx - col * cw
            var cy = r.y0                              // 定位：頂端對齊文字框頂
            val start = col * cpc
            val end = min(start + cpc, chars.length)
            for (i in start until end) {
                drawCharVertical(canvas, chars[i], cx, cy + lh / 2f, fill, stroke, cfg.fontBorder)
                cy += lh
            }
        }
    }

    /** 橫排：列上→下、字左→右、向上對齊；大小填滿放大後的文字框。 */
    private fun drawHorizontal(canvas: Canvas, r: TextRegion, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig) {
        val bw = (r.x1 - r.x0) * cfg.expandW
        val rowRoom = (r.y1 - r.y0) * cfg.expandH
        var size = cfg.fontSizeMin
        var lines = listOf(text)
        var s = min(rowRoom.toInt(), cfg.fontSizeMax)
        while (s >= cfg.fontSizeMin) {
            fill.textSize = s.toFloat()
            val ls = wrapCjk(text, fill, bw)
            val maxW = ls.maxOfOrNull { fill.measureText(it) } ?: 0f
            if (ls.size * s * 1.18f <= rowRoom && maxW <= bw) { size = s; lines = ls; break }
            s--
        }
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        val lh = size * 1.18f
        val tcx = (r.x0 + r.x1) / 2f
        var baseline = r.y0 + size * ASCENT            // 頂端對齊
        for (ln in lines) {
            val tx = tcx - fill.measureText(ln) / 2f
            if (cfg.fontBorder) canvas.drawText(ln, tx, baseline, stroke)
            canvas.drawText(ln, tx, baseline, fill)
            baseline += lh
        }
    }

    private fun drawCharVertical(canvas: Canvas, ch: Char, cx: Float, cyc: Float, fill: Paint, stroke: Paint, border: Boolean) {
        val s = ch.toString()
        val w = fill.measureText(s)
        val fm = fill.fontMetrics
        val baseline = cyc - (fm.ascent + fm.descent) / 2f
        val rotate = ch in ROTATE_CHARS
        if (rotate) { canvas.save(); canvas.rotate(90f, cx, cyc) }
        if (border) canvas.drawText(s, cx - w / 2f, baseline, stroke)
        canvas.drawText(s, cx - w / 2f, baseline, fill)
        if (rotate) canvas.restore()
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

    private const val ASCENT = 0.82f
    private const val ROTATE_CHARS = "ー－—―‐~〜～…‥（）()「」『』【】〔〕［］｛｝〈〉《》＜＞<>｜|：;"
}
