package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.roundToInt

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
                TextOrientation.AUTO -> region.direction == "v" // 跟著偵測到的原文方向（對齊 m-i-t），不再無腦直排
                TextOrientation.VERTICAL -> true
                TextOrientation.HORIZONTAL -> false
            }
            // 斜框：繞區域中心旋轉畫布、用去傾斜框排版。
            // 旋轉方向經 PCA 量測對齊 m-i-t 樣本：region 正角＝文字右端下斜。
            // Android Canvas 正角＝順時針（y 朝下）＝右端下斜 → 直接 +angle（parity 用 PIL、正角逆時針故取 -angle）。
            val rotate = abs(region.angle) >= 1f
            val x0: Float; val y0: Float; val x1: Float; val y1: Float
            if (rotate) {
                x0 = region.cx - region.boxW / 2f; y0 = region.cy - region.boxH / 2f
                x1 = region.cx + region.boxW / 2f; y1 = region.cy + region.boxH / 2f
                canvas.save(); canvas.rotate(region.angle, region.cx, region.cy)
            } else {
                x0 = region.x0; y0 = region.y0; x1 = region.x1; y1 = region.y1
            }
            if (vertical) drawVertical(canvas, x0, y0, x1, y1, text, fill, stroke, cfg)
            else drawHorizontal(canvas, x0, y0, x1, y1, text, fill, stroke, cfg)
            if (rotate) canvas.restore()
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
    private fun drawVertical(canvas: Canvas, x0: Float, y0: Float, x1: Float, y1: Float, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig) {
        val chars = text.filter { it != '\n' }
        if (chars.isEmpty()) return
        val bw = (x1 - x0) * cfg.expandW         // 寬：放大後的文字框寬
        val colRoom = (y1 - y0) * cfg.expandH    // 直欄可用高（從文字框頂往下）
        var size = cfg.fontSizeMin
        var s = min(colRoom.toInt(), cfg.fontSizeMax)
        while (s >= cfg.fontSizeMin) {
            val lh = s * 1.05f; val cw = s * 1.1f
            val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
            if (ceil(chars.length / cpc.toFloat()).toInt() * cw <= bw) { size = s; break }
            s--
        }
        size = maxOf(cfg.fontSizeMin, (size * cfg.fontScale).roundToInt())  // 整體縮小、更 fit
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        stroke.strokeWidth = maxOf(2f, size * STROKE_RATIO)  // 描邊隨字級（取代固定寬）
        val lh = size * 1.05f; val cw = size * 1.1f
        val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
        val columns = splitColumnsV(chars, cpc)       // 禁則：欄不以行頭禁則字開頭
        val cols = columns.size
        val tcx = (x0 + x1) / 2f                  // 定位：水平置中於文字框中心
        val rightCx = tcx + cols * cw / 2f - cw / 2f
        val blockH = columns.maxOf { it.length } * lh // 垂直置中：以最長欄高為塊高，置中於框
        val startCy = (y0 + y1) / 2f - blockH / 2f
        for (col in 0 until cols) {
            val cx = rightCx - col * cw
            var cy = startCy
            for (ch in columns[col]) {
                drawCharVertical(canvas, ch, cx, cy + lh / 2f, fill, stroke, cfg.fontBorder)
                cy += lh
            }
        }
    }

    /** 直排切欄＋行頭禁則：禁則字不置於欄頭、併回前一欄（最多 +2，避免暴衝）。 */
    private fun splitColumnsV(chars: String, cpc: Int): List<String> {
        val cols = ArrayList<String>()
        var i = 0
        val n = chars.length
        while (i < n) {
            var end = minOf(i + cpc, n)
            var ext = 0
            while (end < n && chars[end] in NO_START && ext < 2) { end++; ext++ }
            cols.add(chars.substring(i, end))
            i = end
        }
        return cols
    }

    /** 橫排：列上→下、字左→右、向上對齊；大小填滿放大後的文字框。 */
    private fun drawHorizontal(canvas: Canvas, x0: Float, y0: Float, x1: Float, y1: Float, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig) {
        val bw = (x1 - x0) * cfg.expandW
        val rowRoom = (y1 - y0) * cfg.expandH
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
        size = maxOf(cfg.fontSizeMin, (size * cfg.fontScale).roundToInt())  // 整體縮小、更 fit
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        stroke.strokeWidth = maxOf(2f, size * STROKE_RATIO)  // 描邊隨字級
        lines = wrapCjk(text, fill, bw)  // 縮小後重排
        val lh = size * 1.18f
        val tcx = (x0 + x1) / 2f
        var baseline = (y0 + y1) / 2f - lines.size * lh / 2f + size * ASCENT  // 垂直置中於框
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
            if (cur.isNotEmpty() && paint.measureText(cur.toString() + ch) > maxW && ch !in NO_START) {
                lines.add(cur.toString()); cur.clear()  // 行頭禁則：禁則字不另起行
            }
            cur.append(ch)
        }
        if (cur.isNotEmpty()) lines.add(cur.toString())
        return lines
    }

    private const val ASCENT = 0.82f
    private const val STROKE_RATIO = 0.10f  // 描邊寬＝字級×此比例（隨字級縮放）
    private const val ROTATE_CHARS = "ー－—―‐~〜～…‥（）()「」『』【】〔〕［］｛｝〈〉《》＜＞<>｜|：;"
    // 行頭禁則：不可置於欄/行開頭（收尾標點、小假名）→ 併回前一欄/行（kinsoku）
    private const val NO_START = "、。，．：；！？”’）〕】｝」』》〉…‥ーゝゞヽヾ々ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ"
}
