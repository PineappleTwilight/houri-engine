package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt

/** 排版方向。AUTO＝依內容（CJK 直排、純英數橫排），對應 m-i-t config 的 direction:auto。 */
enum class TextOrientation { VERTICAL, HORIZONTAL, AUTO }

/**
 * 排版（純文字框法，可靠）：定位 + 大小都用文字框，框適度放大（[RenderConfig.expandW]/[RenderConfig.expandH]）給呼吸空間。
 * 對齊 parity/typeset_parity.py（§4 第二層：同輸入近輸出）。
 *   不靠氣泡 flood-fill——相鄰氣泡會連通成一塊、整個算錯，已棄用。
 *   直排：CJK 上→下、欄右→左、向上對齊、每欄少 [RenderConfig.colTrim] 字（縮短欄長、減少凸出）、標點旋轉、短 ASCII 串縱中橫（tate-chu-yoko，見 [RenderConfig.tateChuYoko]）。
 *   橫排：左→右、上→下、向上對齊。
 *   文字色：[RenderConfig.colorMode]=auto 取去字後背景亮度判黑/白字（白底黑字、黑底白字）；OCR color head 色相太雜不採用。
 * ★ 後續：字型可選。
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
            // 譯文字級錨定原文高度：短譯文不脹到 fontSizeMax、長譯文不縮到小於原文太多（保持「翻譯前後字一樣大」）。
            val originalSize = originalFontSize(region)
            if (vertical) drawVertical(canvas, x0, y0, x1, y1, text, fill, stroke, cfg, onArt = region.onArt, originalSize = originalSize)
            else drawHorizontal(canvas, x0, y0, x1, y1, text, fill, stroke, cfg, onArt = region.onArt, originalSize = originalSize)
            if (rotate) canvas.restore()
        }
        return out
    }

    /**
     * 估計區域原文的字級（px）：對每行取 min(行框寬,高)＝文字筆畫厚度（直/橫書皆然），再取中位數。
     * 這是排版「譯文要和原文一樣大」的錨點：短譯文（如英文短句 vs 長日文）不再放大到填滿整個氣泡。
     */
    private fun originalFontSize(region: TextRegion): Int {
        val thicknesses = ArrayList<Float>()
        for (line in region.lines) {
            val q = line.quad
            if (q.size < 4) continue
            fun mid(a: Pt, b: Pt) = Pt(((a.x + b.x) / 2f).toInt().toFloat(), ((a.y + b.y) / 2f).toInt().toFloat())
            val hx = mid(q[0], q[1]).x - mid(q[2], q[3]).x
            val hy = mid(q[0], q[1]).y - mid(q[2], q[3]).y
            val wx = mid(q[1], q[2]).x - mid(q[3], q[0]).x
            val wy = mid(q[1], q[2]).y - mid(q[3], q[0]).y
            val h = hypot(hx.toDouble(), hy.toDouble()).toFloat()
            val w = hypot(wx.toDouble(), wy.toDouble()).toFloat()
            thicknesses.add(minOf(h, w))
        }
        if (thicknesses.isEmpty()) return 0
        thicknesses.sort()
        return thicknesses[thicknesses.size / 2].roundToInt()
    }

    /** 文字色 (fill, outline)：auto＝取去字後背景亮度（暗底白字、亮底黑字，對齊 parity auto_colors）；mono＝黑字白邊；其他＝固定色。 */
    private fun textColors(page: Bitmap, r: TextRegion, cfg: RenderConfig): Pair<Int, Int> {
        if (cfg.colorMode == "mono") return Color.BLACK to Color.WHITE
        // 使用者指定固定文字色（預設純黑）：outline 仍依背景亮度判黑/白，確保任何底色都讀得到。
        if (cfg.colorMode != "auto") return cfg.fixedTextColor to (if (bgLuminance(page, r.x0, r.y0, r.x1, r.y1) < cfg.bgDark) Color.WHITE else Color.BLACK)
        // 壓在畫面上(lama 重建的 busy 背景)：一律黑字+粗白邊。白邊把字框出來，任何雜亂背景都讀得到（對齊 m-i-t 做法）。
        if (r.onArt) return Color.BLACK to Color.WHITE
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

/** 直排：欄右→左、格上→下、向上對齊；大小填滿放大後的文字框、每欄少 colTrim 格。每格＝1 字或 1 個縱中橫短串。 */
    private fun drawVertical(canvas: Canvas, x0: Float, y0: Float, x1: Float, y1: Float, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig, onArt: Boolean = false, originalSize: Int = 0) {
        val chars = text.filter { it != '\n' }
        if (chars.isEmpty()) return
        val cells = toVerticalCells(chars, cfg.tateChuYoko)  // 切格：一般字一格、短 ASCII 串併成一個縱中橫格
        val bw = (x1 - x0) * cfg.expandW         // 寬：放大後的文字框寬
        val colRoom = (y1 - y0) * cfg.expandH    // 直欄可用高（從文字框頂往下）
        var size = cfg.fontSizeMin
        // 起點錨定原文字級：譯文不該比原文大；長譯文仍會往下縮到 fit。
        val cap = if (originalSize > 0) minOf(cfg.fontSizeMax, originalSize) else cfg.fontSizeMax
        var s = min(colRoom.toInt(), cap)
        while (s >= cfg.fontSizeMin) {
            val lh = s * 1.05f; val cw = s * 1.1f
            val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
            if (ceil(cells.size / cpc.toFloat()).toInt() * cw <= bw) { size = s; break }
            s--
        }
        size = maxOf(cfg.fontSizeMin, (size * cfg.fontScale).roundToInt())  // 整體縮小、更 fit
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        stroke.strokeWidth = maxOf(2f, size * (if (onArt) cfg.artStrokeRatio else STROKE_RATIO))  // 描邊隨字級；壓畫面區用更粗白邊
        val lh = size * 1.05f; val cw = size * 1.1f
        val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
        val columns = splitColumnsV(cells, cpc)       // 禁則：欄不以行頭禁則字開頭
        val cols = columns.size
        val tcx = (x0 + x1) / 2f                  // 定位：水平置中於文字框中心
        val rightCx = tcx + cols * cw / 2f - cw / 2f
        val blockH = columns.maxOf { it.size } * lh // 垂直置中：以最長欄格數為塊高，置中於框
        val startCy = (y0 + y1) / 2f - blockH / 2f
        for (col in 0 until cols) {
            val cx = rightCx - col * cw
            var cy = startCy
            for (cell in columns[col]) {
                if (cell.length == 1) {
                    drawCharVertical(canvas, cell[0], cx, cy + lh / 2f, fill, stroke, cfg.fontBorder)
                } else {
                    drawTateChuYoko(canvas, cell, cx, cy + lh / 2f, cw, fill, stroke, cfg.fontBorder)
                }
                cy += lh
            }
        }
    }

    /**
     * 直排切格：一般字一格；連續短 ASCII 串（2–[MAX_TCY] 字的數字/字母/!?）併成一個縱中橫格（tate-chu-yoko）。
     * 單字 ASCII（如獨立「5」）維持單格；過長串（> MAX_TCY，如英文長詞）退回逐字（避免水平壓太扁）。
     */
    private fun toVerticalCells(chars: String, enabled: Boolean): List<String> {
        if (!enabled) return chars.map { it.toString() }
        val cells = ArrayList<String>()
        var i = 0
        val n = chars.length
        while (i < n) {
            if (isTcyChar(chars[i])) {
                var j = i + 1
                while (j < n && isTcyChar(chars[j])) j++
                if (j - i in 2..MAX_TCY) {
                    cells.add(chars.substring(i, j))                       // 一個縱中橫格
                } else {
                    for (k in i until j) cells.add(chars[k].toString())   // 單字或過長：逐字（維持原行為）
                }
                i = j
            } else {
                cells.add(chars[i].toString()); i++
            }
        }
        return cells
    }

    private fun isTcyChar(c: Char): Boolean =
        c in '0'..'9' || c in 'A'..'Z' || c in 'a'..'z' || c == '!' || c == '?'

    /** 直排切欄＋行頭禁則：禁則字（單字標點）不置於欄頭、併回前一欄（最多 +2，避免暴衝）。 */
    private fun splitColumnsV(cells: List<String>, cpc: Int): List<List<String>> {
        val cols = ArrayList<List<String>>()
        var i = 0
        val n = cells.size
        while (i < n) {
            var end = minOf(i + cpc, n)
            var ext = 0
            while (end < n && cells[end].length == 1 && cells[end][0] in NO_START && ext < 2) { end++; ext++ }
            cols.add(cells.subList(i, end).toList())
            i = end
        }
        return cols
    }

    /** 縱中橫：把短 ASCII 串在一個直排格內水平並排、置中於欄心；超出格寬只橫向壓縮（高度不變、與鄰字視覺一致）。 */
    private fun drawTateChuYoko(canvas: Canvas, group: String, cx: Float, cyc: Float, cellW: Float, fill: Paint, stroke: Paint, border: Boolean) {
        val w = fill.measureText(group)
        val fm = fill.fontMetrics
        val baseline = cyc - (fm.ascent + fm.descent) / 2f
        val target = cellW * 0.92f
        val scaleX = if (w > target) target / w else 1f
        canvas.save()
        if (scaleX != 1f) canvas.scale(scaleX, 1f, cx, baseline)  // 只橫向縮、繞欄心
        val tx = cx - w / 2f
        if (border) canvas.drawText(group, tx, baseline, stroke)
        canvas.drawText(group, tx, baseline, fill)
        canvas.restore()
    }

    /** 橫排：列上→下、字左→右、向上對齊；大小填滿放大後的文字框。 */
    private fun drawHorizontal(canvas: Canvas, x0: Float, y0: Float, x1: Float, y1: Float, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig, onArt: Boolean = false, originalSize: Int = 0) {
        val bw = (x1 - x0) * cfg.expandW
        val rowRoom = (y1 - y0) * cfg.expandH
        var size = cfg.fontSizeMin
        var lines = listOf(text)
        // 起點錨定原文字級：譯文不該比原文大；長譯文仍會往下縮到 fit。
        val cap = if (originalSize > 0) minOf(cfg.fontSizeMax, originalSize) else cfg.fontSizeMax
        var s = min(rowRoom.toInt(), cap)
        while (s >= cfg.fontSizeMin) {
            fill.textSize = s.toFloat()
            val ls = wrapCjk(text, fill, (bw - cfg.rowTrim * s).coerceAtLeast(s.toFloat())) // 每行少 rowTrim 字（橫向字數）
            val maxW = ls.maxOfOrNull { fill.measureText(it) } ?: 0f
            if (ls.size * s * 1.18f <= rowRoom && maxW <= bw) { size = s; lines = ls; break }
            s--
        }
        size = maxOf(cfg.fontSizeMin, (size * cfg.fontScale).roundToInt())  // 整體縮小、更 fit
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        stroke.strokeWidth = maxOf(2f, size * (if (onArt) cfg.artStrokeRatio else STROKE_RATIO))  // 描邊隨字級；壓畫面區用更粗白邊
        lines = wrapCjk(text, fill, (bw - cfg.rowTrim * size).coerceAtLeast(size.toFloat()))  // 縮小後重排（含 rowTrim）
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

    /**
     * Wraps text: CJK wraps char-by-char (with line-start kinsoku rule); Latin text only breaks
     * at whitespace so words are never split mid-word (no "hanging letters"). A single word that
     * is wider than the box is unavoidably split. '\n' forces a hard break.
     */
    private fun wrapCjk(text: String, paint: Paint, maxW: Float): List<String> {
        val lines = ArrayList<String>()
        val cur = StringBuilder()
        var lastSpaceIdx = -1 // index in cur right after the last whitespace (safe break point)
        for (ch in text) {
            if (ch == '\n') {
                lines.add(cur.toString()); cur.clear(); lastSpaceIdx = -1
                continue
            }
            if (cur.isNotEmpty()) {
                val w = paint.measureText(cur.toString() + ch)
                // Kinsoku: forbidden start chars (closing punctuation etc.) never start a new line,
                // even if it slightly overflows.
                if (w > maxW && ch !in NO_START) {
                    if (lastSpaceIdx > 0) {
                        // Break after the last word boundary so whole words stay together.
                        lines.add(cur.substring(0, lastSpaceIdx))
                        cur.delete(0, lastSpaceIdx)
                        lastSpaceIdx = -1
                    } else {
                        // No whitespace to break at (CJK or a single over-long word) -> break in place.
                        lines.add(cur.toString()); cur.clear()
                    }
                }
            }
            cur.append(ch)
            if (ch.isWhitespace()) lastSpaceIdx = cur.length // break point right after this space
        }
        if (cur.isNotEmpty()) lines.add(cur.toString())
        return lines
    }

    private const val ASCENT = 0.82f
    private const val STROKE_RATIO = 0.10f  // 描邊寬＝字級×此比例（隨字級縮放）
    private const val MAX_TCY = 4  // 縱中橫一格最多併幾個 ASCII（涵蓋 2 位數年齡、4 位數年份；更長退回逐字避免壓太扁）
    private const val ROTATE_CHARS = "ー－—―‐~〜～…‥（）()「」『』【】〔〕［］｛｝〈〉《》＜＞<>｜|：;"
    // 行頭禁則：不可置於欄/行開頭（收尾標點、小假名）→ 併回前一欄/行（kinsoku）
    private const val NO_START = "、。，．：；！？”’）〕】｝」』》〉…‥ーゝゞヽヾ々ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ"
}
