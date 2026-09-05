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

/** Layout direction. AUTO = based on content (CJK vertical, pure alphanumerics horizontal), corresponds to m-i-t config direction:auto. */
enum class TextOrientation { VERTICAL, HORIZONTAL, AUTO }

/**
 * Typesetting (pure text-box method, reliable): position + size both use text box, box moderately enlarged ([RenderConfig.expandW]/[RenderConfig.expandH]) for breathing room.
 * Aligned with parity/typeset_parity.py (layer 2: same input near output).
 *   Does not rely on bubble flood-fill — adjacent bubbles would connect into one block and be miscomputed, deprecated.
 *   Vertical: CJK top->bottom, columns right->left, top-aligned, each column trims [RenderConfig.colTrim] chars (shorten column, reduce overflow), punctuation rotation, short ASCII strings as tate-chu-yoko (see [RenderConfig.tateChuYoko]).
 *   Horizontal: left->right, top->bottom, top-aligned.
 *   Text color: [RenderConfig.colorMode]=auto picks black/white from background luminance after inpaint (white bg black text / black bg white text); OCR color head hue too noisy not used.
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
                TextOrientation.AUTO -> region.direction == "v" && isCjk(text)
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
     * Estimate original font size (px) for a region: for each line take min(line box width, height) = stroke thickness (both vertical/horizontal), then median.
     * This is the anchor for "translated text should be same size as original": short translations (e.g., short English vs long Japanese) no longer inflate to fill the bubble.
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

    /** Text color (fill, outline): auto = pick from background luminance after inpaint (dark bg white text, light bg black text, aligned with parity auto_colors); mono = black text white outline; other = fixed color. */
    private fun textColors(page: Bitmap, r: TextRegion, cfg: RenderConfig): Pair<Int, Int> {
        if (cfg.colorMode == "mono") return Color.BLACK to Color.WHITE
        // User-specified fixed text color (default pure black): outline still determined by background luminance to ensure readability on any background.
        if (cfg.colorMode != "auto") return cfg.fixedTextColor to (if (bgLuminance(page, r.x0, r.y0, r.x1, r.y1) < cfg.bgDark) Color.WHITE else Color.BLACK)
        // On art (lama-reconstructed busy background): always black text + thick white outline. Outline frames text, readable on any busy background (aligned with m-i-t).
        if (r.onArt) return Color.BLACK to Color.WHITE
        val lum = bgLuminance(page, r.x0, r.y0, r.x1, r.y1)
        return if (lum < cfg.bgDark) Color.WHITE to Color.BLACK else Color.BLACK to Color.WHITE
    }

    /** Average luminance of background after inpaint inside the text box (Rec.601). */
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

/** Vertical: columns right->left, cells top->bottom, top-aligned; size fills enlarged text box, each column trims colTrim chars. Each cell = 1 char or 1 tate-chu-yoko short string. */
    private fun drawVertical(canvas: Canvas, x0: Float, y0: Float, x1: Float, y1: Float, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig, onArt: Boolean = false, originalSize: Int = 0) {
        val chars = text.filter { it != '\n' }
        if (chars.isEmpty()) return
        val cells = toVerticalCells(chars, cfg.tateChuYoko)  // Split into cells: normal char one cell, short ASCII strings merge into one tate-chu-yoko cell
        val bw = (x1 - x0) * cfg.expandW         // Width: enlarged text box width
        val colRoom = (y1 - y0) * cfg.expandH    // Vertical column available height (from top of text box downward)
        var size = cfg.fontSizeMin
        // Anchor start to original font size: translation should not be larger than original; long translations will shrink to fit.
        val cap = if (originalSize > 0) minOf(cfg.fontSizeMax, originalSize) else cfg.fontSizeMax
        var s = min(colRoom.toInt(), cap)
        while (s >= cfg.fontSizeMin) {
            val lh = s * 1.05f; val cw = s * 1.1f
            val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
            if (ceil(cells.size / cpc.toFloat()).toInt() * cw <= bw) { size = s; break }
            s--
        }
        size = maxOf(cfg.fontSizeMin, (size * cfg.fontScale).roundToInt())  // Overall shrink, more fit
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        stroke.strokeWidth = maxOf(2f, size * (if (onArt) cfg.artStrokeRatio else STROKE_RATIO))  // Outline scales with font size; onArt uses thicker outline
        val lh = size * 1.05f; val cw = size * 1.1f
        val cpc = maxOf(1, (colRoom / lh).toInt() - cfg.colTrim)
        val columns = splitColumnsV(cells, cpc)       // Kinsoku: columns should not start with forbidden starting chars
        val cols = columns.size
        val tcx = (x0 + x1) / 2f                  // Position: horizontally centered at text box center
        val rightCx = tcx + cols * cw / 2f - cw / 2f
        val blockH = columns.maxOf { it.size } * lh // Vertically centered: use longest column cell count as block height, centered in box
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
     * Vertical cell splitting: normal char one cell; consecutive short ASCII strings (2-[MAX_TCY] chars digits/letters/!?) merge into one tate-chu-yoko cell.
     * Single ASCII (e.g., lone "5") stays single cell; overlong strings (> MAX_TCY, e.g., long English words) fall back to char-by-char (avoid over-compression).
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

    /** Vertical column split + line-start kinsoku: forbidden start chars (single punctuation) should not start a column, merge back to previous column (max +2 to avoid explosion). */
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

    /** Tate-chu-yoko: lay short ASCII strings horizontally within one vertical cell, centered at column center; if wider than cell only compress horizontally (height unchanged, visually consistent with neighbors). */
    private fun drawTateChuYoko(canvas: Canvas, group: String, cx: Float, cyc: Float, cellW: Float, fill: Paint, stroke: Paint, border: Boolean) {
        val w = fill.measureText(group)
        val fm = fill.fontMetrics
        val baseline = cyc - (fm.ascent + fm.descent) / 2f
        val target = cellW * 0.92f
        val scaleX = if (w > target) target / w else 1f
        canvas.save()
        if (scaleX != 1f) canvas.scale(scaleX, 1f, cx, baseline)  // Only scale horizontally, around column center
        val tx = cx - w / 2f
        if (border) canvas.drawText(group, tx, baseline, stroke)
        canvas.drawText(group, tx, baseline, fill)
        canvas.restore()
    }

    /** 橫排：列上→下、字左→右、向上對齊；大小填滿放大後的文字框。直式原文的狹長框改旋轉 90° 排版，讓譯文沿長軸填滿（不再縮成一條直排柱）。壓在畫面上的自由文字（onArt）一律不轉：跟著偵測方向排，避免水平原文被轉成側躺。 */
    private fun drawHorizontal(canvas: Canvas, x0: Float, y0: Float, x1: Float, y1: Float, text: String, fill: Paint, stroke: Paint, cfg: RenderConfig, onArt: Boolean = false, originalSize: Int = 0) {
        // 對於非CJK英文，即使是直式原文的狹長氣泡也不應無條件旋轉 — 保持水平可讀性。
        // 只有當氣泡極度狹長（高寬比>2.5）且非onArt時才考慮旋轉，否則保持水平排版避免破壞文字框。
        val aspect = if ((x1 - x0) > 1f) (y1 - y0) / (x1 - x0) else 1f
        val portrait = !onArt && aspect > 2.5f && !isCjk(text)
        val wrapW = if (portrait) (y1 - y0) else (x1 - x0)
        val roomH = if (portrait) (x1 - x0) else (y1 - y0)
        val bw = wrapW * cfg.expandW
        val rowRoom = roomH * cfg.expandH
        var size = cfg.fontSizeMin
        var lines = listOf(text)
        val cap = if (originalSize > 0) minOf(cfg.fontSizeMax, originalSize) else cfg.fontSizeMax
        var s = min(rowRoom.toInt(), cap)
        while (s >= cfg.fontSizeMin) {
            fill.textSize = s.toFloat()
            // 窄框保護：rowTrim 對窄框不應過度扣除，否則可用寬度過小導致單詞級 mid-word 斷裂
            val trimW = if (bw < s * 8) cfg.rowTrim * s * 0.3f else cfg.rowTrim * s
            val ls = wrapCjk(text, fill, (bw - trimW).coerceAtLeast(s * 2f))
            val maxW = ls.maxOfOrNull { fill.measureText(it) } ?: 0f
            if (ls.size * s * 1.18f <= rowRoom && maxW <= bw) { size = s; lines = ls; break }
            s--
        }
        size = maxOf(cfg.fontSizeMin, (size * cfg.fontScale).roundToInt())
        fill.textSize = size.toFloat(); stroke.textSize = size.toFloat()
        stroke.strokeWidth = maxOf(2f, size * (if (onArt) cfg.artStrokeRatio else STROKE_RATIO))
        val finalTrim = if (bw < size * 8) cfg.rowTrim * size * 0.3f else cfg.rowTrim * size
        lines = wrapCjk(text, fill, (bw - finalTrim).coerceAtLeast(size * 2f))
        val lh = size * 1.18f
        if (portrait) {
            // 直式框：繞框心旋轉 90°（順時針），譯文沿長軸橫排、由上往下讀，填滿氣泡長邊。
            canvas.save()
            canvas.rotate(90f, (x0 + x1) / 2f, (y0 + y1) / 2f)
        }
        val tcx = (x0 + x1) / 2f
        var baseline = (y0 + y1) / 2f - lines.size * lh / 2f + size * ASCENT  // 垂直置中於框
        for (ln in lines) {
            val tx = tcx - fill.measureText(ln) / 2f
            if (cfg.fontBorder) canvas.drawText(ln, tx, baseline, stroke)
            canvas.drawText(ln, tx, baseline, fill)
            baseline += lh
        }
        if (portrait) canvas.restore()
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
