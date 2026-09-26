package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import kotlin.comparisons.maxOf
import kotlin.math.roundToInt
import kotlinx.coroutines.coroutineScope

/**
 * Text removal. Proven on device = two options, both pure NCNN (`.param`/`.bin`):
 *  - **boxfill (fast inpaint)**: flat-fill inpaint mask pixels per region with background color. Instant, no model run; perfect for clean white bubbles, flat color block on busy art.
 *  - **aot (AI inpaint, default)**: AOT-GAN (m-i-t manga weights) whole page scaled to [InpainterConfig.tileSize] (768) to reconstruct background, whole area.
 *    Fully convolutional -> any size; 768 = quality/memory/hidden-under-translation (§8 inpaint || translation overlap) sweet spot.
 *
 * AOT I/O contract (aligned with parity/inpaint_parity.py): img in [-1,1] and holes zeroed (m-i-t `img*(1-mask)`), mask in {0,1}(1=erase), output in [-1,1].
 * LaMa (whole page scaled 512 always blurry) and per-tile AOT (native res, CPU too expensive) both retired; GPU/Vulkan proven to miscompute AOT-GAN (see memory ncnn-vulkan-fp16).
 */
class Inpainter(
    modelPath: String,
    private val cfg: InpainterConfig = InpainterConfig(),
) : AutoCloseable {

    private var ncnnHandle: Long = 0L
    /** Actual backend in effect; without adb caller verifies via log/image. */
    val ep: String = "NCNN-CPU"

    init {
        check(modelPath.endsWith(".param")) { "Inpaint requires NCNN `.param` model (AOT-GAN): $modelPath" }
        check(NcnnBackend.available) { "NCNN native library not loaded, cannot inpaint" }
        val bin = modelPath.removeSuffix(".param") + ".bin"
        ncnnHandle = NcnnBackend.createNet(modelPath, bin)
        check(ncnnHandle != 0L) { "Failed to load NCNN AOT model: $modelPath" }
    }

    suspend fun inpaint(page: Bitmap, regions: List<TextRegion>, textMask: Bitmap, render: RenderConfig = RenderConfig()): Bitmap = coroutineScope {
        val w = page.width
        val h = page.height
        val result = page.copy(Bitmap.Config.ARGB_8888, true)
        // Mask = entire "expanded text box" of translation regions (see buildSegMask): new translated text (especially long LTR text in vertical boxes rotated 90°) must have clean background at its landing spot.
        val maskPx = buildSegMask(regions, textMask, w, h, render)

        if (cfg.method == "boxfill") {
            // Per-region flat fill with background color: clean white bubbles with no residue, busy areas are flat color blocks (use aot for quality).
            val px = IntArray(w * h); result.getPixels(px, 0, w, 0, 0, w, h)
            val tightPx = IntArray(w * h); textMask.getPixels(tightPx, 0, w, 0, 0, w, h)
            for (r in regions) {
                val s = bgStats(px, tightPx, r, w, h)
                r.onArt = false; r.dbgStd = s.std; r.dbgWhite = s.meanLum // dbg values for sandbox inpaint comparison boxes
                flatFill(result, maskPx, r, s.color, cfg.bboxPad, w, h)
            }
            return@coroutineScope result
        }

        // aot (default): quality-first fast path. Uniform bubbles are solid-filled up front through the
        // glyph mask (never the whole expanded box, which would erase crisp bubble borders) and excluded
        // from the AOT mask; only textured/art regions run AOT-GAN reconstruction and are marked onArt so
        // Renderer uses the white-outline style.
        val px = IntArray(w * h); result.getPixels(px, 0, w, 0, 0, w, h)
        val tightPx = IntArray(w * h); textMask.getPixels(tightPx, 0, w, 0, 0, w, h)
        val artRegions = ArrayList<TextRegion>()
        val uniformRegions = ArrayList<TextRegion>()
        if (cfg.uniformFastPath) {
            for (r in regions) {
                if (BubbleUniformity.classify(px, w, h, r)) {
                    r.onArt = false
                    uniformRegions.add(r)
                } else {
                    r.onArt = true
                    artRegions.add(r)
                }
            }
        } else {
            regions.forEach { it.onArt = true }
            artRegions.addAll(regions)
        }
        // Filled and excluded areas are the same set by construction: both derive from maskPx, the only
        // difference being which regions. Non-uniform regions never enter the fill mask.
        var uniPx: IntArray? = null
        if (uniformRegions.isNotEmpty()) {
            val gate = regionMask(maskPx, uniformRegions, w, h, render)
            for (r in uniformRegions) {
                val color = BubbleUniformity.meanColor(px, w, h, r) ?: bgStats(px, tightPx, r, w, h).color
                flatFill(result, gate, r, color, cfg.bboxPad, w, h)
            }
            uniPx = gate
        }
        if (artRegions.isEmpty()) return@coroutineScope result // All uniform: fills done, skip AOT entirely.
        // AOT mask = full mask minus the solid-filled tight uniform areas (copy: flat-fill gating mask survives).
        val aotMask = maskPx.copyOf()
        val gate = uniPx
        if (gate != null) {
            for (i in gate.indices) if ((gate[i] and 0xFF) > 127) aotMask[i] = Color.BLACK
        }
        val maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        maskBmp.setPixels(aotMask, 0, w, 0, 0, w, h)
        val aotResult = try { runWholeAot(page, maskBmp, w, h) } catch (t: Throwable) { Log.w(TAG, "AOT failed, keeping boxfill fallback", t); null }
        if (aotResult != null) {
            compositePixels(result, aotMask, aotResult)
        } else {
            // AOT failed: flat-fill only the remaining art regions; uniform fills stay untouched (never worse).
            val fresh = IntArray(w * h); result.getPixels(fresh, 0, w, 0, 0, w, h)
            for (r in artRegions) {
                val s = bgStats(fresh, tightPx, r, w, h)
                flatFill(result, maskPx, r, s.color, cfg.bboxPad, w, h)
            }
        }
        maskBmp.recycle()
        result
    }

    /**
     * Warmup: aot method runs NCNN AOT session once on blank small image (first lazy init completes on single thread).
     * boxfill only flat-fills, does not run that session -> no warmup needed (and no cold collision). Call once before concurrent multi-page translation.
     */
    fun warmUp() {
        if (cfg.method == "boxfill") return
        val w = 64
        val h = 64
        val page = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        try {
            runWholeAot(page, mask, w, h)
        } catch (t: Throwable) {
            Log.w(TAG, "Inpaint warmup failed: ${t.message}")
        } finally {
            page.recycle()
            mask.recycle()
        }
    }

    /** Inpaint mask Bitmap (white = to be inpainted). For redraw material/visualization; same mask as inpaint. */
    fun buildMask(page: Bitmap, regions: List<TextRegion>, textMask: Bitmap, render: RenderConfig = RenderConfig()): Bitmap {
        val w = page.width; val h = page.height
        val maskPx = buildSegMask(regions, textMask, w, h, render)
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply { setPixels(maskPx, 0, w, 0, 0, w, h) }
    }

    /**
     * Inpaint mask (white = to be inpainted) = the **glyph strokes** of each region, clipped to that
     * region's expanded box, dilated by [InpainterConfig.maskDilate].
     *
     * Only the letters are removed, which is what makes the repair invisible: AOT-GAN fills a thin
     * glyph-shaped hole from its immediate surroundings, so hair, screentone and panel edges come
     * back intact, and a flat fill merely repaints the bubble it already sits in. Masking the whole
     * expanded box instead (the previous behaviour) handed AOT-GAN a rectangle several times the
     * glyph area — a 30x200 line is detected as ~90x260 by DBNet's 2.3x unclip, and the box
     * expands that again — which is a hole no tile-768 pass can reconstruct faithfully, so it came
     * back as the flat light patch over artwork.
     *
     * The region box is kept as a containment guard, and as a per-region fallback when a region has
     * readable text but the detector's stroke mask somehow missed it — leaving that text behind is
     * worse than removing a slightly oversized area. The fallback is deliberately **not** granted to
     * regions with no readable text: those are exactly the false detections (a face, a panel border),
     * and blanketing their whole box would chew holes in the artwork. Without readable text, glyph
     * coverage is the only evidence of real text, and no coverage means nothing is removed.
     */
    private fun buildSegMask(regions: List<TextRegion>, textMask: Bitmap, w: Int, h: Int, render: RenderConfig): IntArray {
        val radius = (cfg.maskDilate / 2f).roundToInt().coerceAtLeast(1)

        val glyph = IntArray(w * h)
        textMask.getPixels(glyph, 0, w, 0, 0, w, h)
        for (i in glyph.indices) if ((glyph[i] and 0xFF) <= 127) glyph[i] = Color.BLACK
        dilate(glyph, w, h, radius)

        val boxes = ArrayList<RegionRect>(regions.size)
        val allow = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(allow).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (region in regions) {
                val b = maskBox(region, render, w, h)
                boxes.add(b)
                drawRect(b.x0, b.y0, b.x1, b.y1, p)
            }
        }
        val boxPx = IntArray(w * h)
        allow.getPixels(boxPx, 0, w, 0, 0, w, h)
        allow.recycle()

        val mask = IntArray(w * h)
        for (i in mask.indices) {
            val on = (glyph[i] and 0xFF) > 127 && (boxPx[i] and 0xFF) > 127
            mask[i] = if (on) Color.WHITE else Color.BLACK
        }
        for ((index, region) in regions.withIndex()) {
            if (region.sourceText.isBlank()) continue
            if (countMaskIn(mask, w, h, boxes[index]) > MIN_GLYPH_PIXELS) continue
            val b = RegionBounds.clamped(RegionBounds.expanded(region, render.expandW, render.expandH, cfg.bboxPad), w, h)
            val x0 = b.x0.toInt().coerceIn(0, w - 1)
            val y0 = b.y0.toInt().coerceIn(0, h - 1)
            val x1 = b.x1.toInt().coerceIn(x0 + 1, w)
            val y1 = b.y1.toInt().coerceIn(y0 + 1, h)
            for (y in y0 until y1) {
                val row = y * w
                for (x in x0 until x1) mask[row + x] = Color.WHITE
            }
        }
        return mask
    }

    private fun countMaskIn(mask: IntArray, w: Int, h: Int, b: RegionRect): Int {
        val x0 = b.x0.toInt().coerceIn(0, w - 1)
        val y0 = b.y0.toInt().coerceIn(0, h - 1)
        val x1 = b.x1.toInt().coerceIn(x0 + 1, w)
        val y1 = b.y1.toInt().coerceIn(y0 + 1, h)
        var n = 0
        for (y in y0 until y1) {
            val row = y * w
            for (x in x0 until x1) if ((mask[row + x] and 0xFF) > 127) n++
        }
        return n
    }

    /**
     * Expanded mask box for a region: [RegionBounds] expansion/clamping semantics (portrait
     * 2.5 threshold, expandW/expandH swap, bbox pad) plus a bounded K3-inspired 2.5% ring margin.
     * Single source of truth for mask drawing and uniform-fill clearing, so both cover the same area.
     */
    private fun maskBox(region: TextRegion, render: RenderConfig, w: Int, h: Int): RegionRect {
        val ref = maxOf(region.x1 - region.x0, region.y1 - region.y0)
        val pad = RegionBounds.maskPad(cfg.bboxPad, cfg.maskDilate, ref)
        return RegionBounds.expandedClamped(region, w, h, render.expandW, render.expandH, pad)
    }

    /**
     * Restricts a mask to the given regions' [maskBox] rectangles, so a flat fill only ever touches
     * the bubbles it was routed for. Because the source is already the glyph mask, both removal
     * paths clear the same area and neither can paint past a bubble edge.
     */
    private fun regionMask(base: IntArray, regions: List<TextRegion>, w: Int, h: Int, render: RenderConfig): IntArray {
        val allow = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(allow).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (region in regions) {
                val b = maskBox(region, render, w, h)
                drawRect(b.x0, b.y0, b.x1, b.y1, p)
            }
        }
        val boxPx = IntArray(w * h)
        allow.getPixels(boxPx, 0, w, 0, 0, w, h)
        allow.recycle()
        val out = IntArray(w * h)
        for (i in out.indices) {
            val on = (base[i] and 0xFF) > 127 && (boxPx[i] and 0xFF) > 127
            out[i] = if (on) Color.WHITE else Color.BLACK
        }
        return out
    }

    /** Binary mask separable dilation (horizontal then vertical max-filter), radius pixels. Covers stroke anti-aliased edges, gives inpaint margin. */
    private fun dilate(px: IntArray, w: Int, h: Int, radius: Int) {
        if (radius <= 0) return
        val tmp = IntArray(px.size)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var on = false; var k = -radius
                while (k <= radius) { val xx = x + k; if (xx in 0 until w && (px[row + xx] and 0xFF) > 127) { on = true; break }; k++ }
                tmp[row + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
        for (x in 0 until w) {
            for (y in 0 until h) {
                var on = false; var k = -radius
                while (k <= radius) { val yy = y + k; if (yy in 0 until h && (tmp[yy * w + x] and 0xFF) > 127) { on = true; break }; k++ }
                px[y * w + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
    }

    private class BgStat(val meanLum: Float, val std: Float, val color: Int)

    /**
     * Mean luminance + std + average color of "non-text (background)" pixels inside region bbox. tightPx = undilated textMask (can measure white between strokes).
     * boxfill uses [BgStat.color] for flat fill; std/meanLum only for sandbox inpaint comparison boxes (for reference). Use line quad polygon local mask to avoid axis-aligned bbox corner noise.
     */
    private fun bgStats(px: IntArray, tightPx: IntArray, region: TextRegion, w: Int, h: Int): BgStat {
        val x0 = region.x0.toInt().coerceIn(0, w - 1)
        val y0 = region.y0.toInt().coerceIn(0, h - 1)
        val x1 = region.x1.toInt().coerceIn(x0 + 1, w)
        val y1 = region.y1.toInt().coerceIn(y0 + 1, h)
        val bw = x1 - x0; val bh = y1 - y0
        val qmBmp = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888)
        Canvas(qmBmp).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (line in region.lines) {
                val q = line.quad
                if (q.size < 4) continue
                val path = android.graphics.Path().apply {
                    moveTo(q[0].x - x0, q[0].y - y0)
                    for (i in 1..3) lineTo(q[i].x - x0, q[i].y - y0)
                    close()
                }
                drawPath(path, p)
            }
        }
        val qm = IntArray(bw * bh)
        qmBmp.getPixels(qm, 0, bw, 0, 0, bw, bh)
        qmBmp.recycle()
        var n = 0; var sl = 0.0; var sl2 = 0.0; var sr = 0L; var sg = 0L; var sb = 0L
        for (y in 0 until bh) {
            for (x in 0 until bw) {
                if ((qm[y * bw + x] and 0xFF) <= 127) continue
                val gi = (y0 + y) * w + (x0 + x)
                if ((tightPx[gi] and 0xFF) > 127) continue // Text pixel
                val p = px[gi]
                val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
                val lum = 0.299 * r + 0.587 * g + 0.114 * b
                sl += lum; sl2 += lum * lum; sr += r; sg += g; sb += b; n++
            }
        }
        if (n < 16) return BgStat(255f, 0f, Color.WHITE)
        val mean = sl / n
        val std = kotlin.math.sqrt((sl2 / n - mean * mean).coerceAtLeast(0.0))
        return BgStat(mean.toFloat(), std.toFloat(), Color.rgb((sr / n).toInt(), (sg / n).toInt(), (sb / n).toInt()))
    }

    /** White bubble inpaint = flat-fill inpaint mask pixels inside region bbox (expanded by pad) with background color. Uniform white bubbles guarantee no residue. */
    private fun flatFill(result: Bitmap, maskPx: IntArray, region: TextRegion, color: Int, pad: Int, w: Int, h: Int) {
        val x0 = (region.x0.toInt() - pad).coerceIn(0, w - 1)
        val y0 = (region.y0.toInt() - pad).coerceIn(0, h - 1)
        val x1 = (region.x1.toInt() + pad).coerceIn(x0 + 1, w)
        val y1 = (region.y1.toInt() + pad).coerceIn(y0 + 1, h)
        val bw = x1 - x0; val bh = y1 - y0
        val sub = IntArray(bw * bh)
        result.getPixels(sub, 0, bw, x0, y0, bw, bh)
        for (y in 0 until bh) {
            val mrow = (y0 + y) * w + x0
            val row = y * bw
            for (x in 0 until bw) if ((maskPx[mrow + x] and 0xFF) > 127) sub[row + x] = color
        }
        result.setPixels(sub, 0, bw, x0, y0, bw, bh)
    }

    private class WinOut(val x0: Int, val y0: Int, val ww: Int, val wh: Int, val px: IntArray)

    private fun runWholeAot(page: Bitmap, maskBmp: Bitmap, w: Int, h: Int): WinOut? {
        val t = cfg.tileSize
        val imgScaled: Bitmap
        val maskScaled: Bitmap
        if (cfg.preserveAspect) {
            imgScaled = letterboxScale(page, t)
            maskScaled = letterboxScale(maskBmp, t, filter = false)
        } else {
            imgScaled = Bitmap.createScaledBitmap(page, t, t, true)
            maskScaled = Bitmap.createScaledBitmap(maskBmp, t, t, false)
        }
        return try {
            val imgChw = aotImageChw(imgScaled, maskScaled, t)
            val maskArr = maskArr(maskScaled, t)
            val out = FloatArray(3 * t * t)
            val rc = NcnnBackend.inpaintAot(ncnnHandle, imgChw, maskArr, t, out)
            check(rc == 0) { "NCNN AOT inference failed rc=$rc" }
            val resScaled = aotArrToBitmap(out, t)
            val resWin = Bitmap.createScaledBitmap(resScaled, w, h, true)
            val px = IntArray(w * h)
            resWin.getPixels(px, 0, w, 0, 0, w, h)
            if (resWin !== resScaled) resScaled.recycle()
            resWin.recycle()
            WinOut(0, 0, w, h, px)
        } catch (t2: Throwable) {
            Log.w(TAG, "AOT inpaint failed: ${t2.message}"); null
        } finally {
            if (imgScaled !== page) imgScaled.recycle()
            if (maskScaled !== maskBmp) maskScaled.recycle()
        }
    }

    private fun letterboxScale(src: Bitmap, tile: Int, filter: Boolean = true): Bitmap {
        val sw = src.width; val sh = src.height
        val scale = minOf(tile.toFloat() / sw, tile.toFloat() / sh)
        val nw = (sw * scale).toInt().coerceAtLeast(1); val nh = (sh * scale).toInt().coerceAtLeast(1)
        val scaled = Bitmap.createScaledBitmap(src, nw, nh, filter)
        if (nw == tile && nh == tile) return scaled
        val out = Bitmap.createBitmap(tile, tile, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.GRAY)
        val dx = (tile - nw) / 2f; val dy = (tile - nh) / 2f
        canvas.drawBitmap(scaled, dx, dy, null)
        scaled.recycle()
        return out
    }

    private fun aotImageChw(imgBmp: Bitmap, maskBmp: Bitmap, n: Int): FloatArray {
        val area = n * n
        val px = IntArray(area); imgBmp.getPixels(px, 0, n, 0, 0, n, n)
        val mp = IntArray(area); maskBmp.getPixels(mp, 0, n, 0, 0, n, n)
        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            if ((mp[i] and 0xFF) > 127) continue
            val p = px[i]
            chw[i] = ((p shr 16) and 0xFF) / 127.5f - 1f
            chw[area + i] = ((p shr 8) and 0xFF) / 127.5f - 1f
            chw[2 * area + i] = (p and 0xFF) / 127.5f - 1f
        }
        return chw
    }

    private fun maskArr(bmp: Bitmap, n: Int): FloatArray {
        val px = IntArray(n * n); bmp.getPixels(px, 0, n, 0, 0, n, n)
        val m = FloatArray(n * n)
        for (i in px.indices) m[i] = if ((px[i] and 0xFF) > 127) 1f else 0f
        return m
    }

    private fun aotArrToBitmap(arr: FloatArray, n: Int): Bitmap {
        val area = n * n
        val px = IntArray(area)
        for (i in 0 until area) {
            val r = ((arr[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val g = ((arr[area + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val b = ((arr[2 * area + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, n, n, Bitmap.Config.ARGB_8888)
    }

    private fun compositePixels(result: Bitmap, maskPx: IntArray, o: WinOut) {
        val w = result.width
        val cur = IntArray(o.ww * o.wh)
        result.getPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
        val r = cfg.featherRadius.coerceIn(0, 4)
        if (r == 0) {
            for (y in 0 until o.wh) {
                val maskRow = (o.y0 + y) * w + o.x0
                val row = y * o.ww
                for (x in 0 until o.ww) if ((maskPx[maskRow + x] and 0xFF) > 127) cur[row + x] = o.px[row + x]
            }
        } else {
            for (y in 0 until o.wh) {
                val maskRow = (o.y0 + y) * w + o.x0
                val row = y * o.ww
                for (x in 0 until o.ww) {
                    if ((maskPx[maskRow + x] and 0xFF) <= 127) continue
                    var edgeDist = r + 1
                    for (dy in -r..r) for (dx in -r..r) {
                        val nx = x + dx; val ny = y + dy
                        if (nx !in 0 until o.ww || ny !in 0 until o.wh) continue
                        if ((maskPx[(o.y0 + ny) * w + o.x0 + nx] and 0xFF) <= 127) {
                            val d = kotlin.math.abs(dx) + kotlin.math.abs(dy)
                            if (d < edgeDist) edgeDist = d
                        }
                    }
                    if (edgeDist > r) {
                        cur[row + x] = o.px[row + x]
                    } else {
                        val alpha = edgeDist.toFloat() / (r + 1)
                        val src = o.px[row + x]; val dst = cur[row + x]
                        val sr = (src shr 16) and 0xFF; val sg = (src shr 8) and 0xFF; val sb = src and 0xFF
                        val dr = (dst shr 16) and 0xFF; val dg = (dst shr 8) and 0xFF; val db = dst and 0xFF
                        val mr = (sr * (1 - alpha) + dr * alpha).toInt()
                        val mg = (sg * (1 - alpha) + dg * alpha).toInt()
                        val mb = (sb * (1 - alpha) + db * alpha).toInt()
                        cur[row + x] = Color.rgb(mr, mg, mb)
                    }
                }
            }
        }
        result.setPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
    }

    override fun close() {
        if (ncnnHandle != 0L) { NcnnBackend.releaseNet(ncnnHandle); ncnnHandle = 0L }
    }

    companion object {
        private const val TAG = "Inpainter"

        private const val MIN_GLYPH_PIXELS = 16
    }
}
