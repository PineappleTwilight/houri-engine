package li.joye.yakuyomi.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 48px CTC OCR.
 *
 * Ported from manga_translator/ocr/model_48px_ctc.py (+ ocr/common.py, utils/generic.py) @ d5a3eee
 *   Crop: sortPnts determines vertical/horizontal + get_transformed_region (findHomography->warpPerspective->48px strip, vertical rotated 90°)
 *         here uses Android Matrix.setPolyToPoly instead of cv2 perspective (§6).
 *   Preprocess: (x-127.5)/127.5, NCHW, RGB.
 *   Decode: greedy CTC (blank=0, collapse repeats + strip blank) -> lookup dictionary.
 *   ignore_bubble (cfg.ignoreBubble, ported from utils/bubble.py): skip colored/non-bubble SFX text.
 *   Color head not used (colored backgrounds too noisy); text color instead determined by [Renderer] from post-inpaint background luminance.
 */
class Ocr(
    modelPath: String,
    private val dictionary: List<String>,
    private val cfg: OcrConfig = OcrConfig(),
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        // Concurrent mode: one thread per line (intra-op=1), fill cores via N concurrent lines; sequential mode: one line uses NUM_THREADS (current).
        val threads = if (cfg.concurrent) 1 else NUM_THREADS
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            if (cfg.useXnnpack) {
                try {
                    addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                } catch (t: Throwable) {
                    Log.w(TAG, "XNNPACK unavailable, fallback to CPU: ${t.message}")
                }
            }
        }
        session = env.createSession(modelPath, options) // Path load = native memory, does not occupy JVM heap
    }

    /**
     * Perform OCR on each text line, filling direction and text in place. Each line gets [PAD_MARGIN] white border on right (see [stripToTensor]) so CTC does not cut trailing chars.
     * [OcrConfig.concurrent]=true: concurrent multi-line (small tiles under-utilize intra-op -> switch to single-thread, fill cores via concurrency, see init); false: sequential per line (current).
     * Batch padding already rejected (width variance -> padding waste); here is "concurrency" (zero padding), different from batch.
     */
    suspend fun recognize(
        page: Bitmap,
        lines: List<TextLine>,
        bicubic: Boolean = cfg.useBicubic, // Crop scaling interpolation: true=hand-rolled bicubic (saves small kana), false=Canvas bilinear (current)
    ): Unit = coroutineScope {
        val inputName = session.inputNames.first()
        if (cfg.concurrent && lines.size > 1) {
            val sem = Semaphore(cfg.concurrency.coerceAtLeast(1))
            lines.map { line ->
                async(Dispatchers.Default) { sem.withPermit { recognizeOne(page, line, inputName, bicubic) } }
            }.awaitAll()
        } else {
            for (line in lines) recognizeOne(page, line, inputName, bicubic)
        }
    }

    /**
     * Warmup: run OCR session once on a blank strip to complete first lazy initialization of ORT session (arena/EP setup) on a single thread.
     * Call once before concurrent multi-page translation; strip content does not matter (just to trigger one run).
     */
    fun warmUp() {
        val strip = Bitmap.createBitmap(160, cfg.textHeight, Bitmap.Config.ARGB_8888)
        try {
            stripToTensor(strip).use { input ->
                session.run(mapOf(session.inputNames.first() to input)).use { }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "OCR warmup failed: ${t.message}")
        } finally {
            strip.recycle()
        }
    }

    /** Single-line OCR: crop -> preprocess -> CTC -> fill text. Thread-safe: only writes own line, session.run can be concurrent, rest is local/read-only. */
    private fun recognizeOne(page: Bitmap, line: TextLine, inputName: String, bicubic: Boolean) {
        // First expand, then sortPnts: sortPnts determines point order for warp, expand then sort avoids disorder (expansion itself does not change vertical/horizontal determination).
        val quad = if (cfg.stripPad > 0) expandQuad(line.quad, cfg.stripPad, page.width, page.height) else line.quad
        val (ordered, isV) = sortPnts(quad)
        line.direction = if (isV) "v" else "h"
        val strip = transformedRegion(page, ordered, isV, cfg.textHeight, bicubic) ?: return
        if (cfg.ignoreBubble in 1..50 && isIgnore(strip, cfg.ignoreBubble)) {
            strip.recycle()  // Colored / non-bubble SFX text -> skip
            return
        }
        try {
            stripToTensor(strip).use { input ->
                session.run(mapOf(inputName to input)).use { res ->
                    val logits = res.get(OUT_LOGITS).orElseThrow {
                        IllegalStateException("Missing output $OUT_LOGITS")
                    } as OnnxTensor
                    val (text, prob) = ctcDecode(logits)
                    if (prob >= cfg.minProb) line.text = text  // Low-confidence misread -> discard
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "OCR single-line failed: ${t.message}")
        } finally {
            strip.recycle()
        }
    }

    /**
     * Expand each side of quad by [pad] px for OCR crop ([OcrConfig.stripPad]): minAreaRect -> w,h each +2*pad -> corners -> clamp inside image.
     * **Only returns new points, does not modify [TextLine.quad]** => detection box and inpaint mask (via seg strokes) completely unaffected.
     * Aligned with desktop exp_pad.py:expand_quad (cv2.minAreaRect + boxPoints). Saves "thin box clips glyphs -> CTC empty read -> left untranslated".
     */
    private fun expandQuad(pts: List<Pt>, pad: Int, w: Int, h: Int): List<Pt> {
        val rect = Geometry.minAreaRect(pts) ?: return pts
        return rect.expand(pad.toFloat()).corners().map {
            Pt(it.x.coerceIn(0f, (w - 1).toFloat()), it.y.coerceIn(0f, (h - 1).toFloat()))
        }
    }

    /** Aligned with generic.py:sort_pnts — return sorted 4 points and whether vertical. */
    private fun sortPnts(quad: List<Pt>): Pair<List<Pt>, Boolean> {
        val n = quad.size
        var best0 = 0
        var best1 = 0
        // 16 pairs of vectors, take 8th and 10th longest (long edges)
        val norms = FloatArray(n * n)
        for (i in 0 until n) for (j in 0 until n) {
            val vx = quad[i].x - quad[j].x
            val vy = quad[i].y - quad[j].y
            norms[i * n + j] = hypot(vx.toDouble(), vy.toDouble()).toFloat()
        }
        val order = (0 until n * n).sortedBy { norms[it] }
        best0 = order[8]
        best1 = order[10]
        var l0x = quad[best0 / n].x - quad[best0 % n].x
        var l0y = quad[best0 / n].y - quad[best0 % n].y
        val l1x = quad[best1 / n].x - quad[best1 % n].x
        val l1y = quad[best1 / n].y - quad[best1 % n].y
        if (l0x * l1x + l0y * l1y < 0f) { l0x = -l0x; l0y = -l0y }
        val sx = abs((l0x + l1x) / 2f)
        val sy = abs((l0y + l1y) / 2f)
        val isV = sx <= sy

        return if (isV) {
            val byY = quad.sortedBy { it.y }
            val first2 = listOf(byY[0], byY[1]).sortedBy { it.x }
            val last2 = listOf(byY[2], byY[3]).sortedBy { it.x }
            Pair(listOf(first2[0], first2[1], last2[1], last2[0]), true)
        } else {
            val byX = quad.sortedBy { it.x }
            val ls = listOf(byX[0], byX[1]).sortedBy { it.y } // Left: top, bottom
            val rs = listOf(byX[2], byX[3]).sortedBy { it.y } // Right: top, bottom
            Pair(listOf(ls[0], rs[0], rs[1], ls[1]), false)
        }
    }

    /** Aligned with generic.py:Quadrilateral.get_transformed_region. When [bicubic]=true warp uses hand-rolled bicubic sampling (see [bicubicWarp]). */
    private fun transformedRegion(page: Bitmap, pts: List<Pt>, isV: Boolean, th: Int, bicubic: Boolean): Bitmap? {
        // Structure edge midpoints
        val p1x = (pts[0].x + pts[1].x) / 2f; val p1y = (pts[0].y + pts[1].y) / 2f
        val p2x = (pts[2].x + pts[3].x) / 2f; val p2y = (pts[2].y + pts[3].y) / 2f
        val p3x = (pts[1].x + pts[2].x) / 2f; val p3y = (pts[1].y + pts[2].y) / 2f
        val p4x = (pts[3].x + pts[0].x) / 2f; val p4y = (pts[3].y + pts[0].y) / 2f
        val vLen = hypot((p2x - p1x).toDouble(), (p2y - p1y).toDouble()).toFloat()
        val hLen = hypot((p4x - p3x).toDouble(), (p4y - p3y).toDouble()).toFloat()
        val ratio = vLen / max(hLen, 1e-6f)

        val w: Int
        val h: Int
        if (!isV) {
            h = max(th, 2)
            w = max((th / ratio).roundToInt(), 2)
        } else {
            w = max(th, 2)
            h = max((th * ratio).roundToInt(), 2)
        }

        val src = floatArrayOf(
            pts[0].x, pts[0].y, pts[1].x, pts[1].y,
            pts[2].x, pts[2].y, pts[3].x, pts[3].y,
        )
        val dst = floatArrayOf(
            0f, 0f, (w - 1).toFloat(), 0f,
            (w - 1).toFloat(), (h - 1).toFloat(), 0f, (h - 1).toFloat(),
        )
        val m = Matrix()
        if (!m.setPolyToPoly(src, 0, dst, 0, 4)) return null

        // Bicubic: hand-rolled perspective bicubic sampling (Android Canvas only has bilinear); bilinear blurs ~30px vertical strip upscaled to 48px
        // and loses small kana -> OCR misses (even sentence-ending negation -> meaning inversion). Parity measured bicubic 517 chars/100 lines vs bilinear 486/96.
        val region = if (bicubic) {
            bicubicWarp(page, m, w, h, pts) ?: return null
        } else {
            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                .also { Canvas(it).drawBitmap(page, m, Paint(Paint.FILTER_BITMAP_FLAG)) }
        }
        if (!isV) return region

        // Vertical: cv2.ROTATE_90_COUNTERCLOCKWISE -> height becomes 48
        val rot = Matrix().apply { postRotate(-90f) }
        // filter=false (lossless transpose): exact 90° rotation needs no interpolation; filter=true would add another bilinear blur on top of warp's bilinear
        // and blur small kana -> OCR misses. Rotation itself has no scaling, turning off filter is lossless and faster (see stripToTensor sharpening).
        val rotated = Bitmap.createBitmap(region, 0, 0, w, h, rot, false)
        region.recycle()
        return rotated
    }

    // Hand-rolled perspective bicubic warp (a=-0.75 cubic convolution): bit-exact with parity verification script verify_handcubic(a=-0.75)
    // -> 517 chars / 100 lines over confidence threshold (vs current bilinear 486/96), restores small kana blurred by bilinear scaling (including sentence-ending negation -> meaning inversion).
    // Android Canvas has no bicubic so hand-rolled. Performance: strip small (avg ~13K px), extra ~12M multiply-add per page ~ +1-4% OCR time (parity est).
    private fun bicubicWarp(page: Bitmap, forward: Matrix, w: Int, h: Int, pts: List<Pt>): Bitmap? {
        val inv = Matrix()
        if (!forward.invert(inv)) return null
        val iv = FloatArray(9)
        inv.getValues(iv) // 3x3: sx=(iv0*dx+iv1*dy+iv2)/(iv6*dx+iv7*dy+iv8), sy similar (Android Matrix perspective same as cv2)
        // Only crop quad source bounding box (±2 for 4-tap) -> single getPixels, save memory
        val pw = page.width
        val ph = page.height
        val x0 = (floor(pts.minOf { it.x }) - 2f).toInt().coerceIn(0, pw - 1)
        val y0 = (floor(pts.minOf { it.y }) - 2f).toInt().coerceIn(0, ph - 1)
        val x1 = (ceil(pts.maxOf { it.x }) + 2f).toInt().coerceIn(0, pw - 1)
        val y1 = (ceil(pts.maxOf { it.y }) + 2f).toInt().coerceIn(0, ph - 1)
        val cw = x1 - x0 + 1
        val ch = y1 - y0 + 1
        if (cw < 2 || ch < 2) return null
        val src = IntArray(cw * ch)
        page.getPixels(src, 0, cw, x0, y0, cw, ch)
        val out = IntArray(w * h)
        for (dy in 0 until h) {
            for (dx in 0 until w) {
                val den = iv[6] * dx + iv[7] * dy + iv[8]
                val sx = (iv[0] * dx + iv[1] * dy + iv[2]) / den - x0
                val sy = (iv[3] * dx + iv[4] * dy + iv[5]) / den - y0
                out[dy * w + dx] = bicubicSample(src, cw, ch, sx, sy)
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    /** Sample [px] (cw x ch, ARGB) at (sx,sy) with 4x4 bicubic (a=-0.75, clamp borders, per-channel). */
    private fun bicubicSample(px: IntArray, cw: Int, ch: Int, sx: Float, sy: Float): Int {
        val a = -0.75f
        val fx = floor(sx).toInt()
        val fy = floor(sy).toInt()
        var r = 0f
        var g = 0f
        var b = 0f
        for (my in -1..2) {
            val wy = cubicW(sy - (fy + my), a)
            val rowBase = (fy + my).coerceIn(0, ch - 1) * cw
            var rr = 0f
            var gg = 0f
            var bb = 0f
            for (mx in -1..2) {
                val wx = cubicW(sx - (fx + mx), a)
                val p = px[rowBase + (fx + mx).coerceIn(0, cw - 1)]
                rr += wx * ((p shr 16) and 0xFF)
                gg += wx * ((p shr 8) and 0xFF)
                bb += wx * (p and 0xFF)
            }
            r += wy * rr
            g += wy * gg
            b += wy * bb
        }
        return (0xFF shl 24) or
            (r.roundToInt().coerceIn(0, 255) shl 16) or
            (g.roundToInt().coerceIn(0, 255) shl 8) or
            b.roundToInt().coerceIn(0, 255)
    }

    /** Cubic convolution kernel (Horner expansion): |t|<=1 and 1<|t|<2 segments, else 0. */
    private fun cubicW(t: Float, a: Float): Float {
        val x = abs(t)
        return when {
            x <= 1f -> ((a + 2f) * x - (a + 3f)) * x * x + 1f
            x < 2f -> ((a * x - 5f * a) * x + 8f * a) * x - 4f * a
            else -> 0f
        }
    }

    /**
     * Unsharp mask (in-place sharpen [px]=strip ARGB pixels, [w]x[h]): counteracts blur from warping ~30px wide vertical strip upscaled to 48px.
     * That blur smears small kana (ni/i/do/n) into blobs -> CTC collapses -> whole sentence missed (device read "narenai mono nado" as "mono").
     *
     * Method = per-channel separable Gaussian (sigma~1.2, 5-tap) blur, then `orig + AMOUNT*(orig-blur)`, clamp 0..255.
     * Parity measured (this preprocessing + sharpen amount=1.6): whole chapter readable chars 255->431, lines over confidence threshold (minProb 0.5) 60->90 (/101);
     * no side effect on already correct (p~1.0) clean lines. Ported from `parity` verification script verify_myunsharp.py:my_unsharp (bit-exact).
     * Sharpen factor and interpolation both determined by parity benchmark (bilinear+unsharp achieves ~93% of bicubic+unsharp benefit, Android has no bicubic so choose this).
     */
    private fun unsharp(px: IntArray, w: Int, h: Int) {
        val n = w * h
        if (n == 0) return
        val amount = 1.6f
        val k0 = 0.3434f
        val k1 = 0.2428f
        val k2 = 0.0855f // Gaussian sigma~1.2 normalized 5-tap (center/ +-1/ +-2)
        for (shift in intArrayOf(16, 8, 0)) { // R, G, B each sharpened (text mostly grayscale, but colored SFX also handled)
            val ch = FloatArray(n)
            for (i in 0 until n) ch[i] = ((px[i] shr shift) and 0xFF).toFloat()
            val tmp = FloatArray(n)
            for (y in 0 until h) { // Horizontal blur (border clamp copy)
                val row = y * w
                for (x in 0 until w) {
                    tmp[row + x] = k2 * ch[row + max(0, x - 2)] + k1 * ch[row + max(0, x - 1)] +
                        k0 * ch[row + x] + k1 * ch[row + min(w - 1, x + 1)] + k2 * ch[row + min(w - 1, x + 2)]
                }
            }
            for (y in 0 until h) { // Vertical blur + sharpen write back to px (only modify this channel's 8 bits, preserve rest)
                val row = y * w
                val rm2 = max(0, y - 2) * w
                val rm1 = max(0, y - 1) * w
                val rp1 = min(h - 1, y + 1) * w
                val rp2 = min(h - 1, y + 2) * w
                for (x in 0 until w) {
                    val blur = k2 * tmp[rm2 + x] + k1 * tmp[rm1 + x] + k0 * tmp[row + x] +
                        k1 * tmp[rp1 + x] + k2 * tmp[rp2 + x]
                    val orig = ch[row + x]
                    val v = (orig + amount * (orig - blur)).coerceIn(0f, 255f).toInt()
                    val i = row + x
                    px[i] = (px[i] and (0xFF shl shift).inv()) or (v shl shift)
                }
            }
        }
    }

    private fun stripToTensor(strip: Bitmap): OnnxTensor {
        val sw = strip.width
        val h = strip.height
        val w = sw + PAD_MARGIN // Right white border: prevent CTC from cutting trailing chars (Saka -> Sakamoto, neene -> neenee)
        val px = IntArray(sw * h)
        strip.getPixels(px, 0, sw, 0, 0, sw, h)
        if (cfg.ocrUnsharp) unsharp(px, sw, h) // Sharpen: counteract scaling blur, restore missed small kana (see [unsharp], OcrConfig.ocrUnsharp)
        val area = h * w
        val chw = FloatArray(3 * area) { 1f } // White background ((255-127.5)/127.5=1.0), right side stays white
        for (y in 0 until h) {
            val inRow = y * sw
            val outRow = y * w
            for (x in 0 until sw) {
                val p = px[inRow + x]
                chw[outRow + x] = (((p shr 16) and 0xFF) - 127.5f) / 127.5f
                chw[area + outRow + x] = (((p shr 8) and 0xFF) - 127.5f) / 127.5f
                chw[2 * area + outRow + x] = ((p and 0xFF) - 127.5f) / 127.5f
            }
        }
        return OnnxTensor.createTensor(
            env, FloatBuffer.wrap(chw), longArrayOf(1, 3, h.toLong(), w.toLong()),
        )
    }

    /** Greedy CTC (single [1,T,d]) -> read arr then delegate to [ctcDecodeArr]. */
    private fun ctcDecode(logits: OnnxTensor): Pair<String, Float> {
        val shape = (logits.info as TensorInfo).shape // [1, T, dict]
        val t = shape[1].toInt()
        val d = shape[2].toInt()
        val arr = FloatArray(t * d)
        logits.floatBuffer.get(arr, 0, t * d)
        return ctcDecodeArr(arr, t, d)
    }

    /** Greedy CTC (blank=0, collapse repeats) + average confidence, aligned with decode_ctc_top1. Returns (text, prob). */
    private fun ctcDecodeArr(arr: FloatArray, t: Int, d: Int): Pair<String, Float> {
        val sb = StringBuilder()
        var last = BLANK
        var logpSum = 0.0
        var nChars = 0
        for (ti in 0 until t) {
            val base = ti * d
            var best = 0
            var bestV = arr[base]
            for (c in 1 until d) {
                val v = arr[base + c]
                if (v > bestV) { bestV = v; best = c }
            }
            if (best != last && best != BLANK) {
                val ch = dictionary[best]
                sb.append(if (ch == "<SP>") " " else ch)
                // Top-1 log_softmax = bestV - logsumexp(row) = -ln(sum exp(x-bestV))
                var s = 0.0
                for (c in 0 until d) s += Math.exp((arr[base + c] - bestV).toDouble())
                logpSum += -Math.log(s)
                nChars++
            }
            last = best
        }
        val prob = if (nChars > 0) Math.exp(logpSum / nChars).toFloat() else 0f
        return sb.toString() to prob
    }

    /** SFX / non-bubble text determination (ported from utils/bubble.py:is_ignore @ d5a3eee): mixed border color (not clean bubble background) or colored -> skip. */
    private fun isIgnore(strip: Bitmap, ignoreBubble: Int): Boolean {
        val w = strip.width
        val h = strip.height
        if (w < 4 || h < 4) return false
        val px = IntArray(w * h)
        strip.getPixels(px, 0, w, 0, 0, w, h)
        fun dark(i: Int): Boolean {
            val p = px[i]
            return 0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF) < 127f
        }
        var black = 0
        var total = 0
        for (y in 0..1) for (x in 0 until w) { if (dark(y * w + x)) black++; total++ }
        for (y in h - 2 until h) for (x in 0 until w) { if (dark(y * w + x)) black++; total++ }
        for (y in 2 until h - 2) for (x in 0..1) { if (dark(y * w + x)) black++; total++ }
        for (y in 2 until h - 2) for (x in w - 2 until w) { if (dark(y * w + x)) black++; total++ }
        val ratio = if (total > 0) black.toDouble() / total * 100 else 0.0
        if (ratio >= ignoreBubble && ratio <= 100 - ignoreBubble) return true
        return checkColor(px)
    }

    /** Color text determination (ported from utils/bubble.py:check_color): >10 non-grayscale pixels -> True. */
    private fun checkColor(px: IntArray): Boolean {
        var n = 0
        for (p in px) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            val gray = 0.299 * r + 0.587 * g + 0.114 * b
            val dr = r - gray
            val dg = g - gray
            val db = b - gray
            if (dr * dr + dg * dg + db * db > 100) { n++; if (n > 10) return true }
        }
        return false
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "Ocr"
        private const val NUM_THREADS = 4
        private const val BLANK = 0
        private const val OUT_LOGITS = "char_logits"
        private const val PAD_MARGIN = 16  // White border on right side of each strip: gives CTC context, prevents truncating trailing chars
    }
}
