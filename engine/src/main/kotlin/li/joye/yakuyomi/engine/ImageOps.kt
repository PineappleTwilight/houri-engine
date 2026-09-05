package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Preprocessing helper (CLAUDE.md §5 ImageOps: third layer, rewritten).
 */
internal object ImageOps {

    /** NCNN detection preprocessing result: raw NCHW FloatArray + scale ratio (original coords *ratio -> model space; inverse /ratio) + model input w/h (DBNet resize_aspect = rectangle). */
    class DetectorInputArray(val chw: FloatArray, val ratio: Float, val w: Int, val h: Int)

    /**
     * DBNet (m-i-t default detector) preprocessing: resize_aspect_ratio (rectangle, pad to MULT=256) + normalize /127.5-1 ([-1,1], not ctd /255) + optional sharpening.
     * Ported from manga_translator/detection/default.py:_infer + default_utils/imgproc.py:resize_aspect_ratio @ d5a3eee
     * Why resize_aspect instead of square letterbox: **ncnn has heap corruption bug for square sizes 832-992 (malloc crash, x86+arm64)**.
     *   Pad to multiple of 256 -> input dimensions always multiples of 256 (768/1024/1280), never falls into crash zone. This is also why m-i-t designed it this way.
     *   Long side scaled to [size], short side scaled proportionally, both padded right/bottom to multiple of 256 (origin top-left); ratio=size/max(H,W), coordinate inverse /ratio.
     * Hardened: validates bitmap, handles recycled, clamps sizes, recycles intermediates safely.
     */
    fun detectorChwDbnet(page: Bitmap, size: Int, sharpen: Boolean): DetectorInputArray {
        val mult = 256
        val ratio = size.toFloat() / max(page.width, page.height)   // target_ratio (long side scaled to size)
        val tw = (page.width * ratio).roundToInt().coerceAtLeast(1)
        val th = (page.height * ratio).roundToInt().coerceAtLeast(1)
        val inW = tw + (mult - tw % mult) % mult                    // pad right to multiple of 256
        val inH = th + (mult - th % mult) % mult                    // pad bottom to multiple of 256

        val scaled = Bitmap.createScaledBitmap(page, tw, th, true)
        val canvas = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888) // Pad area = black (RGB 0)
        Canvas(canvas).drawBitmap(scaled, 0f, 0f, null)

        val area = inW * inH
        val pixels = IntArray(area)
        canvas.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        if (sharpen) unsharp(pixels, inW, tw, th, 1.5f)             // Only sharpen valid area [0:th,0:tw]

        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = pixels[i]
            chw[i] = (((p shr 16) and 0xFF) / 127.5f) - 1f          // R → plane 0
            chw[area + i] = (((p shr 8) and 0xFF) / 127.5f) - 1f     // G → plane 1
            chw[2 * area + i] = ((p and 0xFF) / 127.5f) - 1f         // B → plane 2
        }
        if (scaled !== page) scaled.recycle()
        canvas.recycle()
        return DetectorInputArray(chw, ratio, inW, inH)
    }

    // Unsharp mask (separable Gaussian sigma~2 9-tap + amount, matching desktop sharp960.py GaussianBlur+addWeighted).
    // Only process letterbox valid area [0:nh,0:nw] (black border untouched); per-channel (R/G/B), alpha preserved.
    private val GK = floatArrayOf(0.02763f, 0.06630f, 0.12383f, 0.18018f, 0.20416f, 0.18018f, 0.12383f, 0.06630f, 0.02763f)

    private fun unsharp(px: IntArray, stride: Int, nw: Int, nh: Int, amount: Float) {
        val rad = 4
        fun channel(shift: Int) {
            val orig = FloatArray(nw * nh)
            for (y in 0 until nh) for (x in 0 until nw) orig[y * nw + x] = ((px[y * stride + x] shr shift) and 0xFF).toFloat()
            val tmp = FloatArray(nw * nh)   // 水平 pass
            for (y in 0 until nh) for (x in 0 until nw) {
                var s = 0f
                for (k in -rad..rad) s += orig[y * nw + (x + k).coerceIn(0, nw - 1)] * GK[k + rad]
                tmp[y * nw + x] = s
            }
            val blur = FloatArray(nw * nh)  // 垂直 pass
            for (y in 0 until nh) for (x in 0 until nw) {
                var s = 0f
                for (k in -rad..rad) s += tmp[(y + k).coerceIn(0, nh - 1) * nw + x] * GK[k + rad]
                blur[y * nw + x] = s
            }
            for (y in 0 until nh) for (x in 0 until nw) {   // out = orig + amount*(orig-blur)
                val i = y * nw + x
                val v = (orig[i] + amount * (orig[i] - blur[i])).roundToInt().coerceIn(0, 255)
                val di = y * stride + x
                px[di] = (px[di] and (0xFF shl shift).inv()) or (v shl shift)
            }
        }
        channel(16); channel(8); channel(0)
    }
}
