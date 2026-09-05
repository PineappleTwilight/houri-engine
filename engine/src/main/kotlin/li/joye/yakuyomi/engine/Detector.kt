package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.util.Log
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * DBNet text line detector (m-i-t default detector, ResNet34+DB head; pure NCNN, product arm64 NCNN required).
 * 1.6-2.5x more correct reads than retired comic-text-detector (proven on device, see memory dbnet-detector-ncnn).
 *
 * Ported from manga_translator/detection/default_utils/ @ d5a3eee
 *   db logits -> sigmoid -> binarize -> connected components -> minAreaRect -> unclip expansion -> rotated quadrilateral.
 * Parameters provided by [DetectorConfig] (§5 first layer).
 */
class Detector(
    modelPath: String,
    private val cfg: DetectorConfig = DetectorConfig(),
) : AutoCloseable {

    private var ncnnHandle: Long = 0L
    /** Actual backend in effect; when no adb, caller writes to log/image to confirm. */
    val ep: String = "NCNN-CPU"

    init {
        check(modelPath.endsWith(".param")) { "Detection requires NCNN `.param` model: $modelPath" }
        check(NcnnBackend.available) { "NCNN native library not loaded, cannot detect" }
        val bin = modelPath.removeSuffix(".param") + ".bin"
        ncnnHandle = NcnnBackend.createNet(modelPath, bin)
        check(ncnnHandle != 0L) { "Failed to load NCNN detection model: $modelPath" }
        Log.i(TAG, "NCNN detector loaded $modelPath")
    }

    /**
     * out0=db (2ch, ch0=raw logits -> Kotlin adds sigmoid), out1=mask (1ch, half/full-res varies by platform, already sigmoid).
     * Post-processing = [linesFromProbMap] (connected components + minAreaRect + unclip; score=component-mean prob = DB box_score_fast).
     */
    fun detect(page: Bitmap): Detection {
        // Hardened input validation - prevent destructive processing of invalid bitmaps
        require(!page.isRecycled) { "Cannot detect on recycled bitmap" }
        require(page.width in 32..8000 && page.height in 32..8000) { "Page size out of bounds ${page.width}x${page.height}" }
        if (ncnnHandle == 0L) throw IllegalStateException("Detector native handle not initialized")
        val pre = try {
            ImageOps.detectorChwDbnet(page, cfg.dbnetInputSize, cfg.detectUnsharp)
        } catch (t: Throwable) {
            throw IllegalStateException("Failed to preprocess page for detection: ${t.message}", t)
        }
        val inW = pre.w
        val inH = pre.h
        require(inW in 32..2048 && inH in 32..2048) { "Preprocessed size out of bounds ${inW}x${inH}" }
        val area = inW * inH
        require(area in 1..(2048 * 2048)) { "Area too large $area" }
        val db = FloatArray(2 * area)
        // Mask size varies by platform (x86 half-res inW/2 x inH/2, arm64 full-res inW x inH) -> allocate full-res upper bound, actual size returned via rc.
        val mask = FloatArray(area)
        val rc = try {
            NcnnBackend.detectDbnet(ncnnHandle, pre.chw, inW, inH, db, mask)
        } catch (t: Throwable) {
            throw IllegalStateException("NCNN DBNet forward failed: ${t.message}", t)
        }
        check(rc > 0) {
            if (rc < 0) {
                "DBNet size mismatch: actual db.w=${(-rc) / 1000} mask.w=${(-rc) % 1000} (buffer db=2x${inW}x$inH, mask<=${inW}x$inH)"
            } else {
                "NCNN DBNet forward empty output/failed rc=$rc"
            }
        }
        val mw = rc / 10000 // JNI returns mask.w*10000+mask.h (actual mask size, do not assume half/full res)
        val mh = rc % 10000
        // db ch0 = raw logits -> sigmoid -> prob (ctd out0 already sigmoid, DBNet not); grid = rectangle inW x inH
        val prob = FloatArray(area)
        for (i in 0 until area) prob[i] = 1f / (1f + exp(-db[i]))
        var lines = linesFromProbMap(
            prob, inW, inH, pre.ratio, page.width, page.height,
            cfg.dbBinThreshold, cfg.dbBoxThreshold, cfg.dbUnclipRatio,
        )
        if (lines.isEmpty() && cfg.dbBoxThreshold > 0.55f) {
            val retry = linesFromProbMap(
                prob, inW, inH, pre.ratio, page.width, page.height,
                (cfg.dbBinThreshold - 0.05f).coerceAtLeast(0.35f),
                (cfg.dbBoxThreshold - 0.15f).coerceAtLeast(0.5f),
                cfg.dbUnclipRatio,
            )
            if (retry.isNotEmpty()) {
                Log.i(TAG, "Detector fallback rescued ${retry.size} lines (relaxed thresholds)")
                lines = retry
            }
        }
        // mask (already sigmoid) -> original-size stroke mask. mask space ratio = pre.ratio * mw/inW (half-res=ratio/2, full-res=ratio, dynamic).
        val textMask = segToMask(mask, mw, mh, pre.ratio * mw.toFloat() / inW, page.width, page.height)
        Log.i(TAG, "DBNet detected ${lines.size} lines (in ${inW}x$inH mask ${mw}x$mh)")
        return Detection(lines, textMask)
    }

    /**
     * Warmup: run detection once on a blank small image to complete first lazy initialization of NCNN detection session on a single thread.
     * Call once before concurrent multi-page translation (see fork TranslationEngineService) to avoid multiple pages hitting uninitialized session simultaneously -> native crash.
     */
    fun warmUp() {
        val blank = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        try {
            detect(blank).textMask.recycle()
        } catch (t: Throwable) {
            Log.w(TAG, "Detection warmup failed: ${t.message}")
        } finally {
            blank.recycle()
        }
    }

    /**
     * Restore seg to original-size binary text mask. Preprocessing = image pasted top-left, pad bottom-right (ImageOps.detectorChwDbnet),
     * so valid area = seg[0:nh, 0:nw] (nw=round(origW*ratio), nh=round(origH*ratio)) -> scale back to original -> threshold.
     * Aligned with parity/seg_validate.py (crop pad -> cv2.resize bilinear -> >segThreshold).
     */
    private fun segToMask(
        s: FloatArray,
        srcW: Int,
        srcH: Int,
        ratio: Float,
        origW: Int,
        origH: Int,
    ): Bitmap {
        val nw = (origW * ratio).roundToInt().coerceIn(1, srcW)
        val nh = (origH * ratio).roundToInt().coerceIn(1, srcH)
        // Valid area to grayscale small image
        val gray = IntArray(nw * nh)
        for (y in 0 until nh) {
            val srow = y * srcW
            val drow = y * nw
            for (x in 0 until nw) {
                val v = (s[srow + x] * 255f).toInt().coerceIn(0, 255)
                gray[drow + x] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }
        }
        val small = Bitmap.createBitmap(gray, nw, nh, Bitmap.Config.ARGB_8888)
        val scaled = Bitmap.createScaledBitmap(small, origW, origH, true) // Bilinear, matching cv2.resize
        // createScaledBitmap returns same object when target size == source size (scaled === small) -> must not recycle small first,
        // otherwise scaled is also recycled and getPixels below crashes ("getPixels on a recycled bitmap").
        // Trigger: page size makes r=min(size/h,size/w)=1.0 (e.g., 720x1024, size=1024) -> nw,nh==origW,origH.
        // So: getPixels first, then recycle scaled only if different object, finally always recycle small.
        val th = (cfg.segThreshold * 255f).toInt()
        val px = IntArray(origW * origH)
        scaled.getPixels(px, 0, origW, 0, 0, origW, origH)
        if (scaled !== small) scaled.recycle()
        small.recycle()
        for (i in px.indices) px[i] = if ((px[i] and 0xFF) > th) MASK_ON else MASK_OFF
        return Bitmap.createBitmap(px, origW, origH, Bitmap.Config.ARGB_8888)
    }

    private fun linesFromProbMap(
        prob: FloatArray,
        gridW: Int,
        gridH: Int,
        ratio: Float,
        origW: Int,
        origH: Int,
        binThresh: Float,
        scoreThresh: Float,
        unclip: Float,
    ): List<TextLine> {
        val thresh = binThresh
        val visited = BooleanArray(prob.size)
        val stack = IntArray(prob.size)
        val out = ArrayList<TextLine>()
        val boundary = ArrayList<Pt>()

        for (seed in prob.indices) {
            if (visited[seed] || prob[seed] <= thresh) continue

            var sp = 0
            stack[sp++] = seed
            visited[seed] = true
            boundary.clear()
            var sum = 0f
            var cnt = 0

            while (sp > 0) {
                val idx = stack[--sp]
                val x = idx % gridW
                val y = idx / gridW
                sum += prob[idx]
                cnt++
                var isBoundary = false

                var dy = -1
                while (dy <= 1) {
                    var dx = -1
                    while (dx <= 1) {
                        if (dx != 0 || dy != 0) {
                            val nx = x + dx
                            val ny = y + dy
                            if (nx in 0 until gridW && ny in 0 until gridH) {
                                val nidx = ny * gridW + nx
                                if (prob[nidx] > thresh) {
                                    if (!visited[nidx]) {
                                        visited[nidx] = true
                                        stack[sp++] = nidx
                                    }
                                } else if (dx == 0 || dy == 0) {
                                    isBoundary = true
                                }
                            } else if (dx == 0 || dy == 0) {
                                isBoundary = true
                            }
                        }
                        dx++
                    }
                    dy++
                }
                if (isBoundary) boundary.add(Pt(x.toFloat(), y.toFloat()))
            }

            val score = if (cnt > 0) sum / cnt else 0f
            if (score < scoreThresh) continue
            val rect = Geometry.minAreaRect(boundary) ?: continue
            if (min(rect.w, rect.h) < cfg.minSide) continue

            val quad = rect.unclip(unclip).corners().map {
                Pt(
                    (it.x / ratio).coerceIn(0f, origW.toFloat()),
                    (it.y / ratio).coerceIn(0f, origH.toFloat()),
                )
            }
            out.add(TextLine(quad, score))
        }
        return out
    }

    override fun close() {
        if (ncnnHandle != 0L) {
            NcnnBackend.releaseNet(ncnnHandle)
            ncnnHandle = 0L
        }
    }

    companion object {
        private const val TAG = "Detector"
        private const val MASK_ON = 0xFFFFFFFF.toInt()
        private const val MASK_OFF = 0xFF000000.toInt()
    }
}

/** Detection result: text lines + original-size fine stroke text mask (for inpainting, § inpaint upgrade). */
class Detection(val lines: List<TextLine>, val textMask: Bitmap)
