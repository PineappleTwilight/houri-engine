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
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * 48px CTC OCR。
 *
 * ported from manga_translator/ocr/model_48px_ctc.py (+ ocr/common.py, utils/generic.py) @ d5a3eee
 *   裁切：sortPnts 定直/橫書 + get_transformed_region（findHomography→warpPerspective→48px 條，直書轉90°）
 *         此處用 Android Matrix.setPolyToPoly 取代 cv2 透視（§6）。
 *   前處理：(x-127.5)/127.5、NCHW、RGB。
 *   解碼：greedy CTC（blank=0、收合重複+去blank）→ 查字典；顏色 head 留 M3。
 */
class Ocr(
    modelBytes: ByteArray,
    private val dictionary: List<String>,
    private val cfg: OcrConfig = OcrConfig(),
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(NUM_THREADS)
            try {
                addXnnpack(mapOf("intra_op_num_threads" to NUM_THREADS.toString()))
            } catch (t: Throwable) {
                Log.w(TAG, "XNNPACK 不可用，退回 CPU：${t.message}")
            }
        }
        session = env.createSession(modelBytes, options)
    }

    /** 對每條文字行做 OCR，就地填入 direction 與 text。 */
    fun recognize(page: Bitmap, lines: List<TextLine>) {
        val inputName = session.inputNames.first()
        for (line in lines) {
            val (ordered, isV) = sortPnts(line.quad)
            line.direction = if (isV) "v" else "h"
            val strip = transformedRegion(page, ordered, isV, cfg.textHeight) ?: continue
            try {
                stripToTensor(strip).use { input ->
                    session.run(mapOf(inputName to input)).use { res ->
                        val logits = res.get(OUT_LOGITS).orElseThrow {
                            IllegalStateException("缺輸出 $OUT_LOGITS")
                        } as OnnxTensor
                        line.text = ctcDecode(logits)
                    }
                }
            } catch (t: Throwable) {
                Log.w(TAG, "OCR 單行失敗：${t.message}")
            } finally {
                strip.recycle()
            }
        }
    }

    /** 對齊 generic.py:sort_pnts —— 回傳排序後 4 點與是否直書。 */
    private fun sortPnts(quad: List<Pt>): Pair<List<Pt>, Boolean> {
        val n = quad.size
        var best0 = 0
        var best1 = 0
        // 16 對向量、取長度排序的第 8、10 名（長邊）
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
            val ls = listOf(byX[0], byX[1]).sortedBy { it.y } // 左：上、下
            val rs = listOf(byX[2], byX[3]).sortedBy { it.y } // 右：上、下
            Pair(listOf(ls[0], rs[0], rs[1], ls[1]), false)
        }
    }

    /** 對齊 generic.py:Quadrilateral.get_transformed_region。 */
    private fun transformedRegion(page: Bitmap, pts: List<Pt>, isV: Boolean, th: Int): Bitmap? {
        // structure 邊中點
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

        val region = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(region).drawBitmap(page, m, Paint(Paint.FILTER_BITMAP_FLAG))
        if (!isV) return region

        // 直書：cv2.ROTATE_90_COUNTERCLOCKWISE → 高度變 48
        val rot = Matrix().apply { postRotate(-90f) }
        val rotated = Bitmap.createBitmap(region, 0, 0, w, h, rot, true)
        region.recycle()
        return rotated
    }

    private fun stripToTensor(strip: Bitmap): OnnxTensor {
        val w = strip.width
        val h = strip.height
        val px = IntArray(w * h)
        strip.getPixels(px, 0, w, 0, 0, w, h)
        val area = w * h
        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = px[i]
            chw[i] = (((p shr 16) and 0xFF) - 127.5f) / 127.5f
            chw[area + i] = (((p shr 8) and 0xFF) - 127.5f) / 127.5f
            chw[2 * area + i] = ((p and 0xFF) - 127.5f) / 127.5f
        }
        return OnnxTensor.createTensor(
            env, FloatBuffer.wrap(chw), longArrayOf(1, 3, h.toLong(), w.toLong()),
        )
    }

    /** greedy CTC（blank=0、收合重複），對齊 decode_ctc_top1。 */
    private fun ctcDecode(logits: OnnxTensor): String {
        val shape = (logits.info as TensorInfo).shape // [1, T, dict]
        val t = shape[1].toInt()
        val d = shape[2].toInt()
        val arr = FloatArray(t * d)
        logits.floatBuffer.get(arr, 0, t * d)
        val sb = StringBuilder()
        var last = BLANK
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
            }
            last = best
        }
        return sb.toString()
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "Ocr"
        private const val NUM_THREADS = 4
        private const val BLANK = 0
        private const val OUT_LOGITS = "char_logits"
    }
}
