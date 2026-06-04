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
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

/** OCR 單行診斷結果（[Ocr.debugOne]）：裁切圖塊 + 一行資訊。 */
class OcrDebug(val strip: Bitmap?, val info: String)

/**
 * 48px CTC OCR。
 *
 * ported from manga_translator/ocr/model_48px_ctc.py (+ ocr/common.py, utils/generic.py) @ d5a3eee
 *   裁切：sortPnts 定直/橫書 + get_transformed_region（findHomography→warpPerspective→48px 條，直書轉90°）
 *         此處用 Android Matrix.setPolyToPoly 取代 cv2 透視（§6）。
 *   前處理：(x-127.5)/127.5、NCHW、RGB。
 *   解碼：greedy CTC（blank=0、收合重複+去blank）→ 查字典。
 *   ignore_bubble（cfg.ignoreBubble，ported from utils/bubble.py）：跳過彩色/非氣泡 SFX 類文字。
 *   顏色 head 不採用（彩底太雜）；文字色改由 [Renderer] 取去字後背景亮度判黑/白。
 */
class Ocr(
    modelPath: String,
    private val dictionary: List<String>,
    private val cfg: OcrConfig = OcrConfig(),
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        // 並發模式：每行單緒（intra-op=1）、靠 N 行並發填核；序列模式：單行用滿 NUM_THREADS（現狀）。
        val threads = if (cfg.concurrent) 1 else NUM_THREADS
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            if (cfg.useXnnpack) {
                try {
                    addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                } catch (t: Throwable) {
                    Log.w(TAG, "XNNPACK 不可用，退回 CPU：${t.message}")
                }
            }
        }
        session = env.createSession(modelPath, options) // 路徑載入＝native 記憶體、不佔 JVM heap
    }

    /**
     * 對每條文字行做 OCR，就地填入 direction 與 text。每條右側加 [PAD_MARGIN] 白邊（見 [stripToTensor]）讓 CTC 不截尾字。
     * [OcrConfig.concurrent]＝true：多行並發（小圖塊吃不滿 intra-op→改單緒、並發填核，見 init）；false：逐行序列（現狀）。
     * 批次 padding 已否決（寬度差→padding 浪費）；此處是「並發」（零 padding），與批次不同。
     */
    suspend fun recognize(page: Bitmap, lines: List<TextLine>): Unit = coroutineScope {
        val inputName = session.inputNames.first()
        if (cfg.concurrent && lines.size > 1) {
            val sem = Semaphore(cfg.concurrency.coerceAtLeast(1))
            lines.map { line ->
                async(Dispatchers.Default) { sem.withPermit { recognizeOne(page, line, inputName) } }
            }.awaitAll()
        } else {
            for (line in lines) recognizeOne(page, line, inputName)
        }
    }

    /** 單行 OCR：裁切→前處理→CTC→填 text。thread-safe：只寫自己的 line、session.run 可並發、其餘皆 local/唯讀。 */
    private fun recognizeOne(page: Bitmap, line: TextLine, inputName: String) {
        val (ordered, isV) = sortPnts(line.quad)
        line.direction = if (isV) "v" else "h"
        val strip = transformedRegion(page, ordered, isV, cfg.textHeight) ?: return
        if (cfg.ignoreBubble in 1..50 && isIgnore(strip, cfg.ignoreBubble)) {
            strip.recycle()  // 彩色/非氣泡 SFX 類文字 → 跳過
            return
        }
        try {
            stripToTensor(strip).use { input ->
                session.run(mapOf(inputName to input)).use { res ->
                    val logits = res.get(OUT_LOGITS).orElseThrow {
                        IllegalStateException("缺輸出 $OUT_LOGITS")
                    } as OnnxTensor
                    val (text, prob) = ctcDecode(logits)
                    if (prob >= cfg.minProb) line.text = text  // 低信心誤讀 → 丟
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "OCR 單行失敗：${t.message}")
        } finally {
            strip.recycle()
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
        val sw = strip.width
        val h = strip.height
        val w = sw + PAD_MARGIN // 右側白邊：避免 CTC 截掉尾字（坂→坂本、ねえね→ねえねえ）
        val px = IntArray(sw * h)
        strip.getPixels(px, 0, sw, 0, 0, sw, h)
        val area = h * w
        val chw = FloatArray(3 * area) { 1f } // 白底（(255-127.5)/127.5=1.0），右邊維持白
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

    /** greedy CTC（單條 [1,T,d]）→ 讀出 arr 後交給 [ctcDecodeArr]。 */
    private fun ctcDecode(logits: OnnxTensor): Pair<String, Float> {
        val shape = (logits.info as TensorInfo).shape // [1, T, dict]
        val t = shape[1].toInt()
        val d = shape[2].toInt()
        val arr = FloatArray(t * d)
        logits.floatBuffer.get(arr, 0, t * d)
        return ctcDecodeArr(arr, t, d)
    }

    /** greedy CTC（blank=0、收合重複）+ 平均信心，對齊 decode_ctc_top1。回傳 (text, prob)。 */
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
                // top-1 的 log_softmax＝bestV − logsumexp(row)＝−ln(Σ exp(x−bestV))
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

    /** SFX/非氣泡文字判定（ported from utils/bubble.py:is_ignore @ d5a3eee）：邊框混色（非乾淨氣泡底）或彩色 → 跳過。 */
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

    /** 彩色文字判定（ported from utils/bubble.py:check_color）：>10 個非灰階像素 → True。 */
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

    /** 診斷：對單行回傳裁切圖塊 + 資訊（圖塊尺寸/輸出 shape/原始文字/prob 或例外），不丟低信心。 */
    fun debugOne(page: Bitmap, line: TextLine): OcrDebug {
        val (ordered, isV) = sortPnts(line.quad)
        val strip = transformedRegion(page, ordered, isV, cfg.textHeight)
            ?: return OcrDebug(null, "strip=null（setPolyToPoly 失敗）")
        val px = IntArray(strip.width * strip.height)
        strip.getPixels(px, 0, strip.width, 0, 0, strip.width, strip.height)
        val mean = if (px.isEmpty()) 0 else px.sumOf { (it shr 16 and 0xFF) + (it shr 8 and 0xFF) + (it and 0xFF) } / (px.size * 3)
        val info = try {
            stripToTensor(strip).use { input ->
                session.run(mapOf(session.inputNames.first() to input)).use { res ->
                    val logits = res.get(OUT_LOGITS).orElseThrow { IllegalStateException("缺 char_logits") } as OnnxTensor
                    val shp = (logits.info as TensorInfo).shape
                    val tt = shp[1].toInt(); val dd = shp[2].toInt()
                    val a = FloatArray(tt * dd); logits.floatBuffer.get(a, 0, tt * dd)
                    var nb = 0
                    for (ti in 0 until tt) {
                        var b = 0; var bv = a[ti * dd]
                        for (c in 1 until dd) if (a[ti * dd + c] > bv) { bv = a[ti * dd + c]; b = c }
                        if (b != BLANK) nb++
                    }
                    val (text, prob) = ctcDecode(logits)
                    "${strip.width}x${strip.height} 均值$mean out=${shp.joinToString("x")} 非空$nb/$tt '${text.take(10)}' p=${"%.2f".format(prob)}"
                }
            }
        } catch (t: Throwable) {
            "✗ ${t.javaClass.simpleName}: ${t.message}"
        }
        return OcrDebug(strip, info)
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "Ocr"
        private const val NUM_THREADS = 4
        private const val BLANK = 0
        private const val OUT_LOGITS = "char_logits"
        private const val PAD_MARGIN = 16  // 每條右側白邊：讓 CTC 有 context、不截尾字
    }
}
