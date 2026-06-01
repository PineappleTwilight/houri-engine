package li.joye.yakuyomi.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * comic-text-detector 文字框偵測器。
 *
 * ported from manga_translator/detection/ctd.py (+ ctd_utils/) @ d5a3eee
 *   I/O：input "images" [1,3,1024,1024]
 *        outputs "blk"[1,64512,7](YOLO, 上游已棄用) / "seg"[1,1,1024,1024](mask, M3) /
 *                "det"[1,2,1024,1024](文字行圖, channel 0 = 文字機率)
 *   前處理：見 ImageOps（letterbox + /255 RGB NCHW）
 *   後處理（上游最新走 DB）：det[:,0] 二值化(0.3) → findContours → minAreaRect →
 *                            unclip 膨脹(×1.5, pyclipper) → minAreaRect → score>0.6
 *
 * ★ M0c 第三層「知情偏離」（§4）：此處用「二值化 + 8-連通元件 → 軸對齊 bbox」近似上述 DB 後處理，
 *   先達 M0「大致正確的文字框」。省略：輪廓近似、minAreaRect 旋轉框、pyclipper unclip 膨脹、
 *   box_score_fast（改用連通元件平均機率）。完整 SegDetectorRepresenter 留待 M1 餵 OCR 前再補。
 */
class Detector(modelBytes: ByteArray) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(NUM_THREADS)
            try {
                // 第三層：XNNPACK 加速（CLAUDE.md §2）。不支援時退回 CPU。
                addXnnpack(mapOf("intra_op_num_threads" to NUM_THREADS.toString()))
                Log.i(TAG, "XNNPACK 已啟用")
            } catch (t: Throwable) {
                Log.w(TAG, "XNNPACK 不可用，退回 CPU：${t.message}")
            }
        }
        session = env.createSession(modelBytes, options)
        Log.i(TAG, "session inputs=${session.inputNames} outputs=${session.outputNames}")
    }

    fun detect(page: Bitmap): List<Box> {
        val inputName = session.inputNames.first()
        val pre = ImageOps.toDetectorInput(env, page, INPUT_SIZE)
        pre.tensor.use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val det = result.get(OUT_DET).orElseThrow {
                    IllegalStateException("模型缺少輸出 '$OUT_DET'，實際有 ${session.outputNames}")
                } as OnnxTensor
                val area = INPUT_SIZE * INPUT_SIZE
                val prob = FloatArray(area)
                det.floatBuffer.get(prob, 0, area) // NCHW：channel 0（文字機率圖）= 前 area 個元素
                val boxes = boxesFromProbMap(prob, INPUT_SIZE, pre.ratio, page.width, page.height)
                Log.i(TAG, "偵測到 ${boxes.size} 個文字行框")
                return boxes
            }
        }
    }

    /**
     * 簡化版 DB 後處理：det channel 0 二值化(THRESH) → 8-連通元件 → 軸對齊 bbox；
     * 元件平均機率 > BOX_THRESH 才留；座標由 letterbox 空間 ÷ratio 映回原圖。
     */
    private fun boxesFromProbMap(
        prob: FloatArray,
        size: Int,
        ratio: Float,
        origW: Int,
        origH: Int,
    ): List<Box> {
        val visited = BooleanArray(prob.size)
        val stack = IntArray(prob.size) // 每像素至多入堆一次，故容量 = 像素數即足
        val boxes = ArrayList<Box>()

        for (seed in prob.indices) {
            if (visited[seed] || prob[seed] <= THRESH) continue

            var sp = 0
            stack[sp++] = seed
            visited[seed] = true
            var minX = size
            var minY = size
            var maxX = -1
            var maxY = -1
            var sum = 0f
            var cnt = 0

            while (sp > 0) {
                val idx = stack[--sp]
                val x = idx % size
                val y = idx / size
                sum += prob[idx]
                cnt++
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y

                var dy = -1
                while (dy <= 1) {
                    var dx = -1
                    while (dx <= 1) {
                        if (dx != 0 || dy != 0) {
                            val nx = x + dx
                            val ny = y + dy
                            if (nx in 0 until size && ny in 0 until size) {
                                val nidx = ny * size + nx
                                if (!visited[nidx] && prob[nidx] > THRESH) {
                                    visited[nidx] = true
                                    stack[sp++] = nidx
                                }
                            }
                        }
                        dx++
                    }
                    dy++
                }
            }

            val score = if (cnt > 0) sum / cnt else 0f
            if (score < BOX_THRESH) continue
            if (min(maxX - minX + 1, maxY - minY + 1) < MIN_SIDE) continue

            val x0 = (minX / ratio).roundToInt().coerceIn(0, origW)
            val y0 = (minY / ratio).roundToInt().coerceIn(0, origH)
            val x1 = ((maxX + 1) / ratio).roundToInt().coerceIn(0, origW)
            val y1 = ((maxY + 1) / ratio).roundToInt().coerceIn(0, origH)
            if (x1 > x0 && y1 > y0) boxes.add(Box(x0, y0, x1 - x0, y1 - y0, score))
        }
        return boxes
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "Detector"

        /** comic-text-detector 推論解析度（標準 1024）。 */
        const val INPUT_SIZE = 1024
        private const val NUM_THREADS = 4

        private const val OUT_DET = "det" // 文字行圖（DB），channel 0 = 文字機率
        // private const val OUT_SEG = "seg"  // mask（M3 inpaint 用）

        // 對齊 ctd.py：SegDetectorRepresenter(thresh=0.3) + 外部 box_thresh=0.6
        private const val THRESH = 0.3f
        private const val BOX_THRESH = 0.6f
        private const val MIN_SIDE = 3
    }
}
