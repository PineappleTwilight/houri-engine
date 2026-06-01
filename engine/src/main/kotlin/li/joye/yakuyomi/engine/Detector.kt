package li.joye.yakuyomi.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import kotlin.math.min

/**
 * comic-text-detector 文字行偵測器。
 *
 * ported from manga_translator/detection/ctd.py (+ ctd_utils/, utils/db_utils.py) @ d5a3eee
 *   I/O：input "images"[1,3,1024,1024]；outputs "blk"(YOLO,棄用)/"seg"(mask,M3)/"det"[1,2,1024,1024]
 *   後處理（DB / SegDetectorRepresenter）：det[:,0] 二值化(0.3) → 連通元件 → minAreaRect →
 *     unclip 膨脹(×1.5) → 旋轉四邊形；元件平均機率 > 0.6 才留；座標 ÷ratio 映回原圖。
 *
 * 對齊註記（§4）：上游用 cv2.findContours→minAreaRect；此處用「連通元件邊界點 → 凸包 → minAreaRect」
 *   等價取框；box_score 用元件平均機率近似 box_score_fast（桌面 parity 驗證框數與上游一致）。
 */
class Detector(modelBytes: ByteArray) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(NUM_THREADS)
            try {
                addXnnpack(mapOf("intra_op_num_threads" to NUM_THREADS.toString()))
                Log.i(TAG, "XNNPACK 已啟用")
            } catch (t: Throwable) {
                Log.w(TAG, "XNNPACK 不可用，退回 CPU：${t.message}")
            }
        }
        session = env.createSession(modelBytes, options)
        Log.i(TAG, "session inputs=${session.inputNames} outputs=${session.outputNames}")
    }

    fun detect(page: Bitmap): List<TextLine> {
        val inputName = session.inputNames.first()
        val pre = ImageOps.toDetectorInput(env, page, INPUT_SIZE)
        pre.tensor.use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val det = result.get(OUT_DET).orElseThrow {
                    IllegalStateException("模型缺少輸出 '$OUT_DET'，實際有 ${session.outputNames}")
                } as OnnxTensor
                val area = INPUT_SIZE * INPUT_SIZE
                val prob = FloatArray(area)
                det.floatBuffer.get(prob, 0, area) // NCHW：channel 0（文字機率圖）
                val lines = linesFromProbMap(prob, INPUT_SIZE, pre.ratio, page.width, page.height)
                Log.i(TAG, "偵測到 ${lines.size} 個文字行")
                return lines
            }
        }
    }

    private fun linesFromProbMap(
        prob: FloatArray,
        size: Int,
        ratio: Float,
        origW: Int,
        origH: Int,
    ): List<TextLine> {
        val visited = BooleanArray(prob.size)
        val stack = IntArray(prob.size)
        val out = ArrayList<TextLine>()
        val boundary = ArrayList<Pt>()

        for (seed in prob.indices) {
            if (visited[seed] || prob[seed] <= THRESH) continue

            var sp = 0
            stack[sp++] = seed
            visited[seed] = true
            boundary.clear()
            var sum = 0f
            var cnt = 0

            while (sp > 0) {
                val idx = stack[--sp]
                val x = idx % size
                val y = idx / size
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
                            if (nx in 0 until size && ny in 0 until size) {
                                val nidx = ny * size + nx
                                if (prob[nidx] > THRESH) {
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
            if (score < BOX_THRESH) continue
            val rect = Geometry.minAreaRect(boundary) ?: continue
            if (min(rect.w, rect.h) < MIN_SIDE) continue

            val quad = rect.unclip(UNCLIP_RATIO).corners().map {
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
        session.close()
    }

    companion object {
        private const val TAG = "Detector"
        const val INPUT_SIZE = 1024
        private const val NUM_THREADS = 4
        private const val OUT_DET = "det"

        // 對齊 ctd.py：SegDetectorRepresenter(thresh=0.3, unclip_ratio=1.5) + 外部 box_thresh=0.6
        private const val THRESH = 0.3f
        private const val BOX_THRESH = 0.6f
        private const val UNCLIP_RATIO = 1.5f
        private const val MIN_SIDE = 3f
    }
}
