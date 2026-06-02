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
 *   det[:,0] 二值化 → 連通元件 → minAreaRect → unclip 膨脹 → 旋轉四邊形。
 * 參數由 [DetectorConfig] 提供（§5 第一層）。
 */
class Detector(
    modelPath: String,
    private val cfg: DetectorConfig = DetectorConfig(),
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8) // 用滿核數（原本固定 4）
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            try {
                addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                Log.i(TAG, "XNNPACK 已啟用（threads=$threads）")
            } catch (t: Throwable) {
                Log.w(TAG, "XNNPACK 不可用，退回 CPU：${t.message}")
            }
        }
        session = env.createSession(modelPath, options) // 從路徑載入＝native 記憶體、不佔 JVM heap
        Log.i(TAG, "session inputs=${session.inputNames} outputs=${session.outputNames}")
    }

    fun detect(page: Bitmap): List<TextLine> {
        val size = cfg.inputSize
        val inputName = session.inputNames.first()
        val pre = ImageOps.toDetectorInput(env, page, size)
        pre.tensor.use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val det = result.get(OUT_DET).orElseThrow {
                    IllegalStateException("模型缺少輸出 '$OUT_DET'，實際有 ${session.outputNames}")
                } as OnnxTensor
                val area = size * size
                val prob = FloatArray(area)
                det.floatBuffer.get(prob, 0, area)
                val lines = linesFromProbMap(prob, size, pre.ratio, page.width, page.height)
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
        val thresh = cfg.textThreshold
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
            if (score < cfg.boxThreshold) continue
            val rect = Geometry.minAreaRect(boundary) ?: continue
            if (min(rect.w, rect.h) < cfg.minSide) continue

            val quad = rect.unclip(cfg.unclipRatio).corners().map {
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
        private const val OUT_DET = "det" // 文字行圖（DB），channel 0 = 文字機率
    }
}
