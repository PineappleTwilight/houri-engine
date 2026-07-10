package li.joye.yakuyomi.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import kotlin.math.min
import kotlin.math.roundToInt

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

    // 後端二選一：.param → NCNN（P2、手機 CPU 比 ORT-XNNPACK 快 ~3.7×）；.onnx → ORT。
    private val useNcnn = modelPath.endsWith(".param")
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var ncnnHandle: Long = 0L
    /** 實際生效的後端（"NCNN-CPU"/"XNNPACK"/"CPU"）；無 adb 時由呼叫端寫進 log/圖確認。 */
    val ep: String

    init {
        if (useNcnn) {
            check(NcnnBackend.available) { "NCNN 原生庫未載入，無法用 .param 偵測模型" }
            val bin = modelPath.removeSuffix(".param") + ".bin"
            ncnnHandle = NcnnBackend.createNet(modelPath, bin, false) // CPU（實測勝 Vulkan）
            check(ncnnHandle != 0L) { "NCNN 偵測模型載入失敗：$modelPath" }
            ep = "NCNN-CPU"
            Log.i(TAG, "NCNN detector loaded $modelPath ep=$ep")
        } else {
            val options = OrtSession.SessionOptions()
            ep = options.applyEp(cfg.intraThreads, TAG)
            session = env.createSession(modelPath, options) // 從路徑載入＝native 記憶體、不佔 JVM heap
            Log.i(TAG, "session inputs=${session!!.inputNames} outputs=${session!!.outputNames} ep=$ep")
        }
    }

    fun detect(page: Bitmap): Detection {
        val size = cfg.inputSize
        val area = size * size
        val prob: FloatArray  // det channel 0＝文字行 DB 圖
        val segArr: FloatArray // seg[1,1,H,W]＝逐像素文字機率
        val ratio: Float

        if (useNcnn) {
            val pre = ImageOps.detectorChw(page, size)
            val det = FloatArray(2 * area) // ch0=det, ch1=blk 邊界（此處只用 ch0）
            val seg = FloatArray(area)
            val rc = NcnnBackend.detect(ncnnHandle, pre.chw, size, det, seg)
            check(rc == 0) { "NCNN 偵測推論失敗 rc=$rc" }
            prob = det.copyOfRange(0, area)
            segArr = seg
            ratio = pre.ratio
        } else {
            val s = session!!
            val pre = ImageOps.toDetectorInput(env, page, size)
            pre.tensor.use { input ->
                s.run(mapOf(s.inputNames.first() to input)).use { result ->
                    val det = result.get(OUT_DET).orElseThrow {
                        IllegalStateException("模型缺少輸出 '$OUT_DET'，實際有 ${s.outputNames}")
                    } as OnnxTensor
                    val p = FloatArray(area)
                    det.floatBuffer.get(p, 0, area)
                    val seg = result.get(OUT_SEG).orElseThrow {
                        IllegalStateException("模型缺少輸出 '$OUT_SEG'，實際有 ${s.outputNames}")
                    } as OnnxTensor
                    val sg = FloatArray(area)
                    seg.floatBuffer.get(sg, 0, area)
                    val lines = linesFromProbMap(p, size, pre.ratio, page.width, page.height)
                    val textMask = segToMask(sg, size, pre.ratio, page.width, page.height)
                    Log.i(TAG, "偵測到 ${lines.size} 個文字行")
                    return Detection(lines, textMask)
                }
            }
            error("unreachable")
        }

        val lines = linesFromProbMap(prob, size, ratio, page.width, page.height)
        // seg → letterbox 還原 → 原圖尺寸細筆畫二值遮罩（去字用）
        val textMask = segToMask(segArr, size, ratio, page.width, page.height)
        Log.i(TAG, "偵測到 ${lines.size} 個文字行")
        return Detection(lines, textMask)
    }

    /**
     * seg 還原成原圖尺寸的二值文字遮罩。前處理 letterbox＝圖貼左上、pad 右下（ImageOps.toDetectorInput），
     * 故有效區＝seg[0:nh, 0:nw]（nw=round(origW*ratio)、nh=round(origH*ratio)）→ 縮回原圖 → 門檻。
     * 對齊 parity/seg_validate.py（裁 pad → cv2.resize 雙線性 → >segThreshold）。
     */
    private fun segToMask(
        s: FloatArray,
        size: Int,
        ratio: Float,
        origW: Int,
        origH: Int,
    ): Bitmap {
        val nw = (origW * ratio).roundToInt().coerceIn(1, size)
        val nh = (origH * ratio).roundToInt().coerceIn(1, size)
        // 有效區轉灰階小圖
        val gray = IntArray(nw * nh)
        for (y in 0 until nh) {
            val srow = y * size
            val drow = y * nw
            for (x in 0 until nw) {
                val v = (s[srow + x] * 255f).toInt().coerceIn(0, 255)
                gray[drow + x] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }
        }
        val small = Bitmap.createBitmap(gray, nw, nh, Bitmap.Config.ARGB_8888)
        val scaled = Bitmap.createScaledBitmap(small, origW, origH, true) // 雙線性，比照 cv2.resize
        // ★ createScaledBitmap 在「目標尺寸＝來源尺寸」時回傳同一物件（scaled === small）→ 不可先 recycle small，
        //   否則等於把 scaled 也 recycle 掉、下面 getPixels 會崩（"getPixels on a recycled bitmap"）。
        //   觸發條件：頁尺寸使 r=min(size/h,size/w)=1.0（如 720×1024、size=1024）→ nw,nh==origW,origH。
        //   故：先 getPixels，再「只在 scaled 為不同物件時」recycle 它，最後一律 recycle small。
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
        session?.close()
        if (ncnnHandle != 0L) {
            NcnnBackend.releaseNet(ncnnHandle)
            ncnnHandle = 0L
        }
    }

    companion object {
        private const val TAG = "Detector"
        private const val OUT_DET = "det" // 文字行圖（DB），channel 0 = 文字機率
        private const val OUT_SEG = "seg" // 逐像素文字筆畫遮罩 [1,1,H,W]
        private const val MASK_ON = 0xFFFFFFFF.toInt()
        private const val MASK_OFF = 0xFF000000.toInt()
    }
}

/** 偵測結果：文字行 ＋ 原圖尺寸的細筆畫文字遮罩（去字用，§去字升級）。 */
class Detection(val lines: List<TextLine>, val textMask: Bitmap)
