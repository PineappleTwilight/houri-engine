package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.util.Log
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * comic-text-detector 文字行偵測器（純 NCNN；ORT 備援已退役移除，產品 arm64 NCNN 必在）。
 *
 * ported from manga_translator/detection/ctd.py (+ ctd_utils/, utils/db_utils.py) @ d5a3eee
 *   det[:,0] 二值化 → 連通元件 → minAreaRect → unclip 膨脹 → 旋轉四邊形。
 * 參數由 [DetectorConfig] 提供（§5 第一層）。
 */
class Detector(
    modelPath: String,
    private val cfg: DetectorConfig = DetectorConfig(),
) : AutoCloseable {

    private var ncnnHandle: Long = 0L
    /** 實際生效的後端；無 adb 時由呼叫端寫進 log/圖確認。 */
    val ep: String = "NCNN-CPU"

    init {
        check(modelPath.endsWith(".param")) { "偵測需 NCNN `.param` 模型：$modelPath" }
        check(NcnnBackend.available) { "NCNN 原生庫未載入，無法偵測" }
        val bin = modelPath.removeSuffix(".param") + ".bin"
        ncnnHandle = NcnnBackend.createNet(modelPath, bin)
        check(ncnnHandle != 0L) { "NCNN 偵測模型載入失敗：$modelPath" }
        Log.i(TAG, "NCNN detector loaded $modelPath")
    }

    fun detect(page: Bitmap): Detection {
        if (cfg.useDbnet) return detectDbnetPath(page)
        val size = cfg.inputSize
        val area = size * size
        val pre = ImageOps.detectorChw(page, size)
        val det = FloatArray(2 * area) // ch0=det, ch1=blk 邊界（此處只用 ch0）
        val seg = FloatArray(area)     // seg＝逐像素文字筆畫機率（去字用）
        val rc = NcnnBackend.detect(ncnnHandle, pre.chw, size, det, seg)
        check(rc == 0) { "NCNN 偵測推論失敗 rc=$rc" }
        val lines = linesFromProbMap(
            det.copyOfRange(0, area), size, size, pre.ratio, page.width, page.height,
            cfg.textThreshold, cfg.boxThreshold, cfg.unclipRatio,
        )
        // seg → letterbox 還原 → 原圖尺寸細筆畫二值遮罩（去字用）
        val textMask = segToMask(seg, size, size, pre.ratio, page.width, page.height)
        Log.i(TAG, "偵測到 ${lines.size} 個文字行")
        return Detection(lines, textMask)
    }

    /**
     * DBNet（m-i-t default 偵測器）路徑。out0=db（2ch，ch0=raw logits→Kotlin 補 sigmoid）、out1=mask（1ch，半解析度、已 sigmoid）。
     * 後處理沿用 [linesFromProbMap]（連通元件+minAreaRect+unclip；ctd 的 score=component-mean prob 正好＝DB box_score_fast），只換門檻參數。
     */
    private fun detectDbnetPath(page: Bitmap): Detection {
        val pre = ImageOps.detectorChwDbnet(page, cfg.dbnetInputSize, cfg.detectUnsharp)
        val inW = pre.w
        val inH = pre.h
        val area = inW * inH
        val db = FloatArray(2 * area)
        val mw = inW / 2
        val mh = inH / 2
        val mask = FloatArray(mw * mh) // mask 半解析度（矩形）
        val rc = NcnnBackend.detectDbnet(ncnnHandle, pre.chw, inW, inH, db, mask)
        check(rc > 0) {
            if (rc < 0 && -rc != 1) {
                "DBNet 尺寸越界：實際 db.w=${-rc / 1000} mask.w=${-rc % 1000}（Kotlin 緩衝假設 db=2×${inW}×$inH mask=${mw}×$mh）＝解析度假設錯"
            } else {
                "NCNN DBNet forward 空輸出/失敗 rc=$rc（out0/out1 空 → blob 名或 forward 出錯，看 logcat 的 dbnet 行）"
            }
        }
        // db ch0 = raw logits → sigmoid → prob（ctd 的 out0 已 sigmoid、DBNet 沒有）；網格＝矩形 inW×inH
        val prob = FloatArray(area)
        for (i in 0 until area) prob[i] = 1f / (1f + exp(-db[i]))
        val lines = linesFromProbMap(
            prob, inW, inH, pre.ratio, page.width, page.height,
            cfg.dbBinThreshold, cfg.dbBoxThreshold, cfg.dbUnclipRatio,
        )
        // mask 半解析度（已 sigmoid）→ 原圖尺寸筆畫遮罩。mask 空間 ratio = pre.ratio / 2。
        val textMask = segToMask(mask, mw, mh, pre.ratio / 2f, page.width, page.height)
        Log.i(TAG, "DBNet 偵測到 ${lines.size} 個文字行（in ${inW}x$inH）")
        return Detection(lines, textMask)
    }

    /**
     * 暖機：對空白小圖跑一次偵測，讓 NCNN 偵測 session 的首次 lazy 初始化在單緒完成。
     * 併發翻多頁前先呼叫一次（見 fork TranslationEngineService），避免多頁同時打進未初始化的 session → 原生 crash。
     */
    fun warmUp() {
        val blank = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        try {
            detect(blank).textMask.recycle()
        } catch (t: Throwable) {
            Log.w(TAG, "偵測暖機失敗：${t.message}")
        } finally {
            blank.recycle()
        }
    }

    /**
     * seg 還原成原圖尺寸的二值文字遮罩。前處理 letterbox＝圖貼左上、pad 右下（ImageOps.detectorChw），
     * 故有效區＝seg[0:nh, 0:nw]（nw=round(origW*ratio)、nh=round(origH*ratio)）→ 縮回原圖 → 門檻。
     * 對齊 parity/seg_validate.py（裁 pad → cv2.resize 雙線性 → >segThreshold）。
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
        // 有效區轉灰階小圖
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

/** 偵測結果：文字行 ＋ 原圖尺寸的細筆畫文字遮罩（去字用，§去字升級）。 */
class Detection(val lines: List<TextLine>, val textMask: Bitmap)
