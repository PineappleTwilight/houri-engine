package li.joye.yakuyomi.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.Log
import java.nio.FloatBuffer
import kotlin.math.roundToInt
import kotlinx.coroutines.coroutineScope

/**
 * LaMa 去字（Koharu mayocream/lama-manga.onnx）。block-aware：逐氣泡裁窗 → tile → LaMa → 貼回。
 * I/O：image[1,3,N,N] + mask[1,1,N,N]（/255、RGB、mask 1=擦）→ output。參數見 [InpainterConfig]。
 * 對齊 parity/inpaint_parity.py。
 */
class Inpainter(
    modelPath: String,
    private val cfg: InpainterConfig = InpainterConfig(),
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(cfg.intraThreads) // 逐區平行時：concurrency × intraThreads ≈ 核數
            try {
                addXnnpack(mapOf("intra_op_num_threads" to cfg.intraThreads.toString()))
            } catch (t: Throwable) {
                Log.w(TAG, "XNNPACK 不可用：${t.message}")
            }
        }
        session = env.createSession(modelPath, opts) // 路徑載入＝native 記憶體、不佔 JVM heap
    }

    suspend fun inpaint(page: Bitmap, regions: List<TextRegion>, textMask: Bitmap): Bitmap = coroutineScope {
        val w = page.width
        val h = page.height
        val result = page.copy(Bitmap.Config.ARGB_8888, true)

        // 遮罩配方法：boxfill / lama逐區＝全解析度 → seg 細筆畫（精準、壓畫面的字只動筆畫不成方塊）；
        // lama整頁＝整張縮 512、細筆畫會被縮到次像素而殘留 → 改用整塊文字框遮罩（整塊在乾淨泡泡上反而擦得更乾淨）。
        val useSeg = !(cfg.method == "lama" && cfg.wholeImage)
        val maskPx = if (useSeg) buildSegMask(regions, textMask, w, h) else buildQuadMask(regions, w, h)
        val maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        maskBmp.setPixels(maskPx, 0, w, 0, 0, w, h)

        if (cfg.method == "boxfill") {
            boxFill(result, maskPx) // 瞬間：就近取色填字筆畫、不跑 LaMa
            maskBmp.recycle()
            return@coroutineScope result
        }
        if (cfg.wholeImage) {
            // lama 整頁：整張縮 512 跑一次（~6s/頁；快、但大/彩色泡泡會糊）
            runWindow(page, maskBmp, intArrayOf(0, 0, w, h))?.let { compositePixels(result, maskPx, it) }
            maskBmp.recycle()
            return@coroutineScope result
        }
        // lama 逐區（序列）：各區裁窗 → 跑 LaMa → 貼回。不平行——純 CPU 核數固定，平行切核不增總算力（見 InpainterConfig 註）。
        for (win in regions.mapNotNull { windowOf(it, w, h) }) {
            runWindow(page, maskBmp, win)?.let { compositePixels(result, maskPx, it) }
        }
        maskBmp.recycle()
        result
    }

    /** 整塊遮罩（舊法）：文字行框 FILL_AND_STROKE。lama 整頁專用——縮 512 後細筆畫殘留，整塊在乾淨泡泡上擦得更乾淨。 */
    private fun buildQuadMask(regions: List<TextRegion>, w: Int, h: Int): IntArray {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(bmp).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply {
                color = Color.WHITE
                style = Paint.Style.FILL_AND_STROKE
                strokeWidth = cfg.maskDilate
            }
            for (region in regions) {
                for (line in region.lines) {
                    val q = line.quad
                    if (q.size < 4) continue
                    val path = Path().apply {
                        moveTo(q[0].x, q[0].y)
                        for (i in 1..3) lineTo(q[i].x, q[i].y)
                        close()
                    }
                    drawPath(path, p)
                }
            }
        }
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        bmp.recycle()
        return px
    }

    /**
     * seg 細遮罩＝seg 細筆畫 ∩ 已保留區的文字行框（allow），再膨脹。回傳 ARGB 像素（白＝要去字）。
     * allow 把去字限制在「翻譯過的區」內，故 seg 抓到的 SFX/未譯文字不會被擦（留原圖、合 SFX 政策與 §11）。
     */
    private fun buildSegMask(regions: List<TextRegion>, textMask: Bitmap, w: Int, h: Int): IntArray {
        val allow = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(allow).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
            for (region in regions) {
                for (line in region.lines) {
                    val q = line.quad
                    if (q.size < 4) continue
                    val path = Path().apply {
                        moveTo(q[0].x, q[0].y)
                        for (i in 1..3) lineTo(q[i].x, q[i].y)
                        close()
                    }
                    drawPath(path, p)
                }
            }
        }
        val mask = IntArray(w * h)
        allow.getPixels(mask, 0, w, 0, 0, w, h)
        allow.recycle()
        val seg = IntArray(w * h)
        textMask.getPixels(seg, 0, w, 0, 0, w, h)
        for (i in mask.indices) {
            mask[i] = if ((mask[i] and 0xFF) > 127 && (seg[i] and 0xFF) > 127) Color.WHITE else Color.BLACK
        }
        dilate(mask, w, h, (cfg.maskDilate / 2f).roundToInt().coerceAtLeast(1))
        return mask
    }

    /** 二值遮罩可分離膨脹（先橫後縱 max-filter），radius 像素。覆蓋筆畫抗鋸齒邊緣、給去字餘裕。 */
    private fun dilate(px: IntArray, w: Int, h: Int, radius: Int) {
        if (radius <= 0) return
        val tmp = IntArray(px.size)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var on = false
                var k = -radius
                while (k <= radius) {
                    val xx = x + k
                    if (xx in 0 until w && (px[row + xx] and 0xFF) > 127) { on = true; break }
                    k++
                }
                tmp[row + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
        for (x in 0 until w) {
            for (y in 0 until h) {
                var on = false
                var k = -radius
                while (k <= radius) {
                    val yy = y + k
                    if (yy in 0 until h && (tmp[yy * w + x] and 0xFF) > 127) { on = true; break }
                    k++
                }
                px[y * w + x] = if (on) Color.WHITE else Color.BLACK
            }
        }
    }

    /**
     * box-fill 去字（就近取色）：每個遮罩像素換成「上下左右最近的非遮罩像素」均值。瞬間、無 LaMa。
     * 跟著局部背景走 ⇒ 多色/漸層泡泡、合併到的相鄰異色框、壓在畫面上的字都不會糊成單一色塊
     * （取代舊「整區一個平均色」；配 seg 細筆畫遮罩，鄰近背景就在筆畫旁、取色準）。
     */
    private fun boxFill(result: Bitmap, maskPx: IntArray) {
        val w = result.width
        val h = result.height
        val px = IntArray(w * h)
        result.getPixels(px, 0, w, 0, 0, w, h)
        val out = px.copyOf()
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val i = row + x
                if ((maskPx[i] and 0xFF) <= 127) continue // 非遮罩，保留原畫面
                var sr = 0; var sg = 0; var sb = 0; var cnt = 0
                var k = x - 1; var d = 0 // 左
                while (k >= 0 && d < FILL_REACH) { val j = row + k; if ((maskPx[j] and 0xFF) <= 127) { val p = px[j]; sr += (p shr 16) and 0xFF; sg += (p shr 8) and 0xFF; sb += p and 0xFF; cnt++; break }; k--; d++ }
                k = x + 1; d = 0 // 右
                while (k < w && d < FILL_REACH) { val j = row + k; if ((maskPx[j] and 0xFF) <= 127) { val p = px[j]; sr += (p shr 16) and 0xFF; sg += (p shr 8) and 0xFF; sb += p and 0xFF; cnt++; break }; k++; d++ }
                k = y - 1; d = 0 // 上
                while (k >= 0 && d < FILL_REACH) { val j = k * w + x; if ((maskPx[j] and 0xFF) <= 127) { val p = px[j]; sr += (p shr 16) and 0xFF; sg += (p shr 8) and 0xFF; sb += p and 0xFF; cnt++; break }; k--; d++ }
                k = y + 1; d = 0 // 下
                while (k < h && d < FILL_REACH) { val j = k * w + x; if ((maskPx[j] and 0xFF) <= 127) { val p = px[j]; sr += (p shr 16) and 0xFF; sg += (p shr 8) and 0xFF; sb += p and 0xFF; cnt++; break }; k++; d++ }
                if (cnt > 0) out[i] = Color.rgb(sr / cnt, sg / cnt, sb / cnt)
            }
        }
        result.setPixels(out, 0, w, 0, 0, w, h)
    }

    private class WinOut(val x0: Int, val y0: Int, val ww: Int, val wh: Int, val px: IntArray)

    /** 由 region 算 ×windowRatio 裁窗（夾邊界）；太小回 null。 */
    private fun windowOf(region: TextRegion, w: Int, h: Int): IntArray? {
        val rw = region.x1 - region.x0
        val rh = region.y1 - region.y0
        val cx = (region.x0 + region.x1) / 2f
        val cy = (region.y0 + region.y1) / 2f
        val r = cfg.windowRatio
        val wx0 = (cx - rw * r / 2f).toInt().coerceIn(0, w - 1)
        val wy0 = (cy - rh * r / 2f).toInt().coerceIn(0, h - 1)
        val wx1 = (cx + rw * r / 2f).toInt().coerceIn(wx0 + 1, w)
        val wy1 = (cy + rh * r / 2f).toInt().coerceIn(wy0 + 1, h)
        val ww = wx1 - wx0
        val wh = wy1 - wy0
        return if (ww < 8 || wh < 8) null else intArrayOf(wx0, wy0, ww, wh)
    }

    /** 對一塊視窗跑一次 LaMa（縮 tile→推論→放回視窗尺寸）；只讀 page/maskBmp ⇒ 平行安全。回傳視窗+輸出像素。 */
    private fun runWindow(page: Bitmap, maskBmp: Bitmap, win: IntArray): WinOut? {
        val tile = cfg.tileSize
        val wx0 = win[0]; val wy0 = win[1]; val ww = win[2]; val wh = win[3]
        val cropBmp = Bitmap.createBitmap(page, wx0, wy0, ww, wh)
        val crop512 = Bitmap.createScaledBitmap(cropBmp, tile, tile, true)
        val maskCropBmp = Bitmap.createBitmap(maskBmp, wx0, wy0, ww, wh)
        val mask512 = Bitmap.createScaledBitmap(maskCropBmp, tile, tile, false)
        val imgTensor = imageToNCHW(crop512, tile)
        val maskTensor = maskTo1CH(mask512, tile)
        return try {
            session.run(mapOf(INPUT_IMAGE to imgTensor, INPUT_MASK to maskTensor)).use { res ->
                val outT = res.get(OUT_NAME).orElseThrow { IllegalStateException("缺輸出 $OUT_NAME") } as OnnxTensor
                val res512 = nchwToBitmap(outT, tile)
                val resWin = Bitmap.createScaledBitmap(res512, ww, wh, true)
                val px = IntArray(ww * wh)
                resWin.getPixels(px, 0, ww, 0, 0, ww, wh)
                res512.recycle()
                resWin.recycle()
                WinOut(wx0, wy0, ww, wh, px)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "去字單窗失敗：${t.message}"); null
        } finally {
            imgTensor.close()
            maskTensor.close()
            cropBmp.recycle()
            crop512.recycle()
            maskCropBmp.recycle()
            mask512.recycle()
        }
    }

    private fun imageToNCHW(bmp: Bitmap, n: Int): OnnxTensor {
        val px = IntArray(n * n)
        bmp.getPixels(px, 0, n, 0, 0, n, n)
        val area = n * n
        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = px[i]
            chw[i] = ((p shr 16) and 0xFF) / 255f
            chw[area + i] = ((p shr 8) and 0xFF) / 255f
            chw[2 * area + i] = (p and 0xFF) / 255f
        }
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), longArrayOf(1, 3, n.toLong(), n.toLong()))
    }

    private fun maskTo1CH(bmp: Bitmap, n: Int): OnnxTensor {
        val px = IntArray(n * n)
        bmp.getPixels(px, 0, n, 0, 0, n, n)
        val m = FloatArray(n * n)
        for (i in px.indices) m[i] = if ((px[i] and 0xFF) > 127) 1f else 0f
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(m), longArrayOf(1, 1, n.toLong(), n.toLong()))
    }

    private fun nchwToBitmap(t: OnnxTensor, n: Int): Bitmap {
        val area = n * n
        val arr = FloatArray(3 * area)
        t.floatBuffer.get(arr, 0, 3 * area)
        val px = IntArray(area)
        for (i in 0 until area) {
            val r = (arr[i] * 255f).toInt().coerceIn(0, 255)
            val g = (arr[area + i] * 255f).toInt().coerceIn(0, 255)
            val b = (arr[2 * area + i] * 255f).toInt().coerceIn(0, 255)
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, n, n, Bitmap.Config.ARGB_8888)
    }

    /** 把視窗 LaMa 輸出貼回 result，只換遮罩內像素（序列呼叫、寫入安全）。 */
    private fun compositePixels(result: Bitmap, maskPx: IntArray, o: WinOut) {
        val w = result.width
        val cur = IntArray(o.ww * o.wh)
        result.getPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
        for (y in 0 until o.wh) {
            val maskRow = (o.y0 + y) * w + o.x0
            val row = y * o.ww
            for (x in 0 until o.ww) {
                if ((maskPx[maskRow + x] and 0xFF) > 127) cur[row + x] = o.px[row + x]
            }
        }
        result.setPixels(cur, 0, o.ww, o.x0, o.y0, o.ww, o.wh)
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "Inpainter"
        private const val INPUT_IMAGE = "image"
        private const val INPUT_MASK = "mask"
        private const val OUT_NAME = "output"
        private const val FILL_REACH = 64 // box-fill 就近取色：每方向最遠找幾像素的非遮罩背景
    }
}
