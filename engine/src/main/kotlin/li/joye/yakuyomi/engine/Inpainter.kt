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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit

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

    suspend fun inpaint(page: Bitmap, regions: List<TextRegion>): Bitmap = coroutineScope {
        val w = page.width
        val h = page.height
        val result = page.copy(Bitmap.Config.ARGB_8888, true)

        val maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(maskBmp).apply {
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
        val maskPx = IntArray(w * h)
        maskBmp.getPixels(maskPx, 0, w, 0, 0, w, h)

        if (cfg.method == "boxfill") {
            boxFill(result, regions, maskPx) // 瞬間：取氣泡底色填字區、不跑 LaMa
            maskBmp.recycle()
            return@coroutineScope result
        }
        // lama 逐區平行：各區裁窗 → 平行跑 LaMa（只讀原圖+遮罩、唯讀安全）→ 收齊後序列貼回
        val sem = Semaphore(cfg.concurrency.coerceAtLeast(1))
        val outs = regions.mapNotNull { windowOf(it, w, h) }
            .map { win -> async(Dispatchers.Default) { sem.withPermit { runWindow(page, maskBmp, win) } } }
            .awaitAll()
        for (o in outs) if (o != null) compositePixels(result, maskPx, o)
        maskBmp.recycle()
        result
    }

    /** box-fill 快速去字：每區的遮罩像素換成該區氣泡底色（bbox 內非遮罩像素均值）。無 LaMa、瞬間。 */
    private fun boxFill(result: Bitmap, regions: List<TextRegion>, maskPx: IntArray) {
        val w = result.width
        val h = result.height
        val px = IntArray(w * h)
        result.getPixels(px, 0, w, 0, 0, w, h)
        for (region in regions) {
            val x0 = region.x0.toInt().coerceIn(0, w - 1)
            val y0 = region.y0.toInt().coerceIn(0, h - 1)
            val x1 = region.x1.toInt().coerceIn(x0 + 1, w)
            val y1 = region.y1.toInt().coerceIn(y0 + 1, h)
            var sr = 0L; var sg = 0L; var sb = 0L; var cnt = 0L
            for (y in y0 until y1) {
                val row = y * w
                for (x in x0 until x1) {
                    val i = row + x
                    if ((maskPx[i] and 0xFF) <= 127) { // 非遮罩＝氣泡底（非文字）
                        val p = px[i]; sr += (p shr 16) and 0xFF; sg += (p shr 8) and 0xFF; sb += p and 0xFF; cnt++
                    }
                }
            }
            val bg = if (cnt > 0) Color.rgb((sr / cnt).toInt(), (sg / cnt).toInt(), (sb / cnt).toInt()) else Color.WHITE
            for (y in y0 until y1) {
                val row = y * w
                for (x in x0 until x1) {
                    val i = row + x
                    if ((maskPx[i] and 0xFF) > 127) px[i] = bg
                }
            }
        }
        result.setPixels(px, 0, w, 0, 0, w, h)
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
    }
}
