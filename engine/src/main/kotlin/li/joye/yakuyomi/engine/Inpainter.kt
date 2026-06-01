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

/**
 * LaMa 去字（Koharu mayocream/lama-manga.onnx）。block-aware：逐氣泡裁窗 → tile → LaMa → 貼回。
 * I/O：image[1,3,N,N] + mask[1,1,N,N]（/255、RGB、mask 1=擦）→ output。參數見 [InpainterConfig]。
 * 對齊 parity/inpaint_parity.py。
 */
class Inpainter(
    modelBytes: ByteArray,
    private val cfg: InpainterConfig = InpainterConfig(),
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(NUM_THREADS)
            try {
                addXnnpack(mapOf("intra_op_num_threads" to NUM_THREADS.toString()))
            } catch (t: Throwable) {
                Log.w(TAG, "XNNPACK 不可用：${t.message}")
            }
        }
        session = env.createSession(modelBytes, opts)
    }

    fun inpaint(page: Bitmap, regions: List<TextRegion>): Bitmap {
        val tile = cfg.tileSize
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

        for (region in regions) {
            val rw = region.x1 - region.x0
            val rh = region.y1 - region.y0
            val cx = (region.x0 + region.x1) / 2f
            val cy = (region.y0 + region.y1) / 2f
            val win = cfg.windowRatio
            val wx0 = (cx - rw * win / 2f).toInt().coerceIn(0, w - 1)
            val wy0 = (cy - rh * win / 2f).toInt().coerceIn(0, h - 1)
            val wx1 = (cx + rw * win / 2f).toInt().coerceIn(wx0 + 1, w)
            val wy1 = (cy + rh * win / 2f).toInt().coerceIn(wy0 + 1, h)
            val ww = wx1 - wx0
            val wh = wy1 - wy0
            if (ww < 8 || wh < 8) continue

            val cropBmp = Bitmap.createBitmap(result, wx0, wy0, ww, wh)
            val crop512 = Bitmap.createScaledBitmap(cropBmp, tile, tile, true)
            val maskCropBmp = Bitmap.createBitmap(maskBmp, wx0, wy0, ww, wh)
            val mask512 = Bitmap.createScaledBitmap(maskCropBmp, tile, tile, false)

            val imgTensor = imageToNCHW(crop512, tile)
            val maskTensor = maskTo1CH(mask512, tile)
            try {
                session.run(mapOf(INPUT_IMAGE to imgTensor, INPUT_MASK to maskTensor)).use { res ->
                    val outT = res.get(OUT_NAME).orElseThrow { IllegalStateException("缺輸出 $OUT_NAME") } as OnnxTensor
                    val res512 = nchwToBitmap(outT, tile)
                    val resWin = Bitmap.createScaledBitmap(res512, ww, wh, true)
                    compositeMasked(result, resWin, maskPx, w, wx0, wy0, ww, wh)
                    res512.recycle()
                    resWin.recycle()
                }
            } catch (t: Throwable) {
                Log.w(TAG, "去字單區失敗：${t.message}")
            } finally {
                imgTensor.close()
                maskTensor.close()
                cropBmp.recycle()
                crop512.recycle()
                maskCropBmp.recycle()
                mask512.recycle()
            }
        }
        maskBmp.recycle()
        return result
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

    private fun compositeMasked(
        result: Bitmap, resWin: Bitmap, maskPx: IntArray,
        w: Int, wx0: Int, wy0: Int, ww: Int, wh: Int,
    ) {
        val rp = IntArray(ww * wh)
        resWin.getPixels(rp, 0, ww, 0, 0, ww, wh)
        val win = IntArray(ww * wh)
        result.getPixels(win, 0, ww, wx0, wy0, ww, wh)
        for (y in 0 until wh) {
            val maskRow = (wy0 + y) * w + wx0
            val winRow = y * ww
            for (x in 0 until ww) {
                if ((maskPx[maskRow + x] and 0xFF) > 127) win[winRow + x] = rp[winRow + x]
            }
        }
        result.setPixels(win, 0, ww, wx0, wy0, ww, wh)
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "Inpainter"
        private const val NUM_THREADS = 4
        private const val INPUT_IMAGE = "image"
        private const val INPUT_MASK = "mask"
        private const val OUT_NAME = "output"
    }
}
