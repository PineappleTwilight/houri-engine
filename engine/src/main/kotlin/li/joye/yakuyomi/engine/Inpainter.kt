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
 * LaMa 去字（Koharu mayocream/lama-manga.onnx，固定 512×512）。
 *
 * I/O：image[1,3,512,512] + mask[1,1,512,512]（/255、RGB、mask 1=擦）→ output[1,3,512,512]。
 * block-aware（對齊 Koharu BALLOON_WINDOW_RATIO，§4 第二層）：逐氣泡把窗放大 ×1.7 → 裁 → 縮 512 →
 *   LaMa → 縮回 → 只在遮罩處貼回。頁面太大、模型只吃 512，故不整頁一次跑。
 * 對齊 parity/inpaint_parity.py。
 */
class Inpainter(modelBytes: ByteArray) : AutoCloseable {

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

    /** 回傳去字後的新 Bitmap（原圖不動）。 */
    fun inpaint(page: Bitmap, regions: List<TextRegion>): Bitmap {
        val w = page.width
        val h = page.height
        val result = page.copy(Bitmap.Config.ARGB_8888, true)

        // 文字遮罩：填充各行 quad（FILL_AND_STROKE 近似膨脹）
        val maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(maskBmp).apply {
            drawColor(Color.BLACK)
            val p = Paint().apply {
                color = Color.WHITE
                style = Paint.Style.FILL_AND_STROKE
                strokeWidth = DILATE
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
            val rw = (region.x1 - region.x0)
            val rh = (region.y1 - region.y0)
            val cx = (region.x0 + region.x1) / 2f
            val cy = (region.y0 + region.y1) / 2f
            val wx0 = (cx - rw * WIN / 2f).toInt().coerceIn(0, w - 1)
            val wy0 = (cy - rh * WIN / 2f).toInt().coerceIn(0, h - 1)
            val wx1 = (cx + rw * WIN / 2f).toInt().coerceIn(wx0 + 1, w)
            val wy1 = (cy + rh * WIN / 2f).toInt().coerceIn(wy0 + 1, h)
            val ww = wx1 - wx0
            val wh = wy1 - wy0
            if (ww < 8 || wh < 8) continue

            val cropBmp = Bitmap.createBitmap(result, wx0, wy0, ww, wh)
            val crop512 = Bitmap.createScaledBitmap(cropBmp, SIZE, SIZE, true)
            val maskCropBmp = Bitmap.createBitmap(maskBmp, wx0, wy0, ww, wh)
            val mask512 = Bitmap.createScaledBitmap(maskCropBmp, SIZE, SIZE, false)

            val imgTensor = imageToNCHW(crop512)
            val maskTensor = maskTo1CH(mask512)
            try {
                session.run(mapOf(INPUT_IMAGE to imgTensor, INPUT_MASK to maskTensor)).use { res ->
                    val out = res.get(OUT_NAME).orElseThrow { IllegalStateException("缺輸出 $OUT_NAME") } as OnnxTensor
                    val res512 = nchwToBitmap(out)
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

    private fun imageToNCHW(bmp: Bitmap): OnnxTensor {
        val px = IntArray(SIZE * SIZE)
        bmp.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
        val area = SIZE * SIZE
        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = px[i]
            chw[i] = ((p shr 16) and 0xFF) / 255f
            chw[area + i] = ((p shr 8) and 0xFF) / 255f
            chw[2 * area + i] = (p and 0xFF) / 255f
        }
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))
    }

    private fun maskTo1CH(bmp: Bitmap): OnnxTensor {
        val px = IntArray(SIZE * SIZE)
        bmp.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
        val m = FloatArray(SIZE * SIZE)
        for (i in px.indices) m[i] = if ((px[i] and 0xFF) > 127) 1f else 0f
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(m), longArrayOf(1, 1, SIZE.toLong(), SIZE.toLong()))
    }

    private fun nchwToBitmap(t: OnnxTensor): Bitmap {
        val area = SIZE * SIZE
        val arr = FloatArray(3 * area)
        t.floatBuffer.get(arr, 0, 3 * area)
        val px = IntArray(area)
        for (i in 0 until area) {
            val r = (arr[i] * 255f).toInt().coerceIn(0, 255)
            val g = (arr[area + i] * 255f).toInt().coerceIn(0, 255)
            val b = (arr[2 * area + i] * 255f).toInt().coerceIn(0, 255)
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, SIZE, SIZE, Bitmap.Config.ARGB_8888)
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
        private const val SIZE = 512
        private const val NUM_THREADS = 4
        private const val WIN = 1.7f
        private const val DILATE = 7f
        private const val INPUT_IMAGE = "image"
        private const val INPUT_MASK = "mask"
        private const val OUT_NAME = "output"
    }
}
