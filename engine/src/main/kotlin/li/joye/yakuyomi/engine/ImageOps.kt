package li.joye.yakuyomi.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 前處理 helper（CLAUDE.md §5 ImageOps：第三層，重寫）。
 */
internal object ImageOps {

    /** NCNN 偵測前處理結果：裸 NCHW FloatArray ＋ 縮放比例（原圖座標 ×ratio→letterbox 空間；反算 ÷ratio）。 */
    class DetectorInputArray(val chw: FloatArray, val ratio: Float)

    /**
     * comic-text-detector 前處理核心（letterbox + /255 + NCHW），回裸 chw + ratio。
     * ported from manga_translator/detection/ctd.py:preprocess_img
     *           + ctd_utils/utils/imgproc_utils.py:letterbox @ d5a3eee
     *   - 等比縮放 r = min(size/h, size/w)，padding 全加在右/下（上游 dw/2,dh/2 被註解 → 不置中）
     *   - 通道 RGB（cv2 路徑 BGR→RGB 後 blobFromImage 不 swap；Android Bitmap 本就 RGB）
     *   - /255、NCHW
     *
     * ★ §10「前處理對齊」：createScaledBitmap(bilinear) 近似 cv2.INTER_LINEAR；
     *   padding 在右/下，故座標反算只需 ÷ratio、無偏移。
     */
    fun detectorChw(page: Bitmap, size: Int): DetectorInputArray {
        val r = min(size.toFloat() / page.height, size.toFloat() / page.width)
        val nw = (page.width * r).roundToInt().coerceAtLeast(1)
        val nh = (page.height * r).roundToInt().coerceAtLeast(1)

        val scaled = Bitmap.createScaledBitmap(page, nw, nh, true)
        // 預設透明黑 (RGB=0) 即 letterbox 黑邊；只取 RGB 不取 alpha
        val canvas = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(canvas).drawBitmap(scaled, 0f, 0f, null)

        val area = size * size
        val pixels = IntArray(area)
        canvas.getPixels(pixels, 0, size, 0, 0, size, size)

        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = pixels[i]
            chw[i] = ((p shr 16) and 0xFF) / 255f           // R → plane 0
            chw[area + i] = ((p shr 8) and 0xFF) / 255f      // G → plane 1
            chw[2 * area + i] = (p and 0xFF) / 255f           // B → plane 2
        }
        if (scaled !== page) scaled.recycle()
        canvas.recycle()
        return DetectorInputArray(chw, r)
    }
}
