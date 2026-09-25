package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** 均勻氣泡判定（1.025 外環取樣、邊緣裁剪、均值歐氏距離、99% 門檻為 15）。 */
class BubbleUniformityTest {

    private fun region(x0: Float, y0: Float, x1: Float, y1: Float): TextRegion {
        val quad = listOf(Pt(x0, y0), Pt(x1, y0), Pt(x1, y1), Pt(x0, y1))
        return TextRegion(listOf(TextLine(quad, 1f)), "h")
    }

    private fun solid(w: Int, h: Int, color: Int): IntArray = IntArray(w * h) { color }

    @Test fun solidWhite_isUniform() {
        val px = solid(60, 60, 0xFFFFFFFF.toInt())
        assertTrue(BubbleUniformity.classify(px, 60, 60, region(10f, 10f, 40f, 40f)))
        assertEquals(0xFFFFFFFF.toInt(), BubbleUniformity.meanColor(px, 60, 60, region(10f, 10f, 40f, 40f)))
    }

    @Test fun solidBlack_isUniform() {
        val px = solid(60, 60, 0xFF000000.toInt())
        assertTrue(BubbleUniformity.classify(px, 60, 60, region(10f, 10f, 40f, 40f)))
    }

    @Test fun gradient_isNotUniform() {
        val w = 60
        val h = 60
        val px = IntArray(w * h) { i ->
            val x = i % w
            val v = (x * 255 / (w - 1)).coerceIn(0, 255)
            (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        assertFalse(BubbleUniformity.classify(px, w, h, region(5f, 20f, 55f, 40f)))
    }

    @Test fun noise_isNotUniform() {
        val w = 40
        val h = 40
        val px = IntArray(w * h) { i ->
            val v = ((i * 7919) % 256 + 256) % 256
            (0xFF shl 24) or (v shl 16) or ((255 - v) shl 8) or (v / 2)
        }
        assertFalse(BubbleUniformity.classify(px, w, h, region(5f, 5f, 35f, 35f)))
    }

    @Test fun edgeRegion_clampsSamples() {
        val px = solid(30, 30, 0xFFFFFFFF.toInt())
        // 貼邊：外環 1.025 會超出圖邊，必須裁剪而不崩潰，且純白仍判均勻
        assertTrue(BubbleUniformity.classify(px, 30, 30, region(0f, 0f, 29f, 29f)))
        assertTrue(BubbleUniformity.classify(px, 30, 30, region(25f, 25f, 29f, 29f)))
    }

    @Test fun tinyRegion_solidStaysUniform() {
        val px = solid(20, 20, 0xFFFFFFFF.toInt())
        assertTrue(BubbleUniformity.classify(px, 20, 20, region(10f, 10f, 11f, 11f)))
    }

    @Test fun degenerateInputs_returnFalse() {
        val px = solid(10, 10, 0xFFFFFFFF.toInt())
        assertFalse(BubbleUniformity.classify(px, 0, 10, region(1f, 1f, 5f, 5f)))
        assertFalse(BubbleUniformity.classify(IntArray(10), 10, 10, region(1f, 1f, 5f, 5f)))
        val dot = region(5f, 5f, 5f, 5f) // 零面積：無有效環點
        assertFalse(BubbleUniformity.classify(px, 10, 10, dot))
    }
}
