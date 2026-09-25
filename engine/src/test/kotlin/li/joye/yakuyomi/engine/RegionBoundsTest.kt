package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.ceil

/** 共享邊界擴張/裁剪（對齊 Inpainter/Renderer：2.5 豎排閾值、expandW/H 互換、bboxPad）。 */
class RegionBoundsTest {

    private fun region(x0: Float, y0: Float, x1: Float, y1: Float): TextRegion {
        val quad = listOf(Pt(x0, y0), Pt(x1, y0), Pt(x1, y1), Pt(x0, y1))
        return TextRegion(listOf(TextLine(quad, 1f)), "h")
    }

    @Test fun portraitThreshold_mirrorsRenderer() {
        assertTrue(RegionBounds.isPortrait(region(0f, 0f, 20f, 60f)))
        assertFalse(RegionBounds.isPortrait(region(0f, 0f, 60f, 20f)))
        assertFalse(RegionBounds.isPortrait(region(0f, 0f, 20f, 50f))) // aspect == 2.5 不算豎排
        assertFalse(RegionBounds.isPortrait(region(5f, 5f, 5.5f, 60f))) // 寬 <=1px 視為橫排
    }

    @Test fun landscapeExpansion_usesExpandWForWidth() {
        val r = RegionBounds.expanded(region(100f, 100f, 200f, 140f), expandW = 1.3f, expandH = 1.5f, bboxPad = 16)
        assertEquals(100f - (50f * 0.3f + 16f), r.x0, 0.01f)
        assertEquals(200f + (50f * 0.3f + 16f), r.x1, 0.01f)
        assertEquals(100f - (20f * 0.5f + 16f), r.y0, 0.01f)
        assertEquals(140f + (20f * 0.5f + 16f), r.y1, 0.01f)
    }

    @Test fun portraitExpansion_swapsAxes() {
        val r = RegionBounds.expanded(region(100f, 100f, 120f, 200f), expandW = 1.3f, expandH = 1.5f, bboxPad = 16)
        // 豎排：寬用 expandH，高用 expandW
        assertEquals(100f - (10f * 0.5f + 16f), r.x0, 0.01f)
        assertEquals(120f + (10f * 0.5f + 16f), r.x1, 0.01f)
        assertEquals(100f - (50f * 0.3f + 16f), r.y0, 0.01f)
        assertEquals(200f + (50f * 0.3f + 16f), r.y1, 0.01f)
    }

    @Test fun expandedClamped_staysInsideImage() {
        val r = RegionBounds.expandedClamped(region(0f, 0f, 40f, 40f), 100, 100)
        assertTrue(r.x0 >= 0f && r.y0 >= 0f && r.x1 <= 100f && r.y1 <= 100f)
        val edge = RegionBounds.expandedClamped(region(90f, 90f, 99f, 99f), 100, 100)
        assertTrue(edge.x0 >= 0f && edge.y0 >= 0f && edge.x1 <= 100f && edge.y1 <= 100f)
        assertTrue(edge.x1 > edge.x0 && edge.y1 > edge.y0)
    }

    @Test fun clamped_oversizedBox_cappedToImage() {
        val r = RegionBounds.clamped(RegionRect(-50f, -50f, 500f, 500f), 100, 80)
        assertEquals(0f, r.x0, 0.01f)
        assertEquals(0f, r.y0, 0.01f)
        assertEquals(100f, r.x1, 0.01f)
        assertEquals(80f, r.y1, 0.01f)
    }

    @Test fun tinyRegion_doesNotCrash() {
        val r = RegionBounds.expandedClamped(region(10f, 10f, 11f, 11f), 100, 100)
        assertTrue(r.x0 >= 0f && r.y0 >= 0f && r.x1 <= 100f && r.y1 <= 100f)
        assertTrue(r.x1 >= r.x0 && r.y1 >= r.y0)
    }

    @Test fun maskPad_coversDilationAndBoundedMargin() {
        val base = 16 + ceil(24f / 2f).toInt()
        assertEquals(base, RegionBounds.maskPad(16, 24f))
        val withMargin = RegionBounds.maskPad(16, 24f, referenceSize = 200f)
        assertTrue(withMargin >= base)
        assertTrue(withMargin <= RegionBounds.MAX_MASK_PAD)
        assertEquals(RegionBounds.MAX_MASK_PAD, RegionBounds.maskPad(16, 24f, referenceSize = 10000f))
    }
}
