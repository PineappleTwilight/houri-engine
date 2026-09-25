package li.joye.yakuyomi.engine

import kotlin.math.ceil

/**
 * Shared region-bound expansion and clamping (pure JVM, Android-free).
 *
 * Mirrors the production behavior currently living in [Inpainter.buildSegMask]
 * and [Renderer.drawHorizontal] without depending on Android graphics:
 * portrait check `aspect > 2.5`, [expandW]/[expandH] orientation swap,
 * per-side [bboxPad], then a pure clamp into image bounds.
 *
 * Used by the inpainter mask and uniform-fill paths; the pure helpers remain independently testable.
 */
data class RegionRect(val x0: Float, val y0: Float, val x1: Float, val y1: Float) {
    val width: Float get() = x1 - x0
    val height: Float get() = y1 - y0
}

object RegionBounds {
    /** Same portrait threshold as Inpainter mask expansion and Renderer horizontal layout. */
    const val PORTRAIT_THRESHOLD = 2.5f

    /** Outward ring factor inspiring [maskPad]: K3 expands bubble corners 2.5% past center. */
    const val RING_SCALE = 1.025f

    /** Upper bound so the K3-inspired safety margin can never blow the mask up. */
    const val MAX_MASK_PAD = 64

    /**
     * Portrait when the region is tall and narrow. Mirrors Inpainter: widths at
     * or below 1px report aspect 1 (landscape) to avoid dividing by near-zero.
     */
    fun isPortrait(region: TextRegion, threshold: Float = PORTRAIT_THRESHOLD): Boolean {
        val w = region.x1 - region.x0
        val h = region.y1 - region.y0
        val aspect = if (w > 1f) h / w else 1f
        return aspect > threshold
    }

    /**
     * Expand the region bbox by the orientation-swapped factors plus [bboxPad]
     * on every side. Portrait swaps long/short axes so the layout width follows
     * the long axis, matching Inpainter/Renderer. Pure: no clamping, no I/O.
     */
    fun expanded(
        region: TextRegion,
        expandW: Float = 1.3f,
        expandH: Float = 1.5f,
        bboxPad: Int = 16,
        portraitThreshold: Float = PORTRAIT_THRESHOLD,
    ): RegionRect {
        val halfW = (region.x1 - region.x0) / 2f
        val halfH = (region.y1 - region.y0) / 2f
        val portrait = isPortrait(region, portraitThreshold)
        val expW = if (portrait) expandH else expandW
        val expH = if (portrait) expandW else expandH
        val dx = halfW * (expW - 1f) + bboxPad
        val dy = halfH * (expH - 1f) + bboxPad
        return RegionRect(region.x0 - dx, region.y0 - dy, region.x1 + dx, region.y1 + dy)
    }

    /** Pure clamp of [rect] into `[0, imageW] x [0, imageH]`, preserving size when possible. */
    fun clamped(rect: RegionRect, imageW: Int, imageH: Int): RegionRect {
        require(imageW > 0 && imageH > 0) { "image bounds must be positive: $imageW x $imageH" }
        val w = rect.x1 - rect.x0
        val h = rect.y1 - rect.y0
        if (w <= 0f || h <= 0f) {
            val cx = rect.x0.coerceIn(0f, imageW.toFloat())
            val cy = rect.y0.coerceIn(0f, imageH.toFloat())
            return RegionRect(cx, cy, cx, cy)
        }
        val cw = minOf(w, imageW.toFloat())
        val ch = minOf(h, imageH.toFloat())
        val nx0 = rect.x0.coerceIn(0f, imageW - cw)
        val ny0 = rect.y0.coerceIn(0f, imageH - ch)
        return RegionRect(nx0, ny0, nx0 + cw, ny0 + ch)
    }

    /** Expand then clamp: the review-ready entry point for future mask/render wiring. */
    fun expandedClamped(
        region: TextRegion,
        imageW: Int,
        imageH: Int,
        expandW: Float = 1.3f,
        expandH: Float = 1.5f,
        bboxPad: Int = 16,
        portraitThreshold: Float = PORTRAIT_THRESHOLD,
    ): RegionRect =
        clamped(expanded(region, expandW, expandH, bboxPad, portraitThreshold), imageW, imageH)

    /**
     * Conservative mask padding in px.
     *
     * Always covers at least `bboxPad + ceil(maskDilate / 2)` (the bbox margin
     * plus the separable-dilation radius used by the inpaint mask). Optionally
     * adds a small K3-inspired safety margin of 2.5% of [referenceSize] (e.g. a
     * region side), capped at [maxPad] so it stays bounded.
     */
    fun maskPad(
        bboxPad: Int,
        maskDilate: Float,
        referenceSize: Float = 0f,
        maxPad: Int = MAX_MASK_PAD,
    ): Int {
        val base = bboxPad + ceil(maskDilate / 2f).toInt()
        val extra = if (referenceSize > 0f) ceil(referenceSize * (RING_SCALE - 1f)).toInt() else 0
        return (base + extra).coerceIn(base, maxOf(base, maxPad))
    }
}
