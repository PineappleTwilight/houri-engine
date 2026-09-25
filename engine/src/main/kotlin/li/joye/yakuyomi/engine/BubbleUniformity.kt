package li.joye.yakuyomi.engine

import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Uniform-bubble classification over raw ARGB pixels (pure JVM, Android-free).
 *
 * Clean-room port of the K3 "simple block" check: sample a 1.025x outward ring
 * around the region (corners pushed 2.5% past the center), clamp samples to the
 * image edges, average RGB, and accept only when nearly every sample sits close
 * to that mean. The classifier is used by the production AOT fast path; its pure
 * sampling helpers remain independently testable.
 *
 * Pixels are `IntArray` in `0xAARRGGBB` row-major order (same layout as the
 * Android bitmap helpers); only R/G/B participate, alpha is ignored.
 */
object BubbleUniformity {
    /** Outward ring factor: corners pushed 2.5% past the region center. */
    const val RING_SCALE = 1.025f

    /** Max Euclidean RGB distance from the mean to count as matching. */
    const val MAX_DISTANCE = 15f

    /** Required fraction of matching samples to call the bubble uniform. */
    const val REQUIRED_RATIO = 0.99f

    /**
     * Classify the region background as uniform (`true`) or not (`false`).
     *
     * Returns `false` (never throws) for empty images, size mismatches, and
     * degenerate regions with no sample points.
     */
    fun classify(
        pixels: IntArray,
        imageW: Int,
        imageH: Int,
        region: TextRegion,
        ringScale: Float = RING_SCALE,
        maxDistance: Float = MAX_DISTANCE,
        requiredRatio: Float = REQUIRED_RATIO,
    ): Boolean {
        if (imageW <= 0 || imageH <= 0) return false
        if (pixels.size < imageW * imageH) return false
        val samples = sampleRing(pixels, imageW, imageH, region, ringScale)
        if (samples.isEmpty()) return false
        var sumR = 0L
        var sumG = 0L
        var sumB = 0L
        for (c in samples) {
            sumR += (c shr 16) and 0xFF
            sumG += (c shr 8) and 0xFF
            sumB += c and 0xFF
        }
        val n = samples.size.toDouble()
        val meanR = sumR / n
        val meanG = sumG / n
        val meanB = sumB / n
        var valid = 0
        for (c in samples) {
            val dr = ((c shr 16) and 0xFF) - meanR
            val dg = ((c shr 8) and 0xFF) - meanG
            val db = (c and 0xFF) - meanB
            if (sqrt(dr * dr + dg * dg + db * db) <= maxDistance) valid++
        }
        if (valid == 0) return false
        return valid.toDouble() / samples.size >= requiredRatio
    }

    /**
     * Mean ring color as `0xFFRRGGBB`, or `null` when there is nothing to sample.
     * Useful for the future flat-fill path; classification itself uses [classify].
     */
    fun meanColor(
        pixels: IntArray,
        imageW: Int,
        imageH: Int,
        region: TextRegion,
        ringScale: Float = RING_SCALE,
    ): Int? {
        if (imageW <= 0 || imageH <= 0) return null
        if (pixels.size < imageW * imageH) return null
        val samples = sampleRing(pixels, imageW, imageH, region, ringScale)
        if (samples.isEmpty()) return null
        var sumR = 0L
        var sumG = 0L
        var sumB = 0L
        for (c in samples) {
            sumR += (c shr 16) and 0xFF
            sumG += (c shr 8) and 0xFF
            sumB += c and 0xFF
        }
        val n = samples.size
        val r = (sumR / n).toInt().coerceIn(0, 255)
        val g = (sumG / n).toInt().coerceIn(0, 255)
        val b = (sumB / n).toInt().coerceIn(0, 255)
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

    /**
     * Sample the outward ring: every line-quad corner (falling back to the
     * axis-aligned bounds corners when the region has no usable quads), scaled
     * past the point-set center by [ringScale] and clamped to image edges.
     */
    private fun sampleRing(
        pixels: IntArray,
        imageW: Int,
        imageH: Int,
        region: TextRegion,
        ringScale: Float,
    ): IntArray {
        val source = ArrayList<Pt>()
        for (line in region.lines) {
            if (line.quad.size >= 4) {
                source.add(line.quad[0])
                source.add(line.quad[1])
                source.add(line.quad[2])
                source.add(line.quad[3])
            }
        }
        if (source.isEmpty()) {
            if (region.x1 <= region.x0 || region.y1 <= region.y0) return IntArray(0)
            source.add(Pt(region.x0, region.y0))
            source.add(Pt(region.x1, region.y0))
            source.add(Pt(region.x1, region.y1))
            source.add(Pt(region.x0, region.y1))
        }
        var cx = 0.0
        var cy = 0.0
        for (p in source) {
            cx += p.x
            cy += p.y
        }
        cx /= source.size
        cy /= source.size
        val out = IntArray(source.size)
        for (i in source.indices) {
            val sx = (cx + (source[i].x - cx) * ringScale).roundToInt().coerceIn(0, imageW - 1)
            val sy = (cy + (source[i].y - cy) * ringScale).roundToInt().coerceIn(0, imageH - 1)
            out[i] = pixels[sy * imageW + sx]
        }
        return out
    }
}
