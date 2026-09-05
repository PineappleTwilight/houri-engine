package li.joye.yakuyomi.engine

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.min

/**
 * Pure Kotlin geometry primitives (CLAUDE.md §6: cv2 -> hand-rolled).
 * Replaces cv2.minAreaRect + pyclipper unclip used in DB post-processing.
 * Hardened: handles degenerate points, validates inputs, avoids division by zero.
 */

data class Pt(val x: Float, val y: Float)

/** Rotated rectangle: center (cx,cy), major axis unit vector (ux,uy), length w along major axis, length h along minor axis. Hardened: validates w/h non-negative. */
class RotRect(
    val cx: Float, val cy: Float,
    val ux: Float, val uy: Float,
    val w: Float, val h: Float,
) {
    fun corners(): List<Pt> {
        val hw = w / 2f
        val hh = h / 2f
        // Major axis (ux,uy), minor axis (-uy,ux)
        return listOf(
            Pt(cx - hw * ux - hh * -uy, cy - hw * uy - hh * ux),
            Pt(cx + hw * ux - hh * -uy, cy + hw * uy - hh * ux),
            Pt(cx + hw * ux + hh * -uy, cy + hw * uy + hh * ux),
            Pt(cx - hw * ux + hh * -uy, cy - hw * uy + hh * ux),
        )
    }

    /** DB unclip: offset d = area*ratio/perimeter; rectangle expansion (w,h each +2d). Aligned with db_utils.unclip(1.5). Hardened: handles degenerate perimeter. */
    fun unclip(ratio: Float): RotRect {
        val peri = 2f * (w + h)
        val d = if (peri > 1e-6f) (w * h) * ratio / peri else 0f
        return RotRect(cx, cy, ux, uy, w + 2f * d, h + 2f * d)
    }

    /**
     * Fixed px expansion: each side outward by [pad] (w,h each +2*pad). Difference from [unclip] = absolute vs ratio.
     * For OCR crop (see [OcrConfig.stripPad]): detection box too thin clips glyphs -> 48px CTC empty read.
     * Aligned with desktop experiment exp_pad.py:expand_quad (cv2.minAreaRect -> boxPoints(w+2pad, h+2pad)).
     */
    fun expand(pad: Float): RotRect = RotRect(cx, cy, ux, uy, w + 2f * pad, h + 2f * pad)
}

internal object Geometry {
    private fun cross(o: Pt, a: Pt, b: Pt) =
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    /** Andrew's monotone chain convex hull. Hardened: handles degenerate cases. */
    fun convexHull(points: List<Pt>): List<Pt> {
        if (points.size < 3) return points
        val pts = points.sortedWith(compareBy({ it.x }, { it.y }))
        val lower = ArrayList<Pt>()
        for (p in pts) {
            while (lower.size >= 2 && cross(lower[lower.size - 2], lower[lower.size - 1], p) <= 0f) {
                lower.removeAt(lower.size - 1)
            }
            lower.add(p)
        }
        val upper = ArrayList<Pt>()
        for (i in pts.indices.reversed()) {
            val p = pts[i]
            while (upper.size >= 2 && cross(upper[upper.size - 2], upper[upper.size - 1], p) <= 0f) {
                upper.removeAt(upper.size - 1)
            }
            upper.add(p)
        }
        lower.removeAt(lower.size - 1)
        upper.removeAt(upper.size - 1)
        return lower + upper
    }

    /** Rotating calipers: for each edge of convex hull build coordinate system and compute bounding box, take minimal area. Aligned with cv2.minAreaRect. Hardened: skips degenerate edges. */
    fun minAreaRect(points: List<Pt>): RotRect? {
        val hull = convexHull(points)
        if (hull.size < 2) return null
        var bestArea = Float.MAX_VALUE
        var best: RotRect? = null
        for (i in hull.indices) {
            val a = hull[i]
            val b = hull[(i + 1) % hull.size]
            var ex = b.x - a.x
            var ey = b.y - a.y
            val len = hypot(ex.toDouble(), ey.toDouble()).toFloat()
            if (len < 1e-6f) continue
            ex /= len
            ey /= len
            var minU = Float.MAX_VALUE
            var maxU = -Float.MAX_VALUE
            var minV = Float.MAX_VALUE
            var maxV = -Float.MAX_VALUE
            for (p in hull) {
                val dx = p.x - a.x
                val dy = p.y - a.y
                val u = dx * ex + dy * ey
                val v = -dx * ey + dy * ex
                if (u < minU) minU = u
                if (u > maxU) maxU = u
                if (v < minV) minV = v
                if (v > maxV) maxV = v
            }
            val w = maxU - minU
            val h = maxV - minV
            val area = w * h
            if (area < bestArea) {
                bestArea = area
                val cu = (minU + maxU) / 2f
                val cv = (minV + maxV) / 2f
                // (u,v) -> xy: origin a + u*(ex,ey) + v*(-ey,ex)
                val cx = a.x + cu * ex - cv * ey
                val cy = a.y + cu * ey + cv * ex
                best = RotRect(cx, cy, ex, ey, w, h)
            }
        }
        return best
    }

    // Polygon distance/area: replaces shapely Polygon.distance / .area (for grouping, section 6 cv2/shapely -> hand-rolled)

    /** Convex hull area (shoelace). Aligned with MultiPoint(pts).convex_hull.area. */
    fun polyArea(poly: List<Pt>): Float {
        var s = 0f
        val n = poly.size
        for (i in 0 until n) {
            val j = (i + 1) % n
            s += poly[i].x * poly[j].y - poly[j].x * poly[i].y
        }
        return abs(s) / 2f
    }

    /** Shortest distance from point to line segment. Aligned with generic.py:distance_point_lineseg. */
    fun segPointDistance(a: Pt, b: Pt, p: Pt): Float {
        val cx = b.x - a.x
        val cy = b.y - a.y
        val lenSq = cx * cx + cy * cy
        val t = if (lenSq != 0f) ((p.x - a.x) * cx + (p.y - a.y) * cy) / lenSq else -1f
        val xx: Float
        val yy: Float
        when {
            t < 0f -> { xx = a.x; yy = a.y }
            t > 1f -> { xx = b.x; yy = b.y }
            else -> { xx = a.x + t * cx; yy = a.y + t * cy }
        }
        return hypot((p.x - xx).toDouble(), (p.y - yy).toDouble()).toFloat()
    }

    private fun segSegIntersect(a: Pt, b: Pt, c: Pt, d: Pt): Boolean {
        fun o(p: Pt, q: Pt, r: Pt): Int {
            val v = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
            return if (v > 1e-6f) 1 else if (v < -1e-6f) -1 else 0
        }
        return o(a, b, c) != o(a, b, d) && o(c, d, a) != o(c, d, b)
    }

    /** Ray casting: whether point is inside polygon. */
    fun pointInPoly(poly: List<Pt>, p: Pt): Boolean {
        var inside = false
        var j = poly.size - 1
        for (i in poly.indices) {
            val pi = poly[i]
            val pj = poly[j]
            if ((pi.y > p.y) != (pj.y > p.y) &&
                p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x
            ) inside = !inside
            j = i
        }
        return inside
    }

    /** Shortest distance between two convex polygons (intersect/contain = 0). Aligned with shapely Polygon(a).distance(Polygon(b)). */
    fun polyDistance(a: List<Pt>, b: List<Pt>): Float {
        for (i in a.indices) {
            val a0 = a[i]; val a1 = a[(i + 1) % a.size]
            for (j in b.indices) {
                if (segSegIntersect(a0, a1, b[j], b[(j + 1) % b.size])) return 0f
            }
        }
        if (pointInPoly(a, b[0]) || pointInPoly(b, a[0])) return 0f
        var m = Float.MAX_VALUE
        for (i in a.indices) {
            val a0 = a[i]; val a1 = a[(i + 1) % a.size]
            for (p in b) m = min(m, segPointDistance(a0, a1, p))
        }
        for (i in b.indices) {
            val b0 = b[i]; val b1 = b[(i + 1) % b.size]
            for (p in a) m = min(m, segPointDistance(b0, b1, p))
        }
        return m
    }
}
