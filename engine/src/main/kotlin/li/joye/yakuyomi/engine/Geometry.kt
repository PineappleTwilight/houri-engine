package li.joye.yakuyomi.engine

import kotlin.math.hypot

/**
 * 純 Kotlin 幾何 primitive（CLAUDE.md §6：cv2 → 手刻）。
 * 取代 ctd_utils 後處理用到的 cv2.minAreaRect + pyclipper unclip。
 */

data class Pt(val x: Float, val y: Float)

/** 旋轉矩形：中心 (cx,cy)、主軸單位向量 (ux,uy)、沿主軸長 w、沿副軸長 h。 */
class RotRect(
    val cx: Float, val cy: Float,
    val ux: Float, val uy: Float,
    val w: Float, val h: Float,
) {
    fun corners(): List<Pt> {
        val hw = w / 2f
        val hh = h / 2f
        // 主軸 (ux,uy)、副軸 (-uy,ux)
        return listOf(
            Pt(cx - hw * ux - hh * -uy, cy - hw * uy - hh * ux),
            Pt(cx + hw * ux - hh * -uy, cy + hw * uy - hh * ux),
            Pt(cx + hw * ux + hh * -uy, cy + hw * uy + hh * ux),
            Pt(cx - hw * ux + hh * -uy, cy - hw * uy + hh * ux),
        )
    }

    /** DB unclip：偏移 d = area*ratio/perimeter；矩形外擴（w,h 各 +2d）。對齊 db_utils.unclip(1.5)。 */
    fun unclip(ratio: Float): RotRect {
        val peri = 2f * (w + h)
        val d = if (peri > 1e-6f) (w * h) * ratio / peri else 0f
        return RotRect(cx, cy, ux, uy, w + 2f * d, h + 2f * d)
    }
}

object Geometry {
    private fun cross(o: Pt, a: Pt, b: Pt) =
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    /** Andrew monotone chain 凸包。 */
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

    /** 旋轉卡尺：對凸包每條邊建座標系算外接框，取面積最小者。對齊 cv2.minAreaRect。 */
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
                // (u,v) → xy：origin a + u*(ex,ey) + v*(-ey,ex)
                val cx = a.x + cu * ex - cv * ey
                val cy = a.y + cu * ey + cv * ex
                best = RotRect(cx, cy, ex, ey, w, h)
            }
        }
        return best
    }
}
