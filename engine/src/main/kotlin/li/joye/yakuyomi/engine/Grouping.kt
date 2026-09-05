package li.joye.yakuyomi.engine

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Text lines -> bubble region grouping (merge before translation, avoid per-line fragmentation).
 *
 * Ported from manga_translator/textline_merge/__init__.py + utils/generic.py @ d5a3eee
 * (verified via parity/mit_grouping.py as spec; section 4 layer 1/2: same input near output).
 *
 * Two stages (missing either causes "nearby bubbles merge into one large block" = color block + text overlap):
 *   1. **Connect edges**: quadrilateralCanMergeRegion (loose: same direction, similar font size, gap < 1x char, edge alignment < 3x char).
 *   2. **MST split**: splitTextRegion — for each connected component build MST, if max edge (gap) is outlier relative to rest, cut there, recurse.
 *      This splits back "adjacent bubbles over-connected in step 1", key to not over-merge.
 *
 * Note: merging stage m-i-t assigned_direction is None (OCR sets it) -> distance always uses v mode; mirrored here.
 * Region text color not computed here (Renderer determines black/white from post-inpaint background luminance).
 */
class TextRegion(
    val lines: List<TextLine>,
    val direction: String,
    /** Region tilt angle (degrees, aligned with m-i-t TextBlock.angle); typesetting rotates along this angle, <3 degrees treated as 0. */
    val angle: Float = 0f,
    /** Region center (mean of all line corners), pivot for rotated typesetting. */
    val cx: Float = 0f,
    val cy: Float = 0f,
    /** De-rotated (text natural coordinates) box size, used as text box for rotated typesetting. */
    val boxW: Float = 0f,
    val boxH: Float = 0f,
) {
    var translatedText: String = ""

    /** Whether this region is "on art" (inpaint uses lama reconstruction, not clean white bubble). Set by [Inpainter], [Renderer] uses it to give black text thick white outline (readable on busy background). */
    var onArt: Boolean = false

    /** Auto routing debug: measured background std / luminance mean for this region (set by [Inpainter] auto branch; -1 = not measured). Sandbox inpaint comparison shows on box for threshold tuning, product does not read. */
    var dbgStd: Float = -1f
    var dbgWhite: Float = -1f

    /** Merged source text (lines already sorted in reading order). Japanese has no spaces, directly concatenated. */
    val sourceText: String get() = lines.joinToString("") { it.text }

    val x0: Float = lines.minOf { ln -> ln.quad.minOf { it.x } }
    val y0: Float = lines.minOf { ln -> ln.quad.minOf { it.y } }
    val x1: Float = lines.maxOf { ln -> ln.quad.maxOf { it.x } }
    val y1: Float = lines.maxOf { ln -> ln.quad.maxOf { it.y } }
}

object Grouping {

    fun group(lines: List<TextLine>): List<TextRegion> {
        val n = lines.size
        if (n == 0) return emptyList()
        val q = lines.map { GQuad(it) }

        // Step 1: loose edge connection -> union-find connected components
        // Hardened: limit total edges to prevent O(n^2) blow-up on dense pages (n>200)
        if (n > 300) {
            // Too many lines, likely noise or very dense page -> return each line as separate region to avoid destroying boxes via over-merging
            return lines.mapIndexed { idx, _ -> TextRegion(listOf(lines[idx]), q[idx].direction) }
        }
        val parent = IntArray(n) { it }
        fun find(x: Int): Int {
            var r = x
            while (parent[r] != r) r = parent[r]
            var c = x
            while (parent[c] != c) { val nxt = parent[c]; parent[c] = r; c = nxt }
            return r
        }
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                if (canMerge(q[i], q[j])) parent[find(i)] = find(j)
            }
        }
        val comps = LinkedHashMap<Int, MutableList<Int>>()
        for (i in 0 until n) comps.getOrPut(find(i)) { mutableListOf() }.add(i)

        // Step 2: MST split for each connected component
        val regionIdx = ArrayList<List<Int>>()
        for (comp in comps.values) regionIdx.addAll(splitTextRegion(q, comp))

        // Step 3: sort reading order within each region, determine direction, compute tilt angle + de-rotated box
        return regionIdx.map { members ->
            val dir = majorityDir(q, members)
            val ordered = if (dir == "h") {
                members.sortedBy { q[it].centroid.y }            // Horizontal: top -> bottom
            } else {
                members.sortedByDescending { q[it].centroid.x }  // Vertical: right -> left
            }
            val angle = regionAngle(q, members)
            val pts = members.flatMap { q[it].pts }
            val cx = pts.map { it.x }.average().toFloat()
            val cy = pts.map { it.y }.average().toFloat()
            val (bw, bh) = orientedBox(pts, cx, cy, angle)
            TextRegion(ordered.map { lines[it] }, dir, angle, cx, cy, bw, bh)
        }
    }

    /** Region angle = mean of line angles -90 (degrees, aligned with textline_merge dispatch); <3 degrees treated as 0. */
    private fun regionAngle(q: List<GQuad>, members: List<Int>): Float {
        val meanRad = members.map { q[it].angle.toDouble() }.average()
        val deg = (Math.toDegrees(meanRad) - 90.0).toFloat()
        return if (abs(deg) < 3f) 0f else deg
    }

    /** De-rotate all corner points of region by R(-angle) and compute axis-aligned range = box size in text natural coordinates. */
    private fun orientedBox(pts: List<Pt>, cx: Float, cy: Float, angleDeg: Float): Pair<Float, Float> {
        val rad = Math.toRadians(angleDeg.toDouble())
        val cos = cos(rad).toFloat()
        val sin = sin(rad).toFloat()
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (p in pts) {
            val dx = p.x - cx; val dy = p.y - cy
            val rx = cos * dx + sin * dy
            val ry = -sin * dx + cos * dy
            if (rx < minX) minX = rx; if (rx > maxX) maxX = rx
            if (ry < minY) minY = ry; if (ry > maxY) maxY = ry
        }
        return (maxX - minX) to (maxY - minY)
    }

    // Edge connection decision: quadrilateral_can_merge_region (parameters used for merge_bboxes_text_region)
    private fun canMerge(
        a: GQuad, b: GQuad,
        ratio: Float = 1.9f,
        discardConnectionGap: Float = 2f,
        charGapTolerance: Float = 1f,       // m-i-t merge_bboxes passed value
        charGapTolerance2: Float = 3f,
        fontSizeRatioTol: Float = 2f,
        aspectRatioTol: Float = 1.3f,
    ): Boolean {
        val charSize = min(a.fontSize, b.fontSize)
        if (charSize <= 0f) return false
        val x1 = a.aabb[0]; val y1 = a.aabb[1]; val w1 = a.aabb[2]; val h1 = a.aabb[3]
        val x2 = b.aabb[0]; val y2 = b.aabb[1]; val w2 = b.aabb[2]; val h2 = b.aabb[3]
        val d = Geometry.polyDistance(a.pts, b.pts)
        if (d > discardConnectionGap * charSize) return false
        if (max(a.fontSize, b.fontSize) / charSize > fontSizeRatioTol) return false
        if (a.aspectRatio > aspectRatioTol && b.aspectRatio < 1f / aspectRatioTol) return false
        if (b.aspectRatio > aspectRatioTol && a.aspectRatio < 1f / aspectRatioTol) return false
        if (a.isApproxAxisAligned && b.isApproxAxisAligned) {
            if (d < charSize * charGapTolerance) {
                if (abs(x1 + w1 / 2f - (x2 + w2 / 2f)) < charGapTolerance2) return true
                if (w1 > h1 * ratio && h2 > w2 * ratio) return false
                if (w2 > h2 * ratio && h1 > w1 * ratio) return false
                if (w1 > h1 * ratio || w2 > h2 * ratio) {  // Horizontal
                    return abs(x1 - x2) < charSize * charGapTolerance2 ||
                        abs(x1 + w1 - (x2 + w2)) < charSize * charGapTolerance2
                } else if (h1 > w1 * ratio || h2 > w2 * ratio) {  // Vertical
                    return abs(y1 - y2) < charSize * charGapTolerance2 ||
                        abs(y1 + h1 - (y2 + h2)) < charSize * charGapTolerance2
                }
                return false
            }
            return false
        }
        // Non-axis-aligned (rotated box) branch
        if (abs(a.angle - b.angle) < 15f * PI.toFloat() / 180f) {
            val fs = min(a.fontSize, b.fontSize)
            if (Geometry.polyDistance(a.pts, b.pts) > fs * charGapTolerance2) return false
            if (abs(a.fontSize - b.fontSize) / fs > 0.25f) return false
            return true
        }
        return false
    }

    // MST split: split_text_region
    private fun splitTextRegion(
        q: List<GQuad>, idx: List<Int>, gamma: Float = 0.5f, sigma: Float = 2f,
    ): List<List<Int>> {
        if (idx.size == 1) return listOf(idx)
        if (idx.size == 2) {
            val fs = max(q[idx[0]].fontSize, q[idx[1]].fontSize)
            val close = q[idx[0]].distance(q[idx[1]]) < (1f + gamma) * fs
            val sameAngle = abs(q[idx[0]].angle - q[idx[1]].angle) < 0.2f * PI.toFloat()
            return if (close && sameAngle) listOf(idx) else listOf(listOf(idx[0]), listOf(idx[1]))
        }
        // Complete graph -> Kruskal MST
        val edges = ArrayList<Edge>()
        for (a in idx.indices) for (b in a + 1 until idx.size) {
            edges.add(Edge(idx[a], idx[b], q[idx[a]].distance(q[idx[b]])))
        }
        edges.sortBy { it.w }
        val par = HashMap<Int, Int>().apply { idx.forEach { put(it, it) } }
        fun find(x: Int): Int { var r = x; while (par[r] != r) r = par[r]!!; return r }
        val mst = ArrayList<Edge>()
        for (e in edges) {
            val ra = find(e.u); val rb = find(e.v)
            if (ra != rb) { par[ra] = rb; mst.add(e); if (mst.size == idx.size - 1) break }
        }
        mst.sortByDescending { it.w }
        val distances = mst.map { it.w }
        val fontsize = idx.map { q[it].fontSize }.average().toFloat()
        val mean = distances.average().toFloat()
        val std = sqrt(distances.map { (it - mean) * (it - mean) }.average()).toFloat()
        val stdThreshold = max(0.3f * fontsize + 5f, 5f)
        val b1 = q[mst[0].u]; val b2 = q[mst[0].v]
        val maxPolyDistance = Geometry.polyDistance(b1.pts, b2.pts)
        val maxCentroidAlignment = min(abs(b1.centroid.x - b2.centroid.x), abs(b1.centroid.y - b2.centroid.y))
        val keep = (distances[0] <= mean + std * sigma || distances[0] <= fontsize * (1f + gamma)) &&
            (std < stdThreshold || (maxPolyDistance == 0f && maxCentroidAlignment < 5f))
        if (keep) return listOf(idx)
        // Cut largest edge (mst[0]) -> connected components of remaining MST edges -> recurse
        val par2 = HashMap<Int, Int>().apply { idx.forEach { put(it, it) } }
        fun find2(x: Int): Int { var r = x; while (par2[r] != r) r = par2[r]!!; return r }
        for (k in 1 until mst.size) { par2[find2(mst[k].u)] = find2(mst[k].v) }
        val sub = LinkedHashMap<Int, MutableList<Int>>()
        for (i in idx) sub.getOrPut(find2(i)) { mutableListOf() }.add(i)
        val ans = ArrayList<List<Int>>()
        for (c in sub.values) ans.addAll(splitTextRegion(q, c, gamma, sigma))
        return ans
    }

    private fun majorityDir(q: List<GQuad>, members: List<Int>): String {
        val counts = members.groupingBy { q[it].direction }.eachCount()
        val top2 = counts.entries.sortedByDescending { it.value }
        if (top2.size == 1) return top2[0].key
        if (top2[0].value == top2[1].value) {  // Tie -> pick direction of line with most extreme aspect ratio
            var maxAr = -100f
            var dir = top2[0].key
            for (i in members) {
                val ar = q[i].aspectRatio
                if (ar > maxAr) { maxAr = ar; dir = q[i].direction }
                if (1f / ar > maxAr) { maxAr = 1f / ar; dir = q[i].direction }
            }
            return dir
        }
        return top2[0].key
    }

    private data class Edge(val u: Int, val v: Int, val w: Float)
}

/** Geometry of a single text line (aligned with generic.py:Quadrilateral grouping subset). pts ordered as [top-left, top-right, bottom-right, bottom-left]. */
private class GQuad(val line: TextLine) {
    val pts: List<Pt>
    val direction: String

    init {
        val (ordered, isV) = sortPnts(line.quad)
        pts = ordered
        direction = if (isV) "v" else "h"
    }

    /** Four edge midpoints p1,p2,p3,p4 (truncated to int coordinates, aligned with m-i-t structure .astype(int)). */
    private val structure: List<Pt> by lazy {
        fun mid(a: Pt, b: Pt) = Pt(((a.x + b.x) / 2f).toInt().toFloat(), ((a.y + b.y) / 2f).toInt().toFloat())
        listOf(mid(pts[0], pts[1]), mid(pts[2], pts[3]), mid(pts[1], pts[2]), mid(pts[3], pts[0]))
    }

    /** [x, y, w, h] (axis-aligned bounding box). */
    val aabb: FloatArray by lazy {
        val mnx = pts.minOf { it.x }; val mny = pts.minOf { it.y }
        floatArrayOf(mnx, mny, pts.maxOf { it.x } - mnx, pts.maxOf { it.y } - mny)
    }

    val fontSize: Float by lazy {
        val v1 = sub(structure[1], structure[0])
        val v2 = sub(structure[3], structure[2])
        min(norm(v2), norm(v1))
    }

    val aspectRatio: Float by lazy {
        val v1 = sub(structure[1], structure[0])
        val v2 = sub(structure[3], structure[2])
        val n1 = norm(v1)
        if (n1 <= 0f) 1f else norm(v2) / n1
    }

    val angle: Float by lazy {
        val v1 = sub(structure[1], structure[0])
        val n1 = norm(v1)
        val cos = if (n1 <= 0f) 1f else (v1.x / n1)
        ((acos(cos.coerceIn(-1f, 1f)) + PI.toFloat()) % PI.toFloat())
    }

    val centroid: Pt by lazy {
        Pt(pts.map { it.x }.average().toFloat(), pts.map { it.y }.average().toFloat())
    }

    val isApproxAxisAligned: Boolean by lazy {
        val v1 = sub(structure[1], structure[0]); val n1 = norm(v1)
        val v2 = sub(structure[3], structure[2]); val n2 = norm(v2)
        if (n1 <= 0f || n2 <= 0f) return@lazy true
        val u1 = Pt(v1.x / n1, v1.y / n1)
        val u2 = Pt(v2.x / n2, v2.y / n2)
        abs(u1.y) < 0.05f || abs(u1.x) < 0.05f || abs(u2.y) < 0.05f || abs(u2.x) < 0.05f
    }

    /** distance_impl (assigned_direction=None -> v mode): pick top/bottom closer corner distance. */
    fun distance(other: GQuad, rho: Float = 0.5f): Float {
        val fs = max(fontSize, other.fontSize)
        val a1 = Geometry.polyArea(Geometry.convexHull(listOf(pts[0], pts[1], other.pts[0], other.pts[1]))) / fs
        val a2 = Geometry.polyArea(Geometry.convexHull(listOf(pts[2], pts[3], other.pts[2], other.pts[3]))) / fs
        var top = true
        if (a1 < fs * rho) top = true
        if (a2 < fs * rho && a2 < a1) top = false
        return if (top) ptDist(pts[0], other.pts[0]) else ptDist(pts[2], other.pts[2])
    }

    private fun sub(a: Pt, b: Pt) = Pt(a.x - b.x, a.y - b.y)
    private fun norm(p: Pt) = hypot(p.x.toDouble(), p.y.toDouble()).toFloat()
    private fun ptDist(a: Pt, b: Pt) = hypot((a.x - b.x).toDouble(), (a.y - b.y).toDouble()).toFloat()
}

/** sort_pnts: determine vertical/horizontal by long edge vector and order 4 points as [top-left, top-right, bottom-right, bottom-left]. Aligned with generic.py:sort_pnts. */
private fun sortPnts(quad: List<Pt>): Pair<List<Pt>, Boolean> {
    val n = quad.size
    val norms = FloatArray(n * n)
    for (i in 0 until n) for (j in 0 until n) {
        val vx = quad[i].x - quad[j].x
        val vy = quad[i].y - quad[j].y
        norms[i * n + j] = hypot(vx.toDouble(), vy.toDouble()).toFloat()
    }
    val order = (0 until n * n).sortedBy { norms[it] }
    val b0 = order[8]; val b1 = order[10]
    var l0x = quad[b0 / n].x - quad[b0 % n].x
    var l0y = quad[b0 / n].y - quad[b0 % n].y
    val l1x = quad[b1 / n].x - quad[b1 % n].x
    val l1y = quad[b1 / n].y - quad[b1 % n].y
    if (l0x * l1x + l0y * l1y < 0f) { l0x = -l0x; l0y = -l0y }
    val sx = abs((l0x + l1x) / 2f)
    val sy = abs((l0y + l1y) / 2f)
    val isV = sx <= sy
    return if (isV) {
        val byY = quad.sortedBy { it.y }
        val first2 = listOf(byY[0], byY[1]).sortedBy { it.x }
        val last2 = listOf(byY[2], byY[3]).sortedBy { it.x }
        Pair(listOf(first2[0], first2[1], last2[1], last2[0]), true)
    } else {
        val byX = quad.sortedBy { it.x }
        val ls = listOf(byX[0], byX[1]).sortedBy { it.y }
        val rs = listOf(byX[2], byX[3]).sortedBy { it.y }
        Pair(listOf(ls[0], rs[0], rs[1], ls[1]), false)
    }
}
