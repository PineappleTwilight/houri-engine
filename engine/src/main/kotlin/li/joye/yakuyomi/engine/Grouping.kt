package li.joye.yakuyomi.engine

import kotlin.math.sqrt

/**
 * 文字行 → 氣泡區分群（翻譯前合併，避免逐行碎裂）。
 *
 * 對齊 manga_translator/utils/generic.py:quadrilateral_can_merge_region 的精神（§4 第二層）。
 * ★ 簡化（第三層偏離）：用「同方向 + 字級相近 + bbox 間距 < 字高」判合併，
 *   省略上游的多邊形距離 / angle / aspect / 對齊性細節。完整版日後再補。
 */
class TextRegion(val lines: List<TextLine>, val direction: String) {
    var translatedText: String = ""

    /** 合併原文（lines 已依閱讀序排好）。日文無空白，直接相接。 */
    val sourceText: String get() = lines.joinToString("") { it.text }

    val x0: Float = lines.minOf { ln -> ln.quad.minOf { it.x } }
    val y0: Float = lines.minOf { ln -> ln.quad.minOf { it.y } }
    val x1: Float = lines.maxOf { ln -> ln.quad.maxOf { it.x } }
    val y1: Float = lines.maxOf { ln -> ln.quad.maxOf { it.y } }
}

object Grouping {
    private const val GAP = 1.0f       // bbox 間距 < GAP×字高 → 合併
    private const val FS_RATIO = 1.5f  // 字級比上限

    fun group(lines: List<TextLine>): List<TextRegion> {
        val n = lines.size
        if (n == 0) return emptyList()

        val bb = lines.map { l ->
            floatArrayOf(
                l.quad.minOf { it.x }, l.quad.minOf { it.y },
                l.quad.maxOf { it.x }, l.quad.maxOf { it.y },
            )
        }
        val fs = bb.map { minOf(it[2] - it[0], it[3] - it[1]) }

        val parent = IntArray(n) { it }
        fun find(x: Int): Int {
            var r = x
            while (parent[r] != r) r = parent[r]
            var c = x
            while (parent[c] != c) { val nx = parent[c]; parent[c] = r; c = nx }
            return r
        }

        for (i in 0 until n) {
            for (j in i + 1 until n) {
                if (lines[i].direction != lines[j].direction) continue
                val cs = minOf(fs[i], fs[j])
                if (cs <= 0f || maxOf(fs[i], fs[j]) / cs > FS_RATIO) continue
                if (bboxGap(bb[i], bb[j]) < cs * GAP) parent[find(i)] = find(j)
            }
        }

        val groups = LinkedHashMap<Int, MutableList<Int>>()
        for (i in 0 until n) groups.getOrPut(find(i)) { mutableListOf() }.add(i)

        return groups.values.map { members ->
            val dir = lines[members[0]].direction
            val ordered = if (dir == "v") {
                members.sortedByDescending { bb[it][2] }          // 直書：右→左
            } else {
                members.sortedWith(compareBy({ bb[it][1] }, { bb[it][0] })) // 橫書：上→下、左→右
            }
            TextRegion(ordered.map { lines[it] }, dir)
        }
    }

    private fun bboxGap(a: FloatArray, b: FloatArray): Float {
        val dx = maxOf(0f, maxOf(a[0], b[0]) - minOf(a[2], b[2]))
        val dy = maxOf(0f, maxOf(a[1], b[1]) - minOf(a[3], b[3]))
        return sqrt(dx * dx + dy * dy)
    }
}
