package li.joye.yakuyomi.engine

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * 文字行 → 氣泡區分群（翻譯前合併，避免逐行碎裂）。
 *
 * 對齊 manga_translator/utils/generic.py:quadrilateral_can_merge_region（§4 第二層）。
 * 移植其「軸對齊分支」：同方向 + 字級相近 + **緊間距(0.6×字高)** + **對齊**（直書頂或底對齊、橫書左或右對齊）才併。
 * ★ 之前的簡化版只看間距、漏了對齊，會把「靠近但分離」的框錯併（色塊 + 超出框）；補對齊後修正。
 *   仍省略上游的多邊形距離 / 非軸對齊(angle) 分支（複雜分鏡的旋轉框少見），那部分留第三層偏離。
 */
class TextRegion(val lines: List<TextLine>, val direction: String) {
    var translatedText: String = ""

    /** 合併原文（lines 已依閱讀序排好）。日文無空白，直接相接。 */
    val sourceText: String get() = lines.joinToString("") { it.text }

    val x0: Float = lines.minOf { ln -> ln.quad.minOf { it.x } }
    val y0: Float = lines.minOf { ln -> ln.quad.minOf { it.y } }
    val x1: Float = lines.maxOf { ln -> ln.quad.maxOf { it.x } }
    val y1: Float = lines.maxOf { ln -> ln.quad.maxOf { it.y } }

    /** 原文字級估計＝各行短邊平均 ≈ 原漫畫字級。排版以此為字級上限，避免「同框大小字」。 */
    val fontSizeHint: Float = lines.map { ln ->
        minOf(
            ln.quad.maxOf { it.x } - ln.quad.minOf { it.x },
            ln.quad.maxOf { it.y } - ln.quad.minOf { it.y },
        )
    }.average().toFloat()
}

object Grouping {
    private const val GAP = 0.6f       // 緊間距（m-i-t char_gap_tolerance）：bboxGap < GAP×字高 才考慮合併
    private const val ALIGN = 1.5f     // 對齊容差（m-i-t char_gap_tolerance2）：直書頂/底、橫書左/右 對齊 < ALIGN×字高
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
                if (bboxGap(bb[i], bb[j]) >= cs * GAP) continue          // 不夠近 → 不併
                // 對齊（m-i-t 軸對齊分支）：直書要頂或底齊、橫書要左或右齊，否則是「靠近但分離」的框、不併
                val a = bb[i]; val b = bb[j]
                val aligned = if (lines[i].direction == "v") {
                    abs(a[1] - b[1]) < cs * ALIGN || abs(a[3] - b[3]) < cs * ALIGN
                } else {
                    abs(a[0] - b[0]) < cs * ALIGN || abs(a[2] - b[2]) < cs * ALIGN
                }
                if (aligned) parent[find(i)] = find(j)
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
