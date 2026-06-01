package li.joye.yakuyomi.engine

/**
 * OpenCC s2twp 簡化版（§10 / §12-8 繁中安全網）。
 *
 * 鏈：STPhrases + STCharacters（簡→繁，最長匹配，詞優先於字）→ TWVariants（繁→台灣字變體）。
 * 字典資料來自 OpenCC data/dictionary（Apache-2.0；檔頭授權保留）。
 * ★ 簡化（第三層偏離）：未含 TWPhrases（台灣詞彙如 軟件→軟體）；該層靠 prompt 的「台灣繁體」要求。
 */
class OpenCCS2twp(stTexts: List<String>, twTexts: List<String>) {

    private val st = buildMap(stTexts)
    private val tw = buildMap(twTexts)
    private val stMax = (st.keys.maxOfOrNull { it.length } ?: 1).coerceAtMost(MAX_KEY)
    private val twMax = (tw.keys.maxOfOrNull { it.length } ?: 1).coerceAtMost(MAX_KEY)

    fun convert(text: String): String = applyDict(applyDict(text, st, stMax), tw, twMax)

    private fun buildMap(texts: List<String>): Map<String, String> {
        val m = HashMap<String, String>()
        for (t in texts) {
            for (raw in t.lineSequence()) {
                if (raw.isEmpty() || raw[0] == '#') continue
                val tab = raw.indexOf('\t')
                if (tab <= 0) continue
                val key = raw.substring(0, tab)
                val value = raw.substring(tab + 1).substringBefore(' ').trim()
                if (value.isNotEmpty()) m.putIfAbsent(key, value) // 先載入的（詞）優先
            }
        }
        return m
    }

    /** 最長匹配：每個位置試最長的 key，命中就替換並前進。 */
    private fun applyDict(s: String, m: Map<String, String>, maxLen: Int): String {
        if (m.isEmpty()) return s
        val sb = StringBuilder(s.length)
        var i = 0
        val n = s.length
        while (i < n) {
            var matched = false
            var len = minOf(maxLen, n - i)
            while (len >= 1) {
                val v = m[s.substring(i, i + len)]
                if (v != null) {
                    sb.append(v)
                    i += len
                    matched = true
                    break
                }
                len--
            }
            if (!matched) {
                sb.append(s[i])
                i++
            }
        }
        return sb.toString()
    }

    companion object {
        private const val MAX_KEY = 16
    }
}
