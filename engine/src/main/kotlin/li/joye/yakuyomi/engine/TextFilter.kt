package li.joye.yakuyomi.engine

// ported from manga_translator/manga_translator.py（翻譯後過濾鏈，~L1323-1352）@ d5a3eee（第一層照搬）

/**
 * 翻譯後過濾（在去字/排版之前）：丟掉「不值得蓋上去」的譯文區，保留原圖（§11 不變式）。
 *
 * 對齊 m-i-t `_run_text_translation` 的 should_filter 鏈：
 *   1. 空白譯文                → 丟
 *   2. 純數字譯文              → 丟
 *   3. [filterText] regex 命中 → 丟（使用者提供，預設 null＝不啟用）
 *   4. 譯文 == 原文（忽略大小寫/前後空白）→ 丟（LLM 漏譯或原樣回傳）
 *
 * 被丟的區不進 [Inpainter]/[Renderer]，原始日文畫面原樣保留——比「去字後蓋回日文」更好。
 * 僅在「實際有翻譯」時套用；無 key 的 debug 路徑（排版日文）不過濾。
 * internal：屬 pipeline 內部步驟，不跨出 library 邊界。
 */
internal object TextFilter {

    /** 回傳「保留」的區（過濾掉 should-filter 的）。 */
    fun apply(regions: List<TextRegion>, filterText: String? = null): List<TextRegion> {
        val re = filterText?.let { runCatching { Regex(it) }.getOrNull() }
        return regions.filter { !shouldFilter(it, re) }
    }

    private fun shouldFilter(region: TextRegion, re: Regex?): Boolean {
        val t = region.translatedText.trim()
        if (t.isEmpty()) return true                                  // 1 空白
        if (t.all { it.isDigit() }) return true                       // 2 純數字
        if (re != null && re.containsMatchIn(t)) return true          // 3 regex
        if (region.sourceText.trim().equals(t, ignoreCase = true)) return true  // 4 譯==原
        return false
    }
}
