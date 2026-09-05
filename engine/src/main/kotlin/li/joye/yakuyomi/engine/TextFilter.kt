package li.joye.yakuyomi.engine

// Ported from manga_translator/manga_translator.py (post-translation filter chain, ~L1323-1352) @ d5a3eee (first layer direct port)

/**
 * Post-translation filtering (before inpaint/typeset): discard "not worth overwriting" translated regions, keep original (section 11 invariant).
 *
 * Aligned with m-i-t `_run_text_translation` should_filter chain:
 *   1. Blank translation                -> discard
 *   2. Pure numeric translation        -> discard
 *   3. [filterText] regex match        -> discard (user-provided, default null = disabled)
 *   4. Translation == source (ignore case/trim) -> discard (LLM missed translation or returned as-is)
 *
 * Discarded regions do not go to [Inpainter]/[Renderer], original Japanese image kept as is — better than "inpaint then re-paste Japanese".
 * Only applied when "actually translated"; debug path without key (typeset Japanese) is not filtered.
 * Internal: belongs to pipeline internal step, does not cross library boundary.
 * Hardened: handles null/empty, overly long text, invalid regex gracefully.
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
