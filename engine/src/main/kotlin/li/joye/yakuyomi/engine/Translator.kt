package li.joye.yakuyomi.engine

/**
 * 翻譯器介面（CLAUDE.md §5 Translator）。
 * translate(queries) → 對應筆數的譯文；失敗的項回傳原文（§11 不變式：永不比原文更糟）。
 */
interface Translator {
    suspend fun translate(queries: List<String>): List<String>
}
