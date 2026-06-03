package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Test

/** 翻譯後過濾鏈（§11：別把空白/數字/未譯的東西蓋回去）。 */
class TextFilterTest {

    private fun region(jp: String, cht: String): TextRegion {
        val quad = listOf(Pt(0f, 0f), Pt(10f, 0f), Pt(10f, 10f), Pt(0f, 10f))
        val line = TextLine(quad, 1f).apply { text = jp }
        return TextRegion(listOf(line), "h").apply { translatedText = cht }
    }

    @Test fun keepsNormalTranslation() {
        assertEquals(1, TextFilter.apply(listOf(region("こんにちは", "你好"))).size)
    }

    @Test fun dropsBlankDigitAndUntranslated() {
        val kept = TextFilter.apply(
            listOf(
                region("あ", ""),         // 空白
                region("123", "123"),     // 純數字（也＝原文）
                region("hello", "hello"), // 譯＝原（忽略大小寫）
                region("ねこ", "貓"),       // 留
            ),
        )
        assertEquals(1, kept.size)
        assertEquals("貓", kept[0].translatedText)
    }

    @Test fun dropsRegexMatch() {
        val kept = TextFilter.apply(
            listOf(region("x", "廣告請洽"), region("y", "正常對白")),
            filterText = "廣告",
        )
        assertEquals(1, kept.size)
        assertEquals("正常對白", kept[0].translatedText)
    }
}
