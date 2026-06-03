package li.joye.yakuyomi.engine

/**
 * 引擎要的三顆模型的本機檔案路徑。用 [resolve] 從一堆 (檔名, 本機路徑) 比對出來——
 * 把「哪個檔是哪顆模型」的命名知識收進引擎，取代各消費端各自寫的檔名比對。
 *
 * 路徑必須是**本機檔案路徑**（非 SAF/content uri）：ORT 直接 `createSession(path)` 走 native 記憶體
 * （勿 `readBytes()` 進 JVM heap，512MB 上限會 OOM；§10）。SAF 來源請先複製到 filesDir 再給路徑。
 */
data class ModelSet(
    /** comic-text-detector：文字偵測 + seg 筆畫遮罩。 */
    val detector: String,
    /** 48px CTC OCR。 */
    val ocr: String,
    /** LaMa 去字（Koharu lama-manga）。 */
    val inpainter: String,
) {
    companion object {
        /**
         * 從 (檔名, 本機路徑) 清單比對出三顆模型；**缺任一顆回 null**（＝模型未備齊，呼叫端應略過翻譯）。
         * 比對規則對齊命名慣例：detector＝含 `detect`/`comictext`、ocr＝含 `ocr`、inpainter＝含 `lama`/`inpaint`（不分大小寫）。
         */
        fun resolve(files: List<Pair<String, String>>): ModelSet? {
            fun find(vararg keys: String): String? = files.firstOrNull { (name, _) ->
                val n = name.lowercase()
                keys.any { n.contains(it) }
            }?.second
            val detector = find("detect", "comictext") ?: return null
            val ocr = find("ocr") ?: return null
            val inpainter = find("lama", "inpaint") ?: return null
            return ModelSet(detector, ocr, inpainter)
        }
    }
}
