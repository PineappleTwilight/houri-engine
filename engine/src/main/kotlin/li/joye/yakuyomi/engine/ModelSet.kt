package li.joye.yakuyomi.engine

/**
 * 引擎要的三顆模型的本機檔案路徑（NCNN 偵測 `.param`/`.bin` + int8 OCR `.onnx` + NCNN AOT 去字 `.param`/`.bin`）。
 * 用 [resolve] 從一堆 (檔名, 本機路徑) 比對出來——把「哪個檔是哪顆模型」的命名知識收進引擎。
 *
 * 路徑必須是**本機檔案路徑**（非 SAF/content uri）：模型走 native 記憶體載入
 * （勿 `readBytes()` 進 JVM heap，512MB 上限會 OOM；§10）。SAF 來源請先複製到 filesDir 再給路徑。
 * ORT 偵測/去字備援與 LaMa 皆已退役移除——偵測與去字一律 NCNN（產品 arm64、NCNN 必在）。
 */
data class ModelSet(
    /** 48px CTC OCR（int8 量化 `.onnx`）；OCR 留 ORT（NCNN 有寬度牆，見 memory litert-gpu-blocked）。 */
    val ocr: String,
    /** comic-text-detector 的 NCNN 版（`.param`，同名 `.bin` 需在旁）。偵測純 NCNN（手機 CPU 比 ORT-XNNPACK 快 ~2.9×）。 */
    val detectorNcnn: String? = null,
    /** AOT-GAN 去字的 NCNN 版（`.param`，同名 `.bin` 需在旁）。去字純 NCNN（整頁固定 tile 768）。 */
    val aotInpainterNcnn: String? = null,
) {
    companion object {
        /**
         * 從 (檔名, 本機路徑) 清單比對出模型；缺 ocr / NCNN 偵測 / NCNN 去字任一 → 回 null（未備齊，呼叫端略過翻譯）。
         * 比對不分大小寫、依副檔名分流：ocr＝`.onnx` 含 `ocr`；偵測＝`.param` 含 `detect`/`comictext`；去字＝`.param` 含 `aot`。
         */
        fun resolve(files: List<Pair<String, String>>): ModelSet? {
            fun find(ext: String, vararg keys: String): String? = files.firstOrNull { (name, _) ->
                val n = name.lowercase()
                n.endsWith(ext) && keys.any { n.contains(it) }
            }?.second
            val ocr = find(".onnx", "ocr") ?: return null
            val detNcnn = find(".param", "detect", "comictext") ?: return null
            val aotNcnn = find(".param", "aot") ?: return null
            return ModelSet(ocr = ocr, detectorNcnn = detNcnn, aotInpainterNcnn = aotNcnn)
        }
    }
}
