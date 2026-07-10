package li.joye.yakuyomi.engine

/**
 * 引擎要的三顆模型的本機檔案路徑。用 [resolve] 從一堆 (檔名, 本機路徑) 比對出來——
 * 把「哪個檔是哪顆模型」的命名知識收進引擎，取代各消費端各自寫的檔名比對。
 *
 * 路徑必須是**本機檔案路徑**（非 SAF/content uri）：ORT 直接 `createSession(path)` 走 native 記憶體
 * （勿 `readBytes()` 進 JVM heap，512MB 上限會 OOM；§10）。SAF 來源請先複製到 filesDir 再給路徑。
 */
data class ModelSet(
    /** comic-text-detector 的 ORT `.onnx`；**選配·備援**——偵測以 NCNN 為主（[detectorNcnn]），此顆僅在 NCNN 不可用時退回（非-arm64 等）。 */
    val detector: String? = null,
    /** 48px CTC OCR（只有 ORT；NCNN 有寬度牆，見 memory litert-gpu-blocked）。 */
    val ocr: String,
    /** LaMa 去字（Koharu lama-manga，ORT）；**已退役、選配**——去字改以 AOT 為主，僅 `method="lama"/"auto"` 才需要（見 [Yakuyomi.create]）。 */
    val inpainter: String? = null,
    /** AOT-GAN 去字（m-i-t inpainting.ckpt 匯出，ORT）；**現行主去字**（`method="aot"/"auto_aot"` 用；逐格 native 或 NCNN 關時走這顆）。 */
    val aotInpainter: String? = null,
    /** comic-text-detector 的 NCNN 版（`.param`，pnnx 轉）；**選配**，有則偵測優先走 NCNN（手機 CPU 比 ORT-XNNPACK 快 ~3.7×）。 */
    val detectorNcnn: String? = null,
    /** AOT-GAN 去字的 NCNN 版（`.param`）；**選配**，有且**整頁模式**才走 NCNN（逐格變動尺寸會讓 ncnn net 崩，見 [Yakuyomi.create]）。 */
    val aotInpainterNcnn: String? = null,
) {
    companion object {
        /**
         * 從 (檔名, 本機路徑) 清單比對出模型；**ocr 缺、無偵測模型、或無去字模型時回 null**（＝未備齊，呼叫端略過翻譯）。
         * 比對規則對齊命名慣例（不分大小寫、依副檔名分流）：
         *   ORT＝`.onnx`：detector(選配備援) 含 `detect`/`comictext`、ocr 含 `ocr`、inpainter(LaMa,退役選配) 含 `lama`、aotInpainter 含 `aot`；
         *   NCNN＝`.param`：detectorNcnn(主) 含 `detect`/`comictext`、aotInpainterNcnn(主) 含 `aot`。
         *   偵測 NCNN 優先、ORT 備援（至少一顆）；去字 aot 優先、lama 退役備援（至少一顆）。
         */
        fun resolve(files: List<Pair<String, String>>): ModelSet? {
            fun find(ext: String, vararg keys: String): String? = files.firstOrNull { (name, _) ->
                val n = name.lowercase()
                n.endsWith(ext) && keys.any { n.contains(it) }
            }?.second
            val ocr = find(".onnx", "ocr") ?: return null
            val detOnnx = find(".onnx", "detect", "comictext")
            val detNcnn = find(".param", "detect", "comictext")
            if (detOnnx == null && detNcnn == null) return null // 至少要一顆偵測模型
            val lama = find(".onnx", "lama")
            val aot = find(".onnx", "aot")
            val aotNcnn = find(".param", "aot")
            if (lama == null && aot == null && aotNcnn == null) return null // 至少要一顆去字模型（含 NCNN AOT）
            return ModelSet(
                detector = detOnnx,
                ocr = ocr,
                inpainter = lama,
                aotInpainter = aot,
                detectorNcnn = detNcnn,
                aotInpainterNcnn = aotNcnn,
            )
        }
    }
}
