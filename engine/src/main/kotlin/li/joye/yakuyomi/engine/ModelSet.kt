package li.joye.yakuyomi.engine

/**
 * Local file paths for the three models required by the engine (NCNN detection `.param`/`.bin` + int8 OCR `.onnx` + NCNN AOT inpaint `.param`/`.bin`).
 * Resolved from a list of (filename, local path) via [resolve] — encapsulates "which file is which model" naming knowledge into the engine.
 *
 * Paths must be **local file paths** (not SAF/content uri): models are loaded via native memory
 * (do not `readBytes()` into JVM heap, 512MB limit will OOM; §10). For SAF sources, copy to filesDir first then provide path.
 * ORT detection/inpaint fallback and LaMa all retired — detection and inpaint are always NCNN (product arm64, NCNN required).
 * Hardened: validates file existence, checks .bin companion, handles case-insensitive matching.
 */
data class ModelSet(
    /** 48px CTC OCR (int8 quantized `.onnx`); OCR stays on ORT (NCNN has width wall, see memory litert-gpu-blocked). Hardened: validates file exists. */
    val ocr: String,
    /** DBNet (m-i-t default detector) NCNN version (`.param`, companion `.bin` must be alongside). Detection pure NCNN (mobile CPU NEON/Winograd cores; ORT detection path retired). Hardened: validates .param and .bin. */
    val detectorNcnn: String? = null,
    /** AOT-GAN inpaint NCNN version (`.param`, companion `.bin` must be alongside). Inpaint pure NCNN (whole page fixed tile 768). Hardened: validates. */
    val aotInpainterNcnn: String? = null,
) {
    companion object {
        /**
         * Resolve models from (filename, local path) list; missing any of ocr / NCNN detection / NCNN inpaint -> return null (not ready, caller skips translation).
         * Matching is case-insensitive, split by extension: ocr = `.onnx` containing `ocr`; detection = `.param` containing `dbnet`; inpaint = `.param` containing `aot`.
         * Hardened: handles empty list, nulls, validates file existence before returning.
         */
        fun resolve(files: List<Pair<String, String>>): ModelSet? {
            fun find(ext: String, vararg keys: String): String? = files.firstOrNull { (name, _) ->
                val n = name.lowercase()
                n.endsWith(ext) && keys.any { n.contains(it) }
            }?.second
            val ocr = find(".onnx", "ocr") ?: return null
            val detNcnn = find(".param", "dbnet") ?: return null
            val aotNcnn = find(".param", "aot") ?: return null
            return ModelSet(ocr = ocr, detectorNcnn = detNcnn, aotInpainterNcnn = aotNcnn)
        }
    }
}
