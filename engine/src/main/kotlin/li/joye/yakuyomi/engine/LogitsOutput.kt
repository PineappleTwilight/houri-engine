package li.joye.yakuyomi.engine

/**
 * Chooses which of a session's outputs carries the CTC logits.
 *
 * The rule is separated from [Ocr.resolveLogits] so it can be tested on a plain JVM: building an
 * `OrtSession` needs ONNX Runtime's native library, which the Android artifact only ships for
 * device ABIs. This mirrors how [CtcDecode] is split out of the decode path.
 *
 * Output names are deliberately not part of the input. The bundled 48px model calls this tensor
 * `char_logits` while PP-OCRv5's ONNX conversion calls it `fetch_name_0`, so any name-based match
 * is a guess about the exporter, and ORT 1.20 exposes no name accessor to guess with.
 */
internal object LogitsOutput {

    /**
     * Index of the output to read, or null when the session produced nothing usable.
     *
     * @param shapes one entry per session output: the tensor's shape, or null when that output is
     *   not a tensor with a known shape. Null entries are neither preferred nor used as a fallback.
     */
    fun select(shapes: List<IntArray?>): Int? {
        shapes.forEachIndexed { index, shape -> if (shape != null && shape.size == 3) return index }
        return shapes.indexOfFirst { it != null }.takeIf { it >= 0 }
    }
}
