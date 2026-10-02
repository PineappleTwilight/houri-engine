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
     * Rank alone does not identify the logits. The bundled model also emits a `color` head as
     * `[N, T, 6]`, which is rank 3 just like the `[N, T, 19264]` logits, so "first rank-3 wins"
     * resolves correctly only while the graph happens to declare the logits first - and picking the
     * color head instead would decode a 6-wide vocabulary into text that looks plausible. Width is
     * the discriminator: a CTC vocabulary runs to tens of thousands of classes, an auxiliary head to
     * single digits.
     *
     * @param shapes one entry per session output: the tensor's shape, or null when that output is
     *   not a tensor with a known shape. Null entries are neither preferred nor used as a fallback.
     */
    fun select(shapes: List<IntArray?>): Int? {
        var best = -1
        var widest = -1
        shapes.forEachIndexed { index, shape ->
            if (shape != null && shape.size == 3 && shape[2] > widest) {
                best = index
                widest = shape[2]
            }
        }
        if (best >= 0) return best
        return shapes.indexOfFirst { it != null }.takeIf { it >= 0 }
    }
}
