package li.joye.yakuyomi.engine

/**
 * Greedy CTC decode, split out of [Ocr] so it is testable without a model.
 *
 * The index-to-character mapping is the one part of OCR that fails silently: a dictionary that does
 * not match the model's vocabulary produces plausible-looking wrong text rather than an error. It
 * was an instance method, which made it untestable, because constructing [Ocr] opens an ONNX
 * session. It is a pure function of the logits and the dictionary, so it lives here instead.
 *
 * [onWarn] exists so the caller can log while the decoder itself stays free of `android.util.Log`,
 * which would otherwise make this untestable off-device.
 */
internal object CtcDecode {

    private const val BLANK = 0
    private const val SPACE_TOKEN = "<SP>"

    /**
     * Decodes [arr] laid out as [t] timesteps of [d] classes.
     *
     * Class 0 is the CTC blank and is never emitted. A run of the same class collapses to one
     * character; a run split by a blank re-emits. The returned probability is the mean per-character
     * log-softmax, so it is comparable across lines of the same model.
     */
    fun greedy(
        arr: FloatArray,
        t: Int,
        d: Int,
        dictionary: List<String>,
        onWarn: (String) -> Unit = {},
    ): Pair<String, Float> {
        if (dictionary.isEmpty() || t <= 0 || d <= 0) return "" to 0f
        if (arr.size < t * d) return "" to 0f
        if (d != dictionary.size) onWarn("CTC dict size mismatch: logits d=$d vs dict ${dictionary.size}")
        val dictSize = dictionary.size
        val sb = StringBuilder()
        var last = BLANK
        var logpSum = 0.0
        var nChars = 0
        for (ti in 0 until t) {
            val base = ti * d
            if (base + d > arr.size) break
            var best = 0
            var bestV = arr[base]
            for (c in 1 until d) {
                val v = arr[base + c]
                if (v > bestV) {
                    bestV = v
                    best = c
                }
            }
            if (best != last && best != BLANK) {
                if (best >= dictSize) {
                    onWarn("CTC best id $best out of dict bounds $dictSize, skipping")
                } else {
                    val ch = dictionary[best]
                    sb.append(if (ch == SPACE_TOKEN) " " else ch)
                    var s = 0.0
                    for (c in 0 until d) s += Math.exp((arr[base + c] - bestV).toDouble())
                    logpSum += -Math.log(s)
                    nChars++
                }
            }
            last = best
        }
        val prob = if (nChars > 0) Math.exp(logpSum / nChars).toFloat().coerceIn(0f, 1f) else 0f
        return sb.toString() to prob
    }
}
