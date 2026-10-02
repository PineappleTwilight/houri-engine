package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp

/**
 * Greedy CTC decoding against a synthetic dictionary, no model required.
 *
 * A dictionary that does not match the model's vocabulary does not fail - it produces wrong text
 * that looks plausible - which is why the index-to-character mapping is pinned here rather than
 * only exercised on a device.
 */
class CtcDecodeTest {

    /** PP-OCRv5 layout: index 0 blank, 1 the space PaddleOCR inserts, then dictionary order. */
    private val paddleDict = listOf("<blank>", "<SP>", "日", "本", "語", "A")

    /** [path] gives the winning class per timestep; 0 is the CTC blank. */
    private fun logits(t: Int, d: Int, path: IntArray): FloatArray {
        val a = FloatArray(t * d) { -5f }
        for (ti in path.indices) a[ti * d + path[ti]] = 5f
        return a
    }

    private fun decode(t: Int, d: Int, dict: List<String>, path: IntArray): String =
        CtcDecode.greedy(logits(t, d, path), t, d, dict).first

    @Test
    fun decodesByIndex() {
        assertEquals("日本語", decode(3, 6, paddleDict, intArrayOf(2, 3, 4)))
    }

    @Test
    fun blankIsNeverEmitted() {
        val (text, prob) = CtcDecode.greedy(logits(4, 6, IntArray(4)), 4, 6, paddleDict)
        assertEquals("", text)
        assertEquals(0f, prob, 0f)
    }

    @Test
    fun repeatsCollapse() {
        assertEquals("日", decode(4, 6, paddleDict, intArrayOf(2, 2, 2, 2)))
    }

    @Test
    fun blankSeparatesRepeats() {
        assertEquals("日日", decode(3, 6, paddleDict, intArrayOf(2, 0, 2)))
    }

    @Test
    fun spaceTokenBecomesSpace() {
        // PaddleOCR keeps its space at index 1, one past the blank. This is the assertion that the
        // offset is honoured: dropping the inserted space shifts every character by one instead.
        assertEquals("日 本", decode(3, 6, paddleDict, intArrayOf(2, 1, 3)))
    }

    @Test
    fun idBeyondDictionaryIsSkipped() {
        // d=8 with a six-entry dictionary: class 7 has no character and is dropped with a warning,
        // while the surrounding in-range classes still decode.
        val warnings = mutableListOf<String>()
        val (text, _) = CtcDecode.greedy(logits(3, 8, intArrayOf(2, 7, 0)), 3, 8, paddleDict) { warnings += it }
        assertEquals("日", text)
        assertTrue(warnings.any { it.contains("out of dict bounds") })
    }

    @Test
    fun dictionarySizeMismatchWarns() {
        val warnings = mutableListOf<String>()
        CtcDecode.greedy(logits(2, 6, intArrayOf(0, 0)), 2, 6, listOf("a", "b")) { warnings += it }
        assertTrue(warnings.any { it.contains("dict size mismatch") })
    }

    @Test
    fun emptyInputsAreSafe() {
        assertEquals("" to 0f, CtcDecode.greedy(FloatArray(0), 0, 0, paddleDict))
        assertEquals("" to 0f, CtcDecode.greedy(FloatArray(0), 2, 6, paddleDict))
        assertEquals("" to 0f, CtcDecode.greedy(FloatArray(12), 2, 6, emptyList()))
    }

    @Test
    fun probabilityIsHighForAPeakedStrip() {
        val (text, prob) = CtcDecode.greedy(logits(2, 6, intArrayOf(2, 3)), 2, 6, paddleDict)
        assertEquals("日本", text)
        assertTrue("expected a confident read, got $prob", prob > 0.9f)
        assertTrue(prob <= 1f)
    }

    @Test
    fun probabilityDropsForAnUncertainStrip() {
        val (_, flat) = CtcDecode.greedy(FloatArray(12), 2, 6, paddleDict)
        val (_, peaked) = CtcDecode.greedy(logits(2, 6, intArrayOf(2, 3)), 2, 6, paddleDict)
        assertTrue("flat $flat should be below peaked $peaked", flat < peaked)
    }

    @Test
    fun softmaxReferenceMatchesTheMeanLogProb() {
        // Independent recomputation: mean over the two emitted timesteps of
        // -log(sum(exp(logit - max))). Both timesteps emit a character here, so the divisor is 2.
        val d = 6
        val arr = FloatArray(2 * d) { -5f }
        arr[0 * d + 2] = 3f
        arr[1 * d + 3] = 3f
        var total = 0.0
        for (ti in 0 until 2) {
            val base = ti * d
            val max = (0 until d).maxOf { arr[base + it] }
            var sum = 0.0
            for (c in 0 until d) sum += exp((arr[base + c] - max).toDouble())
            total += -Math.log(sum)
        }
        val (text, prob) = CtcDecode.greedy(arr, 2, d, paddleDict)
        assertEquals("日本", text)
        assertEquals(exp(total / 2).toFloat(), prob, 1e-5f)
    }
}
