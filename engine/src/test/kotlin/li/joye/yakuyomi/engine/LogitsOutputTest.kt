package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Picking the logits tensor among a session's outputs, with no model and no ONNX Runtime.
 *
 * Getting this wrong is not loud: a wrong-but-existing tensor decodes into plausible text, and a
 * mis-sized vocabulary garbles every character rather than failing, so the choice is pinned here
 * instead of only being exercised on a device.
 */
class LogitsOutputTest {

    /** A real strip decode is [1, T, vocab]; the vocabulary width differs per model. */
    private fun logitsShape(vocab: Int) = intArrayOf(1, 80, vocab)

    @Test
    fun singleLogitsOutputIsUsed() {
        assertEquals(0, LogitsOutput.select(listOf(logitsShape(19264))))
    }

    @Test
    fun ppOcrVocabularyWidthIsAccepted() {
        // Nothing reads the width here, so the two models must both resolve - the point is that
        // no vocabulary constant creeps into the selection rule.
        assertEquals(0, LogitsOutput.select(listOf(logitsShape(18385))))
    }

    @Test
    fun rankThreeIsPreferredOverEarlierOutputs() {
        val shapes = listOf(
            intArrayOf(4),
            intArrayOf(1, 3, 80, 80),
            logitsShape(18385),
        )
        assertEquals(2, LogitsOutput.select(shapes))
    }

    @Test
    fun firstRankThreeWinsWhenSeveralQualify() {
        val shapes = listOf(logitsShape(1), logitsShape(2), logitsShape(3))
        assertEquals(0, LogitsOutput.select(shapes))
    }

    @Test
    fun fallsBackToTheOnlyCandidateWhenNoneIsRankThree() {
        assertEquals(0, LogitsOutput.select(listOf(intArrayOf(1, 2))))
    }

    @Test
    fun nullEntriesAreSkippedWhenFallingBack() {
        val shapes = listOf(null, null, intArrayOf(5))
        assertEquals(2, LogitsOutput.select(shapes))
    }

    @Test
    fun nullEntriesAreSkippedWhenPreferring() {
        val shapes = listOf(null, logitsShape(18385), intArrayOf(9))
        assertEquals(1, LogitsOutput.select(shapes))
    }

    @Test
    fun sessionWithNoOutputsResolvesToNothing() {
        assertNull(LogitsOutput.select(emptyList()))
    }

    @Test
    fun sessionWithOnlyUnusableOutputsResolvesToNothing() {
        assertNull(LogitsOutput.select(listOf(null, null)))
    }

    @Test
    fun rankThreeDetectionIsExact() {
        // 2-D and 4-D neighbours of the real shape must not be mistaken for it.
        assertEquals(1, LogitsOutput.select(listOf(intArrayOf(80, 18385), intArrayOf(1, 80, 18385))))
        assertEquals(1, LogitsOutput.select(listOf(intArrayOf(1, 1, 80, 18385), intArrayOf(1, 80, 18385))))
    }
}
