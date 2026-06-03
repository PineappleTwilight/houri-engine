package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** ModelSet.resolve：把「哪個檔是哪顆模型」的命名比對收進引擎。 */
class ModelSetTest {

    @Test fun resolvesByName() {
        val m = ModelSet.resolve(
            listOf(
                "comictextdetector.pt.onnx" to "/m/det",
                "ocr_48px_ctc.onnx" to "/m/ocr",
                "lama-manga.onnx" to "/m/lama",
            ),
        )!!
        assertEquals("/m/det", m.detector)
        assertEquals("/m/ocr", m.ocr)
        assertEquals("/m/lama", m.inpainter)
    }

    @Test fun caseInsensitive() {
        val m = ModelSet.resolve(listOf("DETECT.onnx" to "d", "OCR.onnx" to "o", "LaMa.onnx" to "l"))!!
        assertEquals("d", m.detector)
        assertEquals("o", m.ocr)
        assertEquals("l", m.inpainter)
    }

    @Test fun nullWhenAnyMissing() {
        assertNull(ModelSet.resolve(listOf("ocr.onnx" to "o", "lama.onnx" to "l")))        // 缺 detector
        assertNull(ModelSet.resolve(listOf("detect.onnx" to "d", "lama.onnx" to "l")))     // 缺 ocr
        assertNull(ModelSet.resolve(emptyList()))
    }
}
