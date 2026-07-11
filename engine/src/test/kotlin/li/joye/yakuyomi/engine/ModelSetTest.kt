package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** ModelSet.resolve：把「哪個檔是哪顆模型」的命名比對收進引擎（純 NCNN 偵測+去字 + int8 OCR）。 */
class ModelSetTest {

    @Test fun resolvesNcnn() {
        val m = ModelSet.resolve(
            listOf(
                "detector_noblk.ncnn.param" to "/m/det.param",
                "detector_noblk.ncnn.bin" to "/m/det.bin",
                "ocr_int8.onnx" to "/m/ocr",
                "mit_aot_fixed512.ncnn.param" to "/m/aot.param",
                "mit_aot_fixed512.ncnn.bin" to "/m/aot.bin",
            ),
        )!!
        assertEquals("/m/det.param", m.detectorNcnn)
        assertEquals("/m/ocr", m.ocr)
        assertEquals("/m/aot.param", m.aotInpainterNcnn)
    }

    @Test fun caseInsensitive() {
        val m = ModelSet.resolve(listOf("DETECT.param" to "d", "OCR.onnx" to "o", "AOT.param" to "a"))!!
        assertEquals("d", m.detectorNcnn)
        assertEquals("o", m.ocr)
        assertEquals("a", m.aotInpainterNcnn)
    }

    @Test fun nullWhenAnyMissing() {
        assertNull(ModelSet.resolve(listOf("ocr.onnx" to "o", "aot.param" to "a")))    // 缺偵測
        assertNull(ModelSet.resolve(listOf("detect.param" to "d", "aot.param" to "a"))) // 缺 ocr
        assertNull(ModelSet.resolve(listOf("detect.param" to "d", "ocr.onnx" to "o")))  // 缺去字
        assertNull(ModelSet.resolve(emptyList()))
    }
}
