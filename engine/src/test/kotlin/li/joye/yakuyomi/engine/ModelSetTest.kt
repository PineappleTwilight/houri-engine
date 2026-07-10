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

    @Test fun resolvesModelsV2Ncnn() {
        // models-v2：NCNN 偵測(.param+.bin) + int8 OCR(.onnx) + NCNN AOT 去字(.param+.bin)、無 ORT 偵測/LaMa。
        val m = ModelSet.resolve(
            listOf(
                "detector_noblk.ncnn.param" to "/m/det.param",
                "detector_noblk.ncnn.bin" to "/m/det.bin",
                "ocr_int8.onnx" to "/m/ocr",
                "mit_aot_fixed512.ncnn.param" to "/m/aot.param",
                "mit_aot_fixed512.ncnn.bin" to "/m/aot.bin",
            ),
        )!! // 不可為 null：純 NCNN 去字也要算「去字模型已備齊」
        assertEquals("/m/det.param", m.detectorNcnn)
        assertEquals("/m/ocr", m.ocr)
        assertEquals("/m/aot.param", m.aotInpainterNcnn)
        assertNull(m.detector)   // 無 ORT 偵測
        assertNull(m.inpainter)  // 無 LaMa
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
