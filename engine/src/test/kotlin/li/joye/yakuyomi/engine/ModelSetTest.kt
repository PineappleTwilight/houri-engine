package li.joye.yakuyomi.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** ModelSet.resolve：把「哪個檔是哪顆模型」的命名比對收進引擎（純 NCNN 偵測+去字 + int8 OCR）。 */
class ModelSetTest {

    @Test fun resolvesNcnn() {
        val m = ModelSet.resolve(
            listOf(
                "dbnet_detect.ncnn.param" to "/m/det.param",
                "dbnet_detect.ncnn.bin" to "/m/det.bin",
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
        val m = ModelSet.resolve(listOf("DBNET.param" to "d", "OCR.onnx" to "o", "AOT.param" to "a"))!!
        assertEquals("d", m.detectorNcnn)
        assertEquals("o", m.ocr)
        assertEquals("a", m.aotInpainterNcnn)
    }

    @Test fun nullWhenAnyMissing() {
        assertNull(ModelSet.resolve(listOf("ocr.onnx" to "o", "aot.param" to "a")))    // 缺偵測
        assertNull(ModelSet.resolve(listOf("dbnet.param" to "d", "aot.param" to "a"))) // 缺 ocr
        assertNull(ModelSet.resolve(listOf("dbnet.param" to "d", "ocr.onnx" to "o")))  // 缺去字
        assertNull(ModelSet.resolve(emptyList()))
    }

    @Test fun ppocrv5IsPreferredOverTheBundledOcr() {
        // Both are .onnx containing "ocr", so a first-hit match would take whichever the caller
        // happened to list first. YakuyomiEngine.loadAlphabet keys off this same choice, so a
        // reversal would pair the 18385-entry dictionary with the 19264-wide model.
        val m = ModelSet.resolve(
            listOf(
                "dbnet_detect.ncnn.param" to "/m/det.param",
                "aot.param" to "/m/aot.param",
                "ocr_int8.onnx" to "/m/ocr_int8.onnx",
                "ppocrv5_rec.onnx" to "/m/ppocrv5_rec.onnx",
            ),
        )!!
        assertEquals("/m/ppocrv5_rec.onnx", m.ocr)
    }

    @Test fun ppocrv5WinsRegardlessOfListOrder() {
        val m = ModelSet.resolve(
            listOf(
                "ppocrv5_rec.onnx" to "/m/ppocrv5_rec.onnx",
                "dbnet.param" to "/m/det.param",
                "aot.param" to "/m/aot.param",
                "ocr_int8.onnx" to "/m/ocr_int8.onnx",
            ),
        )!!
        assertEquals("/m/ppocrv5_rec.onnx", m.ocr)
    }

    @Test fun bundledOcrStillResolvesOnItsOwn() {
        // The optional model is an upgrade, never a replacement: with only the bundled model
        // present the resolve must keep working, or a failed PP-OCRv5 download would disable OCR.
        val m = ModelSet.resolve(
            listOf(
                "dbnet.param" to "/m/det.param",
                "aot.param" to "/m/aot.param",
                "ocr_int8.onnx" to "/m/ocr_int8.onnx",
            ),
        )!!
        assertEquals("/m/ocr_int8.onnx", m.ocr)
    }

    @Test fun upscalerModelsAreNotMistakenForPipelineRoles() {
        // Real-GUG/anime2real ONNX files live in the same models dir and contain "real" and "2",
        // none of which collide with the three pipeline names - but an upscaler .onnx must never be
        // taken as the OCR model just because it is the only other .onnx present.
        val m = ModelSet.resolve(
            listOf(
                "dbnet.param" to "/m/det.param",
                "aot.param" to "/m/aot.param",
                "Real-CUGAN-x4.onnx" to "/m/cugan.onnx",
            ),
        )
        assertNull(m)
    }

    @Test fun ncnnBinFilesAreNeverTakenAsRoles() {
        // resolve only ever matches .param for the two NCNN roles, so a stray .bin cannot stand in
        // for one - the companion-bin check lives in the caller.
        val m = ModelSet.resolve(
            listOf(
                "dbnet_detect.ncnn.bin" to "/m/det.bin",
                "mit_aot_fixed512.ncnn.bin" to "/m/aot.bin",
                "ocr_int8.onnx" to "/m/ocr",
            ),
        )
        assertNull(m)
    }
}
