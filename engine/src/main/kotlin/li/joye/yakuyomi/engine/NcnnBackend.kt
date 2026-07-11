package li.joye.yakuyomi.engine

/**
 * NCNN 推論後端（P1 去字 + P2 偵測）。OCR 留在 ORT（NCNN 把 transformer 相對位置編碼寫死在轉換寬度、native 寬會壞）。
 *
 * handle-based：`createNet` 載入 pnnx 轉出的 .param/.bin 回一個 native handle（ncnn::Net*），pipeline 每頁復用、
 * 收工 `releaseNet`。模型換到手機 CPU 的 NEON/Winograd → 偵測實測比 ORT-XNNPACK 快 ~3.7×（見 memory litert-gpu-blocked）。
 */
internal object NcnnBackend {
    /** 原生庫是否載得起來（缺 .so / 非 arm64 → false，呼叫端可退回 ORT）。 */
    val available: Boolean = try {
        System.loadLibrary("yakuyomi_ncnn")
        true
    } catch (t: Throwable) {
        false
    }

    /** 載入 .param/.bin，回 native handle（純 CPU；NEON/Winograd）；0=失敗。 */
    external fun createNet(paramPath: String, binPath: String): Long

    external fun releaseNet(handle: Long)

    /** 偵測：chw=NCHW[1,3,s,s] → det 填 [2*s*s]（ch0=det, ch1=blk 邊界）、seg 填 [s*s]。回 0=OK。 */
    external fun detect(handle: Long, chw: FloatArray, s: Int, det: FloatArray, seg: FloatArray): Int

    /** 去字 AOT：img=NCHW[3,s,s]（[-1,1] holes-zeroed）+ mask=[s*s] → out 填 [3*s*s]（[-1,1]）。回 0=OK。 */
    external fun inpaintAot(handle: Long, img: FloatArray, mask: FloatArray, s: Int, out: FloatArray): Int
}
