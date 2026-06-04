package li.joye.yakuyomi.engine

import ai.onnxruntime.OrtSession
import android.util.Log

/**
 * 套用執行供應器（EP）到 [OrtSession.SessionOptions]，回傳實際生效的 EP 名（"XNNPACK"/"CPU"）。
 *
 * **無 adb/Logcat**：回傳值由呼叫端寫進 sandbox 可讀 log/圖，才知道每顆模型到底跑在哪個 EP（XNNPACK 退回 CPU 與否一翻兩瞪眼）。
 *
 * XNNPACK 失敗 → 退回預設 CPU EP。
 * （曾試 QNN/Hexagon NPU + fp16，實測對本專案 float 模型不加速、op 回退 CPU＋多 ~7s 編譯＝更慢，已移除；
 *  真要上 NPU 需 int8 QDQ 量化，留待日後——見 memory qnn-fp16-no-npu-accel。）
 */
internal fun OrtSession.SessionOptions.applyEp(threads: Int, tag: String): String {
    setIntraOpNumThreads(threads)
    return try {
        addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
        "XNNPACK"
    } catch (t: Throwable) {
        Log.w(tag, "XNNPACK 不可用，退回 CPU：${t.message}")
        "CPU"
    }
}
