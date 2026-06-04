package li.joye.yakuyomi.engine

import ai.onnxruntime.OrtSession
import android.util.Log

/**
 * 套用執行供應器（EP）到 [OrtSession.SessionOptions]，回傳實際生效的 EP 名（"QNN"/"XNNPACK"/"CPU"）。
 *
 * **無 adb/Logcat**：回傳值由呼叫端寫進 sandbox 可讀 log，才知道每顆模型到底跑在哪個 EP（QNN 退回與否一翻兩瞪眼）。
 *
 * - [useQnn]=true：QNN HTP（Hexagon NPU）。fp32 模型自動以 fp16 推論（enable_htp_fp16_precision 預設＝1，此處明寫）、burst 效能。
 *   QNN 不支援的節點 ORT 自動切回**預設 CPU EP**（非 XNNPACK，故無 OCR 那種 XNNPACK 誤算風險）。失敗（無 -qnn 原生庫/裝置不支援）→ 退回 XNNPACK→CPU。
 * - [useQnn]=false：XNNPACK（CPU，現狀），失敗 → CPU。
 *
 * 需 onnxruntime-android-qnn AAR 才有 QNN 原生 EP；[OrtSession.SessionOptions.addQnn] 本身在標準 ORT Java API 即可編譯。
 */
internal fun OrtSession.SessionOptions.applyEp(useQnn: Boolean, threads: Int, tag: String): String {
    setIntraOpNumThreads(threads)
    var qnnErr: String? = null
    if (useQnn) {
        try {
            addQnn(
                mapOf(
                    "backend_path" to "libQnnHtp.so",   // Hexagon HTP（NPU）後端庫（由 qnn-runtime 打包進 APK）
                    "htp_performance_mode" to "burst",
                    "enable_htp_fp16_precision" to "1", // fp32 → fp16 在 NPU 跑
                ),
            )
            Log.i(tag, "QNN HTP(fp16) 已啟用")
            return "QNN"
        } catch (t: Throwable) {
            qnnErr = "${t.javaClass.simpleName}:${t.message?.take(80)}" // 無 adb → 把失敗原因帶進 ep 字串給可讀 log
            Log.w(tag, "QNN 不可用，退回 XNNPACK/CPU：${t.message}")
        }
    }
    val base = try {
        addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
        "XNNPACK"
    } catch (t: Throwable) {
        Log.w(tag, "XNNPACK 不可用，退回 CPU：${t.message}")
        "CPU"
    }
    return if (qnnErr != null) "$base←QNN失敗($qnnErr)" else base
}
