package li.joye.yakuyomi.engine

/**
 * 診斷用輕量 trace hook（**暫時**，用來抓那隻「送翻譯後閃退、無崩潰畫面」的原生/OOM crash）。
 *
 * 引擎不知道 log 檔在哪、也不碰 Android 儲存/權限——只把「階段訊息」丟給 [sink]（由 app 端接去落盤的 trace 檔，
 * 每行 flush、native crash 也留得住）。[sink]=null（預設）＝零開銷、完全不記，正式版不設就等於沒有。
 *
 * 慣例：一段原生呼叫記 `xxx.enter`（進函式）→ `xxx.call`（緊接原生呼叫前）→ `xxx.exit`（回來後）。
 * 若行程死在原生內，最後一行會是 `xxx.call` 而**沒有** `xxx.exit` → 精準定位死在哪個原生呼叫。
 * 死在鎖上（deadlock/久候）則停在 `xxx.enter`（連 `.call` 都沒到）。
 */
object EngineTrace {
    @Volatile
    @JvmStatic
    var sink: ((String) -> Unit)? = null

    fun log(msg: String) {
        sink?.invoke(msg)
    }
}
