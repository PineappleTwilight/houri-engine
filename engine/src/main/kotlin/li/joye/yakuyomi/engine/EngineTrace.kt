package li.joye.yakuyomi.engine

/**
 * Engine diagnostic trace hook (controlled by app-side diagnostic logging switch; default off = zero overhead).
 *
 * Catches crashes that logcat and built-in crash logs miss: native SIGSEGV/abort, OOM killed by lowmemorykiller
 * SIGKILL — all kill process instantly without crash screen. Engine does not know log file location nor touches Android storage/permissions — only delivers "stage messages"
 * to [sink] (app side writes to trace file, flush per line, survives native crash).
 * **[sink]=null (default) = zero overhead, completely disabled** — when app-side diagnostic switch is off, sink is not set, equivalent to this whole system not existing.
 *
 * Convention: a native call logs `xxx.enter` (enter function) -> `xxx.call` (right before native call) -> `xxx.exit` (after return).
 * If process dies inside native, last line will be `xxx.call` **without** `xxx.exit` -> precisely locate which native call died.
 * Death on lock (deadlock/long wait) stops at `xxx.enter` (not even reaching `.call`).
 * Hardened: volatile sink, null-safe, no allocation when disabled.
 */
object EngineTrace {
    @Volatile
    @JvmStatic
    var sink: ((String) -> Unit)? = null

    fun log(msg: String) {
        sink?.invoke(msg)
    }
}
