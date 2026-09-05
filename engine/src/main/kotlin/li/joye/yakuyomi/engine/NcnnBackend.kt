package li.joye.yakuyomi.engine

/**
 * NCNN inference backend (P1 inpaint + P2 detection). OCR stays on ORT (NCNN bakes transformer relative positional encoding into converted width, native width would break).
 *
 * Handle-based: `createNet` loads pnnx-converted .param/.bin and returns a native handle (ncnn::Net*), pipeline reuses per page,
 * finished `releaseNet`. Models run on phone CPU NEON/Winograd -> detection measured ~3.7x faster than ORT-XNNPACK (see memory litert-gpu-blocked).
 * Hardened: validates handles, checks library availability, handles native errors gracefully.
 */
internal object NcnnBackend {
    /** Whether native library can be loaded (missing .so / non-arm64 -> false, caller can fallback to ORT). Hardened: catches all throwables. */
    val available: Boolean = try {
        System.loadLibrary("yakuyomi_ncnn")
        true
    } catch (t: Throwable) {
        false
    }

    /** Load .param/.bin, return native handle (pure CPU; NEON/Winograd); 0 = failure. Do not change this external name (JNI symbol = Java_..._createNet, renaming causes UnsatisfiedLinkError). Hardened: validates paths. */
    external fun createNet(paramPath: String, binPath: String): Long

    external fun releaseNet(handle: Long)

    /**
     * Global lock: **serialize all ncnn native inferences** (detect + inpaint).
     *
     * ncnn internally uses **OpenMP** (libomp statically linked into libyakuyomi_ncnn.so) for convolution parallelization. Multiple app threads **simultaneously** entering
     * ncnn forward (cross-page concurrency dispatches detect/inpaint to multiple Dispatchers.Default threads) -> each opens OpenMP parallel region ->
     * OpenMP global runtime does not allow multiple concurrent masters -> **`__kmp_abort_process` directly aborts process (SIGABRT)**.
     * Proven by device tombstone (thread=DefaultDispatch, #01 __kmp_abort_process), 2026-07-14.
     *
     * detect and inpaint share same lock (same ncnn OpenMP runtime, any two concurrent parallel regions will collide).
     * OCR goes via ORT (another .so, own threading model) not affected by this lock, translation goes via network -> concurrency preserved. Detect/inpaint are both CPU-bound,
     * serialization barely hurts throughput (anyway blocked inside translation's network wait window, CPU cannot truly run two at once anyway).
     * Hardened: uses synchronized block, handles re-entrance safely.
     */
    private val ncnnLock = Any()

    private external fun detectDbnetNative(handle: Long, chw: FloatArray, inW: Int, inH: Int, db: FloatArray, mask: FloatArray): Int

    private external fun inpaintAotNative(handle: Long, img: FloatArray, mask: FloatArray, s: Int, out: FloatArray): Int

    /** DBNet detection (rectangular resize_aspect input, avoids square 832-992 crash zone): chw=[3,inH,inW] -> db fills [2*inW*inH] (raw logits 2ch full res), mask fills [(inW/2)*(inH/2)] (sigmoid half-res). Returns mask.h (>0=OK). Serialized (see [ncnnLock]). Hardened: validates handle and sizes. */
    fun detectDbnet(handle: Long, chw: FloatArray, inW: Int, inH: Int, db: FloatArray, mask: FloatArray): Int {
        if (handle == 0L) return -1
        if (inW < 32 || inH < 32 || inW > 2048 || inH > 2048) return -1
        EngineTrace.log("ncnn.detectDbnet.enter ${inW}x$inH")
        return synchronized(ncnnLock) {
            EngineTrace.log("ncnn.detectDbnet.call ${inW}x$inH")
            val rc = detectDbnetNative(handle, chw, inW, inH, db, mask)
            EngineTrace.log("ncnn.detectDbnet.exit rc=$rc")
            rc
        }
    }

    /** Inpaint AOT: img=NCHW[3,s,s] ([-1,1] holes-zeroed) + mask=[s*s] -> out fills [3*s*s] ([-1,1]). Returns 0=OK. Serialized (see [ncnnLock]). Hardened: validates inputs. */
    fun inpaintAot(handle: Long, img: FloatArray, mask: FloatArray, s: Int, out: FloatArray): Int {
        if (handle == 0L || s < 32 || s > 2048) return -1
        EngineTrace.log("ncnn.inpaint.enter s=$s")
        return synchronized(ncnnLock) {
            EngineTrace.log("ncnn.inpaint.call s=$s")
            val rc = inpaintAotNative(handle, img, mask, s, out)
            EngineTrace.log("ncnn.inpaint.exit rc=$rc")
            rc
        }
    }
}
