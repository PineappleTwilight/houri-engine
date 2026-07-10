// Yakuyomi 引擎的 NCNN 推論橋（P1 去字 + P2 偵測）。handle-based：模型 createNet 載一次、
// pipeline 每頁復用同一 ncnn::Net（不像 sandbox benchmark 每次重載）。OCR 留 ORT（NCNN 寬度牆）。
#include <jni.h>
#include <android/log.h>
#include <cstring>
#include "net.h"
#include "gpu.h"

#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "yakuyomi_ncnn", __VA_ARGS__)

static bool g_gpu_inited = false;

// 建立/查 GPU：回 GPU 數（0=無、-1=建立失敗）。Vulkan 需要時才呼叫。
static int ensure_gpu() {
    if (!g_gpu_inited) {
        if (ncnn::create_gpu_instance() != 0) return -1;
        g_gpu_inited = true;
    }
    return ncnn::get_gpu_count();
}

extern "C" JNIEXPORT jint JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_gpuCount(JNIEnv*, jobject) {
    return ensure_gpu();
}

// 載入 pnnx 轉出的 .param/.bin，回 native handle（ncnn::Net*）；0=失敗。要 Vulkan 但無 GPU → 退 CPU。
extern "C" JNIEXPORT jlong JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_createNet(
        JNIEnv* env, jobject, jstring paramPath, jstring binPath, jboolean useVulkan) {
    bool vk = (bool) useVulkan;
    if (vk && ensure_gpu() <= 0) vk = false;

    ncnn::Net* net = new ncnn::Net();
    net->opt.use_vulkan_compute = vk;

    const char* pp = env->GetStringUTFChars(paramPath, nullptr);
    const char* bp = env->GetStringUTFChars(binPath, nullptr);
    int r1 = net->load_param(pp);
    int r2 = net->load_model(bp);
    env->ReleaseStringUTFChars(paramPath, pp);
    env->ReleaseStringUTFChars(binPath, bp);
    if (r1 != 0 || r2 != 0) {
        LOGW("ncnn load fail param=%d model=%d", r1, r2);
        delete net;
        return 0;
    }
    return (jlong) net;
}

extern "C" JNIEXPORT void JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_releaseNet(JNIEnv*, jobject, jlong handle) {
    if (handle) delete (ncnn::Net*) handle;
}

// 偵測：in0[3,s,s] → out0(det[2,s,s]) + out1(seg[1,s,s])。填 detOut[2*s*s]+segOut[s*s]。0=OK。
extern "C" JNIEXPORT jint JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_detect(
        JNIEnv* env, jobject, jlong handle,
        jfloatArray chw, jint s, jfloatArray detOut, jfloatArray segOut) {
    if (!handle) return -1;
    ncnn::Net* net = (ncnn::Net*) handle;

    jfloat* in = env->GetFloatArrayElements(chw, nullptr);
    ncnn::Mat inMat(s, s, 3);
    for (int c = 0; c < 3; c++) {
        memcpy(inMat.channel(c), in + (size_t) c * s * s, sizeof(float) * s * s);
    }
    env->ReleaseFloatArrayElements(chw, in, JNI_ABORT);

    ncnn::Mat det, seg;
    ncnn::Extractor ex = net->create_extractor();
    ex.input("in0", inMat);
    ex.extract("out0", det);
    ex.extract("out1", seg);

    jfloat* od = env->GetFloatArrayElements(detOut, nullptr);
    for (int c = 0; c < 2; c++) {
        memcpy(od + (size_t) c * s * s, det.channel(c), sizeof(float) * s * s);
    }
    env->ReleaseFloatArrayElements(detOut, od, 0);

    jfloat* os = env->GetFloatArrayElements(segOut, nullptr);
    memcpy(os, seg.channel(0), sizeof(float) * s * s);
    env->ReleaseFloatArrayElements(segOut, os, 0);
    return 0;
}

// 去字 AOT：in0=img[3,s,s]（[-1,1] holes-zeroed）+ in1=mask[1,s,s] → out0[3,s,s]（[-1,1]）。填 outArr[3*s*s]。
extern "C" JNIEXPORT jint JNICALL
Java_li_joye_yakuyomi_engine_NcnnBackend_inpaintAot(
        JNIEnv* env, jobject, jlong handle,
        jfloatArray img, jfloatArray mask, jint s, jfloatArray outArr) {
    if (!handle) return -1;
    ncnn::Net* net = (ncnn::Net*) handle;

    jfloat* pi = env->GetFloatArrayElements(img, nullptr);
    jfloat* pm = env->GetFloatArrayElements(mask, nullptr);
    ncnn::Mat imgMat(s, s, 3);
    for (int c = 0; c < 3; c++) {
        memcpy(imgMat.channel(c), pi + (size_t) c * s * s, sizeof(float) * s * s);
    }
    ncnn::Mat maskMat(s, s, 1);
    memcpy(maskMat.channel(0), pm, sizeof(float) * s * s);
    env->ReleaseFloatArrayElements(img, pi, JNI_ABORT);
    env->ReleaseFloatArrayElements(mask, pm, JNI_ABORT);

    ncnn::Mat out;
    ncnn::Extractor ex = net->create_extractor();
    ex.input("in0", imgMat);
    ex.input("in1", maskMat);
    ex.extract("out0", out);

    jfloat* o = env->GetFloatArrayElements(outArr, nullptr);
    for (int c = 0; c < 3; c++) {
        memcpy(o + (size_t) c * s * s, out.channel(c), sizeof(float) * s * s);
    }
    env->ReleaseFloatArrayElements(outArr, o, 0);
    return 0;
}
