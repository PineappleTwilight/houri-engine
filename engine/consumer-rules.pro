# ONNX Runtime：保留 JNI 綁定，否則 R8 縮小後 runtime crash（CLAUDE.md §13 提醒）
-keep class ai.onnxruntime.** { *; }

# 本引擎自身的 JNI（NcnnBackend，static symbol Java_li_joye_yakuyomi_engine_*）：
# 類別名與 external 方法名必須原樣保留，否則 R8 縮小後 UnsatisfiedLinkError。
-keep class li.joye.yakuyomi.engine.NcnnBackend { *; }
-keepclasseswithmembernames class li.joye.yakuyomi.engine.** {
    native <methods>;
}
