# ONNX Runtime：保留 JNI 綁定，否則 R8 縮小後 runtime crash（CLAUDE.md §13 提醒）
-keep class ai.onnxruntime.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
