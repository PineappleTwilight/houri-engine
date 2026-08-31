plugins {
    alias(libs.plugins.android.library)
    // AGP 9+ 內建 Kotlin 支援，不再套 kotlin.android（見 kotl.in/gradle/agp-built-in-kotlin）
}

// Yakuyomi fork 以 Gradle composite build（includeBuild）接此引擎，靠 group:name 替換依賴
group = "li.joye.yakuyomi"
version = "0.2.0"

android {
    namespace = "li.joye.yakuyomi.engine"
    compileSdk = 37
    ndkVersion = "28.2.13676358" // NCNN 原生層（Detector/Inpainter）；釘住版本讓 CI/fork submodule 建置一致

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")

        // NCNN 原生後端（P1 去字 + P2 偵測）：arm64-v8a 用客製 20260718 預編庫、
        // armeabi-v7a 用官方 20260526 android-vulkan 預編庫（皆 Vulkan+OpenMP+threads）。
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += listOf("-DANDROID_STL=c++_static")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // 90MB 的 .onnx 不要壓縮：ORT 直接 mmap，省記憶體也較快
    androidResources {
        noCompress += "onnx"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// AGP 9 內建 Kotlin：jvmTarget 改在 kotlin{} 設（取代已移除的 android.kotlinOptions）
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(libs.onnxruntime.android)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.okhttp)

    testImplementation("junit:junit:4.13.2")
}
