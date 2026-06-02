import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

// BYOK：build 時從 repo 根 api-keys.properties（gitignored）讀 key 注入 debug BuildConfig。
// 缺檔則為空字串，App 端會略過翻譯。正式版走 Android Keystore + 設定頁，不走這裡。
val apiKeys = Properties().apply {
    val f = rootProject.file("api-keys.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "li.joye.yakuyomi.sandbox"
    compileSdk = 34

    defaultConfig {
        applicationId = "li.joye.yakuyomi.sandbox"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1-m2"
        buildConfigField(
            "String",
            "DEEPSEEK_API_KEY",
            "\"${apiKeys.getProperty("DEEPSEEK_API_KEY", "")}\"",
        )
        // 只打 arm64-v8a（實機）：砍掉 armeabi-v7a/x86/x86_64 的 ORT native libs ≈ 省 55MB，雲端手動安裝快很多
        ndk { abiFilters += "arm64-v8a" }
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
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    // 模型分離（BYOM）：*.onnx 不打進 APK，改由 app 從使用者選的資料夾載入 → APK 變小
    androidResources {
        ignoreAssetsPattern = "*.onnx"
    }

    sourceSets["main"].java.srcDirs("src/main/kotlin")
}

dependencies {
    implementation(project(":engine"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.kotlinx.coroutines.android)
    implementation("androidx.documentfile:documentfile:1.0.1") // SAF 資料夾讀檔
    implementation("androidx.activity:activity-ktx:1.8.2")      // registerForActivityResult / OpenDocumentTree
}
