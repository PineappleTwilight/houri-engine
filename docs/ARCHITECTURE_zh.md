# 架構

Yakuyomi 怎麼翻譯一頁漫畫、為什麼這樣切、桌面與裝置兩半怎麼搭。
這是引擎 API 文件（[`engine/README_zh.md`](../engine/README_zh.md)）的可出貨架構說明。

[English](ARCHITECTURE.md) · **中文**

---

## 兩半

| | **裝置端**（`engine/`，Kotlin） | **桌面**（`parity/`，Python） |
|---|---|---|
| 角色 | 真正的產品——在手機上跑 | 驗證 harness——在筆電上跑 |
| 語言 | Kotlin + ONNX Runtime + Android Canvas | Python + numpy/cv2 + onnxruntime + PIL |
| 出貨？ | 是（函式庫） | 否（開發用） |
| 目的 | 高效翻頁 | 證明 Kotlin 移植行為跟參考一致 |

引擎是交付物。parity harness 存在的原因：我們把
[manga-image-translator](https://github.com/zyddnys/manga-image-translator)（m-i-t，Python/torch）
重寫成 Kotlin/ONNX——這種移植沒辦法逐行 diff。桌面 harness 用 Python 跑**同樣**的階段，
讓我們在信任 Kotlin 版之前先檢查「同輸入 → 近輸出」。見 [`parity/README_zh.md`](../parity/README_zh.md)。

---

## 一頁的資料流

```
page bitmap
  │
  ├─ Detector (ONNX)   → 文字行（旋轉四邊形）+ 逐像素筆畫遮罩
  │
  ├─ Ocr (ONNX)        → 每行日文（48px CTC、greedy 解碼）
  │
  ├─ Grouping (Kotlin) → 區域（兩階段：寬鬆連邊，再 MST 分裂過度合併的鄰居；閱讀序；傾斜角）
  │
  ├─ Translator (LLM)  → 繁體中文，逐頁、無滾動上文（可選）
  │
  ├─ TextFilter        → 丟空白／純數字／regex 命中／未譯的區
  │
  ├─ Inpainter (ONNX)  → 抹掉原文（boxfill 就近取色，或 LaMa）
  │
  └─ Renderer (Canvas) → 排版譯文（直/橫排、依區域角度旋轉）
  →
翻好的頁面 bitmap
```

並發發生在**翻譯**這步：一章 N 頁各送一個請求、用 `Semaphore` 限上限。其餘都是逐頁、在裝置上 CPU-bound。

各階段一句話：

- **Detector**——comic-text-detector。letterbox 前處理 → ONNX → 框（NMS + unclip）與 `seg` 筆畫遮罩
  （去字時抹細筆畫、不抹成方塊）。
- **Ocr**——48px CTC。裁切每行（透視校正、直書行轉正），辨識、對字元表 greedy 解碼；低於 `minProb` 丟掉。
- **Grouping**——把行併成氣泡大小的區域。兩階段：寬鬆的*連邊*，再用 MST *分裂*只靠傳遞性連起來的鄰居
  （這步就是阻止密集氣泡黏成一塊的關鍵）。同時算每區的閱讀序與傾斜角。
- **Translator**——把 m-i-t 的 `chatgpt.py` prompt + 協定移植成 OpenAI 相容呼叫。逐頁、無跨頁上文
  （效率優先）。逐區失敗就 fallback 回原文。
  **語言對可設定**——目標走 `toLangName`、來源走 OCR 模型 + prompt 標註（預設日→繁中，非寫死）。
- **TextFilter**——m-i-t 的翻譯後過濾：不要把空白、純數字、LLM 原樣回傳的文字蓋回去。
- **Inpainter**——預設 `boxfill`（每遮罩像素取最近非遮罩色）；LaMa（整頁 tile 或逐區 window）為替代。
- **Renderer**——純文字框排版（不做氣泡 flood-fill）：字級自適應、行頭禁則、直/橫排、依背景亮度自動選字色、
  並沿區域傾斜角旋轉畫布。

orchestration 與「永不用更糟的東西覆蓋」不變式都在 `Pipeline.kt`。

---

## 與 manga-image-translator 對齊

我們對齊的是**行為與決策**、不是原始碼。三層，由緊到鬆：

1. **照搬**——跨語言不變的純資料：翻譯 prompt + 協定、各階段閾值/預設、config schema、模型選擇與處理順序。
   省下重推 m-i-t 的調校。
2. **對齊行為、自由實作**——*做什麼*跟 m-i-t、*怎麼寫*隨 Kotlin/ONNX 慣例：偵測後處理、座標反算、遮罩生成、
   分組兩階段、閱讀序、並發翻譯。判準＝同輸入近輸出。
3. **知情偏離（要留紀錄）**——平台逼的或刻意的取捨：ONNX 量化、丟掉 CUDA、無滾動翻譯上文（效率）、
   boxfill 優先去字。每處在程式碼偏離點註記。

被迫改寫（其餘照搬）：torch → ORT `session.run`；cv2/PIL → Bitmap / Canvas / 手刻；numpy → Kotlin；
async httpx/CLI/YAML → OkHttp + coroutines + 自己的設定載入；manga-ocr 自回歸 decode；任何 CUDA/GPU 假設。

---

## 真機實證（在實體硬體上學到的）

- **XNNPACK 會把 OCR 模型算錯** → `OcrConfig.useXnnpack = false`（純 CPU）。detector 與 inpainter 用 XNNPACK 沒問題。
- **模型走 off-heap。** `createSession(path)` 進 native 記憶體——絕不 `readBytes()` 進 JVM heap
  （上限約 512 MB、與裝置 RAM 無關 → OOM）。BYOM 先把 SAF 複製到 `filesDir`。
- **前處理必須跟 Python 匯出逐位元一致**——resize / normalize / NCHW。這是頭號隱形偏離來源；
  parity harness 大半就是為了抓這個。
- **記憶體衛生**——`OnnxTensor` 用完 close；一次只處理一頁。

---

## Repo 結構

```
engine/        Android 函式庫——裝置端 pipeline（產品本體）
  src/main/kotlin/li/joye/yakuyomi/engine/
    Yakuyomi.kt, TranslationEngine.kt, ModelSet.kt   ← 公開入口（facade + 型別）
    Pipeline.kt, Detector.kt, Ocr.kt, Grouping.kt,
    LlmTranslator.kt, Inpainter.kt, Renderer.kt      ← 各階段
    Geometry.kt, ImageOps.kt, TextFilter.kt          ← internal helper
  src/test/kotlin/…                                  ← JVM 單元測試
app-sandbox/   丟棄式測試 app（debug overlay、真機計時）
parity/        桌面 Python 驗證 harness（不出貨）
docs/          本檔
```

產品 reader（[Yakuyomi](https://github.com/joyeli/Yakuyomi)，mihon fork）以 git submodule + Gradle
composite build 引入 `engine`，只加整合層（下載 hook、設定、模型管理）。
