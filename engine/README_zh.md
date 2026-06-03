# `:engine` — Yakuyomi 翻譯引擎

[English](README.md) · **中文**

裝置端漫畫翻譯**函式庫**（Android、純 Kotlin + ONNX Runtime）。
丟一張頁面 bitmap 進去、拿一張翻好的頁面 bitmap 出來。偵測／OCR／去字在裝置上跑，翻譯走雲端 LLM（OpenAI 相容）。

這個模組與 reader 無關：它只做 `translatePage(bitmap) -> PageResult`。覆蓋檔案、marker、resume、跨頁批次
都是**呼叫端**的責任（見 [結果處理](#結果處理)）。產品 reader（[Yakuyomi](https://github.com/joyeli/Yakuyomi)，
mihon fork）透過 Gradle composite build 引入本引擎。

> Group/version：`li.joye.yakuyomi:engine`。Min SDK 26。

---

## 快速上手

```kotlin
// 1. 指向本機上的三顆模型檔（見「模型」）。
val models = ModelSet(
    detector   = "/path/comictextdetector.pt.onnx",
    ocr        = "/path/ocr_48px_ctc.onnx",
    inpainter  = "/path/lama-manga.onnx",
)
// …或讓引擎從一堆檔案裡自己挑：
val models = ModelSet.resolve(localOnnxFiles /* List<Pair<檔名, 路徑>> */) ?: return // 模型沒備齊

// 2. 載入 OCR 字元表（隨引擎 assets 附帶）與你的 API key。
val alphabet: List<String> = assets.open("models/alphabet-all-v5.txt").bufferedReader().readLines()
val apiKey = "<deepseek key>"   // null/空白 = 只跑偵測+OCR+去字、不翻譯（除錯用）

// 3. 建立引擎並翻譯。`use { }` 會釋放原生 ONNX session。
Yakuyomi.create(models, alphabet, apiKey).use { engine ->
    when (val r = engine.translatePage(pageBitmap)) {
        is PageResult.Translated -> writeBack(r.page)        // 成功 → 覆蓋 + 寫 marker
        is PageResult.Skipped    -> { /* 沒東西可翻 → 保留原圖 */ }
        is PageResult.Failed     -> { /* 出錯 → 保留原圖、之後重試 */ }
    }
}
```

`translatePage` 是 `suspend`，請在背景 dispatcher 呼叫。單一引擎實例**非並發安全**；每個實例一次翻一頁。

---

## 模型（BYOM）

引擎**不附帶任何模型權重**；由 host 提供三顆 ONNX + OCR 字元表：

| 角色 | 檔案（常見命名） | 做什麼 | 來源 |
|------|------------------|--------|------|
| detector | `comictextdetector.pt.onnx` | 文字框 + 逐像素筆畫遮罩 | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| ocr | `ocr_48px_ctc.onnx` | 48px CTC 日文 OCR | manga-image-translator |
| inpainter | `lama-manga.onnx` | LaMa 去字 | [Koharu](https://github.com/mayocream/koharu) |

`ModelSet.resolve(files)` 把一份扁平的 `(檔名, 本機路徑)` 清單依命名比對成三個角色
（`detect`/`comictext` → detector、`ocr` → ocr、`lama`/`inpaint` → inpainter），缺任一顆回 `null`——
拿這個當「能不能翻」的判斷即可。

**路徑必須是本機檔案**、非 SAF/content URI：ORT 用 `createSession(path)` 走 native 記憶體。
別把權重 `readBytes()` 進 JVM heap——heap 上限約 512 MB（與裝置實體 RAM 無關），會 OOM。
來源是 SAF 的話，先複製到 `filesDir` 再給路徑。

---

## 設定

全部是有合理預設的 `data class`——只覆寫你要動的：

```kotlin
val config = EngineConfig(
    ocr     = OcrConfig(minProb = 0.5f),                 // 丟低信心 OCR
    inpainter = InpainterConfig(method = "boxfill"),     // "boxfill" | "lama"
    render  = RenderConfig(orientation = TextOrientation.AUTO),
    translator = TranslatorConfig(model = "deepseek-chat", batchSize = 8),
)
Yakuyomi.create(models, alphabet, apiKey, config)
```

幾個重要預設（完整清單見 `Config.kt`）：

- `OcrConfig.useXnnpack = false`——**必須關著**：XNNPACK 在真機上會把 48px CTC 模型算錯（OCR 吐空）。
  detector/inpainter 用 XNNPACK 沒問題。
- `InpainterConfig.method = "boxfill"`——對筆畫遮罩做「就近取色」填補；最快、品質追平最貴的 LaMa 模式。
  `"lama"` + `wholeImage` 切換其他模式。
- `RenderConfig.orientation = AUTO`——跟著每個區域偵測到的方向（直/橫排），再沿區域傾斜角旋轉以貼合原文。
- `TranslatorConfig.provider = "deepseek"`——任何 OpenAI 相容端點（設 `apiBase`/`model`）。

### 語言對（不寫死日→繁中）

預設是日本語 → 繁體中文，但**任意語言對都行**——這幾個一起設：

```kotlin
translator = TranslatorConfig(
    toLangName   = "English",     // 目標——LLM 直接翻成這個
    fromLangName = "Korean",      // 來源語言標註（""＝讓 LLM 自己判）
    sampleSource = "<|1|>…",      // 來源語言的 few-shot 範例（""＝不放範例）
    sampleTarget = "<|1|>…",      // …以及它的譯文（要跟目標同語言）
)
```

**來源**最終取決於你載的 OCR 模型——內建 48px CTC 是日文；要讀別的語言就載別的 OCR 模型 + 字元表（BYOM）。
**目標**純粹是 LLM prompt。`toLangName` 跟 few-shot 要同一個目標語言，否則範例會把輸出帶偏。

---

## 結果處理

`translatePage` 回 `PageResult`（而非裸 bitmap），讓呼叫端能守住核心不變式：**永不用比原圖更糟的東西覆蓋**。

| 變體 | 意義 | 呼叫端該做什麼 |
|------|------|----------------|
| `Translated(page, stats)` | 成功 | 覆蓋檔案、寫「已翻譯」marker |
| `Skipped(reason, stats)`  | 沒東西可翻（無文字／OCR 全空／全被過濾） | 保留原圖、標記略過、不重試 |
| `Failed(reason)`          | 出錯（網路/429/例外） | 保留原圖、**不**寫 marker、之後重試 |

逐區韌性內建：某個氣泡翻譯失敗就保留它的原文日文、整頁其餘照常算繪。`PageStats` 帶各階段計時供 profiling。

---

## 生命週期與執行緒

- `TranslationEngine : AutoCloseable`。`close()` 釋放 detector/ocr/inpainter 的 ONNX session（native 記憶體）。
  一律用 `use { }` 或 `close()`。
- `translatePage` 是 `suspend`、一次一頁、每實例**非並發安全**。
- 引擎**不會** recycle 輸入 bitmap；`Translated.page` 是**另一個新** bitmap。

---

## Pipeline（內部發生什麼）

```
page ─ Detector ─→ 文字行 + 筆畫遮罩
     ─ Ocr ──────→ 每行日文            （就地填回 lines）
     ─ Grouping ─→ 區域（兩階段：連邊 + MST 分裂；閱讀序；傾斜角）
     ─ Translator → 繁體中文            （逐頁、無滾動上文；可選）
     ─ TextFilter → 丟空白/數字/regex/未譯的區
     ─ Inpainter ─→ 抹掉原文
     ─ Renderer ──→ 排版譯文（直/橫排、依角度旋轉）
     → 翻好的頁面 bitmap
```

設計理念與桌面 parity 驗證見 [`docs/ARCHITECTURE_zh.md`](../docs/ARCHITECTURE_zh.md)。
行為對齊 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)。

---

## 進階：直接用底層元件

工廠是建議路徑。要做逐階段除錯（例如偵測 overlay）可以自己 new 各元件、自組 `Pipeline`——
但這樣**你**就要自己管生命週期：

```kotlin
val detector = Detector(models.detector, config.detector)
val detection = detector.detect(page)   // lines + textMask，畫你的 overlay
// …
detector.close()                        // 自己建的就自己 close
```

純 helper（`Geometry`、`ImageOps`、`TextFilter`）是 `internal`——不屬公開 API。
