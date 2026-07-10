# `:engine` — Yakuyomi 翻譯引擎

[English](README.md) ｜ 中文

裝置端漫畫翻譯 library（Android，Kotlin、NCNN + ONNX Runtime）。給它一張頁 bitmap，回一張翻好的頁 bitmap。偵測、OCR、去字在裝置上跑（偵測與去字走 NCNN、OCR 走 ONNX Runtime）；翻譯呼叫雲端 LLM（OpenAI 相容）。

這個 module 跟 reader 無關，只做一件事：`translatePage(bitmap) -> PageResult`。覆蓋檔案、標記、續傳、跨頁批次是呼叫端的事（見[結果處理](#結果處理)）。reader app（[Yakuyomi](https://github.com/joyeli/Yakuyomi) mihon fork）用 Gradle composite build 引入。

Group：`li.joye.yakuyomi:engine`。Min SDK 26。

## 快速開始

```kotlin
// 1. 指向本機模型檔（見模型）。偵測與去字走 NCNN（.param + .bin 成對），
//    OCR 走 ONNX Runtime（.onnx）。最省事是讓引擎從資料夾清單裡挑：
val models = ModelSet.resolve(localModelFiles) ?: return // null = 沒到齊
// 或明確指定各檔（NCNN 角色給 .param 路徑，對應的 .bin 要放在旁邊）：
val models = ModelSet(
    detectorNcnn     = "/path/detector_noblk.ncnn.param",
    ocr              = "/path/ocr_int8.onnx",
    aotInpainterNcnn = "/path/mit_aot_fixed512.ncnn.param",
)

// 2. 載 OCR 字典（在引擎 assets 裡）跟你的 API key。
val alphabet: List<String> = assets.open("models/alphabet-all-v5.txt").bufferedReader().readLines()
val apiKey = "<deepseek key>"   // null/空 = 只偵測+OCR+去字、不翻譯（debug）

// 3. 建引擎、翻譯。use { } 會釋放 native ONNX session。
Yakuyomi.create(models, alphabet, apiKey).use { engine ->
    when (val r = engine.translatePage(pageBitmap)) {
        is PageResult.Translated -> writeBack(r.page)   // 成功：覆蓋 + 標記完成
        is PageResult.Skipped    -> { /* 沒東西可翻：保留原圖 */ }
        is PageResult.Failed     -> { /* 出錯：保留原圖、之後重試 */ }
    }
}
```

`translatePage` 是 `suspend`，從背景 dispatcher 呼叫。一個引擎實例一次處理一頁；不要在同一個實例上並發呼叫 `translatePage`。

## 模型（BYOM）

引擎不帶模型權重。host 提供模型檔加 OCR 字典。偵測與去字走 NCNN（各是 `.param` + `.bin` 一對，兩個都要）；OCR 走 ONNX Runtime：

| 角色 | 檔名（常見） | 後端 | 做什麼 | 來源 |
|---|---|---|---|---|
| detector | `detector_noblk.ncnn.param`（+ `.bin`） | NCNN | 文字框 + 筆畫遮罩 | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| ocr | `ocr_int8.onnx` | ONNX Runtime | 48px CTC 日文 OCR，int8 動態量化 | manga-image-translator |
| inpainter | `mit_aot_fixed512.ncnn.param`（+ `.bin`） | NCNN | AOT-GAN 去字 | [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |

`ModelSet.resolve(files)` 把一份扁平的 `(檔名, 本機路徑)` 清單按名字加副檔名對到各角色（`.param` NCNN：`detect`/`comictext` 對 detector、`aot` 對 inpainter；`.onnx` ORT：`ocr` 對 ocr，外加選配的 `detect`/`comictext` 與 `lama`/`aot` 備援），少了 OCR、偵測、或去字任一顆就回 `null`。拿這個當「能翻了嗎？」的檢查。

路徑必須是本機檔，不能是 SAF/content URI：後端直接從路徑載進 native 記憶體。別用 `readBytes()` 把權重讀進 JVM heap；heap 上限約 512MB（跟裝置 RAM 無關）會 OOM。來源是 SAF 的話，先複製到 `filesDir` 再傳路徑。

## 設定

全是有預設的 `data class`，只覆蓋你要的：

```kotlin
val config = EngineConfig(
    ocr        = OcrConfig(minProb = 0.5f),               // 丟低信心 OCR
    inpainter  = InpainterConfig(method = "auto_aot"),    // "boxfill"（快速）| "auto_aot"（AI）
    render     = RenderConfig(orientation = TextOrientation.AUTO),
    translator = TranslatorConfig(model = "deepseek-chat", batchSize = 8),
)
Yakuyomi.create(models, alphabet, apiKey, config)
```

完整清單、值域、各參數的效果在 [`docs/PARAMETERS_zh.md`](../docs/PARAMETERS_zh.md)。幾個值得知道的預設：

- `OcrConfig.useXnnpack = false`。必須關：XNNPACK 在真機上會把 48px CTC 算錯、OCR 吐空。OCR 是唯一的 ONNX Runtime 模型；偵測器跟去字都跑 NCNN。
- `OcrConfig.concurrent = true`、`concurrency = 8`。OCR 把文字行並發辨識；8 核手機上 OCR 時間大約砍半，輸出不變。
- `InpainterConfig.method = "auto_aot"`、`wholeImage = true`。去字分兩門別：`"boxfill"`（快速去字）把每個字區用就近的背景色平塗——瞬間、平/單色泡泡最乾淨，但壓在畫面上的字會塗成色塊；`"auto_aot"`（AI 去字，預設）把乾淨泡泡平塗、對忙碌區（壓畫面的字）用整頁一次的 AOT-GAN 重建背景（`tileSize = 768`）。
- `RenderConfig.orientation = AUTO`。跟著每區塊偵測到的方向，再沿區塊傾斜角旋轉。
- `TranslatorConfig.provider = "deepseek"`，配 `apiBase` 跟 `model`。任何 OpenAI 相容端點。`LlmProviders.ALL` 內建 manga-image-translator 的 LLM 那組外加 OpenRouter 的預設（全 OpenAI 相容；Gemini 走它的 compat 端點），`LlmModels.list()` 撈服務商的即時模型清單。詳見 [`docs/PROVIDERS_zh.md`](../docs/PROVIDERS_zh.md)。

### 語言對（不寫死日翻繁中）

預設日翻繁中，但任何語言對都行。這幾個一起設：

```kotlin
translator = TranslatorConfig(
    toLangName   = "English",   // 目標：LLM 翻成這個
    fromLangName = "Korean",    // 來源標籤，進 prompt（"" = 讓 LLM 自己判）
    sampleSource = "<|1|>…",    // 來源語言的 few-shot 範例（"" = 不放範例）
    sampleTarget = "<|1|>…",    // 跟它的譯文，用目標語言
)
```

來源是 OCR 模型辨識的東西；內建的 48px CTC 是日文，要讀別的語言就載別的 OCR 模型加字典。目標純粹是 prompt。`toLangName` 跟 few-shot 要同一個目標語言，不然範例會把輸出帶偏。

## 結果處理

`translatePage` 回 `PageResult` 而不是裸 bitmap，讓呼叫端守住核心不變式：絕不用比原圖更糟的東西覆蓋。

| 變體 | 意思 | 呼叫端做什麼 |
|---|---|---|
| `Translated(page, stats)` | 成功 | 覆蓋檔案、寫「已翻譯」標記 |
| `Skipped(reason, stats)` | 沒東西可翻（沒文字／OCR 空／全被過濾） | 保留原圖、標記略過、不重試 |
| `Failed(reason)` | 出錯（網路/429/例外） | 保留原圖、不標記、之後重試 |

逐區韌性內建：某個氣泡翻譯失敗時，引擎改把它的原文畫回去，整頁其餘照翻。`PageStats` 帶各階段計時，加 `wallMs`（實際耗時，比各階段相加短，因為去字跟翻譯請求重疊跑）。

## 生命週期與執行緒

- `TranslationEngine : AutoCloseable`。`close()` 釋放 detector、OCR、inpainter 的 native session（OCR 的 ONNX Runtime session 加偵測器與去字的 NCNN net）。一律 `use { }` 或 `close()`。
- `translatePage` 是 `suspend`，一個實例一次一頁。它內部用 coroutines（並發 OCR、去字跟翻譯重疊），但別在單一實例上並發呼叫。
- 引擎不回收輸入 bitmap。`Translated.page` 是新的 bitmap。

## Pipeline

```
頁  Detector    行 + 筆畫遮罩
    Ocr         每行日文（並發；就地改 lines）
    Grouping    區塊（連邊 + MST 分裂；閱讀序；傾斜角）
    Translator  目標語言（每頁、無滾動上文；可選）
    TextFilter  判定哪些區塊有可用譯文
    Inpainter   抹掉原文（跟 Translator 並發跑）
    Renderer    排版譯文（直/橫排，沿角度）
    -> 翻好的頁 bitmap
```

設計理由跟桌面 parity 驗證見 [`docs/ARCHITECTURE_zh.md`](../docs/ARCHITECTURE_zh.md)。行為對齊 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)。

## 進階：直接用各元件

工廠是建議路徑。要逐階段除錯（例如偵測 overlay）可以自己建各階段、自己組 `Pipeline`，但生命週期得自己管：

```kotlin
val detector = Detector(models.detectorNcnn, config.detector)
val detection = detector.detect(page)   // 行 + textMask，畫你的 overlay
// …
detector.close()                         // 自己建的自己關
```

純 helper（`Geometry`、`ImageOps`、`TextFilter`）是 `internal`，不是公開 API。
