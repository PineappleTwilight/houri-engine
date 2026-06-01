<div align="center">

# Yakuyomi（訳読み）

**漫畫 AI 翻譯 Reader — 裝置端偵測 / OCR / 去字，雲端 LLM 翻譯**

日本語 → 繁體中文（CHT）

> ⚠️ **狀態：規劃中 / 起步階段（M0）** — 尚未有可用版本，介面與細節都可能變動。

</div>

---

## 這是什麼

Yakuyomi 是一個加上 AI 翻譯能力的漫畫閱讀器，fork 自 [yokai](https://github.com/null2264/yokai)。

它把翻譯流程拆成兩半：

- **文字偵測、OCR、去字** → 在**裝置上**離線執行（ONNX Runtime）。
- **翻譯** → 交給**雲端 LLM**（預設 DeepSeek，OpenAI 相容）。

整條 pipeline 在**行為與決策上對齊** [manga-image-translator](https://github.com/zyddnys/manga-image-translator)（以下簡稱 m-i-t）——對齊的是它的判斷與參數，不是逐行複製它的 Python。

最高目標是**效率 / 吞吐**：一章多頁的翻譯並發送出，盡量壓低端到端時間。

## 為什麼是「半離線」

偵測 / OCR / 去字都在裝置上跑，所以這幾步離線可用；唯獨翻譯那一步走雲端 LLM，因此**翻譯當下需要連網**。這是刻意的取捨——換來其餘階段離線、引擎可獨立發佈，代價是接受翻譯半離線。

## 核心特性

- 🔍 **裝置端偵測**：comic-text-detector，輸出文字區塊 bbox（去字遮罩留待後期）。
- 🔠 **裝置端 OCR**：以 48px CTC 模型為主、manga-ocr 為品質備案。
- 🌐 **雲端 LLM 翻譯**：移植 m-i-t 的 prompt 與協定（行數對齊、漏行重試、詞彙表掛勾）。
- 🧹 **去字 / 重排**：先用白塊蓋字頂著（M2），之後換 LaMa inpaint + 氣泡 typesetting（M3）。
- 🇹🇼 **繁體中文輸出**：prompt 強制台灣繁體用語，再過一道 OpenCC `s2twp` 當安全網。
- 🔑 **BYOK（自帶金鑰）**：provider / model / API base / 金鑰 / 目標語言全可設定，**不內建任何 key**；金鑰存在 Android Keystore。
- 💾 **就地覆蓋、可續傳**：僅在翻譯成功時覆蓋原頁、以頁為單位記錄完成狀態；失敗則保留原圖，**永不用比原圖更糟的東西覆蓋下載庫**。

## 一頁的資料流

```
頁 Bitmap
 → 偵測 (ONNX)   → blocks：bbox (+ mask)，依 m-i-t 的 text_region 啟發式排序
 → OCR  (ONNX)   → 每塊整塊辨識 → 日文 sourceText
 → 翻譯 (LLM)    → 逐頁、並發、無滾動上文 → targetText（依 block ID 貼回）
 → 渲染 (Canvas) → M2：白塊蓋字 ／ M3：LaMa inpaint + 氣泡排版
 → 翻好的 Bitmap
```

並發發生在「翻譯」這一格：一章 N 頁各開一個 coroutine 同時送出，用 `Semaphore` 限制同時在飛的請求數（避免 provider 429）。

## 技術選型

| 項目 | 選擇 | 備註 |
|---|---|---|
| Base fork | **yokai** | 整合點在下載層、不動 reader |
| 推論引擎 | **純 Kotlin + ONNX Runtime**（`onnxruntime-android`，含 XNNPACK） | NNAPI 已棄用，不採用 |
| 加速路線 | XNNPACK/CPU → int8 量化 → 需要時再 QNN / LiteRT | — |
| 偵測 | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) | 已有 ONNX |
| OCR | 48px CTC（主）／ [manga-ocr](https://github.com/kha-white/manga-ocr)（備案） | CTC 單次前向、好搬 |
| 去字 | [LaMa](https://github.com/advimman/lama)（漫畫微調版，ONNX） | M3 才上 |
| 翻譯 | 雲端 LLM，移植 m-i-t `chatgpt.py` 的 prompt + 協定 | OpenAI 相容 |
| 預設 provider | **DeepSeek** | 全可改（BYOK） |

ONNX 匯出與 pipeline 細節參考 [Koharu](https://github.com/mayocream/koharu)（Rust + ONNX）。

## 翻譯 Provider（BYOK）

v1 先支援 **OpenAI 相容**那一組（一個 HTTP client 通吃）：

- `openai`、`deepseek`、`groq`、`custom_openai`（OpenRouter / LM Studio / 自架）
- 之後：`gemini`，再來才是非 LLM 的雲端 MT（DeepL / Caiyun / Youdao / Baidu / Papago）當獨立 adapter。

設定頁會暴露：provider、model、API base（custom_openai 用）、API key（每個 provider 一格，存 Keystore）、目標語言。預設 DeepSeek，但全可改。

## Roadmap

開發在獨立的 sandbox app 內進行，引擎為解耦的 Gradle module（`:engine`，對外只有 `translatePage(bitmap): bitmap`），最後一步才掛進 yokai fork。

| 里程碑 | 內容 |
|---|---|
| **M0** | sandbox app + ONNX 載入 detector，對一頁畫出文字框（真機驗證 XNNPACK） |
| **M1** | 接 OCR（48px CTC），debug overlay 印出辨識的日文 |
| **M2** | 接 LLM 翻譯（DeepSeek，逐頁並發）+ 白塊蓋字 → **第一個端到端能動版本** |
| **M3** | 白塊換 LaMa inpaint、陽春排版換氣泡 typesetting → 拚品質 |
| **M4** | 接進 yokai 下載管線、模型下載管理、量化 / 效能、快速 / 品質模式 |

## 隱私

- **BYOK**：不內建任何 API key，金鑰由你自己提供並存於 Android Keystore。
- **裝置端處理**：偵測 / OCR / 去字不離開裝置。
- **翻譯會連網**：OCR 出來的文字會送往**你所設定的** LLM provider 進行翻譯。請自行確認該 provider 的資料政策。

## 對齊 manga-image-translator

m-i-t 是規格，不是要被 1:1 複寫的母本。對齊分三層：

1. **照搬** — prompt 與協定、各階段參數 / 閾值、config schema、模型選擇與處理順序、provider 範圍。
2. **對齊行為、自由實作** — 偵測後處理、座標反算、遮罩生成、閱讀順序排序、並發翻譯（判準：同輸入給相近輸出）。
3. **知情偏離（留紀錄）** — 平台逼的或刻意的取捨，例如 ORT 量化、丟掉 CUDA、M2 白塊先頂著、預設不啟用滾動上文。

每個移植檔的檔頭都會標 `// ported from <python 路徑> @ <commit>`，並以 parity harness 對 Python 輸出比對驗證。

## 致謝

本專案站在這些前人的肩膀上：

- [yokai](https://github.com/null2264/yokai)（base reader）
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator)（翻譯 pipeline 規格）
- [Koharu](https://github.com/mayocream/koharu)（ONNX 匯出 / pipeline 參考）
- [comic-text-detector](https://github.com/dmMaze/comic-text-detector)、[manga-ocr](https://github.com/kha-white/manga-ocr)、[LaMa](https://github.com/advimman/lama)（模型）
- Noto Sans / Serif CJK、Source Han（繁中算繪字型）

## 授權

**待確認。** Yakuyomi fork 自 yokai（Apache-2.0），並使用 m-i-t 與多個第三方模型權重 / 字型。公開發佈前會逐一完成授權稽核（散布條款、模型 host、字型授權），屆時補上正式 `LICENSE`。在此之前請勿假設任何散布授權。

---

<div align="center">
<sub>訳読み — 讀懂那些還沒被翻譯的格子。</sub>
</div>
