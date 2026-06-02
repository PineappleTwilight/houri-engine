<div align="center">

# Yakuyomi（訳読み）— Engine

**漫畫翻譯引擎 — 裝置端偵測 / OCR / 去字（ONNX Runtime）＋ 雲端 LLM 翻譯**

日本語 → 繁體中文（CHT）

[English](README.md) ｜ **中文**

> **本 repo（`yakuyomi-engine`）是翻譯引擎。** 閱讀器本體 —— **Yakuyomi**（一個
> [mihon](https://github.com/mihonapp/mihon) fork）—— 是另一個 repo，以 submodule 引入本引擎（見[倉庫架構](#倉庫架構)）。
>
> **狀態：引擎端到端可動（M0–M3）** 已在真機跑通；接進 reader（M4）進行中。

</div>

---

## 這是什麼

Yakuyomi 是一個**內建 AI 翻譯**、以 [mihon](https://github.com/mihonapp/mihon) 為基底的漫畫閱讀器。工作拆成兩半：

- **文字偵測、OCR、去字** → 在**裝置上**離線執行（ONNX Runtime）—— 這就是**引擎**（本 repo）。
- **翻譯** → 交給**雲端 LLM**（預設 DeepSeek，OpenAI 相容）。

最高目標是**效率 / 吞吐**：一章多頁的翻譯並發送出，盡量壓低端到端時間。

## 倉庫架構

專案分成兩個 repo：

| Repo | 角色 |
|---|---|
| **`yakuyomi-engine`**（本 repo） | 裝置端翻譯引擎：`:engine`（偵測 / OCR / 翻譯 / 去字 / 排版，對外只有 `translatePage(page): PageResult`）＋ 丟棄式 `:app-sandbox`（單獨測試）＋ `parity/` 驗證 harness。reader 無關、可單獨測。 |
| **`Yakuyomi`** —— 一個 [mihon](https://github.com/mihonapp/mihon) fork | 閱讀器本體：mihon 改品牌，加上下載管線 hook ＋ 翻譯設定 ＋ 模型管理。以 **git submodule** 引入 `yakuyomi-engine`，透過 Gradle `includeBuild` 從原始碼建。 |

為什麼分：引擎保持乾淨、可獨立測；app 是**真的 mihon fork**（才進得了 mihon 的 fork 網路）。引擎改動 commit 在這裡，app 端 bump submodule 指標即可。

## 怎麼做的：vibecoding

本專案以 **vibecoding** 打造——方向、取捨與審查由人主導，程式碼主要由 AI（[Claude Code](https://claude.com/claude-code)）撰寫。

引擎是 Kotlin + ONNX Runtime 從頭實作，**repo 內不含 manga-image-translator 的原始碼**；向它對齊的是行為與 prompt（見[下](#對齊-manga-image-translator)），模型則來自第三方（見[模型與來源](#模型與來源)）。

## 為什麼是「半離線」

偵測 / OCR / 去字都在裝置上跑，這幾步離線可用；唯獨翻譯走雲端 LLM，因此**翻譯當下需要連網**。這是刻意的取捨——換來其餘階段離線、引擎可獨立發佈，代價是接受翻譯半離線。

## 核心特性

- 🔍 **裝置端偵測**：comic-text-detector，輸出文字行框 ＋ **逐像素文字筆畫遮罩（seg）**。
- 🔠 **裝置端 OCR**：48px CTC 模型（純 CPU；XNNPACK 會算錯此模型，已關閉）。
- 🌐 **雲端 LLM 翻譯**：沿用 m-i-t 的 prompt 與協定；逐頁、批內並發、`Semaphore` 限流（防 429）。
- 🧹 **裝置端去字**：預設 **box-fill 就近取色**——取 seg 細筆畫遮罩、只填文字筆畫、每個像素取最近的背景色（多色 / 漸層 / 壓在畫面上的字都不會糊成色塊）；另可切 **LaMa**（整頁 / 逐格）。
- ✍️ **排版**：純文字框法，直 / 橫排、字級自適應、**垂直置中、描邊隨字級、行頭禁則（kinsoku）**、標點旋轉、自動黑 / 白字（依去字後背景亮度）。
- 🇹🇼 **繁體中文輸出**：prompt 強制台灣繁體用語，全靠 LLM（**不做 OpenCC 後處理**，接受偶有用語誤差）。
- 🔑 **BYOK（自帶金鑰）**：provider / model / API base / 金鑰 / 目標語言全可設定，**不內建任何 key**；金鑰存 Android Keystore。
- 📦 **BYOM（自帶模型）**：ONNX 不打進 APK，由使用者指向本機資料夾載入（off-heap，避開 JVM heap 上限）。
- 💾 **就地覆蓋、可續傳**：僅在翻譯成功時覆蓋原頁、以頁為單位記錄完成狀態；失敗 / 無字則保留原圖，**永不用比原圖更糟的東西覆蓋下載庫**。

## 一頁的資料流

```
頁 Bitmap
 → 偵測 (ONNX)   → 文字行框 ＋ seg 文字筆畫遮罩；依 m-i-t 的座標啟發式排序
 → OCR  (ONNX)   → 每塊整塊辨識 → 日文 sourceText
 → 分群           → 鄰近且對齊的行併成氣泡區（移植 m-i-t 合併判準）
 → 翻譯 (LLM)    → 逐頁、並發、無滾動上文 → targetText（依 block ID 貼回）
 → 去字 (ONNX/CV) → box-fill 就近取色（預設）／ LaMa inpaint
 → 排版 (Canvas) → 直 / 橫排、置中、描邊、禁則
 → 翻好的 Bitmap
```

引擎對外只有一個進入點 `translatePage(page): PageResult`（成功 / 略過 / 失敗）；**覆蓋原檔、marker、續傳、跨頁批次併發**都是 reader app（Yakuyomi fork 的下載 worker）的事。

## 技術選型

| 項目 | 選擇 | 備註 |
|---|---|---|
| Reader 基底 | **mihon**（Yakuyomi app 是它的 fork） | 整合點在下載層、不動 reader |
| 推論引擎 | **純 Kotlin + ONNX Runtime**（`onnxruntime-android`，含 XNNPACK） | NNAPI 已棄用，不採用 |
| 加速路線 | XNNPACK/CPU → int8 量化 → 需要時再 QNN / LiteRT（NPU/GPU） | 去字 NPU 化是未來壓低 LaMa 耗時的槓桿 |
| 去字（預設） | **box-fill 就近取色** | 瞬間、跟著局部背景、不糊色塊 |
| 去字（可選） | **LaMa**（整頁 / 逐格） | 見 [模型與來源](#模型與來源) |
| 翻譯 | 雲端 LLM，沿用 m-i-t `chatgpt.py` 的 prompt + 協定 | OpenAI 相容 |
| 預設 provider | **DeepSeek** | 全可改（BYOK） |

## 翻譯 Provider（BYOK）

v1 先支援 **OpenAI 相容**那一組（一個 HTTP client 通吃）：

- `openai`、`deepseek`、`groq`、`custom_openai`（OpenRouter / LM Studio / 自架）
- 之後：`gemini`，再來才是非 LLM 的雲端 MT（DeepL / Caiyun / Youdao / Baidu / Papago）當獨立 adapter。

設定頁會暴露：provider、model、API base（custom_openai 用）、API key（每個 provider 一格，存 Keystore）、目標語言。預設 DeepSeek，但全可改。

## 模型與來源

ONNX 權重**不入庫、不打進 APK**（BYOM）。各模型出處：

| 階段 | 模型 | 來源 |
|---|---|---|
| 偵測 | comic-text-detector（輸出 `blk` / `seg` / `det`） | ONNX 取自 [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector)（manga-image-translator 生態） |
| OCR | 48px CTC | 權重出自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)，我們用 `torch.onnx.export` 轉成 ONNX |
| 去字 | LaMa（漫畫微調） | **ONNX `lama-manga.onnx` 取自 [Koharu（mayocream/koharu）](https://github.com/mayocream/koharu)**；底層架構為 [advimman/LaMa](https://github.com/advimman/lama) |
| 字型 | Noto Sans / Serif CJK TC、Source Han | 繁中算繪用，可散布（OFL / Apache） |

ONNX 匯出與 pipeline 細節參考過 [Koharu](https://github.com/mayocream/koharu)（Rust + ONNX）。各模型 / 字型的散布條款待[授權稽核](#授權)逐一確認。

## 對齊 manga-image-translator

m-i-t 是規格，不是要被 1:1 複寫的母本。釘住一個上游 commit（見 `.upstream-ref`），對齊分三層：

1. **照搬** — prompt 與協定、各階段參數 / 閾值、config schema、模型選擇與處理順序、provider 範圍。
2. **對齊行為、自由實作** — 偵測後處理、座標反算、seg 遮罩生成、文字行分群、閱讀順序、並發翻譯（判準：同輸入給相近輸出）。
3. **知情偏離（留紀錄）** — 平台逼的或刻意的取捨，例如 ORT 推論、丟掉 CUDA、去字改 box-fill 就近取色、預設不啟用滾動上文。

每個移植檔的檔頭標 `// ported from <python 路徑> @ <commit>`，並以 `parity/` 的 harness 對 Python 輸出比對驗證後才接進 pipeline。

## Roadmap

開發在本引擎 repo 內進行（附一個獨立的 `:app-sandbox`）；引擎解耦（`:engine`、對外只有 `translatePage`），由 **Yakuyomi** mihon fork 以 submodule 消費。

| 里程碑 | 內容 | 狀態 |
|---|---|---|
| **M0** | sandbox + ONNX detector，對一頁畫出文字框（真機驗 XNNPACK） | ✅ |
| **M1** | 接 OCR（48px CTC），overlay 印出日文 | ✅ |
| **M2** | 接 LLM 翻譯（DeepSeek，逐頁並發）＋ 蓋字 → 端到端能動 | ✅ |
| **M3** | 去字（box-fill 就近取色 / LaMa）＋ 排版（置中 / 描邊 / 禁則）拚品質；引擎收斂成 `translatePage` | ✅ |
| **M4** | 把引擎接進 **Yakuyomi**（mihon）fork 的下載管線、模型下載管理、量化 / 效能、快速 / 品質模式 | ⏳ |

## 隱私

- **BYOK**：不內建任何 API key，金鑰由你自己提供並存於 Android Keystore。
- **裝置端處理**：偵測 / OCR / 去字不離開裝置。
- **翻譯會連網**：OCR 出來的文字會送往**你所設定的** LLM provider 翻譯，請自行確認該 provider 的資料政策。

## 致謝

本專案站在這些前人的肩膀上：

- [mihon](https://github.com/mihonapp/mihon)（Yakuyomi app fork 自它，Apache-2.0）
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator)（翻譯 pipeline 的行為規格與 prompt）
- [Koharu（mayocream/koharu）](https://github.com/mayocream/koharu)（`lama-manga.onnx` 去字模型、ONNX 匯出 / pipeline 參考）
- [comic-text-detector](https://github.com/dmMaze/comic-text-detector)、[manga-ocr](https://github.com/kha-white/manga-ocr)、[LaMa](https://github.com/advimman/lama)（模型 / 架構）
- Noto Sans / Serif CJK、Source Han（繁中算繪字型）
- [Claude Code](https://claude.com/claude-code)（vibecoding 開發協作）

## 授權

**待確認。** 本 repo 的**程式碼為自行以 Kotlin / ORT 實作**（非移植 m-i-t 原始碼），但專案整體仍**沿用 m-i-t 的 prompt 與參數預設**、以 mihon（Apache-2.0）的 fork 形式發佈，並使用多個第三方模型權重 / 字型。公開發佈前會逐一完成授權稽核（m-i-t 條款、各模型 / 字型授權、模型 host），屆時補上正式 `LICENSE`。在此之前請勿假設任何散布授權。

---

<div align="center">
<sub>訳読み — 讀懂那些還沒被翻譯的格子。</sub>
</div>
