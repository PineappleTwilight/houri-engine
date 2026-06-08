# 架構

[English](ARCHITECTURE.md) ｜ 中文

Yakuyomi 怎麼翻一頁、專案為什麼這樣切、裝置端引擎跟桌面驗證工具怎麼搭。引擎的 API 看 [`engine/README_zh.md`](../engine/README_zh.md)。

## 兩半

| | 裝置端（`engine/`，Kotlin） | 桌面（`parity/`，Python） |
|---|---|---|
| 角色 | 產品，跑在手機上 | 驗證工具，跑在筆電上 |
| 技術 | Kotlin、ONNX Runtime、Android Canvas | Python、numpy/cv2、onnxruntime、PIL |
| 出貨 | 是（library） | 否（開發用） |
| 用途 | 翻頁 | 拿 Kotlin port 對照參考實作 |

引擎是要交付的東西。桌面 parity 之所以存在，是因為引擎把 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)（m-i-t，Python/torch）用 Kotlin/ONNX 重寫，這種 port 沒辦法逐行 diff。桌面在 Python 跑同樣的階段，讓我們先確認「同輸入、近輸出」，再信任 Kotlin 版。見 [`parity/README_zh.md`](../parity/README_zh.md)。

## 單頁資料流

```
頁 bitmap
  Detector (ONNX)    文字行（旋轉四邊形）+ 逐像素筆畫遮罩
  Ocr (ONNX)         每行日文（48px CTC，貪婪解碼）
  Grouping (Kotlin)  行 -> 氣泡區塊（連邊，再 MST 分裂；閱讀序；傾斜角）
  Translator (LLM)   目標語言，每頁一個請求，無滾動上文
  TextFilter         判定哪些區塊有可用的譯文
  Inpainter (ONNX)   抹掉原文（平塗，或 LaMa）
  Renderer (Canvas)  把譯文畫回去（直/橫排，沿區塊角度旋轉）
翻好的頁 bitmap
```

每階段一句話：

- **Detector** — comic-text-detector。letterbox 前處理、ONNX，產出框（NMS + unclip）跟一張 `seg` 筆畫遮罩。去字用這張遮罩，抹的是筆畫不是方塊。
- **Ocr** — 48px CTC。把每行裁出來（透視校正、直書行轉正），辨識、對字典貪婪解碼，丟掉低於 `minProb` 的行。文字行並發跑（見下）。
- **Grouping** — 把行併成氣泡大小的區塊，分兩階段：寬鬆的連邊，再用 MST 分裂把只靠傳遞才連起來的鄰居切開。這個分裂就是讓密集氣泡不黏成一塊的關鍵。同時算每個區塊的閱讀序跟傾斜角。
- **Translator** — m-i-t 的 `chatgpt.py` prompt 與協定，走 OpenAI 相容呼叫，每頁一個請求、無跨頁上文。某區塊翻譯失敗就保留它的原文。語言對可設：目標走 `toLangName`、來源由 OCR 模型加 prompt 標籤決定。預設日翻繁中，不寫死。服務商是預設選單、全 OpenAI 相容（Gemini 走它的 compat 端點），各家的模型清單即時撈取——見 [PROVIDERS.md](PROVIDERS_zh.md)。
- **TextFilter** — m-i-t 的譯後過濾。一個區塊算「可用」的條件：譯文非空白、非純數字、不命中過濾 regex、且不等於原文。
- **Inpainter** — 三個模式：`boxfill`（用局部背景色平塗遮罩區）、`auto` 配 `wholeImage` 開（整頁一次 LaMa）或關（逐區 LaMa）。auto 把乾淨泡泡平塗、只把壓在畫面上的字送 LaMa。預設 auto + `wholeImage` 開。
- **Renderer** — 文字框排版，不做氣泡 flood-fill：字級自適應、行頭禁則、直或橫排、文字顏色取去字後背景亮度、畫布沿區塊傾斜角旋轉。

`Pipeline.kt` 管 orchestration，以及「絕不用比原圖更糟的東西覆蓋」這條不變式。

## 並發

三個地方，為了吞吐：

- **OCR** 把一頁的文字行並發跑，每行一條緒。圖塊小、吃不滿多緒推論，所以用不同的行把核填滿比較快。
- **去字跟翻譯重疊。** 去字只要 OCR 過的區塊，這在翻譯回來前就確定了，所以去字在背景 coroutine 上跑、同時 LLM 請求在飛。一頁省下兩者裡較短的那個。這也是為什麼失敗的區塊保留「重貼的原文」而不是原封不動的原圖：把去字跟翻譯結果解耦，才能讓兩者重疊。
- **跨頁** 批次是 reader app 的事：一章的頁各發一個請求、用 `Semaphore` 限流，下載 worker 在你讀到之前先翻好。

## 與 manga-image-translator 對齊

引擎對齊的是行為，不是原始碼。三層，由緊到鬆：

1. **照抄。** 跨語言不變的資料：prompt 與協定、各階段門檻與預設、config schema、模型選擇、處理順序。直接用 m-i-t 的調校，不重推一遍。
2. **對齊行為、自由實作。** 做「什麼」跟 m-i-t，「怎麼寫」隨 Kotlin/ONNX 慣例。偵測後處理、座標反算、遮罩生成、分群兩階段、閱讀序、並發翻譯。判準是「同輸入、近輸出」。
3. **知情偏離，在碼裡註記。** 平台逼的或刻意的取捨：ONNX 取代 torch、不用 CUDA、不做滾動上文、去字模式階梯。

被迫改寫的（其餘照搬）：torch 換 ORT `session.run`；cv2/PIL 換 Bitmap/Canvas/手刻；numpy 換 Kotlin；async httpx/CLI/YAML 換 OkHttp、coroutines、小型 config 載入；manga-ocr 的自回歸 decode；任何 CUDA/GPU 假設。

## 裝置端的現實

真機跑出來的：

- **XNNPACK 會把 OCR 模型算錯**，吐出空字。`OcrConfig.useXnnpack` 維持關（純 CPU）。偵測器跟去字用 XNNPACK 沒問題。
- **模型載 native 記憶體。** `createSession(path)` 把權重讀進 native；用 `readBytes()` 讀進 JVM heap 會撞到每 app 的 heap 上限（約 512MB，跟裝置 RAM 無關）而 OOM。BYOM 先把選的檔複製到 `filesDir` 再傳路徑。
- **前處理要跟 Python 匯出完全一致** — resize、normalize、NCHW 順序。這是最大的隱形分歧來源，也是 parity 工具大半的存在理由。
- **推論執行緒。** 偵測跟 LaMa 用對齊裝置大核數的緒數（測試機 Snapdragon 8 Gen 3 是 6），把慢的小核加進去會讓一次推論更慢而不是更快。

## Repo 結構

```
engine/        Android library，裝置端 pipeline（產品）
  src/main/kotlin/li/joye/yakuyomi/engine/
    Yakuyomi.kt, TranslationEngine.kt, ModelSet.kt   對外入口（facade + 型別）
    Pipeline.kt, Detector.kt, Ocr.kt, Grouping.kt,
    LlmTranslator.kt, Inpainter.kt, Renderer.kt      各階段
    LlmProviders.kt, LlmModels.kt                    供應商預設 + 即時撈模型清單
    Geometry.kt, ImageOps.kt, TextFilter.kt          內部 helper
  src/test/kotlin/…                                  JVM 單元測試
app-sandbox/   丟棄式測試 app（裝置計時、比較圖）
parity/        桌面 Python 驗證工具（不出貨）
docs/          這份，加參數參考
```

reader app（[Yakuyomi](https://github.com/joyeli/Yakuyomi) mihon fork）用 git submodule + Gradle composite build 引入 `engine`，只加整合層：下載 hook、設定、模型管理。
