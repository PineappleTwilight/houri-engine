# 模型

[English](MODELS.md) ｜ 中文

引擎不內建任何模型權重。它需要三顆模型——偵測、OCR、去字（inpaint）——有兩種取得方式：手動（自備模型）或從本 repo 的 release 自動下載。其中兩顆跑 NCNN、以 `.param` + `.bin` 成對交付，OCR 是單一顆 int8 ONNX——總共五個檔。兩者都落在同一個 models 資料夾，下游解析完全一樣。

## 三顆模型

| 角色 | 後端 | 檔案 | 大小 | 授權 | 出處 |
|---|---|---|---|---|---|
| 偵測 | NCNN | `detector_noblk.ncnn.param` + `.bin` | ~41 MB | GPL-3.0 | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| OCR | ONNX（int8） | `ocr_int8.onnx` | ~44 MB | GPL-3.0 | 由 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 權重在本專案匯出 |
| 去字 | NCNN | `mit_aot_fixed512.ncnn.param` + `.bin` | ~11 MB | GPL-3.0 | AOT-GAN，出自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |

**v2 後端。** 偵測與去字從 ONNX Runtime 換成 NCNN（固定 shape、Vulkan + ARM-NEON；真機偵測快 ~2.9×）；OCR 仍走 ONNX Runtime，但改成 int8 動態量化（QUInt8——ARM 快 ~3.6×、對 fp32 有 96.7% CTC parity、165 MB → 44 MB）。裝置端權重總量從 ~470 MB 降到 ~92 MB。NCNN 圖是固定 shape、Vulkan/NPU-capable——GPU/NPU-ready 但尚未啟用；純 CPU 已夠快（SD 8 Gen 3 上約 5 秒/頁）。v1 的 LaMa 去字已退役移除，改由 AOT-GAN（manga-image-translator 的 inpaint）取代。

精確 bytes 與雜湊釘在 [`models.json`](../models.json)：

```
detector_noblk.ncnn.param      18707  sha256 851c33de…c6f72794
detector_noblk.ncnn.bin     41116904  sha256 e9c9c64f…bfb2b7dd
ocr_int8.onnx               43625294  sha256 353e68a5…29fa4c5c
mit_aot_fixed512.ncnn.param    33810  sha256 f21ef860…ee7d32b5
mit_aot_fixed512.ncnn.bin   11366088  sha256 a52db45e…5e3560b6
```

## 散布與授權

這些權重全是 GPL-3.0。本專案在該授權下、附出處歸屬地重新散布它們，純粹是為了「自動下載」的方便。OCR 模型是我們自己對 manga-image-translator 權重做 int8 量化 ONNX 匯出，NCNN 偵測與 AOT-GAN 去字則是我們自己對上游權重的轉檔——沒有上游現成的可散布檔可指，所以由本 repo host。你也可以自己從出處取得原始權重、走自備模型。

## 怎麼取得模型

**自動下載（reader）。** reader 一鍵從本 repo 的 release 抓齊 manifest 列的每個檔，逐檔對 [`models.json`](../models.json) 的 sha256 驗證。結果跟自備模型一樣，只是自動化。

**自備模型（手動）。** 把檔案放進你指給 app 的 models 資料夾（NCNN 角色要 `.param` 與對應的 `.bin` 兩個都放）。它們按檔名 + 副檔名解析——`.param` 含 `detect`/`comictext` → 偵測、`.param` 含 `aot` → 去字、`.onnx` 含 `ocr` → OCR。仍認得選配的 ORT 備援：`.onnx` 含 `detect`/`comictext`、以及退役 LaMa 路徑的 `.onnx` 含 `lama`。

## 驗證

`ModelDownloader.verify(models, dir)` 逐角色回報「本機檔的 size 與 sha256 是否符合我們發行的版本」。因為 manifest 跟檔案一起版本化，雜湊永遠正確——更新權重 = 出新版 manifest，不會有過時雜湊。這能確認你手上的檔案跟我們散布的逐位元相同。

## API

```kotlin
// 引擎 — ModelDownloader
val models = ModelDownloader.fetchManifest()              // models.json -> List<RemoteModel>
ModelDownloader.ensure(models, destDir) { progress -> }   // 下載缺的/不符的，驗 sha256
ModelDownloader.verify(models, destDir)                   // role -> 是否相符（只驗、不下載）
```

引擎只負責「抓、驗、落檔」；下載到哪、何時觸發、進度 UI 都是 reader 的事——跟 LLM 模型清單同一套引擎/reader 分法（見 [PROVIDERS.md](PROVIDERS_zh.md)）。
