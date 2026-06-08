# 模型

[English](MODELS.md) ｜ 中文

引擎不內建任何模型權重。它需要三顆 ONNX，有兩種取得方式——手動（自備模型）或從本 repo 的 release 自動下載。兩者都落在同一個 models 資料夾，下游解析完全一樣。

## 三顆模型

| 角色 | 檔案 | 大小 | 授權 | 出處 |
|---|---|---|---|---|
| 偵測 | `comictextdetector.pt.onnx` | ~95 MB | GPL-3.0 | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| OCR | `ocr_48px_ctc.onnx` | ~165 MB | GPL-3.0 | 由 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 權重在本專案匯出 |
| 去字 | `lama-manga.onnx` | ~207 MB | GPL-3.0（底層 LaMa 為 Apache-2.0） | [Koharu](https://github.com/mayocream/koharu) |

精確 bytes 與雜湊釘在 [`models.json`](../models.json)：

```
comictextdetector.pt.onnx   94669756  sha256 1a86ace7…071d718f
ocr_48px_ctc.onnx          164974063  sha256 3019b406…9b2c35d8
lama-manga.onnx            207482644  sha256 4512adab…876f02a4
```

## 散布與授權

這些權重是 GPL-3.0（底層 LaMa 架構為 Apache-2.0）。本專案在上述授權下、附出處歸屬地重新散布它們，純粹是為了「自動下載」的方便。OCR 模型是我們自己對 manga-image-translator 權重做 `torch.onnx.export` 出來的，沒有上游現成的 ONNX 可指，所以由本 repo host。你也可以自己從出處取得權重、走自備模型。

## 怎麼取得模型

**自動下載（reader）。** reader 一鍵從本 repo 的 release 抓齊三顆，逐顆對 [`models.json`](../models.json) 的 sha256 驗證。結果跟自備模型一樣，只是自動化。

**自備模型（手動）。** 把三顆 `.onnx` 放進你指給 app 的 models 資料夾。它們按檔名解析——`detect`/`comictext` → 偵測、`ocr` → OCR、`lama`/`inpaint` → 去字。

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
