# 模型

[English](MODELS.md) ｜ 中文

引擎不內建任何模型權重。它需要三顆模型——偵測、OCR、去字（inpaint）——有兩種取得方式：手動（自備模型）或從本 repo 的 releases 自動下載。其中兩顆跑 NCNN、以 `.param` + `.bin` 成對交付，OCR 是單一顆 int8 ONNX——總共五個檔。兩者都落在同一個 models 資料夾，下游解析完全一樣。

## 三顆模型

| 角色 | 後端 | 檔案 | 大小 | 授權 | 出處 |
|---|---|---|---|---|---|
| 偵測 | NCNN | `dbnet_detect.ncnn.param` + `.bin` | ~153 MB | GPL-3.0 | DBNet（ResNet34 + DB head），出自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 的 default detector |
| OCR | ONNX（int8） | `ocr_int8.onnx` | ~44 MB | GPL-3.0 | 由 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 權重在本專案匯出 |
| 去字 | NCNN | `mit_aot_fixed512.ncnn.param` + `.bin` | ~11 MB | GPL-3.0 | AOT-GAN，出自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |

**後端。** 偵測與去字跑 NCNN（ARM-NEON；去字是固定 tile）；OCR 走 ONNX Runtime，且是 int8 動態量化（QUInt8——ARM 快 ~3.6×、對 fp32 有 96.7% CTC parity、165 MB → 44 MB）。三顆都跑 CPU——GPU/NPU 試過、對這些模型不管用（NCNN Vulkan 把 AOT-GAN 算成垃圾、LiteRT 編不出來），GPU/Vulkan 路徑已移除。v1 的 LaMa 去字已退役移除，改由 AOT-GAN（manga-image-translator 的 inpaint）取代。

**v3 偵測器。** comic-text-detector 已退役、整條移除；改用 manga-image-translator 的 default detector（DBNet：ResNet34 + DB head），真機讀對的文字多 **1.6–2.5×**。權重維持 fp16 storage——int8 量化實測**完全吐不出框**、在 ARM 上也沒有比較快，因此不採用；這也是為什麼光偵測器就佔了裝置端 ~208 MB 權重裡的 ~153 MB。前處理是 resize_aspect 到 1024、再 pad 到 256 的倍數；這樣得到的**矩形**輸入同時繞開 ncnn 對 832–992 正方形尺寸的 heap corruption。SD 8 Gen 3 上的實測：6 張代表頁、161 個偵測框，偵測 + OCR 共 10.3 秒、讀出其中 160——99.4%。

精確 bytes 與雜湊釘在 [`models.json`](../models.json)：

```
dbnet_detect.ncnn.param        13392  sha256 9e6db2f8…ee1ff7b5
dbnet_detect.ncnn.bin      153010556  sha256 f57bdbed…fcc55c3d
ocr_int8.onnx               43625294  sha256 353e68a5…29fa4c5c
mit_aot_fixed512.ncnn.param    33810  sha256 f21ef860…ee7d32b5
mit_aot_fixed512.ncnn.bin   11366088  sha256 a52db45e…5e3560b6
```

這些雜湊是**散布用的完整性檢查**——用來確認你手上的檔就是我們發行的那個。它們**不是**判斷「重建是否正確」的準則：一次重建可以數值上完全等價、雜湊卻不同（去字那顆就永遠如此）。要從上游 ckpt 重建這些模型，見 [BUILD_MODELS_zh.md](BUILD_MODELS_zh.md)。

## 散布與授權

這些權重全是 GPL-3.0。本專案在該授權下、附出處歸屬地重新散布它們，純粹是為了「自動下載」的方便。OCR 模型是我們自己對 manga-image-translator 權重做 int8 量化 ONNX 匯出，NCNN 偵測與 AOT-GAN 去字則是我們自己對上游權重的轉檔——沒有上游現成的可散布檔可指，所以由本 repo host。你也可以自己從出處取得原始權重、走自備模型。從上游 ckpt 到這五個檔的完整轉檔路徑——腳本、釘住的版本、以及怎麼驗證產出——見 [BUILD_MODELS_zh.md](BUILD_MODELS_zh.md)。

## 怎麼取得模型

**自動下載（reader）。** reader 一鍵抓齊 manifest 列的每個檔，逐檔對 [`models.json`](../models.json) 的 sha256 驗證。每一筆都帶自己的 url：偵測器來自 `models-v3` release，OCR 與去字沒有變動、仍由 `models-v2` 供應。結果跟自備模型一樣，只是自動化。

**自備模型（手動）。** 把檔案放進你指給 app 的 models 資料夾（NCNN 角色要 `.param` 與對應的 `.bin` 兩個都放）。它們按檔名 + 副檔名解析——`.param` 含 `dbnet` → 偵測、`.param` 含 `aot` → 去字、`.onnx` 含 `ocr` → OCR。三個角色缺一即視為未備齊；沒有 ORT 備援、也沒有 LaMa 路徑（兩者皆已移除）。

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
