# Models

English ｜ [中文](MODELS_zh.md)

The engine ships no model weights. It needs three models — a detector, an OCR model, and a text-removal (inpaint) model — supplied two ways: manually (bring your own model) or by auto-download from this repo's release. Two of them run on NCNN and ship as `.param` + `.bin` pairs; OCR is a single int8 ONNX file — five files in all. Both routes land in the same models folder; downstream resolution is identical.

## The three models

| Role | Backend | File(s) | Size | License | Source |
|---|---|---|---|---|---|
| Detection | NCNN | `detector_noblk.ncnn.param` + `.bin` | ~41 MB | GPL-3.0 | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| OCR | ONNX (int8) | `ocr_int8.onnx` | ~44 MB | GPL-3.0 | exported here from [manga-image-translator](https://github.com/zyddnys/manga-image-translator) weights |
| Text removal | NCNN | `mit_aot_fixed512.ncnn.param` + `.bin` | ~11 MB | GPL-3.0 | AOT-GAN from [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |

**v2 backends.** Detection and text removal moved from ONNX Runtime to NCNN (fixed-shape, Vulkan + ARM-NEON; ~2.9× faster detection on device); OCR stays on ONNX Runtime but is now int8 dynamic-quantized (QUInt8 — ~3.6× faster on ARM, 96.7% CTC parity vs fp32, 165 MB → 44 MB). Total on-device weights dropped from ~470 MB to ~92 MB. All three run on the CPU — GPU/NPU was tried and does not work for these models (NCNN's Vulkan path miscomputes the AOT-GAN; LiteRT can't compile them), and plain CPU is fast enough (~5 s/page on an SD 8 Gen 3). v1's LaMa inpaint is retired and removed; AOT-GAN (manga-image-translator's inpaint) replaces it.

Exact bytes and checksums are pinned in [`models.json`](../models.json):

```
detector_noblk.ncnn.param      18707  sha256 851c33de…c6f72794
detector_noblk.ncnn.bin     41116904  sha256 e9c9c64f…bfb2b7dd
ocr_int8.onnx               43625294  sha256 353e68a5…29fa4c5c
mit_aot_fixed512.ncnn.param    33810  sha256 f21ef860…ee7d32b5
mit_aot_fixed512.ncnn.bin   11366088  sha256 a52db45e…5e3560b6
```

## Redistribution and licensing

These weights are all GPL-3.0. This project redistributes them under that license, with attribution to the sources above, purely as a convenience for auto-download. The OCR model is our own int8-quantized ONNX export of manga-image-translator's weights, and the NCNN detector and AOT-GAN inpaint are our own conversions of the upstream weights — there is no upstream distributable to point at, so they are hosted here. If you prefer, obtain the original weights yourself from the sources and use bring-your-own-model.

## Getting the models

**Auto-download (reader).** The reader fetches every file listed in the manifest from this repo's release in one step, verifying each file's sha256 against [`models.json`](../models.json). Same result as bring-your-own-model, just automated.

**Bring your own model (manual).** Put the files in the models folder you point the app at (NCNN roles need both the `.param` and its `.bin`). They are resolved by name and extension — a `.param` matching `detect`/`comictext` → detector, a `.param` matching `aot` → text removal, an `.onnx` matching `ocr` → OCR. Optional ORT fallbacks are still recognised: an `.onnx` matching `detect`/`comictext`, and an `.onnx` matching `lama` for the retired LaMa path.

## Verification

`ModelDownloader.verify(models, dir)` returns, per role, whether the local file's size and sha256 match what we published. Because the manifest is versioned together with the files, the checksum is always correct — updating the weights means a new manifest version, not a stale hash. This confirms the file you hold is byte-for-byte the one we distribute.

## API

```kotlin
// engine — ModelDownloader
val models = ModelDownloader.fetchManifest()              // models.json -> List<RemoteModel>
ModelDownloader.ensure(models, destDir) { progress -> }   // download missing/mismatched, verify sha256
ModelDownloader.verify(models, destDir)                   // role -> ok (verify only, no download)
```

The engine only fetches, verifies, and writes files; where to download, when to trigger, and the progress UI belong to the reader — the same engine/reader split as the LLM model list (see [PROVIDERS.md](PROVIDERS.md)).
