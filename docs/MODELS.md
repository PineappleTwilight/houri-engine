# Models

English ｜ [中文](MODELS_zh.md)

The engine ships no model weights. It needs three models — a detector, an OCR model, and a text-removal (inpaint) model — supplied two ways: manually (bring your own model) or by auto-download from this repo's releases. Two of them run on NCNN and ship as `.param` + `.bin` pairs; OCR is a single int8 ONNX file — five files in all. Both routes land in the same models folder; downstream resolution is identical.

## The three models

| Role | Backend | File(s) | Size | License | Source |
|---|---|---|---|---|---|
| Detection | NCNN | `dbnet_detect.ncnn.param` + `.bin` | ~153 MB | GPL-3.0 | DBNet (ResNet34 + DB head), the default detector of [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |
| OCR | ONNX (int8) | `ocr_int8.onnx` | ~44 MB | GPL-3.0 | exported here from [manga-image-translator](https://github.com/zyddnys/manga-image-translator) weights |
| Text removal | NCNN | `mit_aot_fixed512.ncnn.param` + `.bin` | ~11 MB | GPL-3.0 | AOT-GAN from [manga-image-translator](https://github.com/zyddnys/manga-image-translator) |

**Backends.** Detection and text removal run on NCNN (ARM-NEON); OCR runs on ONNX Runtime and is int8 dynamic-quantized (QUInt8 — ~3.6× faster on ARM, 96.7% CTC parity vs fp32, 165 MB → 44 MB). All three run on the CPU — GPU/NPU was tried and does not work for these models (NCNN's Vulkan path miscomputes the AOT-GAN; LiteRT can't compile them). v1's LaMa inpaint is retired and removed; AOT-GAN (manga-image-translator's inpaint) replaces it.

**v3 detector.** comic-text-detector is retired and removed; DBNet (manga-image-translator's default detector) replaces it, reading 1.6–2.5× more text correctly on device. It is kept in fp16 storage — int8 quantization makes it emit no boxes at all and is no faster on ARM, so it is not used, which is why the detector alone is ~153 MB of the ~208 MB of on-device weights. Input is `resize_aspect` to 1024 padded to a multiple of 256; the resulting rectangular input also steers clear of an ncnn heap-corruption bug on square 832–992 inputs. On an SD 8 Gen 3, detection + OCR over 6 representative pages (161 boxes) takes 10.3 s and reads 160 of them — 99.4%.

Exact bytes and checksums are pinned in [`models.json`](../models.json):

```
dbnet_detect.ncnn.param        13392  sha256 9e6db2f8…ee1ff7b5
dbnet_detect.ncnn.bin      153010556  sha256 f57bdbed…fcc55c3d
ocr_int8.onnx               43625294  sha256 353e68a5…29fa4c5c
mit_aot_fixed512.ncnn.param    33810  sha256 f21ef860…ee7d32b5
mit_aot_fixed512.ncnn.bin   11366088  sha256 a52db45e…5e3560b6
```

These checksums are a distribution integrity check — they confirm the file you hold is the one we published. They are **not** a criterion for judging a rebuild: a rebuilt model can be numerically identical and still hash differently (the inpaint model always does). To rebuild any of these from the upstream checkpoints, see [BUILD_MODELS.md](BUILD_MODELS.md).

## Redistribution and licensing

These weights are all GPL-3.0. This project redistributes them under that license, with attribution to the sources above, purely as a convenience for auto-download. The OCR model is our own int8-quantized ONNX export of manga-image-translator's weights, and the NCNN detector and AOT-GAN inpaint are our own conversions of the upstream weights — there is no upstream distributable to point at, so they are hosted here. If you prefer, obtain the original weights yourself from the sources and use bring-your-own-model. The full conversion path from upstream checkpoint to each of these files — scripts, pinned versions, and how to verify the result — is in [BUILD_MODELS.md](BUILD_MODELS.md).

## Getting the models

**Auto-download (reader).** The reader fetches every file listed in the manifest in one step, verifying each file's sha256 against [`models.json`](../models.json). Each entry carries its own url: the detector comes from the `models-v3` release, while OCR and text removal are unchanged and still served from `models-v2`. Same result as bring-your-own-model, just automated.

**Bring your own model (manual).** Put the files in the models folder you point the app at (NCNN roles need both the `.param` and its `.bin`). They are resolved by name and extension — a `.param` matching `dbnet` → detector, a `.param` matching `aot` → text removal, an `.onnx` matching `ocr` → OCR. There are no ORT fallbacks: the ONNX detector/inpaint paths and LaMa are removed, so detection and text removal are NCNN-only, and all three roles must resolve or the set is rejected.

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
