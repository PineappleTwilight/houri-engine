# Models

English ｜ [中文](MODELS_zh.md)

The engine ships no model weights. It needs three ONNX files, supplied two ways — manually (bring your own model) or by auto-download from this repo's release. Both land in the same models folder; downstream resolution is identical.

## The three models

| Role | File | Size | License | Source |
|---|---|---|---|---|
| Detection | `comictextdetector.pt.onnx` | ~95 MB | GPL-3.0 | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| OCR | `ocr_48px_ctc.onnx` | ~165 MB | GPL-3.0 | exported here from [manga-image-translator](https://github.com/zyddnys/manga-image-translator) weights |
| Text removal | `lama-manga.onnx` | ~207 MB | GPL-3.0 (base LaMa Apache-2.0) | [Koharu](https://github.com/mayocream/koharu) |

Exact bytes and checksums are pinned in [`models.json`](../models.json):

```
comictextdetector.pt.onnx   94669756  sha256 1a86ace7…071d718f
ocr_48px_ctc.onnx          164974063  sha256 3019b406…9b2c35d8
lama-manga.onnx            207482644  sha256 4512adab…876f02a4
```

## Redistribution and licensing

These weights are GPL-3.0 (the base LaMa architecture is Apache-2.0). This project redistributes them under those licenses, with attribution to the sources above, purely as a convenience for auto-download. The OCR model is our own `torch.onnx.export` of manga-image-translator's weights, so there is no upstream ONNX to point at — it is hosted here. If you prefer, obtain the weights yourself from the sources and use bring-your-own-model.

## Getting the models

**Auto-download (reader).** The reader fetches all three from this repo's release in one step, verifying each file's sha256 against [`models.json`](../models.json). Same result as bring-your-own-model, just automated.

**Bring your own model (manual).** Put the three `.onnx` files in the models folder you point the app at. They are resolved by name — `detect`/`comictext` → detector, `ocr` → OCR, `lama`/`inpaint` → text removal.

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
