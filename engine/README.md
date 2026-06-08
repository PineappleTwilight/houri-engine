# `:engine` — Yakuyomi translation engine

English ｜ [中文](README_zh.md)

An on-device manga translation library (Android, Kotlin + ONNX Runtime). Give it a page bitmap, get back a translated page bitmap. Detection, OCR, and text removal run on the device; translation calls a cloud LLM (OpenAI-compatible).

The module is reader-agnostic. Its only job is `translatePage(bitmap) -> PageResult`. Overwriting files, markers, resume, and cross-page batching are the caller's responsibility (see [Result handling](#result-handling)). The reader app (the [Yakuyomi](https://github.com/joyeli/Yakuyomi) mihon fork) consumes it via Gradle composite build.

Group: `li.joye.yakuyomi:engine`. Min SDK 26.

## Quick start

```kotlin
// 1. Point at the three model files on local disk (see Models).
val models = ModelSet(
    detector  = "/path/comictextdetector.pt.onnx",
    ocr       = "/path/ocr_48px_ctc.onnx",
    inpainter = "/path/lama-manga.onnx",
)
// or let the engine pick them out of a folder listing:
val models = ModelSet.resolve(localOnnxFiles) ?: return // null = not all present

// 2. Load the OCR alphabet (in the engine assets) and your API key.
val alphabet: List<String> = assets.open("models/alphabet-all-v5.txt").bufferedReader().readLines()
val apiKey = "<deepseek key>"   // null/blank = detect+OCR+remove only, no translation (debug)

// 3. Create the engine and translate. use { } releases the native ONNX sessions.
Yakuyomi.create(models, alphabet, apiKey).use { engine ->
    when (val r = engine.translatePage(pageBitmap)) {
        is PageResult.Translated -> writeBack(r.page)   // success: overwrite + mark done
        is PageResult.Skipped    -> { /* nothing to translate: keep original */ }
        is PageResult.Failed     -> { /* error: keep original, retry later */ }
    }
}
```

`translatePage` is a `suspend` function; call it from a background dispatcher. One engine instance handles one page at a time; it isn't safe to call `translatePage` concurrently on the same instance.

## Models (BYOM)

The engine ships no model weights. The host supplies three ONNX files plus the OCR alphabet:

| Role | File (typical name) | What it does | Source |
|---|---|---|---|
| detector | `comictextdetector.pt.onnx` | text boxes + stroke mask | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| ocr | `ocr_48px_ctc.onnx` | 48px CTC Japanese OCR | manga-image-translator |
| inpainter | `lama-manga.onnx` | LaMa text removal | [Koharu](https://github.com/mayocream/koharu) |

`ModelSet.resolve(files)` maps a flat `(filename, localPath)` listing to the three roles by name (`detect`/`comictext` to detector, `ocr` to ocr, `lama`/`inpaint` to inpainter) and returns `null` if any is missing. Use that as your "ready to translate?" check.

Paths must be local files, not SAF/content URIs: ORT calls `createSession(path)` into native memory. Don't read weights into the JVM heap with `readBytes()`; the heap is capped around 512 MB regardless of device RAM and will OOM. If the source is SAF, copy to `filesDir` first and pass the path.

## Configuration

Everything is a `data class` with defaults; override only what you need:

```kotlin
val config = EngineConfig(
    ocr        = OcrConfig(minProb = 0.5f),               // drop low-confidence OCR
    inpainter  = InpainterConfig(method = "auto"),        // "boxfill" | "auto" (+ wholeImage)
    render     = RenderConfig(orientation = TextOrientation.AUTO),
    translator = TranslatorConfig(model = "deepseek-chat", batchSize = 8),
)
Yakuyomi.create(models, alphabet, apiKey, config)
```

The full list, with ranges and the effect of each, is in [`docs/PARAMETERS.md`](../docs/PARAMETERS.md). A few defaults worth knowing:

- `OcrConfig.useXnnpack = false`. Must stay off: XNNPACK miscomputes the 48px CTC model on real hardware and OCR returns empty. The detector and inpainter use XNNPACK fine.
- `OcrConfig.concurrent = true`, `concurrency = 8`. OCR recognizes lines in parallel; on an 8-core phone this roughly halves OCR time, with no change to output.
- `InpainterConfig.method = "auto"`, `wholeImage = true`. Flat-fills clean bubbles and runs one whole-image LaMa pass for text over artwork. `"boxfill"` flat-fills everything (fastest, on-art text becomes a colour block); `"auto"` with `wholeImage = false` runs per-region LaMa (slowest, sharpest).
- `RenderConfig.orientation = AUTO`. Follows each region's detected direction, then rotates along the region's skew angle.
- `TranslatorConfig.provider = "deepseek"`, with `apiBase` and `model`. Any OpenAI-compatible endpoint. `LlmProviders.ALL` carries presets for manga-image-translator's LLM set plus OpenRouter (all OpenAI-compatible; Gemini via its compat endpoint), and `LlmModels.list()` fetches a provider's live model list. See [`docs/PROVIDERS.md`](../docs/PROVIDERS.md).

### Language pair (not fixed to JP→CHT)

Default is Japanese to Traditional Chinese, but any pair works. Set these together:

```kotlin
translator = TranslatorConfig(
    toLangName   = "English",   // target: the LLM translates into this
    fromLangName = "Korean",    // source label for the prompt ("" = let the LLM infer)
    sampleSource = "<|1|>…",    // few-shot example in the source language ("" = no example)
    sampleTarget = "<|1|>…",    // and its translation, in the target language
)
```

The source is whatever the OCR model recognizes; the bundled 48px CTC is Japanese, so load a different OCR model and alphabet to read another language. The target is purely the prompt. Keep `toLangName` and the few-shot in the same target language, or the example biases the output.

## Result handling

`translatePage` returns a `PageResult`, not a bare bitmap, so the caller can honour the core invariant: never overwrite the original with something worse.

| Variant | Meaning | What the caller does |
|---|---|---|
| `Translated(page, stats)` | success | overwrite the file, write a "translated" marker |
| `Skipped(reason, stats)` | nothing translatable (no text / OCR empty / all filtered) | keep original, mark skipped, don't retry |
| `Failed(reason)` | error (network/429/exception) | keep original, no marker, retry later |

Per-region resilience is built in: if one bubble fails to translate, the engine draws its source text back instead, and the rest of the page still translates. `PageStats` carries per-stage timings, plus `wallMs` (the actual elapsed time, which is shorter than the stage sum because text removal overlaps the translation request).

## Lifecycle and threading

- `TranslationEngine : AutoCloseable`. `close()` releases the detector/ocr/inpainter ONNX sessions (native memory). Always `use { }` or `close()`.
- `translatePage` is `suspend`, one page at a time per instance. It uses coroutines internally (parallel OCR, removal overlapping translation), but don't call it concurrently on a single instance.
- The input bitmap isn't recycled by the engine. `Translated.page` is a new bitmap.

## Pipeline

```
page  Detector    lines + stroke mask
      Ocr         Japanese text per line (parallel; mutates lines in place)
      Grouping    regions (connect + MST-split; reading order; skew angle)
      Translator  target text (per page, no rolling context; optional)
      TextFilter  decide which regions have a usable translation
      Inpainter   erase original text (runs concurrently with Translator)
      Renderer    typeset translation (vertical/horizontal, angle-aware)
      -> translated page bitmap
```

See [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the design rationale and the desktop-parity validation. Behaviour follows [manga-image-translator](https://github.com/zyddnys/manga-image-translator).

## Advanced: direct component access

The factory is the recommended path. For per-stage debugging (a detection overlay, say) you can construct the stages yourself and assemble a `Pipeline`, but then you own their lifecycle:

```kotlin
val detector = Detector(models.detector, config.detector)
val detection = detector.detect(page)   // lines + textMask, draw your overlay
// …
detector.close()                         // close what you create
```

Pure helpers (`Geometry`, `ImageOps`, `TextFilter`) are `internal`, not part of the public API.
