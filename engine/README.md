# `:engine` — Yakuyomi translation engine

**English** · [中文](README_zh.md)

An on-device manga translation **library** (Android, pure Kotlin + ONNX Runtime).
Give it a page bitmap, get back a translated page bitmap. Detection / OCR / text-removal run
on-device; translation calls a cloud LLM (OpenAI-compatible).

This module is reader-agnostic: its only job is `translatePage(bitmap) -> PageResult`. Overwriting
files, markers, resume, and cross-page batching are the **caller's** responsibility (see
[Result handling](#result-handling)). The product reader ([Yakuyomi](https://github.com/joyeli/Yakuyomi),
a mihon fork) consumes this via Gradle composite build.

> Group/version: `li.joye.yakuyomi:engine`. Min SDK 26.

---

## Quick start

```kotlin
// 1. Point at the three model files on local disk (see "Models").
val models = ModelSet(
    detector   = "/path/comictextdetector.pt.onnx",
    ocr        = "/path/ocr_48px_ctc.onnx",
    inpainter  = "/path/lama-manga.onnx",
)
// …or let the engine pick them out of a folder listing:
val models = ModelSet.resolve(localOnnxFiles /* List<Pair<name, path>> */) ?: return // not ready

// 2. Load the OCR alphabet (bundled with the engine assets) and your API key.
val alphabet: List<String> = assets.open("models/alphabet-all-v5.txt").bufferedReader().readLines()
val apiKey = "<deepseek key>"   // null/blank = detect+OCR+inpaint only, no translation (debug)

// 3. Create the engine and translate. `use { }` releases the native ONNX sessions.
Yakuyomi.create(models, alphabet, apiKey).use { engine ->
    when (val r = engine.translatePage(pageBitmap)) {
        is PageResult.Translated -> writeBack(r.page)        // success → overwrite + mark done
        is PageResult.Skipped    -> { /* nothing to translate → keep original */ }
        is PageResult.Failed     -> { /* error → keep original, retry later */ }
    }
}
```

`translatePage` is a `suspend` function — call it from a background dispatcher. A single engine
instance is **not** concurrency-safe; do one page at a time per instance.

---

## Models (BYOM)

The engine ships **no model weights**; the host supplies three ONNX files + the OCR alphabet:

| Role | File (typical name) | What it does | Source |
|------|---------------------|--------------|--------|
| detector | `comictextdetector.pt.onnx` | text boxes + per-pixel stroke mask | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| ocr | `ocr_48px_ctc.onnx` | 48px CTC Japanese OCR | manga-image-translator |
| inpainter | `lama-manga.onnx` | LaMa text removal | [Koharu](https://github.com/mayocream/koharu) |

`ModelSet.resolve(files)` maps a flat `(filename, localPath)` listing to the three roles by name
(`detect`/`comictext` → detector, `ocr` → ocr, `lama`/`inpaint` → inpainter) and returns `null` if
any is missing — use that as your "ready to translate?" check.

**Paths must be local files**, not SAF/content URIs: ORT does `createSession(path)` into native
memory. Don't read weights into the JVM heap (`readBytes()`) — the heap is capped at ~512 MB
regardless of device RAM and will OOM. If your source is SAF, copy to `filesDir` first, then pass
the path.

---

## Configuration

Everything is a `data class` with sane defaults — override only what you need:

```kotlin
val config = EngineConfig(
    ocr     = OcrConfig(minProb = 0.5f),                 // drop low-confidence OCR
    inpainter = InpainterConfig(method = "boxfill"),     // "boxfill" | "lama"
    render  = RenderConfig(orientation = TextOrientation.AUTO),
    translator = TranslatorConfig(model = "deepseek-chat", batchSize = 8),
)
Yakuyomi.create(models, alphabet, apiKey, config)
```

Notable defaults (full list in `Config.kt`):

- `OcrConfig.useXnnpack = false` — **must stay off**: XNNPACK miscomputes the 48px CTC model on real
  hardware (OCR returns empty). Detector/inpainter use XNNPACK fine.
- `InpainterConfig.method = "boxfill"` — nearest-colour fill on the stroke mask; fastest and ties
  the most expensive LaMa mode on quality. `"lama"` + `wholeImage` toggles the alternatives.
- `RenderConfig.orientation = AUTO` — follow each region's detected direction (vertical/horizontal),
  then rotate along the region's skew angle to match the source.
- `TranslatorConfig.provider = "deepseek"` — any OpenAI-compatible endpoint (set `apiBase`/`model`).

### Language pair (not fixed to JP→CHT)

Default is Japanese → Traditional Chinese, but **any pair works** — set these together:

```kotlin
translator = TranslatorConfig(
    toLangName   = "English",     // target — the LLM translates into this
    fromLangName = "Korean",      // source label for the prompt ("" = let the LLM infer)
    sampleSource = "<|1|>…",      // few-shot example in the source language ("" = no example)
    sampleTarget = "<|1|>…",      // …and its translation, in the target language
)
```

The **source** is ultimately whatever your OCR model recognises — the bundled 48px CTC is Japanese;
load a different OCR model + alphabet (BYOM) to read another language. The **target** is purely the LLM
prompt. Keep `toLangName` and the few-shot in the same target language, or the example biases output.

---

## Result handling

`translatePage` returns a `PageResult` (not a bare bitmap) so the caller can honour the core
invariant: **never overwrite the original with something worse.**

| Variant | Meaning | What the caller should do |
|---------|---------|---------------------------|
| `Translated(page, stats)` | success | overwrite the file, write a "translated" marker |
| `Skipped(reason, stats)`  | nothing translatable (no text / OCR empty / all filtered) | keep original, mark skipped, don't retry |
| `Failed(reason)`          | error (network/429/exception) | keep original, **no** marker, retry later |

Per-region resilience is built in: if one bubble fails to translate it keeps its original Japanese
and the rest of the page still renders. `PageStats` carries per-stage timings for profiling.

---

## Lifecycle & threading

- `TranslationEngine : AutoCloseable`. `close()` releases the detector/ocr/inpainter ONNX sessions
  (native memory). Always `use { }` or `close()`.
- `translatePage` is `suspend`, one page at a time, **not** concurrency-safe per instance.
- The input bitmap is not recycled by the engine; `Translated.page` is a **new** bitmap.

---

## Pipeline (what happens inside)

```
page ─ Detector ─→ lines + stroke mask
     ─ Ocr ──────→ Japanese text per line   (mutates lines in place)
     ─ Grouping ─→ regions (two-stage: connect + MST-split; reading order; skew angle)
     ─ Translator → Traditional Chinese      (per-page, no rolling context; optional)
     ─ TextFilter → drop blank/digit/regex/untranslated regions
     ─ Inpainter ─→ erase original text
     ─ Renderer ──→ typeset translation (vertical/horizontal, angle-aware)
     → translated page bitmap
```

See [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the design rationale and the
desktop-parity validation story. Behaviour is aligned with
[manga-image-translator](https://github.com/zyddnys/manga-image-translator).

---

## Advanced: direct component access

The factory is the recommended path. For per-stage debugging (e.g. a detection overlay) you can
construct the stages yourself and assemble a `Pipeline` — but then **you** own their lifecycle:

```kotlin
val detector = Detector(models.detector, config.detector)
val detection = detector.detect(page)   // lines + textMask, draw your overlay
// …
detector.close()                        // you must close what you create
```

Pure helpers (`Geometry`, `ImageOps`, `TextFilter`) are `internal` — not part of the public API.
