# Architecture

English ｜ [中文](ARCHITECTURE_zh.md)

How Yakuyomi translates a page, why the project is split the way it is, and how the on-device engine relates to the desktop validation harness. For the engine's API, see [`engine/README.md`](../engine/README.md).

## The two halves

| | On-device (`engine/`, Kotlin) | Desktop (`parity/`, Python) |
|---|---|---|
| Role | the product; runs on the phone | a validation harness; runs on a laptop |
| Stack | Kotlin, ONNX Runtime, Android Canvas | Python, numpy/cv2, onnxruntime, PIL |
| Ships | yes (the library) | no (dev only) |
| Purpose | translate pages | check the Kotlin port against the reference |

The engine is the deliverable. The parity harness exists because the engine re-implements [manga-image-translator](https://github.com/zyddnys/manga-image-translator) (m-i-t, Python/torch) in Kotlin/ONNX, and that port can't be diffed line for line. The harness runs the same stages in Python so we can confirm "same input, close output" before trusting the Kotlin version. See [`parity/README.md`](../parity/README.md).

## Per-page data flow

```
page bitmap
  Detector (ONNX)    text lines (rotated quads) + per-pixel stroke mask
  Ocr (ONNX)         Japanese text per line (48px CTC, greedy decode)
  Grouping (Kotlin)  lines -> bubble regions (connect, then MST-split; reading order; skew angle)
  Translator (LLM)   target text, one request per page, no rolling context
  TextFilter         decide which regions have a usable translation
  Inpainter (ONNX)   erase the original text (flat-fill, or LaMa)
  Renderer (Canvas)  draw the translation back (vertical/horizontal, rotated to region angle)
translated page bitmap
```

Each stage in one line:

- **Detector** — comic-text-detector. Letterbox preprocess, ONNX, then boxes (NMS + unclip) and a `seg` stroke mask. Text removal uses the mask so it erases strokes, not solid rectangles.
- **Ocr** — 48px CTC. Crops each line (perspective-corrected, vertical lines rotated), recognizes it, greedy-decodes against the alphabet, drops lines below `minProb`. Lines run concurrently (see below).
- **Grouping** — turns lines into bubble-sized regions in two stages: a permissive connect pass, then an MST split that breaks apart neighbours connected only transitively. The split is what stops dense bubbles from merging into one block. It also computes each region's reading order and skew angle.
- **Translator** — m-i-t's `chatgpt.py` prompt and protocol over an OpenAI-compatible call, one request per page, no cross-page context. A region whose translation fails keeps its source text. The language pair is configurable: target via `toLangName`, source via the OCR model plus a prompt label. Default is Japanese to Traditional Chinese, not hardcoded. Providers are presets, all OpenAI-compatible (Gemini via its compat endpoint), with each provider's model list fetched live — see [PROVIDERS.md](PROVIDERS.md).
- **TextFilter** — m-i-t's post-translation filter. A region is "usable" when its translation isn't blank, isn't bare digits, doesn't match the filter regex, and isn't identical to the source.
- **Inpainter** — three modes: `boxfill` (flat-fill the masked area with the local background colour) and `auto` with `wholeImage` either on (one whole-image LaMa pass) or off (per-region LaMa). Auto flat-fills clean bubbles and sends only on-art text to LaMa. Default is auto with `wholeImage` on.
- **Renderer** — text-box typesetting, no bubble flood-fill: font auto-sizing, kinsoku line breaks, vertical or horizontal, text colour from the cleaned background luminance, and canvas rotation along the region's skew angle.

`Pipeline.kt` holds the orchestration and the invariant that the page is never overwritten with something worse than the original.

## Concurrency

Three places, for throughput:

- **OCR** runs a page's lines concurrently, each on one thread. The strips are small and don't saturate a multi-threaded inference, so filling the cores with separate lines is faster.
- **Text removal overlaps translation.** Removal needs only the OCR'd regions, which are known before translation returns, so the inpaint runs on a background coroutine while the LLM request is in flight. A page saves whichever of the two is shorter. This is why failed regions keep their re-pasted source text rather than the untouched original image: decoupling removal from the translation result is what lets them overlap.
- **Cross-page** batching is the reader app's job: a chapter's pages each fire a request bounded by a `Semaphore`, and the download worker translates ahead of where you read.

## Aligning with manga-image-translator

The engine aligns on behaviour, not source. Three layers, tightest first:

1. **Copy verbatim.** Language-independent data: the prompt and protocol, per-stage thresholds and defaults, the config schema, model choice, and processing order. This reuses m-i-t's tuning instead of re-deriving it.
2. **Match behaviour, implement freely.** The *what* tracks m-i-t; the *how* follows Kotlin/ONNX idioms. Detection post-processing, coordinate inverse-mapping, mask generation, the grouping two-stage, reading order, concurrent translation. The test is "same input, close output".
3. **Informed divergence, recorded in code.** Platform-forced or deliberate trade-offs: ONNX instead of torch, no CUDA, no rolling translation context, the text-removal mode ladder.

Forced rewrites, since everything else is ported as is: torch to ORT `session.run`; cv2/PIL to Bitmap/Canvas/hand-rolled; numpy to Kotlin; async httpx/CLI/YAML to OkHttp, coroutines, and a small config loader; the manga-ocr autoregressive decode; and any CUDA/GPU assumption.

## On-device realities

Learned by running on real hardware:

- **XNNPACK miscomputes the OCR model**, so it returns empty text. `OcrConfig.useXnnpack` stays off (CPU only). The detector and inpainter run on XNNPACK fine.
- **Load models off-heap.** `createSession(path)` reads weights into native memory. Reading them into the JVM heap with `readBytes()` hits the per-app heap cap (around 512 MB regardless of device RAM) and OOMs. BYOM copies the picked file to `filesDir` first, then passes the path.
- **Preprocessing must match the Python export exactly** — resize, normalize, NCHW order. This is the main source of silent divergence, and most of why the parity harness exists.
- **Inference threads.** Detection and LaMa use a thread count tuned to the device's big cores (six on the Snapdragon 8 Gen 3 test device); adding the slow efficiency cores makes a pass slower, not faster.

## Repo layout

```
engine/        Android library, the on-device pipeline (the product)
  src/main/kotlin/li/joye/yakuyomi/engine/
    Yakuyomi.kt, TranslationEngine.kt, ModelSet.kt   public entry (facade + types)
    Pipeline.kt, Detector.kt, Ocr.kt, Grouping.kt,
    LlmTranslator.kt, Inpainter.kt, Renderer.kt      stages
    LlmProviders.kt, LlmModels.kt                    provider presets + live model-list fetch
    Geometry.kt, ImageOps.kt, TextFilter.kt          internal helpers
  src/test/kotlin/…                                  JVM unit tests
app-sandbox/   sandbox test app (device timing, comparison images)
parity/        desktop Python validation harness (not shipped)
docs/          this file, plus the parameter reference
```

The reader app (the [Yakuyomi](https://github.com/joyeli/Yakuyomi) mihon fork) pulls `engine` in as a git submodule and Gradle composite build, and adds only the integration layer: the download hook, settings, and model management.
