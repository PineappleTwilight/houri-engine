# Architecture

English ｜ [中文](ARCHITECTURE_zh.md)

How Yakuyomi translates a page, why the project is split the way it is, and how the on-device engine relates to the desktop validation harness. For the engine's API, see [`engine/README.md`](../engine/README.md).

## The two halves

| | On-device (`engine/`, Kotlin) | Desktop (`parity/`, Python) |
|---|---|---|
| Role | the product; runs on the phone | a validation harness; runs on a laptop |
| Stack | Kotlin, NCNN + ONNX Runtime, Android Canvas | Python, numpy/cv2, onnxruntime, PIL |
| Ships | yes (the library) | no (dev only) |
| Purpose | translate pages | check the Kotlin port against the reference |

The engine is the deliverable. The parity harness exists because the engine re-implements [manga-image-translator](https://github.com/zyddnys/manga-image-translator) (m-i-t, Python/torch) in Kotlin (NCNN + ONNX Runtime), and that port can't be diffed line for line. The harness runs the same stages in Python so we can confirm "same input, close output" before trusting the Kotlin version. See [`parity/README.md`](../parity/README.md).

## Per-page data flow

```
page bitmap
  Detector (NCNN)    text lines (rotated quads) + per-pixel stroke mask
  Ocr (ONNX int8)    Japanese text per line (48px CTC, greedy decode)
  Grouping (Kotlin)  lines -> bubble regions (connect, then MST-split; reading order; skew angle)
  Translator (LLM)   target text, one request per page, no rolling context
  TextFilter         decide which regions have a usable translation
  Inpainter (NCNN)   erase the original text (flat-fill, or AOT-GAN)
  Renderer (Canvas)  draw the translation back (vertical/horizontal, rotated to region angle)
translated page bitmap
```

Each stage in one line:

- **Detector** — DBNet (m-i-t's default detector: ResNet34 + DB head), NCNN, fp16. Preprocess is `resize_aspect` to 1024 on the long side, padded to a multiple of 256 — a *rectangular* input, which also keeps clear of an NCNN heap-corruption bug at square sizes 832–992. Post-process is the DB one: sigmoid, binarize at 0.5, connected components, `minAreaRect`, drop boxes scoring below 0.7, unclip 2.3. The second output becomes the `seg` stroke mask (threshold 0.12, then dilated). Text removal uses the mask so it erases strokes, not solid rectangles. This replaced comic-text-detector, which is gone: on-device, DBNet reads back 1.6–2.5× more text.
- **Ocr** — 48px CTC, a dynamically-quantized int8 ONNX model (~3.6× faster than fp32 on ARM, 165→44 MB, 96.7% CTC parity). Crops each line (hand-rolled bicubic perspective warp, vertical lines rotated), recognizes it, greedy-decodes against the alphabet, drops lines below `minProb`. Before cropping, the detected quad is expanded by `stripPad` (4 px): a too-tight box clips the last character, the CTC then returns an empty string, and the whole region gets dropped untranslated. The padding only affects the OCR crop — the detected box itself is unchanged, so the stroke-based removal mask is untouched. Lines run concurrently (see below).
- **Grouping** — turns lines into bubble-sized regions in two stages: a permissive connect pass, then an MST split that breaks apart neighbours connected only transitively. The split is what stops dense bubbles from merging into one block. It also computes each region's reading order and skew angle.
- **Translator** — m-i-t's `chatgpt.py` prompt and protocol over an OpenAI-compatible call, one request per page, no cross-page context. A region whose translation fails keeps its source text. The language pair is configurable: target via `toLangName`, source via the OCR model plus a prompt label. Default is Japanese to Traditional Chinese, not hardcoded. Providers are presets, all OpenAI-compatible (Gemini via its compat endpoint), with each provider's model list fetched live — see [PROVIDERS.md](PROVIDERS.md).
- **TextFilter** — m-i-t's post-translation filter. A region is "usable" when its translation isn't blank, isn't bare digits, doesn't match the filter regex, and isn't identical to the source.
- **Inpainter** — two modes: `boxfill` (**fast** removal: flat-fill the masked area with the nearest background colour — quick but coarse) and AOT-GAN (**AI** removal, the default: an NCNN whole-image pass at 768 px that reconstructs the background for higher quality). The old per-region (per-tile) path is gone; LaMa is retired.
- **Renderer** — text-box typesetting, no bubble flood-fill: font auto-sizing, kinsoku line breaks, vertical or horizontal, text colour from the cleaned background luminance, and canvas rotation along the region's skew angle.

`Pipeline.kt` holds the orchestration and the invariant that the page is never overwritten with something worse than the original.

## Concurrency

Three places, for throughput:

- **OCR** runs a page's lines concurrently, each on one thread. The strips are small and don't saturate a multi-threaded inference, so filling the cores with separate lines is faster.
- **Text removal overlaps translation.** Removal needs only the OCR'd regions, which are known before translation returns, so the inpaint runs on a background coroutine while the LLM request is in flight. A page saves whichever of the two is shorter. This is why failed regions keep their re-pasted source text rather than the untouched original image: decoupling removal from the translation result is what lets them overlap.
- **Cross-page** batching is the reader app's job, not the engine's: the fork's `PageTranslator` runs a chapter's pages through `translatePage` bounded by a `Semaphore(pipelineDepth)`, and the download worker translates ahead of where you read. (`TranslatorConfig.batchSize` / `batchConcurrent` remain only as a mirror of m-i-t's config schema — nothing in the engine reads them.)

## Aligning with manga-image-translator

The engine aligns on behaviour, not source. Three layers, tightest first:

1. **Copy verbatim.** Language-independent data: the prompt and protocol, per-stage thresholds and defaults, the config schema, model choice, and processing order. This reuses m-i-t's tuning instead of re-deriving it.
2. **Match behaviour, implement freely.** The *what* tracks m-i-t; the *how* follows Kotlin/NCNN/ONNX idioms. Detection post-processing, coordinate inverse-mapping, mask generation, the grouping two-stage, reading order, concurrent translation. The test is "same input, close output".
3. **Informed divergence, recorded in code.** Platform-forced or deliberate trade-offs: NCNN and ONNX instead of torch, no CUDA, no rolling translation context, the two text-removal modes.

Forced rewrites, since everything else is ported as is: torch to NCNN or ORT `session.run`; cv2/PIL to Bitmap/Canvas/hand-rolled; numpy to Kotlin; async httpx/CLI/YAML to OkHttp, coroutines, and a small config loader; the manga-ocr autoregressive decode; and any CUDA/GPU assumption.

## On-device realities

Learned by running on real hardware:

- **XNNPACK miscomputes the OCR model**, so it returns empty text. `OcrConfig.useXnnpack` stays off (CPU only). OCR is the only ONNX Runtime model now; the detector and inpainter run on NCNN.
- **Load models off-heap.** `createSession(path)` reads weights into native memory. Reading them into the JVM heap with `readBytes()` hits the per-app heap cap (around 512 MB regardless of device RAM) and OOMs. BYOM copies the picked file to `filesDir` first, then passes the path.
- **Preprocessing must match the Python export exactly** — resize, normalize, NCHW order. This is the main source of silent divergence, and most of why the parity harness exists.
- **Inference threads.** Detection and AOT-GAN inpainting use a thread count tuned to the device's big cores (six on the Snapdragon 8 Gen 3 test device); adding the slow efficiency cores makes a pass slower, not faster.
- **GPU/NPU was tried and does not work for these models — everything runs on the CPU.** NCNN's Vulkan path *miscomputes* the AOT-GAN inpaint model (garbage output at both fp16 and fp32, worsening with tile size — a shader-level bug for this op mix on Adreno), and the detector loses to CPU on Vulkan anyway; LiteRT cannot even compile these models to the GPU; an NPU (Hexagon) backend would need int8 QDQ, which the OCR model's dynamic width blocks — and the detector quantized to int8 stopped producing boxes at all, with no speed gain on ARM, so it stays fp16. So all three models run on the CPU — detection and text removal on NCNN's mobile kernels (NEON/Winograd), OCR on ONNX Runtime's CPU — which is fast enough (Snapdragon 8 Gen 3, 6 representative pages: detection + OCR 10.3 s total, 160 of 161 detected boxes read back).

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
