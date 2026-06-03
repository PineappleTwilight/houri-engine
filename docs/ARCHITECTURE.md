# Architecture

**English** · [中文](ARCHITECTURE_zh.md)

How Yakuyomi translates a manga page, why it's split the way it is, and how the desktop and
on-device halves relate. This is the shippable companion to the engine API docs
([`engine/README.md`](../engine/README.md)).

---

## The two halves

| | **On-device** (`engine/`, Kotlin) | **Desktop** (`parity/`, Python) |
|---|---|---|
| Role | the real product — runs on the phone | a validation harness — runs on a laptop |
| Language | Kotlin + ONNX Runtime + Android Canvas | Python + numpy/cv2 + onnxruntime + PIL |
| Ships? | yes (the library) | no (dev-only) |
| Purpose | translate pages efficiently | prove the Kotlin port behaves like the reference |

The engine is the deliverable. The parity harness exists because we re-implement
[manga-image-translator](https://github.com/zyddnys/manga-image-translator) (m-i-t, Python/torch)
in Kotlin/ONNX — a port that can't be diffed line-for-line. The desktop harness runs the *same*
stages in Python and lets us check "same input → close output" before trusting the Kotlin version.
See [`parity/README.md`](../parity/README.md).

---

## Per-page data flow

```
page bitmap
  │
  ├─ Detector (ONNX)   → text lines (rotated quads) + per-pixel stroke mask
  │
  ├─ Ocr (ONNX)        → Japanese text per line (48px CTC, greedy decode)
  │
  ├─ Grouping (Kotlin) → regions  (two-stage: loose-connect lines, then MST-split
  │                                 over-merged neighbours; reading order; skew angle)
  │
  ├─ Translator (LLM)  → Traditional Chinese, per page, no rolling context  (optional)
  │
  ├─ TextFilter        → drop blank / pure-digit / regex-matched / untranslated regions
  │
  ├─ Inpainter (ONNX)  → erase original text (boxfill nearest-colour, or LaMa)
  │
  └─ Renderer (Canvas) → typeset translation (vertical/horizontal, rotated to region angle)
  →
translated page bitmap
```

Concurrency lives at the **translation** step: a chapter's N pages each fire a request, bounded by a
`Semaphore`. Everything else is per-page and CPU-bound on-device.

Each stage in one line:

- **Detector** — comic-text-detector. Letterbox-preprocess → ONNX → boxes (NMS + unclip) and a `seg`
  stroke mask used by text-removal (erase thin strokes, not solid rectangles).
- **Ocr** — 48px CTC. Crops each line (perspective-corrected, vertical lines rotated), recognises,
  greedy-decodes against the alphabet; drops below `minProb`.
- **Grouping** — turns lines into bubble-sized regions. Two stages: a permissive *connect* pass, then
  an MST-based *split* that breaks apart neighbours connected only transitively (this is what stops
  dense bubbles merging into one block). Also computes each region's reading order and skew angle.
- **Translator** — ports m-i-t's `chatgpt.py` prompt + protocol to an OpenAI-compatible call. Per
  page, no cross-page context (efficiency-first). Per-region failures fall back to the original text.
- **TextFilter** — m-i-t's post-translation filter: don't paste back blanks, bare numbers, or
  text the LLM returned untranslated.
- **Inpainter** — `boxfill` (nearest non-mask colour per masked pixel) by default; LaMa
  (whole-image tile, or per-region windows) as alternatives.
- **Renderer** — pure text-box typesetting (no bubble flood-fill): font auto-sizing, kinsoku
  line-breaking, vertical/horizontal, auto text colour from background luminance, and canvas
  rotation along the region's skew angle.

The orchestration + the "never overwrite with something worse" invariant live in `Pipeline.kt`.

---

## Aligning with manga-image-translator

We align on **behaviour and decisions**, not source. Three layers, tightest first:

1. **Copy verbatim** — language-independent data: the translation prompt + protocol, per-stage
   thresholds/defaults, the config schema, model choice and processing order. Saves re-deriving
   m-i-t's tuning.
2. **Match behaviour, implement freely** — the *what* tracks m-i-t, the *how* follows Kotlin/ONNX
   idioms: detection post-processing, coordinate inverse-mapping, mask generation, the grouping
   two-stage, reading order, concurrent translation. Test = same input → close output.
3. **Informed divergence (recorded)** — platform-forced or deliberate trade-offs: ONNX quantization,
   no CUDA, no rolling translation context (efficiency), boxfill-first text removal. Each is noted
   in code where it diverges.

Forced rewrites (everything else is ported as-is): torch → ORT `session.run`; cv2/PIL → Bitmap /
Canvas / hand-rolled; numpy → Kotlin; async httpx/CLI/YAML → OkHttp + coroutines + own config; the
manga-ocr autoregressive decode; any CUDA/GPU assumption.

---

## On-device realities (learned on real hardware)

- **XNNPACK miscomputes the OCR model** → `OcrConfig.useXnnpack = false` (CPU only). Detector and
  inpainter are fine with XNNPACK.
- **Load models off-heap.** `createSession(path)` into native memory — never `readBytes()` into the
  JVM heap (capped ~512 MB regardless of device RAM → OOM). BYOM copies SAF → `filesDir` first.
- **Preprocessing must match the Python export bit-for-bit** — resize / normalize / NCHW. This is the
  #1 source of silent divergence; the parity harness exists largely to catch it.
- **Memory hygiene** — close `OnnxTensor`s; process one page at a time.

---

## Repo layout

```
engine/        Android library — the on-device pipeline (the product)
  src/main/kotlin/li/joye/yakuyomi/engine/
    Yakuyomi.kt, TranslationEngine.kt, ModelSet.kt   ← public entry (facade + types)
    Pipeline.kt, Detector.kt, Ocr.kt, Grouping.kt,
    LlmTranslator.kt, Inpainter.kt, Renderer.kt      ← stages
    Geometry.kt, ImageOps.kt, TextFilter.kt          ← internal helpers
  src/test/kotlin/…                                  ← JVM unit tests
app-sandbox/   throwaway test app (debug overlay, on-device timing)
parity/        desktop Python validation harness (not shipped)
docs/          this file
```

The product reader ([Yakuyomi](https://github.com/joyeli/Yakuyomi), a mihon fork) pulls `engine` in
as a git submodule + Gradle composite build, and only adds the integration layer (download hook,
settings, model management).
