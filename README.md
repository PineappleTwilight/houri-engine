# Yakuyomi — manga translation engine

On-device detection, OCR, and text removal (ONNX Runtime) plus cloud-LLM translation. Japanese to Traditional Chinese by default; any source and target language can be set.

English ｜ [中文](README_zh.md)

Status: the engine runs end to end on a real device (milestones M0–M3). Reader integration (M4) is in progress.

This repo is the **engine** (`yakuyomi-engine`). The reader app is a separate [mihon](https://github.com/mihonapp/mihon) fork that pulls the engine in as a submodule; see [Repository layout](#repository-layout).

## What it is

Yakuyomi translates manga pages. Four of the five stages run on the device with ONNX Runtime and Canvas; only translation calls out to a network LLM:

```
page bitmap
  detect    (ONNX)  text-line boxes + per-pixel stroke mask
  OCR       (ONNX)  one forward per line  ->  source text
  group             merge aligned lines into bubble regions
  translate (LLM)   one request per page, batched, rate-limited
  remove    (ONNX)  erase the original text (flat-fill or LaMa)
  typeset   (Canvas) draw the translation back
  translated bitmap
```

The engine exposes one call, `translatePage(page): PageResult` (translated / skipped / failed). Writing the file back, the "translated" marker, resume, and cross-page batching belong to the reader app.

## Goals

- **Throughput.** A reader should not stall on translation. OCR processes a page's lines concurrently, text removal overlaps the translation network wait, and the download worker translates pages ahead of where you read. A page is roughly 10–16 s on a Snapdragon 8 Gen 3, depending on the text-removal mode.
- **Configurable, and ready to be public.** Provider, model, API base, key, and language pair are all settings (bring your own key). Models are loaded from a folder you pick, not bundled (bring your own model). About 20 engine parameters are exposed; see [docs/PARAMETERS.md](docs/PARAMETERS.md).
- **Never make the library worse.** A page is overwritten only when translation succeeds. If a page has no text, or every line fails, or the network drops, the original is kept untouched. Blocks whose translation fails keep their Japanese text instead of being blanked.

## What it can do

- **Detection** — comic-text-detector. Returns text-line quads and a per-pixel stroke mask used to limit text removal to the glyphs.
- **OCR** — a 48px CTC model, one forward per line, decoded greedily. Runs on CPU (XNNPACK miscomputes this model). Lines are recognized concurrently, which on an 8-core phone cuts OCR roughly in half.
- **Translation** — a cloud LLM with the line-numbered protocol from manga-image-translator. DeepSeek by default; any OpenAI-compatible provider works. Per-page requests run concurrently under a semaphore to avoid rate limits. A failed line falls back to its source text rather than breaking the page.
- **Text removal** — three modes, trading speed for quality. Speech bubbles are always flat-filled (clean, no halo); the modes differ only in how text drawn over artwork is handled:

  | Mode | Artwork handling | Speed |
  |---|---|---|
  | BoxFill | flat-fill (becomes a colour block) | fastest |
  | Auto-whole (default) | one whole-image LaMa pass | balanced |
  | Auto-tile | per-region LaMa | slowest, sharpest |

- **Typesetting** — text-box layout, vertical or horizontal, with adaptive font size, vertical centering, outline scaled to the font, line-head kinsoku, and tilt-aware placement (text follows a slanted bubble's angle). Text colour is chosen from the cleaned background (black on light, white on dark).
- **Languages** — Japanese to Traditional Chinese out of the box. Set a different target, source, and few-shot example for any pair. Traditional-Chinese output relies on the prompt; there is no OpenCC post-processing.

## Repository layout

Two repos:

| Repo | Role |
|---|---|
| `yakuyomi-engine` (this one) | the engine: `:engine` (the pipeline, exposing only `translatePage`), a throwaway `:app-sandbox` for testing on a device, and the `parity/` desktop validation harness. No reader code. |
| `Yakuyomi` (a mihon fork) | the reader app: mihon with the download hook, translation settings, and model management. Consumes the engine as a git submodule via Gradle `includeBuild`. |

The engine stays reader-agnostic so it can be tested on its own; the app is a real mihon fork. Engine work is committed here, and the app bumps the submodule pointer.

## Models

Weights are not committed and not packed into the APK. The app loads them from a folder you choose.

| Stage | Model | Source |
|---|---|---|
| Detection | comic-text-detector | [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| OCR | 48px CTC | weights from [manga-image-translator](https://github.com/zyddnys/manga-image-translator), exported to ONNX here |
| Text removal | LaMa (manga fine-tune) | `lama-manga.onnx` from [Koharu](https://github.com/mayocream/koharu); architecture from [advimman/LaMa](https://github.com/advimman/lama) |
| Fonts | Noto Sans/Serif CJK, Source Han | CJK rendering (OFL / Apache) |

## Building

The engine builds as a standard Android Gradle library. The sandbox app (`:app-sandbox`) is the quickest way to exercise the pipeline:

```
./gradlew :app-sandbox:assembleDebug
```

Install it, point it at a folder containing the three `.onnx` models, pick test images, and run a diagnostic or a text-removal comparison. The reader app lives in the separate fork repo.

## Configuration

Every tunable parameter, its range, and the effect of changing it is documented in [docs/PARAMETERS.md](docs/PARAMETERS.md). The reader's translation settings expose the same set, grouped by stage, with the advanced knobs behind a toggle.

## How it relates to manga-image-translator

The engine is a from-scratch Kotlin + ONNX Runtime implementation. It contains no manga-image-translator source code; what it borrows is behaviour — the translation prompt and protocol, the parameter defaults, the model choice and processing order. The details, and the layered alignment policy, are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Credits

- [mihon](https://github.com/mihonapp/mihon) — the reader the app forks (Apache-2.0)
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator) — prompt and behaviour reference
- [Koharu](https://github.com/mayocream/koharu) — the `lama-manga.onnx` model and ONNX-export reference
- [comic-text-detector](https://github.com/dmMaze/comic-text-detector), [LaMa](https://github.com/advimman/lama) — models and architectures
- Noto Sans/Serif CJK, Source Han — fonts

## License

To be decided. The code here is written from scratch in Kotlin/ORT, but the project reuses manga-image-translator's prompts and defaults, forks mihon (Apache-2.0), and uses third-party model weights and fonts. A licensing audit will be done before any public release. Until then, assume no distribution license.
