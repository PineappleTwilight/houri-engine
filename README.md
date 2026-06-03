<div align="center">

# Yakuyomi（訳読み）— Engine

**On-device manga-translation engine — detection / OCR / text-removal (ONNX Runtime) + cloud-LLM translation**

Japanese → Traditional Chinese by default — **any source/target pair is configurable**

**English** ｜ [中文](README_zh.md)

> **This repo (`yakuyomi-engine`) is the translation engine.** The reader app — **Yakuyomi**, a
> [mihon](https://github.com/mihonapp/mihon) fork — is a separate repo that pulls this engine in as a
> submodule (see [Repository layout](#repository-layout)).
>
> **Status: the engine works end-to-end (M0–M3)** on a real device; reader integration (M4) is in progress.

</div>

---

## What it is

Yakuyomi is a manga reader with **built-in AI translation**, based on [mihon](https://github.com/mihonapp/mihon). The work splits in two:

- **Text detection, OCR, text removal** → run **on-device**, offline (ONNX Runtime) — this is the **engine** (this repo).
- **Translation** → handed to a **cloud LLM** (DeepSeek by default, OpenAI-compatible).

The top priority is **efficiency / throughput**: a chapter's pages are translated concurrently to minimize end-to-end time.

## Repository layout

The project spans two repos:

| Repo | Role |
|---|---|
| **`yakuyomi-engine`** (this repo) | the on-device translation engine: `:engine` (detection / OCR / translation / text-removal / typesetting, exposing only `translatePage(page): PageResult`) + a throwaway `:app-sandbox` for isolated testing + the `parity/` validation harness. Reader-agnostic, independently testable. |
| **`Yakuyomi`** — a [mihon](https://github.com/mihonapp/mihon) fork | the reader app: mihon rebranded, with the download-pipeline hook + translation settings + model management. It pulls in `yakuyomi-engine` as a **git submodule**, built from source via Gradle `includeBuild`. |

Why split: the engine stays clean and testable on its own, while the app is a genuine mihon fork (so it's eligible for mihon's fork network). Engine changes are committed here; the app bumps the submodule pointer.

## How it's built: vibecoding

This project is built by **vibecoding** — a human drives direction, trade-offs and review; the code is written mostly by AI ([Claude Code](https://claude.com/claude-code)).

The engine is a from-scratch Kotlin + ONNX Runtime implementation, and **this repo contains no manga-image-translator source code**; what it aligns to m-i-t is its behaviour and prompts (see [below](#aligning-with-manga-image-translator)), while the models are third-party (see [Models & sources](#models--sources)).

## Why "semi-offline"

Detection / OCR / text removal run on-device, so those steps work offline; only translation goes to a cloud LLM, so **translation needs a network connection while it runs**. A deliberate trade-off — the other stages stay offline and the engine can ship independently, at the cost of semi-offline translation.

## Highlights

- 🔍 **On-device detection**: comic-text-detector, emitting text-line boxes **plus a per-pixel text-stroke mask (seg)**.
- 🔠 **On-device OCR**: 48px CTC model (CPU-only; XNNPACK miscomputes this model, so it's disabled).
- 🌐 **Cloud-LLM translation**: reuses m-i-t's prompt and protocol; per-page, in-batch concurrency, `Semaphore` rate-limiting (avoids 429s).
- 🧹 **On-device text removal**: default **box-fill nearest-colour** — uses the seg thin-stroke mask, fills only the glyph strokes, each pixel taking its nearest background colour (multi-colour / gradient / text-over-art never smear into a colour block); **LaMa** (whole-page / per-region) is also selectable.
- ✍️ **Typesetting**: pure text-box layout — vertical / horizontal, adaptive font size, **vertical centering, stroke width scaled to font size, line-head kinsoku**, punctuation rotation, automatic black/white text (by post-removal background luminance).
- 🇹🇼 **Traditional-Chinese output**: the prompt enforces Taiwan-style Traditional Chinese, entirely via the LLM (**no OpenCC post-processing**; occasional wording slips accepted).
- 🔑 **BYOK (bring your own key)**: provider / model / API base / key / target language are all configurable; **no key is bundled**; keys live in the Android Keystore.
- 📦 **BYOM (bring your own model)**: ONNX weights aren't packed into the APK; the user points the app at a local folder (loaded off-heap, dodging the JVM heap cap).
- 💾 **Overwrite-in-place, resumable**: a page is overwritten only when translation succeeds, with per-page completion state; on failure / no text the original is kept — **never overwrite the library with something worse than the original**.

## A page's data flow

```
Page Bitmap
 → Detect (ONNX)    → text-line boxes + seg stroke mask; sorted by m-i-t's coordinate heuristic
 → OCR    (ONNX)    → whole-block recognition per box → Japanese sourceText
 → Group            → merge nearby, aligned lines into bubble regions (ported m-i-t merge rule)
 → Translate (LLM)  → per-page, concurrent, no rolling context → targetText (pasted back by block ID)
 → Remove (ONNX/CV) → box-fill nearest-colour (default) / LaMa inpaint
 → Typeset (Canvas) → vertical / horizontal, centered, stroke, kinsoku
 → Translated Bitmap
```

The engine exposes a single entry point, `translatePage(page): PageResult` (translated / skipped / failed); **overwriting the original, markers, resume, and cross-page batch concurrency** are the reader app's job (the Yakuyomi fork's download worker).

## Tech choices

| Item | Choice | Notes |
|---|---|---|
| Reader base | **mihon** (the Yakuyomi app is a fork) | integration sits at the download layer; the reader is untouched |
| Inference | **pure Kotlin + ONNX Runtime** (`onnxruntime-android`, with XNNPACK) | NNAPI deprecated, not used |
| Acceleration | XNNPACK/CPU → int8 quantization → QNN / LiteRT (NPU/GPU) when needed | NPU is the future lever to cut LaMa's cost |
| Text removal (default) | **box-fill nearest-colour** | instant, follows local background, no colour block |
| Text removal (optional) | **LaMa** (whole-page / per-region) | see [Models & sources](#models--sources) |
| Translation | cloud LLM, reusing m-i-t `chatgpt.py`'s prompt + protocol | OpenAI-compatible |
| Default provider | **DeepSeek** | fully changeable (BYOK) |

## Translation providers (BYOK)

v1 first supports the **OpenAI-compatible** group (one HTTP client covers all):

- `openai`, `deepseek`, `groq`, `custom_openai` (OpenRouter / LM Studio / self-hosted)
- then `gemini`, and later the non-LLM cloud MTs (DeepL / Caiyun / Youdao / Baidu / Papago) as separate adapters.

The settings screen exposes: provider, model, API base (for custom_openai), API key (one slot per provider, stored in Keystore), and target language. DeepSeek by default, all changeable.

## Models & sources

ONNX weights are **not committed and not packed into the APK** (BYOM). Provenance:

| Stage | Model | Source |
|---|---|---|
| Detection | comic-text-detector (outputs `blk` / `seg` / `det`) | ONNX from [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector) (manga-image-translator ecosystem) |
| OCR | 48px CTC | weights from [manga-image-translator](https://github.com/zyddnys/manga-image-translator), exported to ONNX by us via `torch.onnx.export` |
| Text removal | LaMa (manga fine-tune) | **`lama-manga.onnx` from [Koharu (mayocream/koharu)](https://github.com/mayocream/koharu)**; underlying architecture [advimman/LaMa](https://github.com/advimman/lama) |
| Fonts | Noto Sans / Serif CJK TC, Source Han | for CJK rendering, redistributable (OFL / Apache) |

ONNX export and pipeline details were informed by [Koharu](https://github.com/mayocream/koharu) (Rust + ONNX). Redistribution terms for each model / font are pending the [licensing audit](#license).

## Aligning with manga-image-translator

m-i-t is the spec, not a master to be copied line-for-line. We pin one upstream commit (see `.upstream-ref`) and align in three layers:

1. **Copy verbatim** — prompt & protocol, per-stage parameters / thresholds, config schema, model selection & processing order, provider scope.
2. **Match behaviour, implement freely** — detection post-processing, coordinate back-projection, seg mask generation, line grouping, reading order, concurrent translation (criterion: same input → near-identical output).
3. **Informed divergence (recorded)** — platform-forced or deliberate trade-offs, e.g. ORT inference, dropping CUDA, box-fill nearest-colour for removal, rolling context off by default.

Every ported file is headed with `// ported from <python path> @ <commit>`, and is validated against the Python output by the `parity/` harness before being wired into the pipeline.

## Roadmap

Development happens in this engine repo (with a standalone `:app-sandbox`); the engine is decoupled (`:engine`, exposing only `translatePage`) and is consumed by the **Yakuyomi** mihon fork as a submodule.

| Milestone | Scope | Status |
|---|---|---|
| **M0** | sandbox + ONNX detector, draw text boxes on a page (verify XNNPACK on a real device) | ✅ |
| **M1** | wire OCR (48px CTC), overlay the recognized Japanese | ✅ |
| **M2** | wire LLM translation (DeepSeek, per-page concurrency) + cover text → end-to-end working | ✅ |
| **M3** | text removal (box-fill nearest-colour / LaMa) + typesetting (centering / stroke / kinsoku); engine consolidated into `translatePage` | ✅ |
| **M4** | hook the engine into the **Yakuyomi** (mihon) fork's download pipeline, model download management, quantization / perf, fast / quality modes | ⏳ |

## Privacy

- **BYOK**: no API key is bundled; you provide your own, stored in the Android Keystore.
- **On-device processing**: detection / OCR / text removal never leave the device.
- **Translation goes online**: the OCR'd text is sent to **the LLM provider you configure** for translation — check that provider's data policy yourself.

## Credits

This project stands on the shoulders of:

- [mihon](https://github.com/mihonapp/mihon) (the reader the Yakuyomi app is forked from, Apache-2.0)
- [manga-image-translator](https://github.com/zyddnys/manga-image-translator) (behavioural spec & prompts for the translation pipeline)
- [Koharu (mayocream/koharu)](https://github.com/mayocream/koharu) (the `lama-manga.onnx` removal model; ONNX-export / pipeline reference)
- [comic-text-detector](https://github.com/dmMaze/comic-text-detector), [manga-ocr](https://github.com/kha-white/manga-ocr), [LaMa](https://github.com/advimman/lama) (models / architectures)
- Noto Sans / Serif CJK, Source Han (CJK rendering fonts)
- [Claude Code](https://claude.com/claude-code) (vibecoding development)

## License

**TBD.** The code in this repo is **self-implemented in Kotlin / ORT** (not a port of m-i-t source), but the project as a whole still **reuses m-i-t's prompts and parameter defaults**, ships as a fork of mihon (Apache-2.0), and uses several third-party model weights / fonts. A licensing audit (m-i-t terms, each model / font license, model hosting) will be completed before any public release, at which point a proper `LICENSE` will be added. Until then, assume no distribution license.

---

<div align="center">
<sub>訳読み — read the panels that haven't been translated yet.</sub>
</div>
