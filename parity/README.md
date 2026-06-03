# `parity/` — desktop validation harness

**English** · [中文](README_zh.md)

**Not shipped.** This is a developer-only Python harness that runs the same pipeline stages as the
Kotlin `:engine`, so we can check the on-device port behaves like the reference
([manga-image-translator](https://github.com/zyddnys/manga-image-translator), m-i-t) — *before*
trusting it on a device.

Why it exists: the engine re-implements m-i-t (Python/torch) in Kotlin/ONNX. That port can't be
diffed line-for-line, so correctness is "same input → close output". These scripts produce that
reference output (and, for grouping, an automated cross-language assertion). See
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md#the-two-halves).

---

## Setup

```bash
pip install -r parity/requirements.txt    # numpy, opencv-python, onnxruntime, pillow,
                                          # networkx, shapely  (torch/opencc only for tools)
```

External inputs the scripts expect (developer machine):

| What | Where (current) | Notes |
|------|-----------------|-------|
| ONNX models | `engine/src/main/assets/models/*.onnx` | same files the engine loads |
| OCR alphabet | `/tmp/ocr-ctc/alphabet-all-v5.txt` | char dictionary for CTC decode |
| m-i-t clone | `/mnt/d/Gits/manga-image-translator` | source of the vendored spec / `SegDetectorRepresenter` |
| test pages | `~/OneDrive/Manga/yakuyomi/test/raw/*.jpg`, m-i-t outputs in `…/test/mit/` | comparison fixtures |
| API key | `api-keys.properties` (repo root, gitignored) | `DEEPSEEK_API_KEY=` for translation |

Outputs land in `parity/out/` (cached JSON + comparison PNGs; gitignored).

> These paths are currently hard-coded per script — centralizing them is a follow-up.

---

## What's here

**End-to-end**
- `pipeline_parity.py <img…>` — full chain detect→OCR→group→translate→inpaint→typeset for a page.
  The main driver; writes `out/final_<name>.png` + caches intermediates.

**Per-stage parity** (run/inspect one stage)
- `ctd_reference.py [page]` — detection: faithful (m-i-t post-processing) vs simplified, side by side.
- `ocr_parity.py` — 48px CTC recognition on detected boxes.
- `group_exp.py <name…>` — grouping: our regions vs m-i-t's, drawn as boxes.
- `translate_parity.py` — OCR'd JP → DeepSeek → CHT.
- `merge_translate_parity.py` — line-merge then translate.
- `inpaint_parity.py` — LaMa text removal on regions.
- `typeset_parity.py [v|h|auto]` / `retypeset.py <name…>` — typesetting (retypeset = re-render from
  cache without re-calling the LLM; for tuning layout fast).

**Vendored spec** (ground truth, copied from m-i-t — keep in sync with `.upstream-ref`)
- `mit_grouping.py` — m-i-t's two-stage grouping (`merge_bboxes_text_region`), self-contained.
- `ctd_reference.py` — also pulls m-i-t's detection post-processing.

**Tools**
- `export_ocr_onnx.py` — export the 48px CTC checkpoint to ONNX (build-time, needs torch).
- `seg_validate.py` — inspect the detector's `seg` stroke mask at thresholds.
- `emit_grouping_fixture.py` — generate the Kotlin grouping test fixture (see below).

---

## The cross-language grouping test

The one automated parity check spans both languages:

```
emit_grouping_fixture.py                          # desktop: detect real pages, group with mit_grouping,
   → engine/src/test/kotlin/.../GroupingFixture.kt #   emit detected lines + expected regions as Kotlin
                                                   #
gradlew :engine:testDebugUnitTest                 # device-side: feed the same lines to Kotlin Grouping,
   → GroupingParityTest                            #   assert regions (bbox ±2px) + angle (±1°) match
```

So a change to the Kotlin grouping (or a re-sync of `mit_grouping.py`) is caught automatically: edit,
re-run `emit_grouping_fixture.py`, run the test. Other stages are still validated visually (compare
`out/*.png` against `…/test/mit/`).

---

## Typical loop

1. `pipeline_parity.py raw/002.jpg raw/012.jpg` — end-to-end, eyeball `out/final_*.png` vs `mit/`.
2. Tuning layout only? edit `typeset_parity.py`, `retypeset.py 002 012` (no LLM call).
3. Touched grouping? `emit_grouping_fixture.py` then `:engine:testDebugUnitTest`.
4. Synced m-i-t? bump `mit_grouping.py` / `.upstream-ref`, re-run the relevant parity, fix to green.
