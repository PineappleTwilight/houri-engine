# `parity/` — desktop validation harness

**English** · [中文](README_zh.md)

Not shipped. A developer-only Python harness that runs the same pipeline stages as the Kotlin
`:engine`, so we can check the on-device port matches the reference
([manga-image-translator](https://github.com/zyddnys/manga-image-translator), m-i-t) before trusting
it on a device.

The engine re-implements m-i-t (Python/torch) in Kotlin/ONNX, and that port can't be diffed line for
line, so correctness means "same input, close output". These scripts produce the reference output,
and for grouping an automated cross-language assertion. See
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md#the-two-halves).

---

## Setup

```bash
pip install -r parity/requirements.txt    # numpy, opencv-python, onnxruntime, pillow,
                                          # networkx, shapely  (torch/opencc only for tools)
```

All paths live in one place — **`parity/paths.py`** — and the machine-specific ones are
**env-overridable**, so a different machine / CI / public user can run without editing scripts:

| What | `paths.py` | env override | default |
|------|-----------|--------------|---------|
| ONNX models | `MODELS` | — (repo-relative) | `engine/src/main/assets/models` |
| OCR alphabet | `ALPHABET` | `YAKU_ALPHABET` / `YAKU_OCR_CTC_DIR` | `/tmp/ocr-ctc/alphabet-all-v5.txt` |
| OCR checkpoint | `OCR_CKPT` | `YAKU_OCR_CKPT` / `YAKU_OCR_CTC_DIR` | `/tmp/ocr-ctc/ocr-ctc.ckpt` |
| m-i-t clone | `MIT_CLONE` | `YAKU_MIT_CLONE` | `/mnt/d/Gits/manga-image-translator` |
| test pages + m-i-t outputs | `RAW_DIR` / `MIT_DIR` | `YAKU_TEST_DIR` | `~/OneDrive/Manga/yakuyomi/test/{raw,mit}` |
| API key | `API_KEYS` | — (repo root, gitignored) | `api-keys.properties` (`DEEPSEEK_API_KEY=`) |

```bash
# e.g. point at a different test dir + m-i-t clone, no script edits:
YAKU_TEST_DIR=~/manga-test YAKU_MIT_CLONE=~/src/mit python3 pipeline_parity.py raw/002.jpg
```

Outputs land in `parity/out/` (cached JSON + comparison PNGs; gitignored).

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
