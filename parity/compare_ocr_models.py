#!/usr/bin/env python3
"""
Frozen-quad OCR comparison: bundled int8 CTC vs PP-OCRv5 mobile_rec.

Both models are 48px CTC recognisers, run over the same 30 quads from
`fixtures/faithful_boxes.json` (page demo03.png, sha256 pinned in the fixture), each decoded with
its own alphabet since the two vocabularies are matched to the logits by index.

The crop and decode are imported from `ocr_parity.py` rather than reimplemented. That script is the
engine's own desktop validation harness, so reusing it means this comparison cannot drift from
production behaviour - an earlier hand-port of the same logic diverged enough to read plausible
garbage out of *both* models. Importing also sidesteps `minAreaRect`, whose exact corner convention
is not part of the engine's public surface.

The bundled model needs no alphabet argument here: `ocr_parity` holds its own module-level ALPHABET,
the same asset the app ships. PP-OCRv5's 18385-entry list lives Houri-side, so it is passed in.

This measures *recognition only* - no detection, inpainting or translation. It says nothing about a
page where detection missed a line, and the confidence numbers are the models' own per-character CTC
scores, not calibrated probabilities of correctness.

Usage:
    python3 parity/compare_ocr_models.py
    MTL_OCR_INT8=/path/ocr_int8.onnx MTL_OCR_PPOCRV5=/path/ppocrv5_rec.onnx \
        python3 parity/compare_ocr_models.py
"""
import argparse
import json
import os
import resource
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODELS = {
    "bundled_int8": os.environ.get("MTL_OCR_INT8", "ocr_int8.onnx"),
    "ppocrv5": os.environ.get("MTL_OCR_PPOCRV5", "ppocrv5_rec.onnx"),
}
# Houri ships PP-OCRv5's alphabet; the engine asset belongs to the bundled model.
# ROOT is <repo>/external/yakuyomi-engine, so the Houri app module is two levels up.
PP_OCR_ALPHABET = os.path.join(
    os.path.dirname(os.path.dirname(ROOT)), "yakuyomi-engine/src/main/assets/yakuyomi_alphabet_ppocrv5.txt"
)
FIXTURE = os.path.join(HERE, "fixtures", "faithful_boxes.json")


def read_alphabet(path):
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line != "\n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="engine default is 0.5; below that a line is discarded")
    ap.add_argument("--limit", type=int, default=0, help="only the first N quads (0 = all)")
    ap.add_argument("--mem-cap-gb", type=float, default=4.0)
    args = ap.parse_args()

    cap = int(args.mem_cap_gb * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    import cv2
    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, HERE)
    import ocr_parity as op
    import paths

    fx = json.load(open(FIXTURE, encoding="utf-8"))
    page = cv2.imread(os.path.join(ROOT, fx["image_path"]), cv2.IMREAD_COLOR)
    if page is None:
        sys.exit(f"cannot read {fx['image_path']}")
    ph, pw = page.shape[:2]

    dicts = {
        "bundled_int8": read_alphabet(paths.ALPHABET),
        "ppocrv5": read_alphabet(PP_OCR_ALPHABET),
    }

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False

    print(f"page {fx['image']} {pw}x{ph}   quads {len(fx['boxes'])}")

    sessions = {}
    for key, path in MODELS.items():
        if not os.path.isfile(path):
            sys.exit(f"missing model {path} (override with MTL_OCR_INT8 / MTL_OCR_PPOCRV5)")
        sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        sessions[key] = sess
        widths = [o.shape[2] for o in sess.get_outputs() if isinstance(o.shape[2], int)]
        width = max(widths) if widths else None
        if width != len(dicts[key]):
            # A mismatch would read as plausible wrong text rather than fail, so refuse outright.
            sys.exit(f"{key}: logits width {width} != alphabet {len(dicts[key])} - refusing to compare")
        print(f"  {key:<14} logits {width}  alphabet {len(dicts[key])}  ok")

    boxes = fx["boxes"][: args.limit] if args.limit else fx["boxes"]
    if args.limit:
        print(f"  --limit {args.limit}")

    strips = op.build_strips(page, boxes)
    print(f"  strips built: {len(strips)}")

    results = {}
    for key, sess in sessions.items():
        inp = sess.get_inputs()[0].name
        rows = []
        for s in strips:
            outs = sess.run(None, {inp: s["x"].astype(np.float32)})
            # Mirrors LogitsOutput.select: among rank-3 outputs the widest vocabulary is the logits.
            logits = max((o for o in outs if o.ndim == 3), key=lambda o: o.shape[-1])[0]
            colors = next((o[0] for o in outs if o.ndim == 3 and o.shape[-1] == 6), None)
            decoded = op.ctc_decode(logits, dicts[key], colors)
            text, prob = decoded[0], decoded[1]
            rows.append((s["i"], s["dir"], text, prob))
        results[key] = rows

    keys = list(MODELS)
    a, b = keys
    by_index = {k: {i: (t, p) for i, _, t, p in results[k]} for k in keys}
    indices = [s["i"] for s in strips]

    print(f"\n  {'i':<3}{'bundled_int8':<30}{'ppocrv5':<30}")
    print("  " + "-" * 63)
    for i in indices:
        ta, pa = by_index[a][i]
        tb, pb = by_index[b][i]
        mark = "  " if ta == tb else "≠ "
        print(f"  {i:<3}{mark}{ta[:28]:<30}{tb[:28]:<30}")
        if ta != tb:
            print(f"     {' ' * 33}p={pa:.3f} / {pb:.3f}")

    same = sum(1 for i in indices if by_index[a][i][0] == by_index[b][i][0])
    nonempty = {k: sum(1 for i in indices if by_index[k][i][0]) for k in keys}
    kept = {k: sum(1 for i in indices if by_index[k][i][1] >= args.min_prob) for k in keys}
    chars = {k: sum(len(by_index[k][i][0]) for i in indices) for k in keys}

    print(f"\n  strips              {len(strips)}")
    print(f"  lines read          {a}={nonempty[a]:>3}  {b}={nonempty[b]:>3}")
    print(f"  lines >= min-prob   {a}={kept[a]:>3}  {b}={kept[b]:>3}   (min_prob={args.min_prob})")
    print(f"  total characters    {a}={chars[a]:>5}  {b}={chars[b]:>5}")
    print(f"  identical lines     {same}/{len(indices)}")
    print("\n  Recognition only - no detection, inpainting or translation. Confidence is\n"
          "  the models' own mean per-character CTC score, not a probability of correctness.")


if __name__ == "__main__":
    main()