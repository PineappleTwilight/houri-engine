#!/usr/bin/env python3
"""驗證用：比較「我們現在的 group()」vs「m-i-t 真正的 merge_bboxes_text_region」分組結果。
只畫區域框（不翻譯、不去字），快速看密集泡泡有沒有被正確分開。用法：group_exp.py 002 012 ..."""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onnxruntime as ort
import pipeline_parity as pp
from mit_grouping import Quadrilateral, merge_bboxes_text_region

RAW = "/home/joyel/OneDrive/Manga/yakuyomi/test/raw"
OUT = os.path.join(os.path.dirname(HERE), "parity/out")
COLS = [(255, 0, 0), (0, 160, 0), (0, 0, 255), (255, 140, 0), (150, 0, 200),
        (0, 160, 200), (200, 0, 120), (120, 120, 0), (0, 120, 120), (200, 80, 80)]


def mit_group(res, W, H):
    quads = []
    for r in res:
        pts = np.array(r["quad"], dtype=np.float64)
        fg = r.get("fg") or (0, 0, 0)
        bg = r.get("bg") or (255, 255, 255)
        quads.append(Quadrilateral(pts, r["text"], 0.9,
                                   tuple(int(c) for c in fg), tuple(int(c) for c in bg)))
    regions = []
    for txtlns, fgc, bgc in merge_bboxes_text_region(quads, W, H):
        x0 = min(t.aabb.x for t in txtlns)
        y0 = min(t.aabb.y for t in txtlns)
        x1 = max(t.aabb.x + t.aabb.w for t in txtlns)
        y1 = max(t.aabb.y + t.aabb.h for t in txtlns)
        d = Counter([t.direction for t in txtlns]).most_common(1)[0][0]
        regions.append({"dir": d, "n": len(txtlns), "bbox": [int(x0), int(y0), int(x1), int(y1)]})
    return regions


def draw(rgb, regions):
    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im)
    for i, r in enumerate(regions):
        x0, y0, x1, y1 = r["bbox"]
        dr.rectangle([x0, y0, x1, y1], outline=COLS[i % len(COLS)], width=5)
    return np.array(im)


def main():
    det = ort.InferenceSession(f"{pp.MODELS}/comictextdetector.pt.onnx", providers=["CPUExecutionProvider"])
    ocr = ort.InferenceSession(f"{pp.MODELS}/ocr_48px_ctc.onnx", providers=["CPUExecutionProvider"])
    seg_rep = pp.import_seg_rep()(thresh=0.3)
    dic = [s[:-1] for s in open(pp.ALPHABET, encoding="utf-8").readlines()]
    for name in sys.argv[1:]:
        rgb = cv2.cvtColor(cv2.imread(os.path.join(RAW, f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        quads = pp.detect(det, seg_rep, rgb)
        res = pp.ocr_all(ocr, dic, rgb, quads)
        ours = pp.group(res)
        mit = mit_group(res, W, H)
        a = draw(rgb, ours)
        b = draw(rgb, mit)
        sep = np.full((H, 8, 3), 0, np.uint8)
        cv2.imwrite(os.path.join(OUT, f"grp_{name}.png"),
                    cv2.cvtColor(np.hstack([a, sep, b]), cv2.COLOR_RGB2BGR))
        print(f"{name}: {len(quads)} 框 → 我們 {len(ours)} 區 / m-i-t {len(mit)} 區 → grp_{name}.png")


if __name__ == "__main__":
    main()
