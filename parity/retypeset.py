#!/usr/bin/env python3
"""只重跑排版（從快取的去字圖 + 區域），不重打 DeepSeek。用法：retypeset.py demo1 demo2 ..."""
import os, sys, json
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import typeset_parity as ts

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parity/out")

for name in sys.argv[1:]:
    im = Image.open(os.path.join(OUT, f"inpainted_{name}.png")).convert("RGB")
    dr = ImageDraw.Draw(im)
    for r in json.load(open(os.path.join(OUT, f"cache_{name}.json"), encoding="utf-8")):
        cht = r.get("cht", "")
        x0, y0, x1, y1 = r["bbox"]
        if (x1 - x0) < 8 or (y1 - y0) < 8 or ts.should_filter(r.get("jp", ""), cht, ts.FILTER_TEXT):
            continue
        tb = (x0, y0, x1, y1)
        if ts.is_cjk(cht):
            ts.draw_v(im, dr, cht, tb)
        else:
            ts.draw_h(dr, cht, tb)
    im.save(os.path.join(OUT, f"final_{name}.png"))
    print(f"{name} retypeset")
