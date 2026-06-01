#!/usr/bin/env python3
"""
M3 排版 parity：把繁中譯文排進去字後的氣泡（自動字級 + CJK 斷行 + 置中 + 描邊）。
第一版（陽春橫排）；直排 / 氣泡內縮 / 字級更聰明的自適應留後續精修。
"""
import os, sys, json
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import translate_parity as tp

OUT = tp.OUT
INPAINTED = os.path.join(OUT, "inpainted.png")
REGIONS = os.path.join(OUT, "merged_results.json")
FONT = tp.find_font()


def wrap_cjk(text, font, maxw):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        if font.getlength(cur + ch) > maxw and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def fit(text, bw, bh):
    for size in range(min(int(bh), 46), 8, -1):
        font = ImageFont.truetype(FONT, size)
        lines = wrap_cjk(text, font, bw)
        lh = size * 1.25
        if len(lines) * lh <= bh and max((font.getlength(l) for l in lines), default=0) <= bw:
            return font, lines, lh
    font = ImageFont.truetype(FONT, 9)
    return font, wrap_cjk(text, font, bw), 11


def main():
    im = Image.open(INPAINTED).convert("RGB")
    dr = ImageDraw.Draw(im)
    for r in json.load(open(REGIONS, encoding="utf-8")):
        cht = r.get("cht", "")
        if not cht:
            continue
        x0, y0, x1, y1 = r["bbox"]
        bw, bh = (x1 - x0) * 1.1, (y1 - y0) * 1.15  # 略放寬，原文字框偏緊
        if bw < 8 or bh < 8:
            continue
        font, lines, lh = fit(cht, bw, bh)
        ty = y0 + ((y1 - y0) - len(lines) * lh) / 2
        for ln in lines:
            tx = x0 + ((x1 - x0) - font.getlength(ln)) / 2
            dr.text((tx, ty), ln, fill=(0, 0, 0), font=font,
                    stroke_width=2, stroke_fill=(255, 255, 255))
            ty += lh
    im.save(os.path.join(OUT, "translated.png"))
    print(f"排版完成 → translated.png（font: {FONT}）")


if __name__ == "__main__":
    main()
