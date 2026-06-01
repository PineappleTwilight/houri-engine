#!/usr/bin/env python3
"""
M3 排版 parity：把譯文排進去字後的氣泡。支援 橫排(h) / 直排(v) 兩模式（對應使用者可切換設定）。
  直排：CJK 字由上而下、欄由右而左（漫畫原生）。
  橫排：左到右、上到下（英文等）。
兩者皆自動字級 + 斷行 + 置中 + 描邊。用法：python3 typeset_parity.py [v|h]
"""
import os, sys, json, math
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import translate_parity as tp

OUT = tp.OUT
INPAINTED = os.path.join(OUT, "inpainted.png")
REGIONS = os.path.join(OUT, "merged_results.json")
FONT = tp.find_font()
MODE = sys.argv[1] if len(sys.argv) > 1 else "v"


def wrap_h(text, font, maxw):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        if cur and font.getlength(cur + ch) > maxw:
            lines.append(cur); cur = ""
        cur += ch
    if cur:
        lines.append(cur)
    return lines


def draw_h(dr, text, box):
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * 1.1, (y1 - y0) * 1.15
    for size in range(min(int(bh), 46), 8, -1):
        font = ImageFont.truetype(FONT, size)
        lines = wrap_h(text, font, bw)
        lh = size * 1.25
        if len(lines) * lh <= bh and max((font.getlength(l) for l in lines), default=0) <= bw:
            break
    ty = y0 + ((y1 - y0) - len(lines) * lh) / 2
    for ln in lines:
        tx = x0 + ((x1 - x0) - font.getlength(ln)) / 2
        dr.text((tx, ty), ln, font=font, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
        ty += lh


def draw_v(dr, text, box):
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * 1.1, (y1 - y0) * 1.15
    chars = [c for c in text if c != "\n"]
    font = ImageFont.truetype(FONT, 9)
    cols, cpc, lh, cw = 1, 1, 11, 11
    for size in range(min(int(bh), 46), 8, -1):
        lh = size * 1.05            # 每字垂直步進
        cw = size * 1.18            # 每欄寬
        cpc = max(1, int(bh // lh)) # 每欄字數
        cols = math.ceil(len(chars) / cpc)
        if cols * cw <= bw:
            font = ImageFont.truetype(FONT, size)
            break
    total_w = cols * cw
    right_cx = x0 + ((x1 - x0) - total_w) / 2 + total_w - cw / 2  # 最右欄中心
    for c in range(cols):
        cx = right_cx - c * cw
        seg = chars[c * cpc:(c + 1) * cpc]
        total_h = len(seg) * lh
        cy = y0 + ((y1 - y0) - total_h) / 2
        for ch in seg:
            w = font.getlength(ch)
            dr.text((cx - w / 2, cy), ch, font=font, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
            cy += lh


def main():
    im = Image.open(INPAINTED).convert("RGB")
    dr = ImageDraw.Draw(im)
    for r in json.load(open(REGIONS, encoding="utf-8")):
        cht = r.get("cht", "")
        x0, y0, x1, y1 = r["bbox"]
        if not cht or (x1 - x0) < 8 or (y1 - y0) < 8:
            continue
        (draw_v if MODE == "v" else draw_h)(dr, cht, (x0, y0, x1, y1))
    dst = os.path.join(OUT, f"translated_{MODE}.png")
    im.save(dst)
    print(f"排版完成（{MODE}）→ {dst}")


if __name__ == "__main__":
    main()
