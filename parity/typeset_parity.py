#!/usr/bin/env python3
"""
M3 排版 parity（純文字框，可靠）：定位 + 大小都用文字框，框適度放大給呼吸空間。
不靠氣泡 flood-fill（相鄰氣泡會連通出錯）。直排向上對齊、標點旋轉。
用法：python3 typeset_parity.py [v|h|auto]
"""
import os, sys, json, math, re
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import translate_parity as tp

OUT = tp.OUT
INPAINTED = os.path.join(OUT, "inpainted.png")
REGIONS = os.path.join(OUT, "merged_results.json")
FONT = os.path.join(tp.ROOT, "engine/src/main/assets/fonts/NotoSansMonoCJK.ttc")
MODE = sys.argv[1] if len(sys.argv) > 1 else "auto"
ROTATE = set("ー－—―‐~〜～…‥（）()「」『』【】〔〕［］｛｝〈〉《》＜＞<>｜|：;")
EXP_W, EXP_H = 1.3, 1.5  # 文字框放大倍率（寬 / 直欄高度）
COL_TRIM = 2             # 直排每欄少放幾字（縮短欄長、減少凸出；欄變多→字級自動縮）
FILTER_TEXT = None       # config.filter_text：regex 命中譯文則濾掉該區（預設不啟用）


def is_cjk(text):
    return any(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF
               or 0x3400 <= ord(c) <= 0x4DBF or 0xFF00 <= ord(c) <= 0xFFEF for c in text)


def should_filter(jp, cht, filter_text=None):
    """翻譯後過濾鏈（對齊 m-i-t manga_translator.py ~L1323）：空白/數字/regex/譯==原 → 丟。"""
    t = (cht or "").strip()
    if not t:
        return True
    if t.isnumeric():
        return True
    if filter_text and re.search(filter_text, t):
        return True
    if (jp or "").strip().lower() == t.lower():
        return True
    return False


def blit_rotated(im, ch, font, cx, cy, size):
    pad = int(size * 1.6)
    tmp = Image.new("RGBA", (pad * 2, pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((pad, pad), ch, font=font, fill=(0, 0, 0), anchor="mm",
                             stroke_width=2, stroke_fill=(255, 255, 255))
    im.paste(tmp.rotate(-90, expand=False), (int(cx - pad), int(cy - pad)), tmp.rotate(-90, expand=False))


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


def draw_v(im, dr, text, textbox):
    tx0, ty0, tx1, ty1 = textbox
    tw, th = tx1 - tx0, ty1 - ty0
    bw = tw * EXP_W
    col_room = th * EXP_H
    chars = [c for c in text if c != "\n"]
    if not chars:
        return
    size = 9
    for s in range(min(int(col_room), 70), 8, -1):
        lh, cw = s * 1.05, s * 1.1
        cpc = max(1, int(col_room // lh) - COL_TRIM)
        if math.ceil(len(chars) / cpc) * cw <= bw:
            size = s; break
    font = ImageFont.truetype(FONT, size)
    lh, cw = size * 1.05, size * 1.1
    cpc = max(1, int(col_room // lh) - COL_TRIM)
    cols = math.ceil(len(chars) / cpc)
    tcx = (tx0 + tx1) / 2                       # 定位：水平置中於文字框中心
    right_cx = tcx + cols * cw / 2 - cw / 2
    for c in range(cols):
        cx = right_cx - c * cw
        cy = ty0                                # 定位：頂端對齊文字框頂
        for ch in chars[c * cpc:(c + 1) * cpc]:
            cyc = cy + lh / 2
            if ch in ROTATE:
                blit_rotated(im, ch, font, cx, cyc, size)
            else:
                w = font.getlength(ch)
                dr.text((cx - w / 2, cyc - size * 0.62), ch, font=font, fill=(0, 0, 0),
                        stroke_width=2, stroke_fill=(255, 255, 255))
            cy += lh


def draw_h(dr, text, textbox):
    tx0, ty0, tx1, ty1 = textbox
    tw, th = tx1 - tx0, ty1 - ty0
    bw = tw * EXP_W
    row_room = th * EXP_H
    size, lines = 9, [text]
    for s in range(min(int(row_room), 70), 8, -1):
        font = ImageFont.truetype(FONT, s)
        ls = wrap_h(text, font, bw)
        if len(ls) * s * 1.18 <= row_room and max((font.getlength(l) for l in ls), default=0) <= bw:
            size, lines = s, ls; break
    font = ImageFont.truetype(FONT, size)
    lh = size * 1.18
    tcx = (tx0 + tx1) / 2
    ty = ty0
    for ln in lines:
        dr.text((tcx - font.getlength(ln) / 2, ty), ln, font=font, fill=(0, 0, 0),
                stroke_width=2, stroke_fill=(255, 255, 255))
        ty += lh


def main():
    im = Image.open(INPAINTED).convert("RGB")
    dr = ImageDraw.Draw(im)
    for r in json.load(open(REGIONS, encoding="utf-8")):
        cht = r.get("cht", "")
        x0, y0, x1, y1 = r["bbox"]
        if (x1 - x0) < 8 or (y1 - y0) < 8 or should_filter(r.get("jp", ""), cht, FILTER_TEXT):
            continue
        mode = "v" if (MODE == "auto" and is_cjk(cht)) else ("h" if MODE == "auto" else MODE)
        if mode == "v":
            draw_v(im, dr, cht, (x0, y0, x1, y1))
        else:
            draw_h(dr, cht, (x0, y0, x1, y1))
    im.save(os.path.join(OUT, f"translated_{MODE}.png"))
    print(f"排版完成（{MODE}）")


if __name__ == "__main__":
    main()
