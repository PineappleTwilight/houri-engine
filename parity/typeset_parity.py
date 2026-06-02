#!/usr/bin/env python3
"""
M3 排版 parity（純文字框，可靠）：定位 + 大小都用文字框，框適度放大給呼吸空間。
不靠氣泡 flood-fill（相鄰氣泡會連通出錯）。直排向上對齊、標點旋轉。
用法：python3 typeset_parity.py [v|h|auto]
"""
import os, sys, json, math, re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import translate_parity as tp

OUT = tp.OUT
INPAINTED = os.path.join(OUT, "inpainted.png")
REGIONS = os.path.join(OUT, "merged_results.json")
FONT = os.path.join(tp.ROOT, "engine/src/main/assets/fonts/NotoSansMonoCJK.ttc")
MODE = sys.argv[1] if len(sys.argv) > 1 else "auto"
ROTATE = set("ー－—―‐~〜～…‥（）()「」『』【】〔〕［］｛｝〈〉《》＜＞<>｜|：;")
# 行頭禁則：不可置於欄/行開頭（收尾標點、小假名）→ 併回前一欄/行（kinsoku）
NO_START = set("、。，．：；！？”’）〕】｝」』》〉…‥ーゝゞヽヾ々ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ")
EXP_W, EXP_H = 1.3, 1.5  # 文字框放大倍率（寬 / 直欄高度）
COL_TRIM = 3             # 直排每欄少放幾字（縮短欄長、減少凸出；欄變多→字級自動縮）★與 Kotlin RenderConfig.colTrim 同步
SHRINK = 0.85            # 算好字級後再整體縮放（<1＝更小、更 fit 格子；留邊距）
STROKE = 0.10            # 描邊寬＝字級×此比例（隨字級縮放，取代固定寬）
FILTER_TEXT = None       # config.filter_text：regex 命中譯文則濾掉該區（預設不啟用）
COLOR_MODE = "auto"      # 文字色：auto（預設，取去字後背景亮度→黑/白字，最穩）| mono | polarity | hue
BG_DARK = 110            # auto：去字後背景平均亮度 < 此值＝暗底 → 白字


def is_cjk(text):
    return any(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF
               or 0x3400 <= ord(c) <= 0x4DBF or 0xFF00 <= ord(c) <= 0xFFEF for c in text)


def color_difference(rgb1, rgb2):
    """CIE76 ΔE（cv2 LAB，L 權重 0.392）。對齊 m-i-t utils/generic2.py:color_difference。"""
    c1 = np.array(rgb1, np.uint8).reshape(1, 1, 3)
    c2 = np.array(rgb2, np.uint8).reshape(1, 1, 3)
    d = cv2.cvtColor(c1, cv2.COLOR_RGB2LAB).astype(np.float32) - cv2.cvtColor(c2, cv2.COLOR_RGB2LAB).astype(np.float32)
    d[..., 0] *= 0.392
    return float(np.linalg.norm(d, axis=2).item())


def fg_bg_compare(fg, bg):
    """對齊 m-i-t rendering/__init__.py:fg_bg_compare：fg/bg 太近就把 bg 翻白(fg暗)/黑(fg亮)。"""
    fg = tuple(int(v) for v in fg)
    bg = tuple(int(v) for v in bg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if (sum(fg) / 3) <= 127 else (0, 0, 0)
    return fg, bg


def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def auto_colors(npimg, bbox):
    """取去字後背景在 bbox 內的平均亮度 → 暗底白字、亮底黑字（最穩，保證可讀）。"""
    x0, y0, x1, y1 = (int(v) for v in bbox)
    crop = npimg[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
    if crop.size == 0:
        return (0, 0, 0), (255, 255, 255)
    lum = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    return ((255, 255, 255), (0, 0, 0)) if lum.mean() < BG_DARK else ((0, 0, 0), (255, 255, 255))


def resolve_colors(npimg, bbox, fg, bg):
    """依 COLOR_MODE 回傳 (fill, outline)。auto 用背景亮度；其餘走 text_colors（OCR color head）。"""
    if COLOR_MODE == "auto":
        return auto_colors(npimg, bbox)
    return text_colors(fg, bg)


def text_colors(fg, bg):
    """回傳 (fill, outline)。COLOR_MODE：
       mono     = 一律黑字白邊（不看 color head）
       polarity = color head 只判明暗極性 → 純黑或純白字（乾淨、看齊 m-i-t；白底黑字/黑底白字都對）
       hue      = 保留 m-i-t 原始色相 + fg_bg_compare 安全網（忠實，但彩底易染濁）"""
    fg = tuple(int(v) for v in fg)
    bg = tuple(int(v) for v in bg)
    if COLOR_MODE == "mono":
        return (0, 0, 0), (255, 255, 255)
    if COLOR_MODE == "hue":
        return fg_bg_compare(fg, bg)
    lf, lb = _lum(fg), _lum(bg)
    dark = lf < lb if abs(lf - lb) > 20 else lf < 128
    return ((0, 0, 0), (255, 255, 255)) if dark else ((255, 255, 255), (0, 0, 0))


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


def blit_rotated(im, ch, font, cx, cy, size, fg=(0, 0, 0), bg=(255, 255, 255), sw=2):
    pad = int(size * 1.6)
    tmp = Image.new("RGBA", (pad * 2, pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((pad, pad), ch, font=font, fill=tuple(fg), anchor="mm",
                             stroke_width=sw, stroke_fill=tuple(bg))
    im.paste(tmp.rotate(-90, expand=False), (int(cx - pad), int(cy - pad)), tmp.rotate(-90, expand=False))


def wrap_h(text, font, maxw):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        if cur and font.getlength(cur + ch) > maxw and ch not in NO_START:  # 行頭禁則：禁則字不另起行
            lines.append(cur); cur = ""
        cur += ch
    if cur:
        lines.append(cur)
    return lines


def split_columns_v(chars, cpc):
    """直排切欄＋行頭禁則：禁則字不置於欄頭，併回前一欄（最多 +2，避免暴衝）。"""
    cols, i, n = [], 0, len(chars)
    while i < n:
        end = min(i + cpc, n)
        ext = 0
        while end < n and chars[end] in NO_START and ext < 2:
            end += 1; ext += 1
        cols.append(chars[i:end])
        i = end
    return cols


def draw_v(im, dr, text, textbox, fg=(0, 0, 0), bg=(255, 255, 255)):
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
    size = max(9, int(round(size * SHRINK)))   # 整體縮小、更 fit
    font = ImageFont.truetype(FONT, size)
    lh, cw = size * 1.05, size * 1.1
    cpc = max(1, int(col_room // lh) - COL_TRIM)
    columns = split_columns_v(chars, cpc)       # 禁則：欄不以行頭禁則字開頭
    cols = len(columns)
    sw = max(2, round(size * STROKE))            # 描邊隨字級
    tcx = (tx0 + tx1) / 2                        # 定位：水平置中於文字框中心
    right_cx = tcx + cols * cw / 2 - cw / 2
    block_h = max(len(c) for c in columns) * lh  # 垂直置中：以最長欄高為塊高，置中於框
    start_cy = (ty0 + ty1) / 2 - block_h / 2
    for c, col in enumerate(columns):
        cx = right_cx - c * cw
        cy = start_cy
        for ch in col:
            cyc = cy + lh / 2
            if ch in ROTATE:
                blit_rotated(im, ch, font, cx, cyc, size, fg, bg, sw)
            else:
                w = font.getlength(ch)
                dr.text((cx - w / 2, cyc - size * 0.62), ch, font=font, fill=tuple(fg),
                        stroke_width=sw, stroke_fill=tuple(bg))
            cy += lh


def draw_h(dr, text, textbox, fg=(0, 0, 0), bg=(255, 255, 255)):
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
    size = max(9, int(round(size * SHRINK)))   # 整體縮小、更 fit
    font = ImageFont.truetype(FONT, size)
    lines = wrap_h(text, font, bw)             # 縮小後重排
    lh = size * 1.18
    tcx = (tx0 + tx1) / 2
    sw = max(2, round(size * STROKE))
    ty = (ty0 + ty1) / 2 - len(lines) * lh / 2   # 垂直置中於框
    for ln in lines:
        dr.text((tcx - font.getlength(ln) / 2, ty), ln, font=font, fill=tuple(fg),
                stroke_width=sw, stroke_fill=tuple(bg))
        ty += lh


def main():
    im = Image.open(INPAINTED).convert("RGB")
    dr = ImageDraw.Draw(im)
    npimg = np.array(im)
    for r in json.load(open(REGIONS, encoding="utf-8")):
        cht = r.get("cht", "")
        x0, y0, x1, y1 = r["bbox"]
        if (x1 - x0) < 8 or (y1 - y0) < 8 or should_filter(r.get("jp", ""), cht, FILTER_TEXT):
            continue
        fg, bg = resolve_colors(npimg, (x0, y0, x1, y1), r.get("fg") or (0, 0, 0), r.get("bg") or (255, 255, 255))
        mode = "v" if (MODE == "auto" and is_cjk(cht)) else ("h" if MODE == "auto" else MODE)
        if mode == "v":
            draw_v(im, dr, cht, (x0, y0, x1, y1), fg, bg)
        else:
            draw_h(dr, cht, (x0, y0, x1, y1), fg, bg)
    im.save(os.path.join(OUT, f"translated_{MODE}.png"))
    print(f"排版完成（{MODE}）")


if __name__ == "__main__":
    main()
