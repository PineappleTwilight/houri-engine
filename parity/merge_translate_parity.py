#!/usr/bin/env python3
"""
行合併 parity（品質精修）：同氣泡的 OCR 行先合併再翻，解決逐行碎裂。
合併判準 = quadrilateral_can_merge_region 的對齊精神之簡化版（同方向 + 字級相近 + bbox 間距 < 字高）。
翻譯/ s2twp 沿用 translate_parity。
"""
import os, sys, re, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import translate_parity as tp
import opencc

ROOT = tp.ROOT
OUT = tp.OUT
OCR_JSON = os.path.join(OUT, "ocr_results.json")
IMG = tp.IMG
GAP = 1.0          # 合併：bbox 間距 < GAP×字高
FS_RATIO = 1.5     # 字級比上限


def bbox(quad):
    a = np.array(quad, float)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def font_size(b):
    return min(b[2] - b[0], b[3] - b[1])


def bbox_gap(A, B):
    dx = max(0.0, max(A[0], B[0]) - min(A[2], B[2]))
    dy = max(0.0, max(A[1], B[1]) - min(A[3], B[3]))
    return (dx * dx + dy * dy) ** 0.5


def main():
    s2twp = opencc.OpenCC("s2twp")
    key = tp.read_key()
    res = json.load(open(OCR_JSON, encoding="utf-8"))
    n = len(res)
    bb = [bbox(r["quad"]) for r in res]
    fs = [font_size(b) for b in bb]

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            if res[i]["dir"] != res[j]["dir"]:
                continue
            cs = min(fs[i], fs[j])
            if cs <= 0 or max(fs[i], fs[j]) / cs > FS_RATIO:
                continue
            if bbox_gap(bb[i], bb[j]) < cs * GAP:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    regions = []
    for members in groups.values():
        d = res[members[0]]["dir"]
        if d == "v":
            members.sort(key=lambda i: -bb[i][2])           # 直書：右→左
        else:
            members.sort(key=lambda i: (bb[i][1], bb[i][0]))  # 橫書：上→下
        jp = "".join(res[i]["text"] for i in members)
        xs0 = min(bb[i][0] for i in members); ys0 = min(bb[i][1] for i in members)
        xs1 = max(bb[i][2] for i in members); ys1 = max(bb[i][3] for i in members)
        regions.append({"dir": d, "jp": jp, "bbox": [xs0, ys0, xs1, ys1], "n": len(members)})

    queries = [r["jp"] for r in regions]
    user = "\n".join(f"<|{i + 1}|>{q}" for i, q in enumerate(queries))
    messages = [
        {"role": "system", "content": tp.CHAT_SYSTEM_TEMPLATE.format(to_lang=tp.TO_LANG)},
        {"role": "user", "content": tp.SAMPLE_IN},
        {"role": "assistant", "content": tp.SAMPLE_OUT},
        {"role": "user", "content": user},
    ]
    print(f"{n} 行 → 合併成 {len(regions)} 區，送 DeepSeek…")
    raw = re.sub(r'(</think>)?<think>.*?</think>', '', tp.call_deepseek(key, messages), flags=re.DOTALL)
    trans = {}
    for line in raw.splitlines():
        m = re.match(r'^\s*<\|(\d+)\|>\s*(.*)$', line)
        if m:
            trans[int(m.group(1))] = m.group(2).strip()

    for i, r in enumerate(regions):
        r["cht"] = s2twp.convert(trans.get(i + 1, ""))
        print(f"[區{i:2d}|{r['n']}行|{r['dir']}] {r['jp']}  →  {r['cht']}")

    font = tp.find_font()
    if font:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.open(IMG).convert("RGB")
        dr = ImageDraw.Draw(im)
        fnt = ImageFont.truetype(font, 26)
        for r in regions:
            if not r["cht"]:
                continue
            x0, y0, x1, y1 = map(int, r["bbox"])
            dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
            dr.text((x0, max(0, y0 - 28)), r["cht"], fill=(200, 30, 30), font=fnt,
                    stroke_width=3, stroke_fill=(255, 255, 255))
        im.save(os.path.join(OUT, "merged_overlay.png"))
        print(f"overlay → {OUT}/merged_overlay.png")
    json.dump(regions, open(os.path.join(OUT, "merged_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
