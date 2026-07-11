#!/usr/bin/env python3
# 跨頁併發流水線展示圖（Option A·單圖·標籤中英合併）。數字＝sandbox 實測(boxfill·demo01)。
from PIL import Image, ImageDraw, ImageFont

FONT = "/mnt/c/Windows/Fonts/msjh.ttc"
def f(sz):
    try: return ImageFont.truetype(FONT, sz)
    except Exception: return ImageFont.load_default()

W, H = 1760, 566
BG = (255, 255, 255)
INK = (28, 31, 38); SUB = (120, 126, 138)
CPU = (59, 130, 246); NET = (245, 158, 11); INP = (16, 185, 129)
HL = (219, 234, 254); BLUE = (37, 99, 235)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
fT, fTe, fB, fBe, fN, fS = f(42), f(27), f(28), f(20), f(21), f(19)

def duo(x, y, zh, en, fz, fe, cz=INK, ce=SUB, gap=12):
    d.text((x, y), zh, font=fz, fill=cz)
    xe = x + d.textlength(zh, font=fz) + gap
    d.text((xe, y + (fz.size - fe.size) - 2), en, font=fe, fill=ce)

# ── 標題（中英合併）──
duo(40, 24, "跨頁併發流水線", "Cross-page pipeline", fT, fTe)
d.text((40, 78), "同一引擎、多頁重疊翻譯，整章 ~2× 快 · one engine, pages overlapped, ~2× faster / chapter", font=fN, fill=SUB)

X0, XMAX = 190, 1490
SEQ, PIPE, PAGE = 17.2, 9.2, 4.3
CPU_S, NET_S = 2.4, 1.9
SC = (XMAX - X0) / SEQ
BARH, GAP = 30, 8

def seg(x, y, s, c, lab=None):
    w = s * SC
    d.rounded_rectangle([x, y, x + w, y + BARH], radius=5, fill=c)
    if lab and w > 42: d.text((x + 7, y + 5), lab, font=f(17), fill=(255, 255, 255))
    return x + w

def pg(x, y, tag):
    x1 = seg(x, y, CPU_S, CPU, tag); seg(x1, y, NET_S, NET)
    d.rounded_rectangle([x1, y + BARH - 5, x1 + 0.2 * SC, y + BARH], radius=2, fill=INP)

# ── 循序 循序 Sequential ──
ys = 146
d.text((40, ys - 4), "循序", font=fB, fill=INK)
d.text((40, ys + 31), "Sequential", font=fS, fill=SUB)
x = X0
for i in range(4): pg(x, ys, f"P{i+1}"); x += PAGE * SC
d.text((x + 12, ys - 4), "17.2s", font=fB, fill=INK)
d.text((x + 12, ys + 31), "4.3s/頁 pg", font=fS, fill=SUB)

# ── 跨頁 Pipeline ──
yp = 256
stag = (PIPE - PAGE) / 3
ptop, pbot = yp - 6, yp + 4 * (BARH + GAP) + 2
ov_a, ov_b = X0 + CPU_S * SC, X0 + PAGE * SC
d.rectangle([ov_a, ptop, ov_b, pbot], fill=HL)
d.text((40, yp - 4), "跨頁", font=fB, fill=INK)
d.text((40, yp + 31), "Pipeline", font=fS, fill=SUB)
d.text((40, yp + 56), "深度 depth 4", font=f(17), fill=SUB)
for i in range(4): pg(X0 + i * stag * SC, yp + i * (BARH + GAP), f"P{i+1}")
pend = X0 + PIPE * SC
for yy in range(ptop, pbot, 8): d.line([pend, yy, pend, yy + 4], fill=SUB, width=2)
d.text((pend - 108, pbot + 8), "9.2s · 2.3s/頁 pg", font=fN, fill=INK)
# 重疊註記（中英合併）
duo(ov_a - 6, ptop - 28, "重疊＝併發", "Overlap = concurrency", f(20), f(18), cz=BLUE, ce=BLUE, gap=8)
d.text((ov_a - 6, pbot + 50), "第1頁等翻譯(網路)時，第2頁 CPU 已開跑 · page 2's CPU runs while page 1 waits on the network",
       font=f(17), fill=SUB)

# ── 大 2× + faster（中英合併）──
d.text((pend + 92, yp + 2), "2×", font=f(120), fill=BLUE)
duo(pend + 102, yp + 146, "更快", "faster", fB, fBe, cz=SUB, ce=SUB, gap=8)

# ── 圖例（中英合併）──
ly, lx = 512, 40
leg = [(CPU, "偵測/OCR", "Detect/OCR (CPU)"), (NET, "翻譯", "Translate (network)"), (INP, "去字", "Text removal (hidden)")]
for c, zh, en in leg:
    d.rounded_rectangle([lx, ly, lx + 22, ly + 22], radius=4, fill=c)
    d.text((lx + 30, ly + 1), zh, font=fS, fill=INK)
    xe = lx + 30 + d.textlength(zh, font=fS) + 8
    d.text((xe, ly + 3), en, font=f(17), fill=SUB)
    lx = xe + d.textlength(en, font=f(17)) + 52

img.save("/mnt/d/Gits/Yakuyomi/docs/img/crosspage_showcase.png")
print("saved", img.size)
