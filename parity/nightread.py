#!/usr/bin/env python3
"""nightread.py — 夜讀重繪（單頁）：DBNet 偵測 → 三分區遮罩 → 合成暗色閱讀頁。

這是什麼：把白底漫畫頁「重繪」成適合夜間閱讀的暗色頁（非濾鏡、非反相），
桌面原型（scratchpad dm_detect/dm_v4/dm_art/dm_final 系列）的正式收斂版。
三分區處理：
  留白（貼頁邊白＝頁邊距/格溝）→ 填深 BG + 格框描亮；
  氣泡內部                     → 深底 BG、文字筆畫畫亮 INK（原圖墨度當 alpha、
                                 邊緣天然抗鋸齒）、輪廓描亮；
  畫面（其餘）                 → D2：高光滾降 LUT（單調、保序）+ 自適應墨線增亮
                                 （只在局部背景偏暗處拉筆畫、cap<紙白 ⇒ 不反相）。

設計紅線（不可違反）：畫面絕不反相（只允許單調映射壓暗）；框白填深、字反白
（氣泡＝深底亮字）。最壞情況只是「某區沒變暗」，絕不出現負片畫面。

相對原型的三個修法（2026-08 校準，11 張測試頁量測定閾）：
  修法1  氣泡白元件「整頁面積上限」——併入白色連通元件前先查它在整頁的佔比
         （BUBBLE_COMP_MAX_FRAC）與「相對文字窗的局部性」（BUBBLE_LOCAL_K），
         超標＝不是氣泡（格內背景白）不併；改成整顆元件併入（不裁窗截斷），
         原型的 regrow 事後補救隨之移除。滅：demo01 黑塊、demo02 灰縫、
         ch34_014 方塊化。
  修法2  頁型判別降級——長直格框線（形態學開運算）太少＝無框/白背景頁
         （demo04/05 這型），命中則背景不填深、只重繪氣泡，畫面照 D2 壓暗。
         判準：H/V 線各 ≥ FRAME_MIN_EACH 且合計 ≥ FRAME_MIN_SUM（px/千像素；
         校準：有框頁最弱 demo02=1.19/5.77，無框頁最強 demo04=0.86/3.36）。
  修法3  留白遮罩格框感知——貼頁邊白元件逐顆分類：厚芯（距離變換 > CORE_R）
         大量「深入頁內」（距頁邊 > deep_px）＝格內白（如出血格的天空），
         或「深入頁內且包住線稿」（小洞內墨密度 ≥ DEEP_INK_RATIO）＝白包畫，
         皆改判畫面（D2 壓暗、不填深）。校準：真留白網絡 coreDeep 0.00–0.08，
         問題格（ch34_006 老人格 0.58 / ch34_010 第1格 0.40 / demo06 教堂 0.91）。

偵測路徑＝export_dbnet_ncnn.build_model（m-i-t TextDetection @ .upstream-ref、
detect-20241225.ckpt，paths.fetch 自動下載+驗 sha256）torch eager 前向 ＋
m-i-t dbnet_utils.SegDetectorRepresenter 後處理 ＋ mit_grouping 兩階段區域合併
——與引擎同款前處理（長邊 1024、pad 右下到 256 倍數、/127.5-1）。

用法：
  python3 nightread.py <頁圖> [-o 輸出夾]      # 預設輸出 parity/out/nightread/
輸出（皆帶頁名前綴）：_regions.json / _seg.png / _bubble.png / _gutter.png /
  _final.png / _cmp.png（三聯：原圖｜成品｜遮罩視覺化）。
批次（多頁 + 白面積表）用 nightread_batch.py。
"""
import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_dbnet_ncnn as ex                                    # noqa: E402
from mit_grouping import Quadrilateral, merge_bboxes_text_region  # noqa: E402
import paths                                                      # noqa: E402

OUT_DEFAULT = os.path.join(paths.OUT, "nightread")

# ── 設計常數（真機 A/B 要調的旋鈕全在這）─────────────────────────────
# 顏色/位準
BG = 16                # 深底（留白/氣泡底）
INK = 240              # 亮字
STROKE = 3             # 描邊半徑（格框/氣泡輪廓描亮 band）
# 畫面 D2 曲線
DIM_FLOOR, DIM_CEIL = 30, 140   # 壓暗值域（紙白 255 → 140）
ROLLOFF_G = 0.55                # 高光滾降 y = floor+(ceil-floor)*x^g（g<1 凹）
GLOW_STRENGTH = 55              # 自適應墨線增亮強度
GLOW_CAP = 112                  # 增亮上限（< DIM_CEIL ⇒ 線永遠比紙暗、不反相）
# 偵測/遮罩
SEG_TH = 0.12          # 筆畫遮罩二值化閾（引擎 Config.segThreshold 同款）
WHITE_TH = 235         # 「白」的灰階下限（氣泡白/留白共用同一份連通元件）
BUBBLE_PAD = 40        # 文字區 bbox 外擴的搜尋窗
WHITE_MEASURE_TH = 200 # 白面積統計閾（回報用，不進演算法）
GUTTER_MIN_AREA_FRAC = 0.0006   # 留白元件最小面積（整頁佔比）
# 修法1：氣泡白元件上限
BUBBLE_COMP_MAX_FRAC = 0.07     # 元件整頁佔比上限（6–8% 帶，取中偏上）
BUBBLE_LOCAL_K = 4.0            # 元件面積 ≤ K × 文字搜尋窗面積（局部性）
# 修法2：頁型判別（長直格框線，px/千像素）
FRAME_LINE_L_DIV = 5            # 線長 = min(W,H)//DIV（至少 60px）
FRAME_DARK_TH = 100             # 「框線暗」灰階上限
FRAME_MIN_EACH = 1.0            # H、V 各自下限
FRAME_MIN_SUM = 4.5             # H+V 合計下限
# 修法3：留白元件分類（gutter vs 格內白）
CORE_R = 26                     # 厚芯：距離變換 > CORE_R（真格溝半寬遠小於此）
DEEP_EDGE_FRAC = 0.07           # deep_px = max(64, 0.07*min(W,H))（頁邊距帶寬）
IN_PANEL_CORE_FRAC = 0.15       # 規則A：厚芯佔元件 ≥ 此
IN_PANEL_CORE_DEEP = 0.25       #        且厚芯深入頁內比例 ≥ 此 ⇒ 格內白
DEEP_INK_DEEP = 0.5             # 規則B：厚芯深入比例 ≥ 此
DEEP_INK_RATIO = 0.02           #        且小洞內墨/元件面積 ≥ 此 ⇒ 白包畫
HOLE_MAX_FRAC = 0.01            # 「小洞」上限（整頁佔比；大洞＝整格，不算包線稿）
INK_DARK_TH = 128               # 洞內「墨」灰階上限

_model = None
_dbnet_utils = None


def _load_dbnet_utils():
    """m-i-t dbnet_utils 無相對 import，以檔案載入。"""
    p = os.path.join(ex.MIT, "manga_translator/detection/default_utils/dbnet_utils.py")
    spec = importlib.util.spec_from_file_location("mit_dbnet_utils", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mit_dbnet_utils"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_model():
    """DBNet（torch eager）＋後處理模組，模組級快取（批次只載一次）。"""
    global _model, _dbnet_utils
    if _model is None:
        ex.fetch_ckpt()                       # paths.fetch：缺檔下載 + sha256 驗證
        _model = ex.build_model()
        _dbnet_utils = _load_dbnet_utils()
    return _model, _dbnet_utils


# ── 偵測 ────────────────────────────────────────────────────────────

def detect(img_bgr):
    """DBNet 前向 + m-i-t 後處理 + mit_grouping 區域合併。

    回傳 (lines, regions, seg)：文字行 Quadrilateral、區域 dict（bbox/angle/lines）、
    seg 筆畫二值遮罩（bool、原圖解析度）。
    """
    import torch
    model, du = get_model()
    H, W = img_bgr.shape[:2]
    chw, inW, inH, ratio = ex.preprocess(img_bgr)      # 引擎同款前處理
    th_, tw_ = int(round(H * ratio)), int(round(W * ratio))
    with torch.no_grad():
        db, mask = model(torch.from_numpy(chw[None]))
    db = db.sigmoid().numpy()                          # m-i-t default.py：模型外 sigmoid
    mask = mask.numpy()[0, 0]

    rep = du.SegDetectorRepresenter(thresh=0.5, box_thresh=0.7, unclip_ratio=2.3)
    boxes, scores = rep({"shape": [(inH, inW)]}, db)
    boxes, scores = boxes[0], scores[0]
    lines = []
    if boxes.size:
        idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
        for pts, sc in zip(boxes[idx].astype(np.float64), np.asarray(scores)[idx]):
            q = pts / ratio                            # pad 在右下 ⇒ 除 ratio 即原圖座標
            q[:, 0] = np.clip(q[:, 0], 0, W - 1)
            q[:, 1] = np.clip(q[:, 1], 0, H - 1)
            if cv2.contourArea(q.astype(np.float32)) > 16:
                lines.append(Quadrilateral(q.astype(int), "", float(sc)))

    regions = []
    for txtlns, _, _ in merge_bboxes_text_region(list(lines), W, H):
        x0 = int(min(t.aabb.x for t in txtlns)); y0 = int(min(t.aabb.y for t in txtlns))
        x1 = int(max(t.aabb.x + t.aabb.w for t in txtlns))
        y1 = int(max(t.aabb.y + t.aabb.h for t in txtlns))
        ang = float(np.degrees(np.mean([t.angle for t in txtlns])) - 90)
        if abs(ang) < 3:
            ang = 0.0
        regions.append({
            "bbox": [x0, y0, x1, y1],
            "angle": round(ang, 1),
            "lines": [{"quad": t.pts.tolist(), "score": round(float(t.prob), 4)}
                      for t in txtlns],
        })

    # seg 筆畫遮罩：半解析 → canvas → 裁 pad → 原圖 → 閾值（對齊 seg_validate/引擎）
    m_canvas = cv2.resize(mask, (inW, inH), interpolation=cv2.INTER_LINEAR)
    m_full = cv2.resize(m_canvas[:th_, :tw_], (W, H), interpolation=cv2.INTER_LINEAR)
    seg = m_full > SEG_TH
    return lines, regions, seg


# ── 遮罩：白元件分類（修法2/3）＋氣泡（修法1）───────────────────────

def page_is_frameless(g):
    """修法2：長直格框線存在性。回傳 (frameless, h_px_per_k, v_px_per_k)。

    暗像素對「長水平/垂直線」形態學開運算＝只留貼直的長線（格框）；
    無框/白背景頁（demo04/05 型）兩向都近零。
    """
    H, W = g.shape
    dark = (g < FRAME_DARK_TH).astype(np.uint8)
    L = max(60, min(W, H) // FRAME_LINE_L_DIV)
    lh = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    lv = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    hk, vk = 1000.0 * lh.mean(), 1000.0 * lv.mean()
    frameless = not (min(hk, vk) >= FRAME_MIN_EACH and hk + vk >= FRAME_MIN_SUM)
    return frameless, hk, vk


def _hole_ink_ratio(comp_u8, g):
    """元件「小洞內墨」/元件面積：白元件包住的線稿量（修法3 規則B 的訊號）。

    填洞（1px 零邊框 + 從外 floodFill）→ 洞＝沒被外部填到的非元件像素；
    只計小洞（< HOLE_MAX_FRAC 頁面；大洞＝被留白環住的整格，不是包線稿）。
    """
    ff = np.pad(comp_u8, 1)
    m = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 2)
    holes = (ff[1:-1, 1:-1] == 0).astype(np.uint8)     # 非元件且外部填不到＝洞
    hn, hlab, hstats, _ = cv2.connectedComponentsWithStats(holes, 8)
    ink = 0
    for j in range(1, hn):
        if hstats[j, cv2.CC_STAT_AREA] < HOLE_MAX_FRAC * g.size:
            ink += int((g[hlab == j] < INK_DARK_TH).sum())
    return ink / max(int(comp_u8.sum()), 1)


def classify_white_components(g):
    """整頁白（>=WHITE_TH）連通元件一次算完，供留白與氣泡共用。

    回傳 (lab, stats, gutter_ids, panel_ids)：
      gutter_ids＝判定為留白（頁邊距/格溝）的元件 → 填深；
      panel_ids ＝貼頁邊但屬「格內白」的元件（修法3）→ 當畫面壓暗、氣泡也不併。
    """
    H, W = g.shape
    white = (g >= WHITE_TH).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    dist = cv2.distanceTransform(white, cv2.DIST_L2, 5)   # 白內距最近非白（元件間互不影響）
    deep_px = max(64, int(round(DEEP_EDGE_FRAC * min(W, H))))
    min_area = int(g.size * GUTTER_MIN_AREA_FRAC)

    gutter_ids, panel_ids = set(), set()
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        x, y, cw, ch = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if not (x <= 2 or y <= 2 or x + cw >= W - 2 or y + ch >= H - 2):
            continue                                    # 不貼頁邊 ⇒ 非留白候選
        comp = (lab == i)
        core = comp & (dist > CORE_R)                   # 厚芯：比格溝半寬還厚的部分
        core_frac = core.sum() / a
        if core.any():
            ys, xs = np.nonzero(core)
            edge_d = np.minimum(np.minimum(xs, W - 1 - xs), np.minimum(ys, H - 1 - ys))
            core_deep = float((edge_d > deep_px).mean())  # 厚芯深入頁內（非頁邊距帶）比例
        else:
            core_deep = 0.0
        # 修法3：規則A＝厚芯大量深入頁內（出血格天空）；規則B＝深入且包住線稿（白包畫）
        in_panel = (core_frac >= IN_PANEL_CORE_FRAC and core_deep >= IN_PANEL_CORE_DEEP)
        if not in_panel and core_deep >= DEEP_INK_DEEP:
            in_panel = _hole_ink_ratio(comp.astype(np.uint8), g) >= DEEP_INK_RATIO
        (panel_ids if in_panel else gutter_ids).add(i)
    return lab, stats, gutter_ids, panel_ids


def build_bubble_mask(g, regions, seg, lab, stats, excluded_ids):
    """氣泡內部遮罩（修法1）：每文字區 bbox+BUBBLE_PAD 窗內，找「貼著（外擴後）
    文字筆畫」的白色連通元件，通過守門則整顆併入（不裁窗 ⇒ 無截斷方塊，
    原型 regrow 補救移除）。守門（不併＝該區只保留筆畫，安全降級）：
      整頁佔比 ≤ BUBBLE_COMP_MAX_FRAC（格內背景白太大，不是氣泡）
      面積 ≤ BUBBLE_LOCAL_K × 搜尋窗（局部性：氣泡跟它的字同尺度）
      不在 excluded_ids（留白/格內白元件）
    """
    H, W = g.shape
    seg_u8 = seg.astype(np.uint8) * 255
    seg_dil = cv2.dilate(seg_u8, np.ones((9, 9), np.uint8))  # 筆畫外擴→碰得到氣泡白底
    bubble = np.zeros((H, W), bool)
    merged, rejected = set(), set()
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        cx0, cy0 = max(0, x0 - BUBBLE_PAD), max(0, y0 - BUBBLE_PAD)
        cx1, cy1 = min(W, x1 + BUBBLE_PAD), min(H, y1 + BUBBLE_PAD)
        win_area = (cx1 - cx0) * (cy1 - cy0)
        lab_c = lab[cy0:cy1, cx0:cx1]
        touch = np.unique(lab_c[(seg_dil[cy0:cy1, cx0:cx1] > 0) & (lab_c > 0)])
        for i in touch:
            if i in merged:
                continue
            a = int(stats[i, cv2.CC_STAT_AREA])
            if (a > BUBBLE_COMP_MAX_FRAC * g.size or a > BUBBLE_LOCAL_K * win_area
                    or i in excluded_ids):
                rejected.add(int(i))
                continue
            bubble |= lab == i
            merged.add(int(i))
        bubble[y0:y1, x0:x1] |= seg[y0:y1, x0:x1]       # 區內筆畫本身一定算氣泡內容
    return bubble, merged, rejected


# ── 合成（畫面 D2 / 留白 / 氣泡）────────────────────────────────────

def lut_rolloff(floor=DIM_FLOOR, ceil=DIM_CEIL, gpow=ROLLOFF_G):
    """高光滾降 LUT：y = floor + (ceil-floor)*x^g。嚴格單調 ⇒ 保序、零負片感；
    亮部（紙白）壓得重、中暗部（網點/陰影）對比留得比線性多。"""
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(floor + (ceil - floor) * np.power(x, gpow), 0, 255).astype(np.uint8)


def ink_line_mask(g, seg=None, bh_ksize=7, bh_gain=45.0, dark_lo=40, dark_hi=185):
    """軟性墨線遮罩 0..1：blackhat（細暗線構）×暗度權重 ∪ DBNet seg。
    實心黑塊內部為 0（只認邊緣細線 ⇒ 增亮不掉大塊黑的對比）。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_ksize, bh_ksize))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    soft = np.clip(bh / bh_gain, 0.0, 1.0)
    soft *= np.clip((dark_hi - g.astype(np.float32)) / (dark_hi - dark_lo), 0.0, 1.0)
    if seg is not None:
        soft = np.maximum(soft, seg.astype(np.float32))
    return soft


def ink_glow(dimmed, ink_soft, strength=GLOW_STRENGTH, cap=GLOW_CAP, bg_thr=95, bg_sigma=8.0):
    """自適應墨線增亮：只在「局部背景偏暗」處把筆畫往亮拉，夾在 cap 之下
    （cap < 紙白位準 DIM_CEIL ⇒ 線永遠比紙暗、不反相）。"""
    out = dimmed.astype(np.float32)
    gain = np.float32(strength) * ink_soft
    bg = cv2.GaussianBlur(dimmed, (0, 0), bg_sigma).astype(np.float32)
    gain *= np.clip((bg_thr - bg) / bg_thr, 0.0, 1.0)
    lifted = np.minimum(out + gain, np.maximum(out, np.float32(cap)))
    return np.clip(lifted, 0, 255).astype(np.uint8)


def scene_final(g, seg):
    """畫面區最終處理＝D2：rolloff LUT + 自適應墨線增亮。"""
    return ink_glow(lut_rolloff()[g], ink_line_mask(g, seg))


def ink_alpha(g, gain):
    """原圖墨度（1-亮度）×gain 夾 [0,1]：把墨線「轉亮」時的 alpha（邊緣抗鋸齒）。"""
    return np.clip((1.0 - g.astype(np.float32) / 255.0) * gain, 0.0, 1.0)


def paint_gutter(out, g, gutter):
    """留白填深 + 格框描亮：外擴 STROKE 的 band 內原本是墨線（格框）的像素轉亮。"""
    out[gutter] = BG
    k = np.ones((STROKE * 2 + 1,) * 2, np.uint8)
    band = (cv2.dilate(gutter.astype(np.uint8), k) > 0) & ~gutter
    a = ink_alpha(g, 1.6)
    out[band] = np.maximum(out[band], BG + a[band] * (INK - BG))
    return out


def paint_bubbles(out, g, bubble, seg, text_pad=2):
    """氣泡重繪：內部填深、文字畫亮（墨度 alpha）、輪廓描亮。"""
    out[bubble] = BG
    kt = np.ones((text_pad * 2 + 1,) * 2, np.uint8)
    text = (cv2.dilate((seg & bubble).astype(np.uint8), kt) > 0) & bubble
    out[text] = np.maximum(out[text], BG + ink_alpha(g, 1.4)[text] * (INK - BG))
    ko = np.ones((STROKE * 2 + 1,) * 2, np.uint8)
    band = (cv2.dilate(bubble.astype(np.uint8), ko) > 0) & ~bubble
    out[band] = np.maximum(out[band], BG + ink_alpha(g, 1.6)[band] * (INK - BG))
    return out


def compose(g, gutter, bubble, seg, frameless):
    """整頁合成：D2 畫面 →（有框頁才）留白填深 → 氣泡重繪。回傳 uint8。"""
    out = scene_final(g, seg).astype(np.float32)
    if not frameless:                                   # 修法2：無框頁背景不填深
        out = paint_gutter(out, g, gutter)
    out = paint_bubbles(out, g, bubble, seg)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── 出圖/IO ─────────────────────────────────────────────────────────

def _label(img, text, bar_h=48, scale=0.9):
    """圖上加白底黑字標籤列（cv2.putText 無 CJK ⇒ 英文標籤）。"""
    im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    bar = np.full((bar_h, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (10, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, im])


def _hcat(cols, sep_w=6, sep_v=128):
    h = max(c.shape[0] for c in cols)
    padded = []
    for c in cols:
        if c.shape[0] < h:
            c = cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded.append(c)
    sep = np.full((h, sep_w, 3), sep_v, np.uint8)
    row = padded[0]
    for c in padded[1:]:
        row = np.hstack([row, sep, c])
    return row


def mask_viz(img_bgr, gutter, panel_scene, bubble, seg, regions):
    """遮罩視覺化：留白=黃、修法3改判畫面的格內白=橘、氣泡=綠、筆畫=紅、區域框=洋紅。"""
    viz = img_bgr.copy()
    for m, col in ((gutter, (0, 200, 200)), (panel_scene, (0, 128, 255)),
                   (bubble, (0, 160, 0))):
        viz[m] = (viz[m] * 0.5 + np.array(col) * 0.5).astype(np.uint8)
    viz[seg] = (0, 0, 255)
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        cv2.rectangle(viz, (x0, y0), (x1, y1), (255, 0, 255), 2)
    return viz


def run_page(page_path, outdir=OUT_DEFAULT, col_w=1000):
    """單頁一條龍：偵測 → 遮罩 → 合成 → 落檔。回傳統計 dict（批次表用）。"""
    name = os.path.splitext(os.path.basename(page_path))[0]
    os.makedirs(outdir, exist_ok=True)
    img = cv2.imread(page_path)                        # 彩頁也吃（偵測吃 BGR）
    assert img is not None, page_path
    g = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
    H, W = g.shape

    lines, regions, seg = detect(img)
    frameless, hk, vk = page_is_frameless(g)
    lab, stats, gutter_ids, panel_ids = classify_white_components(g)
    gutter = np.isin(lab, sorted(gutter_ids)) if gutter_ids else np.zeros((H, W), bool)
    panel_scene = np.isin(lab, sorted(panel_ids)) if panel_ids else np.zeros((H, W), bool)
    bubble, merged, rejected = build_bubble_mask(
        g, regions, seg, lab, stats, gutter_ids | panel_ids)
    final = compose(g, gutter, bubble, seg, frameless)

    pref = os.path.join(outdir, name)
    with open(f"{pref}_regions.json", "w", encoding="utf-8") as f:
        json.dump({
            "image": page_path, "width": W, "height": H,
            "pageType": "frameless" if frameless else "framed",
            "frameLines": {"h": round(hk, 3), "v": round(vk, 3)},
            "detector": "DBNet (m-i-t default @ .upstream-ref, detect-20241225.ckpt) "
                        "torch eager; text_th=0.5 box_th=0.7 unclip=2.3; "
                        "regions=mit_grouping.merge_bboxes_text_region",
            "lines": [{"quad": t.pts.tolist(), "score": round(float(t.prob), 4)}
                      for t in lines],
            "regions": regions,
        }, f, ensure_ascii=False, indent=1)
    cv2.imwrite(f"{pref}_seg.png", seg.astype(np.uint8) * 255)
    cv2.imwrite(f"{pref}_bubble.png", bubble.astype(np.uint8) * 255)
    cv2.imwrite(f"{pref}_gutter.png", gutter.astype(np.uint8) * 255)
    cv2.imwrite(f"{pref}_final.png", final)

    wb = float((g >= WHITE_MEASURE_TH).mean())
    wa = float((final >= WHITE_MEASURE_TH).mean())
    cols = []
    for im, t in ((img, f"{name} original"),
                  (final, f"night rebuild ({'FRAMELESS: bubbles only' if frameless else 'framed'})"),
                  (mask_viz(img, gutter, panel_scene, bubble, seg, regions),
                   "masks: gutter=y panelwhite=o bubble=g seg=r")):
        s = col_w / im.shape[1]
        im2 = cv2.resize(im, (col_w, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cols.append(_label(im2, t, bar_h=46, scale=0.8))
    cv2.imwrite(f"{pref}_cmp.png", _hcat(cols))

    st = {"page": name, "pageType": "frameless" if frameless else "framed",
          "regions": len(regions), "whiteBefore": round(wb, 4), "whiteAfter": round(wa, 4),
          "gutterFrac": round(float(gutter.mean()), 4),
          "panelWhiteFrac": round(float(panel_scene.mean()), 4),
          "bubbleFrac": round(float(bubble.mean()), 4),
          "bubbleCompsMerged": len(merged), "bubbleCompsRejected": len(rejected),
          "cmp": f"{pref}_cmp.png"}
    print(f"[{name}] {st['pageType']}  regions={st['regions']}  "
          f"white {wb:.3f}->{wa:.3f}  gutter={st['gutterFrac']:.3f}  "
          f"panelWhite={st['panelWhiteFrac']:.3f}  bubble={st['bubbleFrac']:.3f}  "
          f"comps merged/rejected={len(merged)}/{len(rejected)}", flush=True)
    return st


def main():
    ap = argparse.ArgumentParser(description="夜讀重繪（單頁）")
    ap.add_argument("page", help="頁圖路徑")
    ap.add_argument("-o", "--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    run_page(a.page, a.outdir)


if __name__ == "__main__":
    main()
