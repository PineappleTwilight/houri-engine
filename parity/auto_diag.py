#!/usr/bin/env python3
"""revised auto 去字 + 視覺化檢驗：
  規則：每區看背景(非遮罩像素)——**白(亮)且均勻=對話框→boxfill(預設)**；否則(臉/頭髮/壓畫面)→lama逐格。
  輸出：去字圖上把每個去字區框起來、標方法+判定數值(std/white)，給人眼檢驗路由對不對。
  綠框=boxfill、紅框=lama。閾值 env 可調：YAKU_STDT(均勻門檻,預設24)、YAKU_WHITET(白門檻,預設190)。
用法：auto_diag.py <img>"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import paths  # noqa: E402
from ctd_reference import import_seg_rep  # noqa: E402
from inpaint_dev import boxfill, detect_seg, group, run_window, seg_mask, window_of  # noqa: E402

DIL = 12
PAD = int(os.environ.get("YAKU_PAD", "16"))          # 泡泡 bbox 外擴：涵蓋貼右邊界的假名
STD_T = float(os.environ.get("YAKU_STDT", "24"))     # 背景亮度 std < 此 = 均勻
WHITE_T = float(os.environ.get("YAKU_WHITET", "190"))  # 背景亮度均值 ≥ 此 = 白
LAMA = f"{paths.MODELS}/lama-manga.onnx"


def bg_stats(img, mask, bbox):
    """區 bbox 內非遮罩(背景)像素的亮度均值 + std。"""
    x0, y0, x1, y1 = bbox
    sub = img[y0:y1, x0:x1]; sm = mask[y0:y1, x0:x1]
    bg = sub[sm <= 127]
    if len(bg) < 16:
        return 255.0, 0.0  # 幾乎全遮罩→當白泡(boxfill 安全)
    lum = 0.114 * bg[:, 0] + 0.587 * bg[:, 1] + 0.299 * bg[:, 2]  # BGR
    return float(lum.mean()), float(lum.std())


def is_bubble(mean, std):
    """對話框 = 背景白(亮)且均勻。"""
    return std < STD_T and mean >= WHITE_T


def clear_box(mask, bbox):
    x0, y0, x1, y1 = bbox
    mask[y0:y1, x0:x1] = 0


def main():
    img = sys.argv[1]
    bgr = cv2.imread(img)
    if bgr is None:
        sys.exit(f"❌ 讀不到圖：{img}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    det = ort.InferenceSession(f"{paths.MODELS}/comictextdetector.pt.onnx", providers=["CPUExecutionProvider"])
    quads, segm = detect_seg(det, import_seg_rep()(thresh=0.3), rgb)
    regions = group(quads, w, h)
    smask = seg_mask(regions, segm, w, h, dil=DIL, use_seg=True, bbox_allow=True, bbox_pad=PAD)  # bbox-allow+pad＝涵蓋假名
    tight = np.where(segm > 0.3, 255, 0).astype(np.uint8)  # 未膨脹 seg：量背景用（筆畫間白量得到、不被去字遮罩蓋住）

    # 每區判定（背景量測用 tight seg，非膨脹去字遮罩）
    for r in regions:
        mean, std = bg_stats(bgr, tight, r["bbox"])
        r["mean"], r["std"] = mean, std
        r["bubble"] = is_bubble(mean, std)
    bub = [r for r in regions if r["bubble"]]
    nonbub = [r for r in regions if not r["bubble"]]
    print(f"{len(regions)} 區：boxfill(泡泡) {len(bub)} / lama(非泡泡) {len(nonbub)}  | STD_T={STD_T} WHITE_T={WHITE_T}")
    for r in regions:
        print(f"  bbox={r['bbox']}  white={r['mean']:5.1f} std={r['std']:5.1f}  → {'boxfill' if r['bubble'] else 'LAMA'}")

    # 去字：泡泡 平塗背景白、非泡泡 lama逐格
    out = bgr.copy()
    # 泡泡＝已確認均勻白，直接「平塗背景色」填遮罩：保證無殘留。
    # （boxfill 就近取色在大遮罩中心會因 FILL_REACH=64 搆不到邊緣 → 中間留字；對均勻區用平塗更對。）
    for r in bub:
        x0 = max(0, r["bbox"][0] - PAD); y0 = max(0, r["bbox"][1] - PAD)
        x1 = min(w, r["bbox"][2] + PAD); y1 = min(h, r["bbox"][3] + PAD)
        bgpx = bgr[y0:y1, x0:x1].reshape(-1, 3)[tight[y0:y1, x0:x1].reshape(-1) <= 127]
        color = bgpx.mean(0) if len(bgpx) >= 16 else np.array([255.0, 255.0, 255.0])
        m = smask[y0:y1, x0:x1]
        reg = out[y0:y1, x0:x1]; reg[m > 127] = color; out[y0:y1, x0:x1] = reg
    if nonbub:
        lama = ort.InferenceSession(LAMA, providers=["CPUExecutionProvider"])
        lamamask = smask.copy()
        for r in bub:
            clear_box(lamamask, r["bbox"])  # 泡泡移出 lama
        for r in nonbub:
            win = window_of(r["bbox"], w, h)
            if win:
                run_window(lama, rgb, lamamask, win, out)

    cv2.imwrite(f"{paths.OUT}/dev_autodiag_clean.png", out)

    # 疊框+標籤：綠=boxfill、紅=lama
    ann = out.copy()
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        green, red = (0, 170, 0), (0, 0, 230)
        col = green if r["bubble"] else red
        cv2.rectangle(ann, (x0, y0), (x1, y1), col, 2)
        tag = ("box" if r["bubble"] else "LAMA") + f" w{r['mean']:.0f} s{r['std']:.0f}"
        ty = y0 - 4 if y0 > 14 else y1 + 16
        cv2.putText(ann, tag, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.imwrite(f"{paths.OUT}/dev_autodiag.png", ann)
    print("→ dev_autodiag.png (框+標方法) + dev_autodiag_clean.png (純去字)")


if __name__ == "__main__":
    main()
