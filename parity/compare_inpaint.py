#!/usr/bin/env python3
"""去字模型×方法比較 + 計時：Koharu-lama / AOT × 整頁 / 逐格，固定 DIL=12 遮罩。
給「模型去留」決策：看品質差多少、各花多少時間。
  lama整頁＝quad整塊遮罩(縮512細筆畫會殘)；其餘＝seg細遮罩(逐格/AOT原解析度不縮、不殘)。
桌面 CPU 計時＝相對參考(裝置會更慢，但比例大致成立；裝置實測見 CLAUDE.md：lama整頁~18s、逐格~76s)。
用法：compare_inpaint.py <img>"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import paths  # noqa: E402
from ctd_reference import import_seg_rep  # noqa: E402
from inpaint_dev import bg_std, detect_seg, group, run_aot, run_window, seg_mask, window_of  # noqa: E402

DIL = 12
AOT = "/tmp/aot/inpainting_aot.onnx"
LAMA = f"{paths.MODELS}/lama-manga.onnx"


def load(p):
    return ort.InferenceSession(p, providers=["CPUExecutionProvider"])


def aot_region(aot, rgb, smask, wins, bgr):
    out = bgr.copy()
    for x0, y0, x1, y1 in wins:
        mc = smask[y0:y1, x0:x1]
        if mc.max() == 0:
            continue
        db = cv2.cvtColor(run_aot(aot, rgb[y0:y1, x0:x1], mc), cv2.COLOR_RGB2BGR)
        reg = out[y0:y1, x0:x1].copy()
        reg[mc > 127] = db[mc > 127]
        out[y0:y1, x0:x1] = reg
    return out


def main():
    img = sys.argv[1]
    bgr = cv2.imread(img)
    if bgr is None:
        sys.exit(f"❌ 讀不到圖：{img}（檔案不存在或路徑錯）")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    det = load(f"{paths.MODELS}/comictextdetector.pt.onnx")
    seg_rep = import_seg_rep()(thresh=0.3)
    quads, segm = detect_seg(det, seg_rep, rgb)
    regions = group(quads, w, h)
    smask = seg_mask(regions, segm, w, h, dil=DIL, use_seg=True)
    qmask = seg_mask(regions, segm, w, h, dil=DIL, use_seg=False)
    wins = [win for win in (window_of(r["bbox"], w, h) for r in regions) if win]
    print(f"{len(quads)} 行 → {len(regions)} 區、{len(wins)} 窗；DIL={DIL}")

    results, times = {}, {}

    lama = load(LAMA)
    out = bgr.copy(); t = time.time(); run_window(lama, rgb, qmask, (0, 0, w, h), out)
    times["lama整頁"] = time.time() - t; results["lama整頁"] = out
    out = bgr.copy(); t = time.time()
    for win in wins:
        run_window(lama, rgb, smask, win, out)
    times["lama逐格"] = time.time() - t; results["lama逐格"] = out
    del lama

    if os.path.exists(AOT):
        aot = load(AOT)
        t = time.time(); res = run_aot(aot, rgb, smask)
        times["aot整頁"] = time.time() - t; results["aot整頁"] = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        t = time.time(); out = aot_region(aot, rgb, smask, wins, bgr)
        times["aot逐格"] = time.time() - t; results["aot逐格"] = out
        del aot
    else:
        print(f"⚠ AOT 模型不在 {AOT}")

    order = ["lama整頁", "lama逐格", "aot整頁", "aot逐格"]
    order = [k for k in order if k in results]
    print("\n=== 處理時間（桌面 CPU、去字步驟） ===")
    for k in order:
        print(f"  {k:10s} {times[k]:6.1f}s")
    for k in order:
        cv2.imwrite(f"{paths.OUT}/cmp_{k}.png", results[k])

    # montage：整頁 + 臉/頭髮裁切，各標方法+時間
    tdir = os.path.dirname(os.path.abspath(img))
    mit = next((cv2.imread(p) for p in (f"{tdir}/mit.png", f"{tdir}/mit.jpg") if os.path.exists(p)), None)

    def cell(im, label, H, fx=None, crop=None):
        if im is None:
            im = np.full((200, 200, 3), 200, np.uint8)
        if crop:
            x0, y0, x1, y1 = crop; im = im[y0:y1, x0:x1]
        if fx:
            im = cv2.resize(im, None, fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
        else:
            s = H / im.shape[0]; im = cv2.resize(im, (int(im.shape[1] * s), H))
        im = im.copy()
        cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(im, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return im

    def row(cells, H):
        sep = np.full((H, 4, 3), 255, np.uint8); r = []
        for i, c in enumerate(cells):
            r.append(c)
            if i < len(cells) - 1:
                r.append(sep)
        return np.hstack(r)

    ascii_lab = {"lama整頁": "lama-whole", "lama逐格": "lama-tile", "aot整頁": "aot-whole", "aot逐格": "aot-tile"}
    H = 900
    full = [cell(bgr, "raw", H)] + [cell(results[k], f"{ascii_lab[k]} {times[k]:.0f}s", H) for k in order] + [cell(mit, "MIT", H)]
    cv2.imwrite(f"{paths.OUT}/dev_modelcmp_full.png", row(full, H))
    # 細節裁切＝自動挑背景最忙(臉/頭髮、bg_std 最高)的區，對解析度免疫
    busiest = max(regions, key=lambda r: bg_std(bgr, smask, r["bbox"]), default=None)
    if busiest:
        bx0, by0, bx1, by1 = busiest["bbox"]; pad = 30
        cx = (max(0, bx0 - pad), max(0, by0 - pad), min(w, bx1 + pad), min(h, by1 + pad))
    else:
        cx = (0, 0, w, h)
    fc = [cell(bgr, "raw", 0, fx=1.4, crop=cx)] + \
         [cell(results[k], f"{ascii_lab[k]} {times[k]:.0f}s", 0, fx=1.4, crop=cx) for k in order] + \
         [cell(mit, "MIT", 0, fx=1.4, crop=cx)]
    Hc = fc[0].shape[0]
    cv2.imwrite(f"{paths.OUT}/dev_modelcmp_face.png", row(fc, Hc))
    print("\n→ dev_modelcmp_full.png + dev_modelcmp_face.png")


if __name__ == "__main__":
    main()
