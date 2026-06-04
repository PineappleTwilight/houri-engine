#!/usr/bin/env python3
"""桌面去字 harness：detect+seg+group → 去字（boxfill / lama / auto）→ 輸出去字-only，
看 boxfill 殘留/白塊、調 auto 門檻。不需 OCR/翻譯（去字只要 regions+seg）。
用法：inpaint_dev.py <img> [boxfill|lama|auto]"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onnxruntime as ort  # noqa: E402
import paths  # noqa: E402
from ctd_reference import import_seg_rep, letterbox  # noqa: E402
from mit_grouping import Quadrilateral, merge_bboxes_text_region  # noqa: E402

INPUT, LAMA = 1024, 512
WIN = float(os.environ.get("YAKU_WIN", "1.7"))      # lama 視窗 context 倍率（可 env 覆蓋）
SEG_T = float(os.environ.get("YAKU_SEGT", "0.3"))   # seg 二值門檻（低=抓更多筆畫/小字）
DIL = int(os.environ.get("YAKU_DIL", "3"))          # 遮罩膨脹半徑（大=蓋到邊緣）
FILL_REACH = 64
AUTO_STD = float(os.environ.get("YAKU_STDT", "24.0"))  # auto 路由門檻（低=更多區走 lama；引擎側用 6）


def detect_seg(det, seg_rep, rgb):
    h, w = rgb.shape[:2]
    lb, _, dw, dh = letterbox(rgb, (INPUT, INPUT))
    blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (INPUT, INPUT)).astype(np.float32)
    _blk, seg, det_o = det.run(["blk", "seg", "det"], {"images": blob})
    det2 = det_o[..., :det_o.shape[2] - dh, :det_o.shape[3] - dw]
    bb, sc = seg_rep(None, det2, height=h, width=w)
    quads = [b for b, s in zip(bb[0], sc[0]) if s > 0.6]
    nh, nw = INPUT - dh, INPUT - dw
    segm = cv2.resize(seg[0, 0, :nh, :nw], (w, h))
    return quads, segm


def group(quads, w, h):
    qs = [Quadrilateral(np.array(q, float), "", 1.0) for q in quads]
    out = []
    for txtlns, _, _ in merge_bboxes_text_region(qs, w, h):
        x0 = min(t.aabb.x for t in txtlns); y0 = min(t.aabb.y for t in txtlns)
        x1 = max(t.aabb.x + t.aabb.w for t in txtlns); y1 = max(t.aabb.y + t.aabb.h for t in txtlns)
        out.append({"bbox": [int(x0), int(y0), int(x1), int(y1)], "quads": [t.pts.tolist() for t in txtlns]})
    return out


def seg_mask(regions, segm, w, h, dil=None, use_seg=True, bbox_allow=False, bbox_pad=0):
    dil = DIL if dil is None else dil
    allow = np.zeros((h, w), np.uint8)
    for r in regions:
        if bbox_allow:  # 用區域 bbox 矩形當 allow＝涵蓋漢字旁的注音假名（行框太緊會漏）；pad 再外擴涵蓋貼邊假名
            x0, y0, x1, y1 = r["bbox"]
            cv2.rectangle(allow, (x0 - bbox_pad, y0 - bbox_pad), (x1 + bbox_pad, y1 + bbox_pad), 255, -1)
        else:
            for q in r["quads"]:
                cv2.fillPoly(allow, [np.array(q, np.int32)], 255)
    m = np.where((segm > SEG_T) & (allow > 0), 255, 0).astype(np.uint8) if use_seg else allow.copy()
    if dil > 0:
        m = cv2.dilate(m, np.ones((dil * 2 + 1, dil * 2 + 1), np.uint8))
    return m


def boxfill(img, mask):
    """引擎 boxFill 1:1：每遮罩像素 = 上下左右最近非遮罩像素均值。"""
    out = img.copy()
    h, w = img.shape[:2]
    ys, xs = np.where(mask > 127)
    for y, x in zip(ys.tolist(), xs.tolist()):
        acc = np.zeros(3, np.float64); cnt = 0
        for dx in (-1, 1):
            k = x + dx; d = 0
            while 0 <= k < w and d < FILL_REACH:
                if mask[y, k] <= 127: acc += img[y, k]; cnt += 1; break
                k += dx; d += 1
        for dy in (-1, 1):
            k = y + dy; d = 0
            while 0 <= k < h and d < FILL_REACH:
                if mask[k, x] <= 127: acc += img[k, x]; cnt += 1; break
                k += dy; d += 1
        if cnt > 0:
            out[y, x] = (acc / cnt).astype(np.uint8)
    return out


def bg_std(img, mask, bbox):
    """region bbox 內、非遮罩(背景)像素的亮度 std。低=白底泡泡、高=臉/背景。"""
    x0, y0, x1, y1 = bbox
    sub = img[y0:y1, x0:x1]; sm = mask[y0:y1, x0:x1]
    bg = sub[sm <= 127]
    if len(bg) < 16:
        return 0.0
    lum = 0.114 * bg[:, 0] + 0.587 * bg[:, 1] + 0.299 * bg[:, 2]  # BGR
    return float(lum.std())


def window_of(bbox, w, h, ratio=WIN):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rw, rh = x1 - x0, y1 - y0
    wx0 = max(0, int(cx - rw * ratio / 2)); wy0 = max(0, int(cy - rh * ratio / 2))
    wx1 = min(w, int(cx + rw * ratio / 2)); wy1 = min(h, int(cy + rh * ratio / 2))
    return (wx0, wy0, wx1, wy1) if (wx1 - wx0 >= 8 and wy1 - wy0 >= 8) else None


def run_window(lama, rgb, mask, win, out):
    """一塊視窗跑 LaMa（縮512→推論→貼回，只換遮罩像素）。rgb=RGB、out=BGR。"""
    x0, y0, x1, y1 = win
    crop = rgb[y0:y1, x0:x1]; cm = mask[y0:y1, x0:x1]
    if cm.max() == 0:
        return
    ch, cw = crop.shape[:2]
    ii = (cv2.resize(crop, (LAMA, LAMA), interpolation=cv2.INTER_AREA).astype(np.float32) / 255).transpose(2, 0, 1)[None]
    mi = (cv2.resize(cm, (LAMA, LAMA), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255)[None, None]
    rr = lama.run(["output"], {"image": ii, "mask": mi})[0][0]
    rr = cv2.resize(np.clip(rr.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8), (cw, ch), interpolation=cv2.INTER_LINEAR)
    rr_bgr = cv2.cvtColor(rr, cv2.COLOR_RGB2BGR)
    reg = out[y0:y1, x0:x1]
    reg[cm > 127] = rr_bgr[cm > 127]
    out[y0:y1, x0:x1] = reg


def run_aot(aot, rgb, mask):
    """m-i-t AOT-GAN 去字（整張、原解析度）。img[-1,1]+mask{0,1,1=擦}→輸出[-1,1]。rgb/回傳=RGB。"""
    h, w = rgb.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    img = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    m = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    img_n = (img.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
    m_n = (m >= 128).astype(np.float32)[None, None]  # {0,1}，1=擦
    img_n = img_n * (1 - m_n)                          # ★ 把洞歸零再餵（m-i-t inpainting_lama_mpe L92）
    out = aot.run(["output"], {"img": img_n, "mask": m_n})[0][0]
    out = (((out.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8))[:h, :w]
    res = rgb.copy()
    res[mask > 127] = out[mask > 127]
    return res


def main():
    img_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "boxfill"
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    det = ort.InferenceSession(f"{paths.MODELS}/comictextdetector.pt.onnx", providers=["CPUExecutionProvider"])
    seg_rep = import_seg_rep()(thresh=0.3)
    quads, segm = detect_seg(det, seg_rep, rgb)
    regions = group(quads, w, h)
    mask = seg_mask(regions, segm, w, h, use_seg=(method != "boxfill_quad"))
    print(f"{len(quads)} 行 → {len(regions)} 區；遮罩像素 {int((mask>127).sum())}")
    # 每區背景 std（auto 判定預覽）
    busy = [r for r in regions if bg_std(bgr, mask, r["bbox"]) >= AUTO_STD]
    print(f"auto 門檻 std={AUTO_STD} → 忙碌(走lama) {len(busy)} 區 / 均勻(走boxfill) {len(regions)-len(busy)} 區")
    for r in regions:
        s = bg_std(bgr, mask, r["bbox"])
        print(f"  bbox={r['bbox']}  bg_std={s:5.1f}  {'lama' if s>=AUTO_STD else 'boxfill'}")
    if method in ("boxfill", "boxfill_quad"):
        out = boxfill(bgr, mask)
    elif method == "aot":
        aot = ort.InferenceSession("/tmp/aot/inpainting_aot.onnx", providers=["CPUExecutionProvider"])
        out = cv2.cvtColor(run_aot(aot, rgb, mask), cv2.COLOR_RGB2BGR)
    elif method == "aot_region":  # AOT 逐格（每區裁窗各跑一次）
        aot = ort.InferenceSession("/tmp/aot/inpainting_aot.onnx", providers=["CPUExecutionProvider"])
        out = bgr.copy()
        for r in regions:
            win = window_of(r["bbox"], w, h)
            if not win:
                continue
            x0, y0, x1, y1 = win
            mcrop = mask[y0:y1, x0:x1]
            if mcrop.max() == 0:
                continue
            db = cv2.cvtColor(run_aot(aot, rgb[y0:y1, x0:x1], mcrop), cv2.COLOR_RGB2BGR)
            reg = out[y0:y1, x0:x1].copy()
            reg[mcrop > 127] = db[mcrop > 127]
            out[y0:y1, x0:x1] = reg
    else:
        lama = ort.InferenceSession(f"{paths.MODELS}/lama-manga.onnx", providers=["CPUExecutionProvider"])
        out = bgr.copy()
        if method == "auto":
            boxmask = mask.copy()
            busy_r = [r for r in regions if bg_std(bgr, mask, r["bbox"]) >= AUTO_STD]
            for r in busy_r:  # 忙碌區從 boxfill 移除、改 lama
                x0, y0, x1, y1 = r["bbox"]; boxmask[y0:y1, x0:x1] = 0
            out = boxfill(out, boxmask)
            for r in busy_r:
                win = window_of(r["bbox"], w, h)
                if win:
                    run_window(lama, rgb, mask, win, out)
        else:  # lama：全部逐區 lama
            for r in regions:
                win = window_of(r["bbox"], w, h)
                if win:
                    run_window(lama, rgb, mask, win, out)
    name = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(f"{paths.OUT}/dev_{name}_{method}.png", out)
    # 遮罩疊圖（紅）看蓋得準不準
    ov = bgr.copy(); ov[mask > 127] = (0, 0, 255)
    cv2.imwrite(f"{paths.OUT}/dev_{name}_mask.png", ov)
    print(f"→ dev_{name}_{method}.png + dev_{name}_mask.png")


if __name__ == "__main__":
    main()
