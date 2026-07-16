#!/usr/bin/env python3
"""
M3 去字 parity：用 Koharu 的 lama-manga.onnx（512×512）block-aware 去字。
- 遮罩：偵測框（faithful_boxes.json）填充 + 膨脹。
- 視窗：合併後的氣泡區（merged_results.json）bbox 放大 1.7×（對齊 Koharu BALLOON_WINDOW_RATIO）。
- 每區裁出 → 縮 512 → LaMa(image,mask) → 貼回（只在遮罩處覆蓋）。
"""
import os, json
import numpy as np
import cv2
import onnxruntime as ort

from paths import ROOT, OUT, MODELS, SANDBOX_PAGE as IMG, FAITHFUL_BOXES  # 集中路徑，見 paths.py
LAMA = os.path.join(MODELS, "lama-manga.onnx")
SIZE = 512
WIN = 1.7


def main():
    lama = ort.InferenceSession(LAMA, providers=["CPUExecutionProvider"])
    img = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # 遮罩：所有偵測框填充 + 膨脹
    mask = np.zeros((H, W), np.uint8)
    for b in json.load(open(FAITHFUL_BOXES, encoding="utf-8"))["boxes"]:
        cv2.fillPoly(mask, [np.array(b["quad"], np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)

    regions = json.load(open(os.path.join(OUT, "merged_results.json"), encoding="utf-8"))
    out = img.copy()
    n = 0
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ww, hh = (x1 - x0) * WIN, (y1 - y0) * WIN
        wx0, wy0 = max(0, int(cx - ww / 2)), max(0, int(cy - hh / 2))
        wx1, wy1 = min(W, int(cx + ww / 2)), min(H, int(cy + hh / 2))
        if wx1 - wx0 < 8 or wy1 - wy0 < 8:
            continue
        crop = out[wy0:wy1, wx0:wx1]
        cmask = mask[wy0:wy1, wx0:wx1]
        if cmask.max() == 0:
            continue
        ch, cw = crop.shape[:2]
        cr = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        mr = cv2.resize(cmask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        img_in = (cr.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        mask_in = (mr.astype(np.float32) / 255.0)[None, None]
        res = lama.run(["output"], {"image": img_in, "mask": mask_in})[0][0]  # (3,512,512)
        res = np.clip(res.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        res = cv2.resize(res, (cw, ch), interpolation=cv2.INTER_LINEAR)
        m = (cmask > 0)[..., None]
        out[wy0:wy1, wx0:wx1] = np.where(m, res, crop)
        n += 1

    cv2.imwrite(os.path.join(OUT, "inpainted.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    cmp = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)])
    cv2.imwrite(os.path.join(OUT, "inpaint_compare.png"), cmp)
    print(f"去字完成：{n} 區 → inpainted.png / inpaint_compare.png")


if __name__ == "__main__":
    main()
