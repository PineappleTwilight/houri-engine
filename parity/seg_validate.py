"""
seg 細筆畫遮罩驗證（M3 去字升級的規格）。
comictextdetector 吐 blk/seg/det，我們本來只用 det（文字行框）。seg[1,1,1024,1024]=逐像素文字機率。
這支把 seg letterbox 還原→原圖尺寸→上色疊圖，確認：①筆畫對齊文字 ②夠細（不是整塊）③挑門檻。
Kotlin Detector 要逐位元對齊這裡的前處理與還原。

⚠ comictextdetector 已於 commit 163ee2b **退役**（DBNet 為唯一偵測器），該模型不在任何 models
  release 裡 ⇒ 這支凍在歷史、跑不動很正常，留著當當初 seg 遮罩的規格紀錄。
"""
import os
import cv2
import numpy as np
import onnxruntime as ort

from paths import ROOT, OUT, MODELS, SANDBOX_TEST as ASSETS  # 集中路徑，見 paths.py
MODEL = os.path.join(MODELS, "comictextdetector.pt.onnx")
os.makedirs(OUT, exist_ok=True)
SZ = 1024

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])


def seg_mask(img):
    """letterbox→model→seg 還原回原圖尺寸的 float 機率圖 (H,W)。"""
    h, w = img.shape[:2]
    r = min(SZ / h, SZ / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    rs = cv2.resize(img, (nw, nh))
    canvas = np.zeros((SZ, SZ, 3), np.uint8)
    canvas[:nh, :nw] = rs                       # 圖貼左上、pad 在右/下
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]
    _blk, seg, _det = sess.run(["blk", "seg", "det"], {"images": blob})
    segm = seg[0, 0, :nh, :nw]                   # 裁掉 pad
    return cv2.resize(segm, (w, h))              # 回原圖尺寸


for name in ("demo04.png", "demo03.png"):   # 舊名 demo3.png / page.png，commit ea3e166 改名（demo2.png 已剃除）
    img = cv2.imread(os.path.join(ASSETS, name))
    segm = seg_mask(img)
    for th in (0.3, 0.5):
        m = (segm > th).astype(np.uint8)
        red = img.copy()
        red[m > 0] = (0, 0, 255)
        blend = cv2.addWeighted(img, 0.45, red, 0.55, 0)
        cv2.imwrite(os.path.join(OUT, f"seg_{name[:-4]}_th{th}.png"), blend)
        print(f"{name} th={th}: 文字像素佔 {100*m.mean():.2f}%")

print("done ->", OUT)
