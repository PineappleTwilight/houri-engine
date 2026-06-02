#!/usr/bin/env python3
"""
OCR parity (M1)：對偵測到的文字行跑 48px CTC OCR ONNX，讀出日文。
裁切複製自 manga_translator/utils/generic.py:sort_pnts + Quadrilateral.get_transformed_region @ d5a3eee
CTC 解碼複製自 model_48px_ctc.py:decode_ctc_top1（greedy, blank=0, 收合重複+去blank）
"""
import os, json, glob
import numpy as np
import cv2
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "parity/out")
MODEL = os.path.join(OUT, "ocr_48px_ctc.onnx")
ALPHABET = "/tmp/ocr-ctc/alphabet-all-v5.txt"
IMG = os.path.join(ROOT, "app-sandbox/src/main/assets/test/page.png")
BOXES = os.path.join(OUT, "faithful_boxes.json")
TEXT_H, BLANK = 48, 0


def sort_pnts(pts):
    pts = np.array(pts, dtype=np.float64)
    pv = (pts[:, None] - pts[None]).reshape((16, -1))
    lsv = pv[np.argsort(np.linalg.norm(pv, axis=1))[[8, 10]]]
    if (lsv[0] * lsv[1]).sum() < 0:
        lsv[0] = -lsv[0]
    struc = np.abs(lsv.mean(axis=0))
    is_v = struc[0] <= struc[1]
    if is_v:
        pts = pts[np.argsort(pts[:, 1])]
        pts = pts[[*np.argsort(pts[:2, 0]), *(np.argsort(pts[2:, 0])[::-1] + 2)]]
        return pts, True
    pts = pts[np.argsort(pts[:, 0])]
    out = np.zeros_like(pts)
    out[[0, 3]] = sorted(pts[[0, 1]], key=lambda p: p[1])
    out[[1, 2]] = sorted(pts[[2, 3]], key=lambda p: p[1])
    return out, False


def transformed_region(img, pts, direction, th):
    p = pts.astype(np.float32)
    l1a, l1b = (p[0] + p[1]) / 2, (p[2] + p[3]) / 2
    l2a, l2b = (p[1] + p[2]) / 2, (p[3] + p[0]) / 2
    ratio = np.linalg.norm(l1b - l1a) / max(np.linalg.norm(l2b - l2a), 1e-6)
    src = pts.astype(np.int64).copy()
    im_h, im_w = img.shape[:2]
    x1, y1 = max(int(src[:, 0].min()), 0), max(int(src[:, 1].min()), 0)
    x2, y2 = min(int(src[:, 0].max()), im_w), min(int(src[:, 1].max()), im_h)
    crop = img[y1:y2, x1:x2]
    src[:, 0] -= x1
    src[:, 1] -= y1
    if direction == 'h':
        h, w = max(int(th), 2), max(int(round(th / ratio)), 2)
    else:
        w, h = max(int(th), 2), max(int(round(th * ratio)), 2)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M, _ = cv2.findHomography(src.astype(np.float32), dst, cv2.RANSAC, 5.0)
    if M is None:
        return None
    region = cv2.warpPerspective(crop, M, (w, h))
    if direction == 'v':
        region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return region


def ctc_decode(logits, dictionary, colors=None):
    """greedy CTC（blank=0、收合重複+去blank）。
    傳入 colors（[T,6]＝fg_rgb+bg_rgb，未 clamp）則回傳 (text, prob, fg, bg)；
    色＝保留的非空白 char 對應 timestep 取色、clip 0..1、整行平均 ×255（對齊 model_48px_ctc.decode_ctc_top1）。"""
    lp = logits - logits.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))  # log_softmax
    idx = lp.argmax(1)
    chars, probs, last = [], [], BLANK
    fgs, bgs = [], []
    for t in range(len(idx)):
        c = int(idx[t])
        if c != last and c != BLANK:
            ch = dictionary[c]
            sp = ch == '<SP>'
            chars.append(' ' if sp else ch)
            probs.append(lp[t, c])
            if colors is not None and not sp:
                cv = np.clip(colors[t], 0.0, 1.0)
                fgs.append(cv[:3]); bgs.append(cv[3:6])
        last = c
    text = ''.join(chars)
    prob = float(np.exp(np.mean(probs))) if probs else 0.0
    if colors is None:
        return text, prob
    fg = tuple(int(round(v * 255)) for v in (np.mean(fgs, axis=0) if fgs else (0.0, 0.0, 0.0)))
    bg = tuple(int(round(v * 255)) for v in (np.mean(bgs, axis=0) if bgs else (1.0, 1.0, 1.0)))
    return text, prob, fg, bg


def find_font():
    for c in (glob.glob("/mnt/d/Gits/manga-image-translator/fonts/*.[to][tc][cf]") +
              ["/mnt/c/Windows/Fonts/YuGothM.ttc", "/mnt/c/Windows/Fonts/msgothic.ttc",
               "/mnt/c/Windows/Fonts/meiryo.ttc", "/mnt/c/Windows/Fonts/msmincho.ttc"]):
        if os.path.exists(c):
            return c
    return None


def main():
    dictionary = [s[:-1] for s in open(ALPHABET, encoding="utf-8").readlines()]
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    img = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
    boxes = json.load(open(BOXES, encoding="utf-8"))["boxes"]

    results = []
    for i, b in enumerate(boxes):
        try:
            pts, is_v = sort_pnts(b["quad"])
            direction = 'v' if is_v else 'h'
            region = transformed_region(img, pts, direction, TEXT_H)
            if region is None or region.shape[1] < 2:
                continue
            x = np.transpose((region.astype(np.float32) - 127.5) / 127.5, (2, 0, 1))[None]
            char_logits, _ = sess.run(["char_logits", "color"], {"image": x})
            text, prob = ctc_decode(char_logits[0], dictionary)
            if text.strip():
                results.append({"i": i, "dir": direction, "prob": round(prob, 3), "text": text,
                                "quad": b["quad"]})
                print(f"[{i:2d}] {direction} p={prob:.2f}  {text}")
        except Exception as e:
            print(f"[{i:2d}] error: {e}")

    json.dump(results, open(os.path.join(OUT, "ocr_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    font = find_font()
    print(f"\nfont: {font}")
    if font:
        try:
            from PIL import Image, ImageDraw, ImageFont
            pim = Image.fromarray(img).convert("RGB")
            dr = ImageDraw.Draw(pim)
            fnt = ImageFont.truetype(font, 22)
            for r in results:
                q = np.array(r["quad"], np.int32)
                dr.line([tuple(map(int, p)) for p in q] + [tuple(map(int, q[0]))], fill=(255, 0, 0), width=3)
                x0, y0 = int(q[:, 0].min()), int(q[:, 1].min())
                dr.text((x0, max(0, y0 - 24)), r["text"], fill=(0, 110, 255), font=fnt)
            pim.save(os.path.join(OUT, "ocr_overlay.png"))
            print(f"overlay → {OUT}/ocr_overlay.png")
        except Exception as e:
            print(f"render skipped: {e}")
    print(f"讀出 {len(results)} 行")


if __name__ == "__main__":
    main()
