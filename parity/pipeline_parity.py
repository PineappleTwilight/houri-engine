#!/usr/bin/env python3
"""
端到端 parity：一張圖 → 偵測→OCR→行合併→翻譯→s2twp→去字→排版 → 成品頁。
跨情境測試用。用法：python3 pipeline_parity.py <img1> [img2 ...]
重用既有 parity 模組的函式（detect/ocr/typeset），orchestration 內聯。
"""
import os, sys, re, json
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import translate_parity as tp
import typeset_parity as ts
from ctd_reference import letterbox, import_seg_rep
from ocr_parity import sort_pnts, transformed_region, ctc_decode, is_ignore

ROOT, OUT = tp.ROOT, tp.OUT
MODELS = os.path.join(ROOT, "engine/src/main/assets/models")
ALPHABET = "/tmp/ocr-ctc/alphabet-all-v5.txt"
INPUT, BOX_THRESH, TEXT_H, LAMA, WIN, GAP, FS_RATIO = 1024, 0.6, 48, 512, 1.7, 1.0, 1.5
IGNORE_BUBBLE = 0  # config.ocr.ignore_bubble：1–50 開啟，跳過彩色/非氣泡 SFX 類文字（預設 0＝關）
OCR_PROB = 0.5     # config.ocr.prob：OCR 平均信心 < 此值就丟（剃除低信心誤讀；m-i-t 預設 0.5）


def detect(sess, seg_rep, rgb):
    h, w = rgb.shape[:2]
    lb, _, dw, dh = letterbox(rgb, (INPUT, INPUT))
    blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (INPUT, INPUT))
    _, _, det = sess.run(["blk", "seg", "det"], {"images": blob.astype(np.float32)})
    det = det[..., :det.shape[2] - dh, :det.shape[3] - dw]
    bb, sc = seg_rep(None, det, height=h, width=w)
    return [b for b, s in zip(bb[0], sc[0]) if s > BOX_THRESH]


def ocr_all(sess, dic, rgb, quads):
    out = []
    for b in quads:
        pts, is_v = sort_pnts(b)
        d = "v" if is_v else "h"
        reg = transformed_region(rgb, pts, d, TEXT_H)
        if reg is None or reg.shape[1] < 2:
            continue
        if is_ignore(reg, IGNORE_BUBBLE):  # 跳過彩色/非氣泡 SFX 類文字（預設關）
            continue
        x = np.transpose((reg.astype(np.float32) - 127.5) / 127.5, (2, 0, 1))[None]
        cl, col = sess.run(["char_logits", "color"], {"image": x})
        t, prob, fg, bg = ctc_decode(cl[0], dic, col[0])
        if t.strip() and prob >= OCR_PROB:  # 低信心誤讀（如把 SFX 框成文字）剃除
            out.append({"dir": d, "text": t, "quad": np.array(b).tolist(), "fg": list(fg), "bg": list(bg)})
    return out


def _bbox(q):
    a = np.array(q, float)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def group(res):
    n = len(res)
    bb = [_bbox(r["quad"]) for r in res]
    fs = [min(b[2] - b[0], b[3] - b[1]) for b in bb]
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if res[i]["dir"] != res[j]["dir"]:
                continue
            cs = min(fs[i], fs[j])
            if cs <= 0 or max(fs[i], fs[j]) / cs > FS_RATIO:
                continue
            dx = max(0, max(bb[i][0], bb[j][0]) - min(bb[i][2], bb[j][2]))
            dy = max(0, max(bb[i][1], bb[j][1]) - min(bb[i][3], bb[j][3]))
            if (dx * dx + dy * dy) ** 0.5 < cs * GAP:
                par[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    regions = []
    for mem in groups.values():
        d = res[mem[0]]["dir"]
        mem.sort(key=(lambda i: -bb[i][2]) if d == "v" else (lambda i: (bb[i][1], bb[i][0])))
        regions.append({
            "dir": d, "jp": "".join(res[i]["text"] for i in mem), "n": len(mem),
            "bbox": [min(bb[i][0] for i in mem), min(bb[i][1] for i in mem),
                     max(bb[i][2] for i in mem), max(bb[i][3] for i in mem)],
            "quads": [res[i]["quad"] for i in mem],
            # region 色＝各成員行平均（對齊 m-i-t update_font_colors 的逐行平均）
            "fg": [int(np.mean([res[i]["fg"][k] for i in mem])) for k in range(3)],
            "bg": [int(np.mean([res[i]["bg"][k] for i in mem])) for k in range(3)],
        })
    return regions


def translate(regions, key, filter_text=None):
    if not regions:
        return regions
    user = "\n".join(f"<|{i + 1}|>{r['jp']}" for i, r in enumerate(regions))
    msgs = [{"role": "system", "content": tp.CHAT_SYSTEM_TEMPLATE.format(to_lang=tp.TO_LANG)},
            {"role": "user", "content": tp.SAMPLE_IN}, {"role": "assistant", "content": tp.SAMPLE_OUT},
            {"role": "user", "content": user}]
    raw = re.sub(r"(</think>)?<think>.*?</think>", "", tp.call_deepseek(key, msgs), flags=re.DOTALL)
    trans = {}
    for line in raw.splitlines():
        m = re.match(r"^\s*<\|(\d+)\|>\s*(.*)$", line)
        if m:
            trans[int(m.group(1))] = m.group(2).strip()
    for i, r in enumerate(regions):
        r["cht"] = trans.get(i + 1, "")
    # 翻譯後過濾（m-i-t filter chain）：被丟的區不進去字、保留原圖
    kept = [r for r in regions if not ts.should_filter(r["jp"], r.get("cht", ""), filter_text)]
    if len(kept) != len(regions):
        print(f"  filter: 丟棄 {len(regions) - len(kept)} 區（空白/數字/regex/未譯）")
    return kept


def inpaint(sess, rgb, regions):
    h, w = rgb.shape[:2]
    out = rgb.copy()
    mask = np.zeros((h, w), np.uint8)
    for r in regions:
        for q in r["quads"]:
            cv2.fillPoly(mask, [np.array(q, np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ww, hh = (x1 - x0) * WIN, (y1 - y0) * WIN
        wx0, wy0 = max(0, int(cx - ww / 2)), max(0, int(cy - hh / 2))
        wx1, wy1 = min(w, int(cx + ww / 2)), min(h, int(cy + hh / 2))
        if wx1 - wx0 < 8 or wy1 - wy0 < 8:
            continue
        crop, cm = out[wy0:wy1, wx0:wx1], mask[wy0:wy1, wx0:wx1]
        if cm.max() == 0:
            continue
        ch, cw = crop.shape[:2]
        ii = (cv2.resize(crop, (LAMA, LAMA), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.).transpose(2, 0, 1)[None]
        mi = (cv2.resize(cm, (LAMA, LAMA), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.)[None, None]
        rr = sess.run(["output"], {"image": ii, "mask": mi})[0][0]
        rr = cv2.resize(np.clip(rr.transpose(1, 2, 0) * 255., 0, 255).astype(np.uint8), (cw, ch), interpolation=cv2.INTER_LINEAR)
        out[wy0:wy1, wx0:wx1] = np.where((cm > 0)[..., None], rr, crop)
    return out


def typeset(rgb, regions):
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    npimg = np.asarray(im)
    for r in regions:
        cht = r.get("cht", "")
        x0, y0, x1, y1 = r["bbox"]
        if (x1 - x0) < 8 or (y1 - y0) < 8 or ts.should_filter(r.get("jp", ""), cht, ts.FILTER_TEXT):
            continue
        tb = (x0, y0, x1, y1)
        fg, bg = ts.resolve_colors(npimg, tb, r.get("fg") or (0, 0, 0), r.get("bg") or (255, 255, 255))
        if ts.is_cjk(cht):
            ts.draw_v(im, dr, cht, tb, fg, bg)
        else:
            ts.draw_h(dr, cht, tb, fg, bg)
    return im


def main():
    det = ort.InferenceSession(f"{MODELS}/comictextdetector.pt.onnx", providers=["CPUExecutionProvider"])
    ocr = ort.InferenceSession(f"{MODELS}/ocr_48px_ctc.onnx", providers=["CPUExecutionProvider"])
    lama = ort.InferenceSession(f"{MODELS}/lama-manga.onnx", providers=["CPUExecutionProvider"])
    seg_rep = import_seg_rep()(thresh=0.3)
    dic = [s[:-1] for s in open(ALPHABET, encoding="utf-8").readlines()]
    key = tp.read_key()
    for path in sys.argv[1:]:
        name = os.path.splitext(os.path.basename(path))[0]
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        quads = detect(det, seg_rep, rgb)
        regions = group(ocr_all(ocr, dic, rgb, quads))
        regions = translate(regions, key, ts.FILTER_TEXT)
        cleaned = inpaint(lama, rgb, regions)
        # 快取中間結果，之後可只重跑排版（retypeset.py）不必重打 DeepSeek
        Image.fromarray(cleaned).save(os.path.join(OUT, f"inpainted_{name}.png"))
        json.dump([{"dir": r["dir"], "jp": r["jp"], "n": r["n"], "bbox": r["bbox"], "cht": r.get("cht", ""),
                    "fg": r.get("fg"), "bg": r.get("bg")} for r in regions],
                  open(os.path.join(OUT, f"cache_{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        final = typeset(cleaned, regions)
        dst = os.path.join(OUT, f"final_{name}.png")
        final.save(dst)
        print(f"{name}: {len(quads)} 框 → {len(regions)} 區 → {dst}")


if __name__ == "__main__":
    main()
