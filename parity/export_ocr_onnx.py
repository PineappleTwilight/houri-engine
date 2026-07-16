#!/usr/bin/env python3
"""
把 48px CTC OCR 的 OCR.forward 匯出成 ONNX（重建鏈第 1 棒）。
ported spec: manga_translator/ocr/model_48px_ctc.py:OCR @ d5a3eee
  輸入 image[N,3,48,W]（normalize (x-127.5)/127.5）→ char_logits[N,T,dict] + color[N,T,6]
  解碼（greedy CTC, blank=0）留在 Kotlin/parity 端做。

重建鏈：
  ocr-ctc.ckpt  --[本腳本]--------------->  ocr_48px_ctc.onnx (fp32, ~165MB, opset 17, 動態軸 N/W)
                --[quantize_ocr_int8.py]->  ocr_int8.onnx      (int8, ~44MB＝產品實際載的那顆)

ckpt 缺檔會自動抓（見 ensure_ocr_ctc）。這顆比 DBNet/AOT 多一步解壓：上游把
ckpt + alphabet 包成 ocr-ctc.zip 發（_MODEL_MAPPING['archive']）。

實測（2026-07-16，torch 2.1.1+cu121 / x86 WSL2）：
  產出 ocr_48px_ctc.onnx = 164,974,063 B
    sha256 3019b406dfc7b3dedf3d0a17a8b5f78c4b483712e3acaa3987db078d9b2c35d8（重跑一致）
  這條可重現性綁 torch 版本 —— 換版本大概率只是數值等價、非逐位元相同（那不是失敗）。

用法：
  python3 parity/export_ocr_onnx.py
"""
import importlib.util
import os
import sys
import types
import zipfile

import torch

# 集中路徑 + 共用的下載/驗 hash（三支轉換腳本同一支 fetch），見 paths.py
from paths import (ALPHABET, CKPT_DIR, MIT_CLONE as MIT, OCR_CKPT as CKPT, OCR_CTC_DIR,
                   OUT as _OUTDIR, fetch, sha256_of)

OUT = os.path.join(_OUTDIR, "ocr_48px_ctc.onnx")

# 上游權重（m-i-t ocr/model_48px_ctc.py:Model48pxCTCOCR._MODEL_MAPPING @ d5a3eee 逐字照抄）。
# ⚠️ 上游宣告的 hash 是 **zip 本身**的；解壓出來的兩個檔上游沒宣告 hash ⇒ 我們也不自己編一個
#    塞進來（自算的 hash 只能證明「解壓沒壞」、證明不了「這是上游那顆」）。zip 驗過就夠。
CKPT_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip"
CKPT_SHA256 = "fc61c52f7a811bc72c54f6be85df814c6b60f63585175db27cb94a08e0c30101"
MEMBERS = ("ocr-ctc.ckpt", "alphabet-all-v5.txt")   # ＝上游 _MODEL_MAPPING['archive'] 的兩個 key


def ensure_ocr_ctc():
    """確保 ocr-ctc.ckpt（+ alphabet）在手：缺檔就抓 zip → 驗 sha256 → 解壓到 OCR_CTC_DIR。

    與 DBNet/AOT 的差別＝多一步解壓（上游這顆是 zip 發的）。zip 留在快取夾當快取，
    下次重跑 fetch 驗過 hash 就跳過下載。
    """
    if os.path.exists(CKPT) and os.path.exists(ALPHABET):
        print(f"ocr-ctc ✓ 已在手（ckpt {CKPT}）")
        return
    zip_path = os.path.join(CKPT_DIR, "ocr-ctc.zip")
    fetch(CKPT_URL, zip_path, CKPT_SHA256, label="ocr-ctc.zip")

    os.makedirs(OCR_CTC_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for member in MEMBERS:
            # 照 basename 找（別信 zip 內的路徑：巢狀夾會落空、`../` 會寫出夾外＝zip-slip）
            hits = [n for n in names if os.path.basename(n) == member]
            if not hits:
                raise SystemExit(f"ocr-ctc.zip 裡找不到 {member}（namelist={names}）")
            dst = os.path.join(OCR_CTC_DIR, member)
            with z.open(hits[0]) as src, open(dst, "wb") as fp:
                fp.write(src.read())
            print(f"  解壓 {hits[0]} → {dst}（{os.path.getsize(dst):,} B）")

    # env 覆蓋（YAKU_OCR_CKPT）可能指到別處 ⇒ 解壓完仍要確認我們真的餵得到那個路徑
    missing = [p for p in (CKPT, ALPHABET) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "解壓完成、但這些路徑仍不存在（是不是 YAKU_OCR_CKPT / YAKU_ALPHABET 指到別處？）：\n  "
            + "\n  ".join(missing) + f"\n解壓目的地＝{OCR_CTC_DIR}")


def load_OCR():
    """只取 OCR nn.Module：stub 掉 model_48px_ctc.py 頂部的 m-i-t 套件 import。"""
    for n in ["manga_translator", "manga_translator.ocr", "manga_translator.utils"]:
        m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    cfg = types.ModuleType("manga_translator.config"); cfg.OcrConfig = object
    sys.modules["manga_translator.config"] = cfg
    common = types.ModuleType("manga_translator.ocr.common"); common.OfflineOCR = object
    sys.modules["manga_translator.ocr.common"] = common
    u = sys.modules["manga_translator.utils"]
    u.TextBlock = object; u.Quadrilateral = object; u.AvgMeter = object; u.chunks = lambda *a, **k: None
    bub = types.ModuleType("manga_translator.utils.bubble"); bub.is_ignore = lambda *a, **k: False
    sys.modules["manga_translator.utils.bubble"] = bub
    spec = importlib.util.spec_from_file_location(
        "manga_translator.ocr.model_48px_ctc", f"{MIT}/manga_translator/ocr/model_48px_ctc.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["manga_translator.ocr.model_48px_ctc"] = mod
    spec.loader.exec_module(mod)
    return mod.OCR


def main():
    ensure_ocr_ctc()
    os.makedirs(_OUTDIR, exist_ok=True)   # 空白 clone 沒有 parity/out
    OCR = load_OCR()
    dictionary = [s[:-1] for s in open(ALPHABET, encoding="utf-8").readlines()]
    print(f"dict size = {len(dictionary)}")

    model = OCR(dictionary, 768)
    sd = torch.load(CKPT, map_location="cpu")
    sd = sd["model"] if "model" in sd else sd
    for k in ["encoders.layers.0.pe.pe", "encoders.layers.1.pe.pe", "encoders.layers.2.pe.pe"]:
        sd.pop(k, None)
    res = model.load_state_dict(sd, strict=False)
    print(f"load_state_dict: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.unexpected_keys:
        print("  unexpected[:5]:", res.unexpected_keys[:5])
    model.eval()

    dummy = torch.zeros(1, 3, 48, 256)
    with torch.no_grad():
        cl, cv = model(dummy)
    print(f"forward OK: char_logits{tuple(cl.shape)} color{tuple(cv.shape)}  (T per 256px = {cl.shape[1]})")

    torch.onnx.export(
        model, dummy, OUT,
        input_names=["image"], output_names=["char_logits", "color"],
        dynamic_axes={"image": {0: "N", 3: "W"}, "char_logits": {0: "N", 1: "T"}, "color": {0: "N", 1: "T"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"exported → {OUT}（{os.path.getsize(OUT):,} B）")
    print(f"  sha256 {sha256_of(OUT)}")
    print(f"下一棒：python3 parity/quantize_ocr_int8.py  → ocr_int8.onnx（產品實際載的那顆）")


if __name__ == "__main__":
    main()
