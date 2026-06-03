#!/usr/bin/env python3
"""
把 48px CTC OCR 的 OCR.forward 匯出成 ONNX（M1）。
ported spec: manga_translator/ocr/model_48px_ctc.py:OCR @ d5a3eee
  輸入 image[N,3,48,W]（normalize (x-127.5)/127.5）→ char_logits[N,T,dict] + color[N,T,6]
  解碼（greedy CTC, blank=0）留在 Kotlin/parity 端做。
"""
import sys, types, importlib.util, os
import torch

from paths import MIT_CLONE as MIT, OCR_CKPT as CKPT, ALPHABET, OUT as _OUTDIR  # 集中路徑，見 paths.py
OUT = os.path.join(_OUTDIR, "ocr_48px_ctc.onnx")


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
    print(f"exported → {OUT} ({os.path.getsize(OUT) // 1024 // 1024} MB)")


if __name__ == "__main__":
    main()
