#!/usr/bin/env python3
"""
把 m-i-t 的預設偵測器（DBNet：ResNet34 backbone + DB head）從上游 ckpt 轉成 NCNN。
產出 = 引擎跑的 dbnet_detect.ncnn.param / .bin（models.json 的 detector 角色、models-v3 release）。

ported spec: manga_translator/detection/default_utils/DBNet_resnet34.py:TextDetection @ d5a3eee
             載入方式對齊 manga_translator/detection/default.py:DefaultDetector._load @ d5a3eee

流程：detect-20241225.ckpt → nn.Module → torch.jit.trace → pnnx → *.ncnn.param/.bin

介面（blob 名必須一致 —— 引擎 Detector.kt / ncnn_jni.cpp 就是照這個吃的）：
  in0  [1,3,H,W]  RGB、NCHW、normalize (x/127.5 - 1)
                  （引擎前處理 ImageOps.detectorChwDbnet：長邊 resize 到 1024、
                    pad 右下到 256 倍數、pad 區＝黑）
  out0 [1,2,H,W]  db（**全解析**）：ch0 = shrink_map **raw logits（未 sigmoid）**、
                  ch1 = threshold_map（已 sigmoid）。
                  ch0 未 sigmoid 是上游行為：default.py:23 `db = db.sigmoid()` 在模型外做
                  → 引擎 Detector.kt:59 自己套 sigmoid。改這裡會讓框全爆。
  out1 [1,1,?,?]  mask：文字筆畫遮罩（模型內含 Sigmoid、已是機率）。
                  ★ 尺寸隨平台不定：本機 x86 實測 = 半解析 H/2×W/2、arm64 實測 = 全解析 H×W
                    → 引擎 Detector.kt 是「配全解析上限緩衝 + 由 JNI 回實際尺寸」動態讀的，
                      別在任何一端寫死（commit 7c62f78 就是修這個越界）。

⚠️ 不要量化：int8 實測完全吐不出框、ARM 上也沒更快 → 維持 pnnx 預設 fp16 storage（bin ~146MiB）。
⚠️ trace shape 不會被烘進 param：本網路 fully-conv（產出 param 只有 Convolution/Deconvolution/
   Pooling/Concat/Split/ReLU/BinaryOp，無 Reshape/Interp）⇒ 換尺寸照跑。
   但仍用引擎實跑的 768×1024 矩形 trace，理由見 TRACE_W/TRACE_H 註解。

實測（2026-07-16，torch 2.1.1+cu121 / torchvision 0.16.1 / pnnx 1.0.20260526 / x86 WSL2）：
  產出與現行 models-v3 release 的 dbnet_detect.ncnn.* **逐位元相同**
    param 13,392 B      sha256 9e6db2f8c6b0662ab00eb2100b3373d3c984a235eaac0e61c0b2a484ee1ff7b5
    bin   153,010,556 B sha256 f57bdbede7764a534c56e88be0269602259a7fcd47e54e8b7d954fd0fcc55c3d
  （＝models.json 的值。這條可重現性不保證跨 torch/pnnx 版本，換版本大概率只是數值等價、
    非逐位元相同 —— 那不是失敗，看下面的容差。）
  vs torch eager（真圖 006.jpg、768×1024）：
    out0 sigmoid(ch0) maxdiff 0.0034（mean 4.5e-05、corr 0.9999997）
    out1 mask maxdiff 0.272 但 mean 6.9e-06、僅 0.004% 像素差 >0.05、二值化@0.5 只翻 ~3/196k 像素
    ＝ncnn fp16 storage 在 sigmoid 陡處的捨入，對框/遮罩無實質影響。

用法：
  python3 parity/export_dbnet_ncnn.py              # 匯出 + 驗證
  python3 parity/export_dbnet_ncnn.py --skip-verify
ckpt 缺檔會自動從 m-i-t beta-0.3 release 下載（308MiB）+ 驗 sha256。
"""
import argparse
import importlib.util
import os
import subprocess
import sys

import numpy as np
import torch

# 集中路徑 + 共用的下載/驗 hash（三支轉換腳本同一支 fetch），見 paths.py
from paths import DET_CKPT as CKPT, MIT_CLONE as MIT, OUT as _OUTDIR, PNNX, fetch, sha256_of

# 上游權重（m-i-t detection/default.py:_MODEL_MAPPING @ d5a3eee 逐字照抄）
CKPT_URL = 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt'
CKPT_SHA256 = '67ce1c4ed4793860f038c71189ba9630a7756f7683b1ee5afb69ca0687dc502e'

# trace 用的輸入尺寸＝引擎實跑的樣子：dbnetInputSize=1024（Config.kt:30）、
# 1351×1920 的頁 → resize_aspect 到 721×1024 → pad 右下到 256 倍數 → 768×1024。
# 為何是矩形而非 1024×1024：**ncnn 對正方形 832–992 的輸入有 heap corruption**
# （x86 + arm64 皆重現，見 memory dbnet-detector-ncnn）→ 引擎才用 pad-到-256-倍數的矩形
# （維度永遠是 256 倍數 ⇒ 繞開那個帶）。trace shape 不影響產出的 param（fully-conv），
# 但照著引擎的真實形狀 trace，pnnx 的 shape 推導/圖優化才跟上線情境一致。
TRACE_W, TRACE_H = 768, 1024

OUTDIR = os.path.join(_OUTDIR, 'dbnet')     # parity/out/dbnet（已 gitignore）
TRACED = os.path.join(OUTDIR, 'dbnet.pt')   # pnnx 依輸入檔名決定輸出 → dbnet.ncnn.param/.bin
PARAM = os.path.join(OUTDIR, 'dbnet.ncnn.param')
BIN = os.path.join(OUTDIR, 'dbnet.ncnn.bin')


def fetch_ckpt():
    """確保 ckpt 在手：缺檔就從上游 release 下載（308MiB，慢），一律驗 sha256。"""
    fetch(CKPT_URL, CKPT, CKPT_SHA256, label='detect-20241225.ckpt')


def _pkg_shell(name, path):
    """
    建 package module（帶 __path__）但「不執行」它的 __init__ body。

    為什麼：manga_translator/__init__.py 會 `from .manga_translator import *`（把 translators
    → tiktoken → openai 整包拖進來）、manga_translator/detection/__init__.py 會 import
    paddle_rust → `from rusty_manga_image_translator import ...`（套件沒裝）。
    我們只要 DBNet_resnet34 這個純 torch 模組。先把 shell 放進 sys.modules，
    之後相對 import（DBNet_resnet34 的 `from . import DBHead`）靠 parent 的 __path__
    正常解析，但那些 __init__ 永不執行。純 in-memory，不碰使用者 clone 的任何檔案。
    """
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, '__init__.py'), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # 故意不呼叫 spec.loader.exec_module(mod)
    return mod


def load_TextDetection():
    """從釘住的 m-i-t clone 取 TextDetection nn.Module（不動 clone 的任何檔案/git 狀態）。"""
    if not os.path.isdir(MIT):
        raise SystemExit(f'找不到 m-i-t clone：{MIT}（用 YAKU_MIT_CLONE 指定）')
    base = os.path.join(MIT, 'manga_translator')
    _pkg_shell('manga_translator', base)
    _pkg_shell('manga_translator.detection', os.path.join(base, 'detection'))
    _pkg_shell('manga_translator.detection.default_utils', os.path.join(base, 'detection', 'default_utils'))

    name = 'manga_translator.detection.default_utils.DBNet_resnet34'
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(base, 'detection', 'default_utils', 'DBNet_resnet34.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.TextDetection


def build_model():
    TextDetection = load_TextDetection()
    model = TextDetection()                       # pretrained=None → resnet34 不下載 ImageNet 權重
    sd = torch.load(CKPT, map_location='cpu')
    sd = sd['model'] if 'model' in sd else sd     # default.py:45 同款
    model.load_state_dict(sd)                     # strict：權重對不上就要炸，不要默默半載
    model.eval()                                  # ★ 必須：DBHead.forward 用 self.training 分支
                                                  #   （train 會多吐 binary_maps → out0 變 3 channel）
    return model


def export():
    os.makedirs(OUTDIR, exist_ok=True)
    model = build_model()

    dummy = torch.zeros(1, 3, TRACE_H, TRACE_W)
    with torch.no_grad():
        db, mask = model(dummy)
    print(f'torch forward ✓ in{tuple(dummy.shape)} → db{tuple(db.shape)} mask{tuple(mask.shape)}')
    assert db.shape[1] == 2, f'db 應為 2 channel（eval 分支），實得 {db.shape[1]} → model.eval() 沒生效？'

    traced = torch.jit.trace(model, dummy)
    traced.save(TRACED)
    print(f'traced → {TRACED} ({os.path.getsize(TRACED):,} B)')

    # pnnx：預設 fp16=1（ncnn bin 存 fp16）＝我們要的；不加任何量化。
    cmd = [PNNX, os.path.basename(TRACED), f'inputshape=[1,3,{TRACE_H},{TRACE_W}]']
    print(f'$ (cd {OUTDIR} && {" ".join(cmd)})')
    r = subprocess.run(cmd, cwd=OUTDIR, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    print('\n'.join(f'  | {ln}' for ln in tail[-12:]))
    if r.returncode != 0 or not (os.path.exists(PARAM) and os.path.exists(BIN)):
        raise SystemExit(f'pnnx 失敗（rc={r.returncode}）')

    print(f'\nncnn → {PARAM} ({os.path.getsize(PARAM):,} B)\n      {BIN} ({os.path.getsize(BIN):,} B)')
    return PARAM, BIN


# ────────────────────────────── 驗證 ──────────────────────────────

def preprocess(img_bgr, size=1024, mult=256):
    """對齊引擎 ImageOps.detectorChwDbnet：長邊縮到 size、pad 右下到 mult 倍數、RGB、/127.5-1、NCHW。"""
    import cv2
    h, w = img_bgr.shape[:2]
    ratio = size / max(w, h)
    tw, th = max(1, round(w * ratio)), max(1, round(h * ratio))
    inW = tw + (mult - tw % mult) % mult
    inH = th + (mult - th % mult) % mult
    scaled = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((inH, inW, 3), dtype=np.uint8)     # pad 區＝黑，圖貼左上
    canvas[:th, :tw] = scaled
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1) / 127.5 - 1.0)
    return chw, inW, inH, ratio


def ncnn_forward(param, bin_, chw):
    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.load_param(param)
    net.load_model(bin_)
    ex = net.create_extractor()
    ex.input('in0', ncnn.Mat(chw))
    out = {}
    for name in ('out0', 'out1'):
        rc, m = ex.extract(name)
        if rc != 0:
            raise SystemExit(f'ncnn extract {name} 失敗 rc={rc}')
        out[name] = np.array(m)
    del ex, net
    return out


def verify(param, bin_):
    """三方比對：我們的 ncnn vs 現有 release 模型 vs torch eager（來源真相）。"""
    import cv2
    # 預設測試頁＝repo 內的 sandbox 測試圖 ⇒ 從空白 clone 也驗得動（別寫死個人路徑）。
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = os.environ.get(
        'YAKU_DBNET_TESTPAGE',
        os.path.join(_repo, 'app-sandbox/src/main/assets/test/ch34_006.jpg'))
    # 現有 release 權重的比對是**選配**（空白 clone 沒有它）：沒指到就只跟 torch eager 比。
    ref_dir = os.environ.get('YAKU_DBNET_REF', '')
    ref_param = os.path.join(ref_dir, 'dbnet_detect.ncnn.param')
    ref_bin = os.path.join(ref_dir, 'dbnet_detect.ncnn.bin')

    img = cv2.imread(page)
    if img is None:
        print(f'⚠ 跳過驗證：讀不到測試頁 {page}（YAKU_DBNET_TESTPAGE 可指定）')
        return
    chw, inW, inH, _ratio = preprocess(img)
    print(f'\n驗證用頁 {page}\n  {img.shape[1]}x{img.shape[0]} → 輸入 {inW}x{inH}')

    ours = ncnn_forward(param, bin_, chw)
    print(f'  我們的 ncnn ✓ out0{ours["out0"].shape} out1{ours["out1"].shape}')

    # (1) vs torch eager（真相；ncnn fp16 storage ⇒ 容差放寬）
    with torch.no_grad():
        db_t, mask_t = build_model()(torch.from_numpy(chw[None]))
    db_t, mask_t = db_t[0].numpy(), mask_t[0].numpy()
    d0 = np.abs(ours['out0'] - db_t).max()
    d1 = np.abs(ours['out1'] - mask_t).max()
    # 只比較「有訊號處」的機率誤差才有意義：logits 尾端絕對值大、fp16 相對誤差會放大
    p_ours = 1 / (1 + np.exp(-ours['out0'][0]))
    p_t = 1 / (1 + np.exp(-db_t[0]))
    dp = np.abs(p_ours - p_t).max()
    print(f'\n  [1] vs torch eager：out0 logits maxdiff={d0:.4f}｜'
          f'sigmoid(ch0) maxdiff={dp:.5f}｜out1 maxdiff={d1:.5f}')

    # (2) vs 現有 release 模型（逐位元不會同 —— pnnx/torch 版本、浮點細節；比數值等價）
    if not (os.path.exists(ref_param) and os.path.exists(ref_bin)):
        print(f'  [2] 跳過：找不到現有模型 {ref_param}（YAKU_DBNET_REF 可指定）')
        return
    ref = ncnn_forward(ref_param, ref_bin, chw)
    r0 = np.abs(ours['out0'] - ref['out0']).max()
    r1 = np.abs(ours['out1'] - ref['out1']).max()
    p_ref = 1 / (1 + np.exp(-ref['out0'][0]))
    rp = np.abs(p_ours - p_ref).max()
    same = (ours['out0'] == ref['out0']).all()
    print(f'  [2] vs 現有 release 模型：out0 logits maxdiff={r0:.6f}｜'
          f'sigmoid(ch0) maxdiff={rp:.7f}｜out1 maxdiff={r1:.7f}｜逐位元相同={same}')
    for f, tag in ((param, '我們'), (ref_param, '現有')):
        print(f'      {tag} param {os.path.getsize(f):,} B  sha256 {sha256_of(f)[:16]}…')
    for f, tag in ((bin_, '我們'), (ref_bin, '現有')):
        print(f'      {tag} bin   {os.path.getsize(f):,} B  sha256 {sha256_of(f)[:16]}…')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-verify', action='store_true')
    args = ap.parse_args()
    fetch_ckpt()
    param, bin_ = export()
    if not args.skip_verify:
        verify(param, bin_)
    print(f'\n完成。上線用檔名＝dbnet_detect.ncnn.param / .bin（models.json detector 角色）：\n'
          f'  cp {PARAM} <models>/dbnet_detect.ncnn.param\n'
          f'  cp {BIN} <models>/dbnet_detect.ncnn.bin')


if __name__ == '__main__':
    main()
