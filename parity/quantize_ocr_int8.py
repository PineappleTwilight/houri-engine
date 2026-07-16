#!/usr/bin/env python3
"""
把 48px CTC OCR 的 fp32 ONNX 動態量化成 int8（QUInt8）→ ocr_int8.onnx（產品實際載的 OCR 權重）。

重建鏈（本腳本是第 2 棒）：
  ocr-ctc.ckpt  --[export_ocr_onnx.py]-->  ocr_48px_ctc.onnx (fp32, ~165MB, opset 17, 動態軸 N/W)
                --[本腳本]------------->  ocr_int8.onnx      (int8,  ~44MB)
model spec: manga_translator/ocr/model_48px_ctc.py:OCR @ d5a3eee
  輸入 image[N,3,48,W]（normalize (x-127.5)/127.5）→ char_logits[N,T,dict] + color[N,T,6]

為什麼是「動態」量化、而不是 QDQ / 靜態量化：
  這個模型的輸入寬度 W 是動態的（每行文字 strip 的寬度隨字數而變），
  靜態量化需要一組「固定 shape」的校準集去離線算 activation 的 scale/zero-point，
  對 W 動態的模型不成立。動態量化只把「權重」離線量成 uint8，
  activation 的 scale/zero-point 在 runtime 依實際張量現算（DynamicQuantizeLinear）
  → 免校準集、免固定 shape，代價是每次推論多一點量化開銷。

weight_type=QUInt8（非預設的 QInt8）：
  這是 release 上那顆 ocr_int8.onnx 的做法——**已用逐位元比對反證**（見下方「可重現性」）：
  用 QUInt8 產出的 sha256 與 release 完全相同，所以當初就是這個設定。
  （ORT 的一般建議也是 ARM/x86 走 u8u8 的 ConvInteger 路徑；但本機是 x86，
   「ARM 上快 3.6x」這個數字**沒有在本機驗證過**，那是既有文件的宣稱，要驗得上真機。）

⚠ 兩個非跑不可的前置（沒有就直接失敗，不是可有可無的最佳化）：
  1) 必須先 quant_pre_process（常數摺疊）：torch 匯出的圖裡，layer4.5/conv1 的權重是
     `Conv <- Identity <- initializer`，ORT 的 Conv 量化器只認「input[1] 直接是 initializer」、
     不會穿過 Identity → 不預處理會炸 `ValueError: Expected onnx::Conv_1267 to be an initializer`。
     （export 時已經 do_constant_folding=True 了也還是會留這顆 Identity。）
  2) quant_pre_process 必須 skip_symbolic_shape=True：符號形狀推論碰到動態 W 會算不下去
     （`Cannot determine if floor(floor(W/2)/2) - 1 < 0` → `Incomplete symbolic shape inference`）。
     動態量化本來就不需要形狀推論，跳過無損；要的只是它的常數摺疊那一段。

可重現性（2026-07-16 實測）：
  同一顆 fp32 ONNX 進來，這條路徑產出的 int8 與 release models-v2 上那顆
  **逐位元相同**（sha256 353e68a5…29fa4c5c、43,625,294 B）。
  即量化這步是決定性的；能不能重現整條鏈，取決於前一棒 export_ocr_onnx.py 的 torch 版本
  （實測 torch 2.1.1+cu121 → fp32 sha256 3019b406…）。

用法：
  python3 parity/quantize_ocr_int8.py                    # 量化 + 驗證
  python3 parity/quantize_ocr_int8.py --no-verify        # 只量化
  python3 parity/quantize_ocr_int8.py --ref <ocr_int8.onnx>   # 另外跟既有(release)的 int8 對比
  YAKU_OCR_INT8_REF=<path> python3 parity/quantize_ocr_int8.py

驗證分兩層（真實 strip 拿不到時自動降級，不會假裝驗過）：
  A. 真 strip：有 --page + --boxes → 對每行跑 fp32 vs int8，比 CTC 解碼出的文字
     （逐行 exact match =「CTC parity」）。**預設兩者都在 repo 內、開箱即跑**：
       --page   app-sandbox/src/main/assets/test/demo03.png（paths.SANDBOX_PAGE）
       --boxes  parity/fixtures/faithful_boxes.json（paths.FAITHFUL_BOXES，30 框）
  B. 合成輸入：隨機張量 → 只驗 load + 數值等價（max|Δlogits|、argmax 一致率）

只想重驗 parity 數字、不重跑量化 → 用 parity/ocr_parity.py（吃現成的兩顆 onnx）。

alphabet：paths.ALPHABET 缺 ckpt 那份時會自動退回 repo 內 engine 資產那份
  （engine/src/main/assets/models/alphabet-all-v5.txt，與上游逐位元相同）⇒ 本腳本不必抓 zip。
  只有前一棒 export_ocr_onnx.py 需要 ckpt，那支會自己抓 ocr-ctc.zip + 驗 sha256 + 解壓
  （落 $YAKU_CKPT_DIR/ocr-ctc，預設 parity/out/ckpt/ocr-ctc）⇒ 整條鏈零手動下載。

跑完不留垃圾：量化那兩步一律在 parity/out 下的暫存 cwd 跑、跑完連夾刪（scratch_cwd()）。
  走對的路本來就不寫 cwd；真正會噴 157MiB 垃圾的是「沒帶 skip_symbolic_shape → 炸掉沒人收」
  那條，這個 guard 連例外路徑都收得乾淨。細節見 scratch_cwd 的 docstring。
"""
import argparse
import contextlib
import json
import os
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

# 集中路徑 + 共用的 sha256（與三支轉換腳本同一支），見 paths.py
from paths import OUT, ALPHABET, SANDBOX_PAGE, FAITHFUL_BOXES, sha256_of

FP32 = os.path.join(OUT, "ocr_48px_ctc.onnx")
INT8 = os.path.join(OUT, "ocr_int8.onnx")
BOXES = FAITHFUL_BOXES   # 入庫的 30 框 fixture（來歷見該檔 _provenance）

# release models-v2 上那顆的權威數字（models.json）——拿來對照，不是拿來要求逐位元相同
REF_SIZE = 43_625_294
REF_SHA = "353e68a5506a6b8967905cd9b3c59e67708df1bc6812e105aa54d4e829fa4c5c"

TEXT_H = 48


@contextlib.contextmanager
def scratch_cwd():
    """把 cwd 挪到 parity/out 下的暫存夾再跑量化，跑完（含炸掉）連夾刪掉。

    要防的是什麼（2026-07-16 實測釐清，別照舊說法轉述）：
      * **走對的路（skip_symbolic_shape=True）其實不往 cwd 寫任何東西** —— 實測 cwd 全乾淨。
      * 會噴垃圾的是**走錯、然後炸掉**那條：quant_pre_process 沒帶 skip_symbolic_shape 時，
        符號形狀推論會先把模型連權重 dump 到 cwd，再因動態 W 算不下去而丟例外
        （`Cannot determine if floor(floor(W/2)/2) - 1 < 0`）→ **例外沒人收拾、垃圾留在 cwd**：
          - uuid1 命名的 onnx external-data dump ~157MiB（onnx/external_data_helper.py:136
            `file_name = str(uuid.uuid1())`）—— **連 .gitignore 都擋不到**（*.onnx 擋不住 uuid 檔名）
          - sym_shape_infer_temp.onnx（這個才被 *.onnx 擋掉）
        從 repo 根跑 → 157MiB 未追蹤垃圾直接掉進 repo 根（先前就這樣髒過一次）。
    這些寫入在 library 內部、沒有 API 可改路徑 ⇒ 只能把 cwd 挪開。用 TemporaryDirectory ⇒
    **即使中間那步丟例外也照樣收乾淨**（實測：炸掉當下夾內確有 uuid dump，離開後 parity/out
    與 repo 根皆零殘留）＝日後有人改壞這裡也不會再髒到工作區。
    """
    old = os.getcwd()
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quant.", dir=OUT) as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(old)


def quantize(src, dst, keep_pre=None):
    """fp32 ONNX → int8 動態量化（權重 QUInt8）。兩步，缺一不可（見檔頭 ⚠）。"""
    t0 = time.time()
    # chdir 前先絕對化，否則相對路徑會跟著 cwd 跑掉
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    pre = os.path.abspath(keep_pre) if keep_pre else (dst + ".pre.tmp")
    with scratch_cwd():
        # 1) 常數摺疊：把 `Conv <- Identity <- initializer` 收成 initializer，量化器才吃得下。
        #    skip_symbolic_shape=True：動態 W 會讓符號形狀推論算不下去，且量化不需要它。
        quant_pre_process(src, pre, skip_symbolic_shape=True)
        # 2) 動態量化
        quantize_dynamic(pre, dst, weight_type=QuantType.QUInt8)
    if keep_pre is None:
        os.remove(pre)
    return time.time() - t0


# ── 驗證：取真 strip ─────────────────────────────────────────────

def load_strips(page, boxes_path):
    """用偵測器產的 quad 切出 48px 文字 strip（與 ocr_parity.py 同一套裁切邏輯）。"""
    import cv2  # 只有真 strip 這條路要 cv2
    from ocr_parity import sort_pnts, transformed_region  # 重用，別複製第二份裁切碼

    img = cv2.cvtColor(cv2.imread(page), cv2.COLOR_BGR2RGB)
    boxes = json.load(open(boxes_path, encoding="utf-8"))["boxes"]
    strips = []
    for i, b in enumerate(boxes):
        pts, is_v = sort_pnts(b["quad"])
        direction = "v" if is_v else "h"
        region = transformed_region(img, pts, direction, TEXT_H)
        if region is None or region.shape[1] < 2:
            continue
        x = np.transpose((region.astype(np.float32) - 127.5) / 127.5, (2, 0, 1))[None]
        strips.append((i, direction, x))
    return strips


def decode(logits, dictionary):
    from ocr_parity import ctc_decode  # greedy CTC（blank=0），與產品/parity 同一份
    return ctc_decode(logits, dictionary)


def sess_of(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def verify(fp32, int8, ref, page, boxes_path):
    print("\n" + "=" * 72)
    print("驗證")
    print("=" * 72)

    models = {"fp32": fp32, "int8": int8}
    if ref:
        models["ref-int8"] = ref
    sessions = {}
    for tag, p in models.items():
        try:
            sessions[tag] = sess_of(p)
            print(f"  載入 {tag:8s} ✓  {p}")
        except Exception as e:
            print(f"  載入 {tag:8s} ✗  {e}")
            return
    print()

    # ── 取輸入：優先真 strip，拿不到就降級成合成輸入 ──
    strips, real = [], False
    if page and boxes_path and os.path.exists(page) and os.path.exists(boxes_path):
        try:
            strips = load_strips(page, boxes_path)
            real = True
            print(f"輸入：真 strip {len(strips)} 行（{os.path.basename(page)}）")
        except Exception as e:
            print(f"真 strip 取不到（{e}）→ 降級成合成輸入")
    if not strips:
        rng = np.random.default_rng(0)
        strips = [(i, "h", rng.standard_normal((1, 3, 48, w), dtype=np.float32))
                  for i, w in enumerate((64, 128, 256, 512))]
        print(f"輸入：合成隨機張量 {len(strips)} 筆（真 strip 不可得 → 只驗 load + 數值等價，"
              f"不驗文字）")
    print()

    # ── 逐 strip 前向 ──
    outs = {tag: [] for tag in sessions}
    times = {tag: 0.0 for tag in sessions}
    for _i, _d, x in strips:
        for tag, s in sessions.items():
            t0 = time.time()
            cl, _cv = s.run(["char_logits", "color"], {"image": x})
            times[tag] += time.time() - t0
            outs[tag].append(cl[0])

    # ── 數值等價 ──
    print("-- 數值等價（vs fp32）--")
    for tag in sessions:
        if tag == "fp32":
            continue
        dmax = mism = tot = 0.0
        for a, b in zip(outs["fp32"], outs[tag]):
            n = min(a.shape[0], b.shape[0])
            if a.shape[0] != b.shape[0]:
                print(f"   ! {tag} T 對不上：{a.shape} vs {b.shape}（取前 {n}）")
            d = np.abs(a[:n] - b[:n])
            dmax = max(dmax, float(d.max()))
            mism += float((a[:n].argmax(1) != b[:n].argmax(1)).sum())
            tot += n
        agree = 100.0 * (1 - mism / tot) if tot else 0.0
        print(f"  {tag:8s}  max|Δlogits| = {dmax:7.3f}   argmax 一致 = {agree:6.2f}% "
              f"({int(tot - mism)}/{int(tot)} timestep)")
    print(f"\n-- 前向耗時（x86 CPU，僅供參考；ARM 才是產品場景）--")
    for tag in sessions:
        print(f"  {tag:8s}  {times[tag] * 1000:7.1f} ms / {len(strips)} 筆")

    # ── 文字級 CTC parity（只有真 strip 有意義）──
    if not real:
        print("\n（合成輸入 → 跳過 CTC 文字 parity）")
        return
    if not os.path.exists(ALPHABET):
        print(f"\n（找不到 alphabet {ALPHABET} → 跳過 CTC 文字 parity；"
              f"設 YAKU_OCR_CTC_DIR 或 YAKU_ALPHABET）")
        return

    dictionary = [s[:-1] for s in open(ALPHABET, encoding="utf-8").readlines()]
    texts = {tag: [decode(cl, dictionary) for cl in outs[tag]] for tag in sessions}

    print(f"\n-- CTC 文字 parity（fp32 vs int8，逐行 exact match）--")
    same = 0
    for k, (i, d, _x) in enumerate(strips):
        t32, p32 = texts["fp32"][k]
        t8, p8 = texts["int8"][k]
        eq = t32 == t8
        same += eq
        mark = "=" if eq else "≠"
        line = f"  [{i:2d}] {d} {mark} fp32 p={p32:.2f} {t32!r}"
        if not eq:
            line += f"\n         int8 p={p8:.2f} {t8!r}"
        print(line)
    n = len(strips)
    print(f"\n  exact-match: {same}/{n} = {100.0 * same / n:.1f}%")

    if "ref-int8" in texts:
        same_r = sum(texts["int8"][k][0] == texts["ref-int8"][k][0] for k in range(n))
        same_r32 = sum(texts["fp32"][k][0] == texts["ref-int8"][k][0] for k in range(n))
        print(f"  我方 int8 vs ref-int8 : {same_r}/{n} = {100.0 * same_r / n:.1f}%")
        print(f"  ref-int8 vs fp32      : {same_r32}/{n} = {100.0 * same_r32 / n:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="48px CTC OCR fp32 ONNX → int8 動態量化")
    ap.add_argument("--src", default=FP32, help=f"fp32 ONNX（預設 {FP32}）")
    ap.add_argument("--dst", default=INT8, help=f"輸出 int8 ONNX（預設 {INT8}）")
    ap.add_argument("--ref", default=os.environ.get("YAKU_OCR_INT8_REF", ""),
                    help="既有(release)的 int8 ONNX，拿來對照（可用 YAKU_OCR_INT8_REF）")
    ap.add_argument("--page", default=SANDBOX_PAGE, help="驗證用的頁圖（預設 repo 內 demo03.png）")
    ap.add_argument("--boxes", default=BOXES, help="該頁的 quad（預設 parity/fixtures/faithful_boxes.json）")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit(f"找不到 fp32 ONNX：{args.src}\n先跑：python3 parity/export_ocr_onnx.py")

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    # 註：models.json 的 "165MB -> 44MB" 是十進位 MB；這裡印 MiB（157.3 → 41.6），同一個東西。
    src_mb = os.path.getsize(args.src) / 1024 / 1024
    print(f"src  {args.src}  ({src_mb:.1f} MiB)")
    print(f"     sha256 {sha256_of(args.src)}")
    print("量化中（QUInt8 動態）…")
    took = quantize(args.src, args.dst)

    size = os.path.getsize(args.dst)
    dst_sha = sha256_of(args.dst)
    print(f"\ndst  {args.dst}  ({size / 1024 / 1024:.1f} MiB, {size:,} B)  {took:.1f}s")
    print(f"     sha256 {dst_sha}")
    print(f"     壓縮   {src_mb:.1f} MiB → {size / 1024 / 1024:.1f} MiB "
          f"（{src_mb * 1024 * 1024 / size:.2f}x）")
    d = size - REF_SIZE
    print(f"     vs release models-v2 ({REF_SIZE:,} B)：{d:+,} B "
          f"（{100.0 * d / REF_SIZE:+.3f}%）")
    if dst_sha == REF_SHA:
        print(f"     ✓ 與 release models-v2 那顆 **逐位元相同**（sha256 對上）")
    else:
        # 逐位元不同不等於壞掉：量化器/torch 版本、浮點細節都會變 sha。
        # 判準是下面的驗證：載得起來 + 數值等價 + CTC 文字 parity。
        print(f"     ⚠ sha256 與 release 不同（expect {REF_SHA[:16]}…）")
        print(f"       這不必然是失敗——換 torch/ORT 版本就會變；判準看下面的驗證")

    if not args.no_verify:
        verify(args.src, args.dst, args.ref, args.page, args.boxes)


if __name__ == "__main__":
    main()
