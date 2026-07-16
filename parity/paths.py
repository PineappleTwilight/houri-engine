#!/usr/bin/env python3
"""parity harness 的集中路徑設定（取代各腳本硬編 → 單一真相）＋上游 ckpt 取得。

外部路徑用環境變數覆蓋，方便換機器 / CI / 公開後別人也跑得動。
另提供 fetch()／sha256_of()：三支模型轉換腳本（DBNet / AOT / OCR）共用的下載+驗 hash
一支（別再各寫一套）；各腳本只保留自己那顆的 url + sha256 常數（照抄上游 _MODEL_MAPPING）。

env 覆蓋：
  YAKU_TEST_DIR     測試頁與 m-i-t 對照輸出的根（預設 ~/OneDrive/Manga/yakuyomi/test）
  YAKU_OCR_CTC_DIR  48px CTC 的 alphabet/checkpoint 夾（預設 $YAKU_CKPT_DIR/ocr-ctc；
                    內容來自上游 ocr-ctc.zip，見 docs/BUILD_MODELS.md）
  YAKU_ALPHABET     直接指定 alphabet 檔（預設 $YAKU_OCR_CTC_DIR/alphabet-all-v5.txt，
                    缺檔時退回 repo 內 engine 資產那份 —— 兩者逐位元相同，見 ALPHABET）
  YAKU_BOXES        OCR parity 用的 quad fixture（預設 parity/fixtures/faithful_boxes.json）
  YAKU_OCR_CKPT     直接指定 ckpt（預設 $YAKU_OCR_CTC_DIR/ocr-ctc.ckpt）
  YAKU_MIT_CLONE    manga-image-translator clone（預設 /mnt/d/Gits/manga-image-translator）
  YAKU_CKPT_DIR     上游 ckpt 下載快取夾（預設 parity/out/ckpt，已 gitignore）
  YAKU_INPAINT_CKPT 直接指定去字 ckpt（預設 $YAKU_CKPT_DIR/inpainting.ckpt；缺檔時
                    export_aot_ncnn.py 會自動從 m-i-t beta-0.3 release 下載 + 驗 sha256）
  YAKU_DET_CKPT     直接指定偵測 ckpt（預設 $YAKU_CKPT_DIR/detect-20241225.ckpt；缺檔時
                    export_dbnet_ncnn.py 會自動從 m-i-t beta-0.3 release 下載 + 驗 sha256）
  YAKU_PNNX         pnnx 執行檔（預設 ~/.local/bin/pnnx；`pip install pnnx` 會裝這支）
"""
import hashlib
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))   # parity/
ROOT = os.path.dirname(HERE)                          # repo 根
OUT = os.path.join(HERE, "out")                       # parity/out（輸出/快取，已 gitignore）
FIXTURES = os.path.join(HERE, "fixtures")             # parity/fixtures（入庫的驗證素材）


def _env(name, default):
    return os.environ.get(name, default)


# —— repo 內（本就可攜，集中為單一真相）——
MODELS = os.path.join(ROOT, "engine/src/main/assets/models")
FONT = os.path.join(ROOT, "engine/src/main/assets/fonts/NotoSansMonoCJK.ttc")
SANDBOX_TEST = os.path.join(ROOT, "app-sandbox/src/main/assets/test")
# 舊名 page.png：commit ea3e166「測試圖重整」把它**改名**成 demo03.png（不是刪除，位元完全相同
# sha256 dde9ae9e…）。OCR parity 的 30 框 fixture 就是對這張圖產的，兩者是一組。
SANDBOX_PAGE = os.path.join(SANDBOX_TEST, "demo03.png")
API_KEYS = os.path.join(ROOT, "api-keys.properties")

# OCR parity 的 quad fixture（30 框，入庫；來歷與「為何凍結」見該檔的 _provenance）
FAITHFUL_BOXES = _env("YAKU_BOXES", os.path.join(FIXTURES, "faithful_boxes.json"))

# —— 外部（每台機器不同，env 可覆蓋）——
TEST_DIR = _env("YAKU_TEST_DIR", os.path.expanduser("~/OneDrive/Manga/yakuyomi/test"))
RAW_DIR = os.path.join(TEST_DIR, "raw")   # 測試原頁
MIT_DIR = os.path.join(TEST_DIR, "mit")   # m-i-t 對照輸出

MIT_CLONE = _env("YAKU_MIT_CLONE", "/mnt/d/Gits/manga-image-translator")

# 上游 ckpt 快取（模型轉換腳本用；parity/out 已 gitignore ⇒ 大檔不入庫）
CKPT_DIR = _env("YAKU_CKPT_DIR", os.path.join(OUT, "ckpt"))
INPAINT_CKPT = _env("YAKU_INPAINT_CKPT", os.path.join(CKPT_DIR, "inpainting.ckpt"))
DET_CKPT = _env("YAKU_DET_CKPT", os.path.join(CKPT_DIR, "detect-20241225.ckpt"))

# 48px CTC 的 ckpt/alphabet（來自上游 ocr-ctc.zip，解壓到同一個 ckpt 快取夾）
OCR_CTC_DIR = _env("YAKU_OCR_CTC_DIR", os.path.join(CKPT_DIR, "ocr-ctc"))
OCR_CKPT = _env("YAKU_OCR_CKPT", os.path.join(OCR_CTC_DIR, "ocr-ctc.ckpt"))

# alphabet：解壓 ocr-ctc.zip 那份優先；缺檔時退回 repo 內 engine 資產那份。
# 兩者逐位元相同（sha256 c1295ae1962e69e35b5b225a0405d1f3432e368c9941d23bfd3acda12654da33）＝
# 上游 alphabet-all-v5.txt 原檔 ⇒ 純解碼的腳本（ocr_parity.py）不必抓 ckpt zip 就能從空白 clone 跑。
# 匯出 ONNX（export_ocr_onnx.py）仍需要 ckpt，那條路的 zip 免不了。
_ALPHABET_CKPT = os.path.join(OCR_CTC_DIR, "alphabet-all-v5.txt")
_ALPHABET_REPO = os.path.join(MODELS, "alphabet-all-v5.txt")
ALPHABET = _env("YAKU_ALPHABET",
                _ALPHABET_CKPT if os.path.exists(_ALPHABET_CKPT) else _ALPHABET_REPO)

# pnnx（torch trace → ncnn 的轉換器）：`pip install pnnx` 裝進 ~/.local/bin
PNNX = _env("YAKU_PNNX", os.path.expanduser("~/.local/bin/pnnx"))


# —— 上游 ckpt 取得（三支轉換腳本共用；各腳本只留自己那顆的 url + sha256）——

def sha256_of(path, chunk=1 << 20):
    """檔案 sha256（串流讀 —— ckpt 有 300MB，別整顆讀進記憶體）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dst, sha256=None, label=None):
    """確保 dst 在手：缺檔（或 hash 不符）就下載，給了 sha256 就一律驗過才回。

    行為：
      已存在 + hash 對    → 直接回（不重下）
      已存在 + hash 不符  → 當成壞檔/錯檔，重下一次；仍不符 → SystemExit（絕不硬用）
      已存在 + 沒給 hash  → 直接回（無從驗證，會印警告）
    下載走 <dst>.part 再 os.replace ⇒ 中斷/失敗不會留半截檔冒充成品（下次重跑會重下）。

    ⚠️ sha256 一律照抄上游 _MODEL_MAPPING 宣告的值，別自己算一個填進去 ——
    自己算的只能證明「檔案沒在下載途中壞掉」，證明不了「這是上游那顆」。
    """
    name = label or os.path.basename(dst)
    if os.path.exists(dst):
        if sha256 is None:
            print(f"{name} ✓ {dst}（{os.path.getsize(dst):,} B；未給 hash ⇒ 沒驗）")
            return dst
        got = sha256_of(dst)
        if got == sha256:
            print(f"{name} ✓ {dst}（{os.path.getsize(dst):,} B、sha256 對上上游宣告值）")
            return dst
        print(f"⚠️ {name} sha256 不符（{got[:16]}… != {sha256[:16]}…）→ 重新下載")

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".part"
    print(f"下載 {url}\n  → {dst}")
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as fp:
            total = int(r.headers.get("Content-Length") or 0)
            step = max(total // 10, 1 << 25) if total else (1 << 25)   # 每 ~10% 或 32MiB 印一行
            done = mark = 0
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                fp.write(block)
                done += len(block)
                if done >= mark:
                    pct = f"（{100.0 * done / total:.0f}%）" if total else ""
                    print(f"  … {done:,} B{pct}", flush=True)
                    mark += step
        # 連線中途斷掉時 read() 是回 b'' 收工、**不會丟例外** ⇒ 不比對長度的話，
        # 半截檔會就這樣 os.replace 成「成品」（實測會）。在 replace 前擋掉，
        # 快取夾就永遠不會出現半截檔（有 sha 也只是下一輪才發現，不如現在就別放進去）。
        if total and done != total:
            raise SystemExit(f"{name} 下載不完整：{done:,}/{total:,} B —— 連線中斷？重跑即重下")
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)     # 別留半截檔 —— 下次跑會被誤認成「已下載」
        raise

    size = os.path.getsize(dst)
    if sha256 is None:
        print(f"{name} ✓ 下載完成（{size:,} B；未給 hash ⇒ 沒驗）")
        return dst
    got = sha256_of(dst)
    if got != sha256:
        raise SystemExit(f"{name} 下載後 sha256 不符！\n  期望 {sha256}\n  實得 {got}\n  ({dst})")
    print(f"{name} ✓ 下載完成（{size:,} B、sha256 驗過）")
    return dst
