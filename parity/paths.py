#!/usr/bin/env python3
"""parity harness 的集中路徑設定（取代各腳本硬編 → 單一真相）。
外部路徑用環境變數覆蓋，方便換機器 / CI / 公開後別人也跑得動。

env 覆蓋：
  YAKU_TEST_DIR     測試頁與 m-i-t 對照輸出的根（預設 ~/OneDrive/Manga/yakuyomi/test）
  YAKU_OCR_CTC_DIR  48px CTC 的 alphabet/checkpoint 夾（預設 /tmp/ocr-ctc）
  YAKU_ALPHABET     直接指定 alphabet 檔（預設 $YAKU_OCR_CTC_DIR/alphabet-all-v5.txt）
  YAKU_OCR_CKPT     直接指定 ckpt（預設 $YAKU_OCR_CTC_DIR/ocr-ctc.ckpt）
  YAKU_MIT_CLONE    manga-image-translator clone（預設 /mnt/d/Gits/manga-image-translator）
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))   # parity/
ROOT = os.path.dirname(HERE)                          # repo 根
OUT = os.path.join(HERE, "out")                       # parity/out（輸出/快取）


def _env(name, default):
    return os.environ.get(name, default)


# —— repo 內（本就可攜，集中為單一真相）——
MODELS = os.path.join(ROOT, "engine/src/main/assets/models")
FONT = os.path.join(ROOT, "engine/src/main/assets/fonts/NotoSansMonoCJK.ttc")
SANDBOX_TEST = os.path.join(ROOT, "app-sandbox/src/main/assets/test")
SANDBOX_PAGE = os.path.join(SANDBOX_TEST, "page.png")
API_KEYS = os.path.join(ROOT, "api-keys.properties")

# —— 外部（每台機器不同，env 可覆蓋）——
TEST_DIR = _env("YAKU_TEST_DIR", os.path.expanduser("~/OneDrive/Manga/yakuyomi/test"))
RAW_DIR = os.path.join(TEST_DIR, "raw")   # 測試原頁
MIT_DIR = os.path.join(TEST_DIR, "mit")   # m-i-t 對照輸出

OCR_CTC_DIR = _env("YAKU_OCR_CTC_DIR", "/tmp/ocr-ctc")
ALPHABET = _env("YAKU_ALPHABET", os.path.join(OCR_CTC_DIR, "alphabet-all-v5.txt"))
OCR_CKPT = _env("YAKU_OCR_CKPT", os.path.join(OCR_CTC_DIR, "ocr-ctc.ckpt"))

MIT_CLONE = _env("YAKU_MIT_CLONE", "/mnt/d/Gits/manga-image-translator")
