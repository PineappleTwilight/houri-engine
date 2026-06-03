# `parity/` — 桌面驗證 harness

[English](README.md) · **中文**

**不出貨。** 這是開發專用的 Python harness，跑跟 Kotlin `:engine` 同樣的 pipeline 階段，
讓我們能檢查裝置端移植是否跟參考（[manga-image-translator](https://github.com/zyddnys/manga-image-translator)，
m-i-t）行為一致——*在*信任它上機之前。

為什麼存在：引擎把 m-i-t（Python/torch）重寫成 Kotlin/ONNX。那種移植沒辦法逐行 diff，所以正確性是
「同輸入 → 近輸出」。這些腳本產出那份參考輸出（grouping 還有自動化跨語言斷言）。
見 [`../docs/ARCHITECTURE_zh.md`](../docs/ARCHITECTURE_zh.md#兩半)。

---

## 設定

```bash
pip install -r parity/requirements.txt    # numpy, opencv-python, onnxruntime, pillow,
                                          # networkx, shapely（torch/opencc 只給工具用）
```

腳本預期的外部輸入（開發機）：

| 是什麼 | 位置（目前） | 備註 |
|--------|--------------|------|
| ONNX 模型 | `engine/src/main/assets/models/*.onnx` | 跟引擎載入的同一批 |
| OCR 字元表 | `/tmp/ocr-ctc/alphabet-all-v5.txt` | CTC 解碼用字典 |
| m-i-t clone | `/mnt/d/Gits/manga-image-translator` | 規格本 / `SegDetectorRepresenter` 來源 |
| 測試頁 | `~/OneDrive/Manga/yakuyomi/test/raw/*.jpg`，m-i-t 輸出在 `…/test/mit/` | 比對 fixture |
| API key | `api-keys.properties`（repo 根、gitignored） | 翻譯用 `DEEPSEEK_API_KEY=` |

輸出落在 `parity/out/`（快取 JSON + 比對 PNG；gitignored）。

> 這些路徑目前各腳本各自硬編——集中化是後續工作。

---

## 有什麼

**端到端**
- `pipeline_parity.py <img…>`——整條 detect→OCR→group→translate→inpaint→typeset。
  主驅動；寫 `out/final_<name>.png` + 快取中間結果。

**逐階段 parity**（跑/檢視單一階段）
- `ctd_reference.py [page]`——偵測：faithful（m-i-t 後處理）vs simplified，並排。
- `ocr_parity.py`——對偵測框做 48px CTC 辨識。
- `group_exp.py <name…>`——分組：我們的區域 vs m-i-t 的，畫成框。
- `translate_parity.py`——OCR 出的日文 → DeepSeek → 繁中。
- `merge_translate_parity.py`——先併行再翻。
- `inpaint_parity.py`——對區域跑 LaMa 去字。
- `typeset_parity.py [v|h|auto]` / `retypeset.py <name…>`——排版（retypeset = 從快取重排、不重打 LLM；快速調版用）。

**規格本**（ground truth，從 m-i-t 複製——跟 `.upstream-ref` 同步）
- `mit_grouping.py`——m-i-t 的兩階段分組（`merge_bboxes_text_region`），自含。
- `ctd_reference.py`——也拉 m-i-t 的偵測後處理。

**工具**
- `export_ocr_onnx.py`——把 48px CTC checkpoint 匯出成 ONNX（build-time，需 torch）。
- `seg_validate.py`——在不同閾值下檢視偵測器的 `seg` 筆畫遮罩。
- `emit_grouping_fixture.py`——產生 Kotlin 分組測試 fixture（見下）。

---

## 跨語言分組測試

唯一一個橫跨兩語言的自動化 parity 檢查：

```
emit_grouping_fixture.py                          # 桌面：偵測真實頁面、用 mit_grouping 分組、
   → engine/src/test/kotlin/.../GroupingFixture.kt #   把偵測到的行 + 期望區域 emit 成 Kotlin
                                                   #
gradlew :engine:testDebugUnitTest                 # 裝置端：把同樣的行餵給 Kotlin Grouping，
   → GroupingParityTest                            #   斷言區域（bbox ±2px）+ 角度（±1°）吻合
```

所以動了 Kotlin 分組（或重新同步 `mit_grouping.py`）會被自動抓到：改、重跑 `emit_grouping_fixture.py`、跑測試。
其餘階段仍靠目視驗證（拿 `out/*.png` 對 `…/test/mit/`）。

---

## 典型流程

1. `pipeline_parity.py raw/002.jpg raw/012.jpg`——端到端，目視 `out/final_*.png` vs `mit/`。
2. 只調版？改 `typeset_parity.py`、`retypeset.py 002 012`（不打 LLM）。
3. 動了分組？`emit_grouping_fixture.py` 然後 `:engine:testDebugUnitTest`。
4. 同步了 m-i-t？bump `mit_grouping.py` / `.upstream-ref`，重跑相關 parity、修到綠。
