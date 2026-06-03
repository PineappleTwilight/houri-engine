"""import smoke：抓「from paths import …」之類的破壞。只列 import 時無重量級副作用的模組
（不載模型、不打 API、不需 torch/opencc，也沒有 module 層級執行碼）。"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAFE = [
    "paths",
    "mit_grouping",
    "ctd_reference",
    "ocr_parity",
    "translate_parity",
    "typeset_parity",
    "pipeline_parity",
    "inpaint_parity",
    "group_exp",
    "emit_grouping_fixture",
]


@pytest.mark.parametrize("mod", SAFE)
def test_imports_clean(mod):
    importlib.import_module(mod)
