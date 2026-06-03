"""mit_grouping 的兩階段（連邊 + MST 分裂）行為——分組的規格本，純幾何好測。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mit_grouping import Quadrilateral, merge_bboxes_text_region


def _q(x0, y0, x1, y1):
    return Quadrilateral(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), "t", 1.0)


def _regions(quads, w=1000, h=1000):
    return list(merge_bboxes_text_region(quads, w, h))


def test_single_line_is_one_region():
    assert len(_regions([_q(0, 0, 200, 30)])) == 1


def test_close_aligned_lines_merge():
    # 兩條左對齊橫行、行距小 → 併成 1 區
    assert len(_regions([_q(0, 0, 200, 30), _q(0, 34, 200, 64)])) == 1


def test_three_stacked_lines_one_region():
    assert len(_regions([_q(0, 0, 200, 30), _q(0, 34, 200, 64), _q(0, 68, 200, 98)])) == 1


def test_far_lines_split():
    # 距離很遠 → 2 區（連通塊各自獨立）
    assert len(_regions([_q(0, 0, 200, 30), _q(0, 700, 200, 730)])) == 2
