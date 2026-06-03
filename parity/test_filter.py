"""typeset_parity.should_filter——翻譯後過濾鏈（別把空白/數字/未譯的蓋回去）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import typeset_parity as ts


def test_blank_filtered():
    assert ts.should_filter("あ", "")


def test_pure_digit_filtered():
    assert ts.should_filter("一", "123")


def test_untranslated_filtered():
    assert ts.should_filter("hello", "hello")


def test_regex_filtered():
    assert ts.should_filter("x", "廣告請洽", "廣告")


def test_normal_kept():
    assert not ts.should_filter("ねこ", "貓")
