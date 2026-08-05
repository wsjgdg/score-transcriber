"""简谱 OCR 字形解析单元测试（纯函数，不依赖 OCR 引擎，可在 CI 运行）。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omr_jianpu as O  # noqa: E402


def G(ch, x=10, y=100, w=20, h=20):
    """构造单个字形（左上角原点；数字本体高 20、center y=110）。"""
    return {"char": ch, "x": x, "y": y, "w": w, "h": h}


# --- parse_jianpu_glyphs ---
def test_plain_quarter():
    # 中音 1，四分音符（bpm120 -> beat_sec=0.5）
    assert O.parse_jianpu_glyphs([G("1")], bpm=120) == [
        (0.0, 0.5, 60, 1.0, None)
    ]


def test_high_octave_dot():
    # 1^：数字上方点 -> 高八度 (+12) = 72
    g = [G("1"), {"char": ".", "x": 10, "y": 70, "w": 5, "h": 5}]
    assert O.parse_jianpu_glyphs(g, bpm=120) == [(0.0, 0.5, 72, 1.0, None)]


def test_low_octave_dot():
    # 1_：数字下方点 -> 低八度 (-12) = 48
    g = [G("1"), {"char": ".", "x": 10, "y": 130, "w": 5, "h": 5}]
    assert O.parse_jianpu_glyphs(g, bpm=120) == [(0.0, 0.5, 48, 1.0, None)]


def test_rest():
    assert O.parse_jianpu_glyphs([G("0")], bpm=120) == [
        (0.0, 0.5, -1, 1.0, None)
    ]


def test_consecutive_degrees():
    g = [G(c, 10 + i * 25) for i, c in enumerate("123")]
    assert O.parse_jianpu_glyphs(g, bpm=120) == [
        (0.0, 0.5, 60, 1.0, None),
        (0.5, 1.0, 62, 1.0, None),
        (1.0, 1.5, 64, 1.0, None),
    ]


def test_eighth_underline():
    # 1-：下方减时线 -> 八分音符（0.5 拍 = 0.25s @120bpm）
    g = [G("1"), {"char": "_", "x": 10, "y": 130, "w": 20, "h": 4}]
    assert O.parse_jianpu_glyphs(g, bpm=120) == [(0.0, 0.25, 60, 1.0, None)]


def test_dotted_quarter():
    # 1.：右侧同高附点 -> 附点四分（1.5 拍 = 0.75s @120bpm）
    g = [G("1"), {"char": ".", "x": 35, "y": 100, "w": 5, "h": 5}]
    assert O.parse_jianpu_glyphs(g, bpm=120) == [(0.0, 0.75, 60, 1.0, None)]


def test_barline_ignored():
    g = [G("1"), {"char": "|", "x": 40, "y": 95, "w": 4, "h": 30},
         G("2", 60)]
    assert O.parse_jianpu_glyphs(g, bpm=120) == [
        (0.0, 0.5, 60, 1.0, None),
        (0.5, 1.0, 62, 1.0, None),
    ]


def test_double_underline_sixteenth():
    # 两条减时线 -> 十六分（0.25 拍 = 0.125s）
    g = [G("1"),
         {"char": "_", "x": 10, "y": 130, "w": 20, "h": 4},
         {"char": "_", "x": 10, "y": 138, "w": 20, "h": 4}]
    assert O.parse_jianpu_glyphs(g, bpm=120) == [(0.0, 0.125, 60, 1.0, None)]


def test_empty():
    assert O.parse_jianpu_glyphs([]) == []
