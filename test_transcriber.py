"""transcriber 纯函数单元测试（不依赖音频/模型，CI 随 requirements 安装依赖后运行）。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transcriber as T  # noqa: E402


# --- tokenize_lyrics ---
def test_tokenize_lyrics_space_split():
    assert T.tokenize_lyrics("你 好 世 界") == ["你", "好", "世", "界"]


def test_tokenize_lyrics_char_split_no_space():
    # 无空格按字符切分（CJK 一字一音）
    assert T.tokenize_lyrics("你好") == ["你", "好"]


def test_tokenize_lyrics_english_word():
    assert T.tokenize_lyrics("hello world") == ["hello", "world"]


def test_tokenize_lyrics_strips_punct():
    assert T.tokenize_lyrics("，。！？") == []           # 纯标点剔除
    assert T.tokenize_lyrics("（你好）") == ["你", "好"]  # 无空格按字切分，剔除包裹标点（中文按字）


def test_tokenize_lyrics_empty():
    assert T.tokenize_lyrics("") == []
    assert T.tokenize_lyrics("   ") == []


# --- align_lyrics_by_order ---
def test_align_lyrics_by_order():
    groups = [{"onset": 0.0}, {"onset": 1.0}, {"onset": 2.0}]
    assert T.align_lyrics_by_order(groups, ["a", "b"]) == ["a", "b", None]
    assert T.align_lyrics_by_order(groups, ["a", "b", "c", "d"]) == ["a", "b", "c"]


def test_align_lyrics_by_order_no_tokens():
    groups = [{"onset": 0.0}, {"onset": 1.0}]
    assert T.align_lyrics_by_order(groups, []) == [None, None]


# --- 音高/时值映射 ---
def test_pitch_to_jianpu_c4():
    assert T._pitch_to_jianpu(60) == "1"


def test_pitch_to_jianpu_octave_dots():
    assert T._pitch_to_jianpu(72) == "1" + T._DOT_ABOVE     # 高八度
    assert T._pitch_to_jianpu(48) == "1" + T._DOT_BELOW     # 低八度


def test_dur_to_lily():
    assert T._dur_to_lily(1.0) == "4"
    assert T._dur_to_lily(0.5) == "8"
    assert T._dur_to_lily(2.0) == "2"


# --- _build_groups ---
def test_build_groups_separate_and_chord():
    notes = [(0.0, 0.5, 60, 0.9, None), (0.5, 1.0, 64, 0.8, None)]
    assert len(T._build_groups(notes, 120)) == 2
    # 同 1/4 拍网格内的两音应合并为和弦，pitches 升序
    chord = [(0.0, 0.5, 60, 0.9, None), (0.01, 0.5, 64, 0.8, None)]
    g2 = T._build_groups(chord, 120)
    assert len(g2) == 1
    assert g2[0]["pitches"] == [60, 64]


def test_build_groups_conf_mean():
    chord = [(0.0, 0.5, 60, 0.9, None), (0.0, 0.5, 64, 0.7, None)]
    g = T._build_groups(chord, 120)[0]
    assert abs(g["conf"] - 80.0) < 1e-6


# --- build_jianpu ---
def test_build_jianpu_header_and_truncate():
    notes = [(i * 0.5, i * 0.5 + 0.5, 60, 0.9, None) for i in range(5)]
    jp = T.build_jianpu(notes, 120, 4)
    assert jp.startswith("1=C")
    many = [(i * 0.5, i * 0.5 + 0.5, 60, 0.9, None)
            for i in range(T.MAX_JIANPU_EVENTS + 50)]
    assert "截断" in T.build_jianpu(many, 120, 4)


def test_build_jianpu_reuses_groups():
    # 传入预构建 groups 应与原地构建结果一致（避免重复分组）
    notes = [(0.0, 0.5, 60, 0.9, None), (0.5, 1.0, 64, 0.8, None)]
    groups = T._build_groups(notes, 120)
    assert T.build_jianpu(notes, 120, 4, groups=groups) == \
        T.build_jianpu(notes, 120, 4)


# --- _sanitize_lily_lyric (LilyPond 歌词安全转义) ---
def test_sanitize_lily_lyric():
    assert T._sanitize_lily_lyric("a$b#c%") == "abc"      # 危险字符移除
    assert T._sanitize_lily_lyric("  hello  world  ") == "hello world"
    assert T._sanitize_lily_lyric("") == ""


# --- 选调 / 移调 (#17) ---
def test_resolve_key_default_and_unknown():
    toff, label, lily, sharps, delta = T.resolve_key("C")
    assert (toff, label, lily, sharps, delta) == (0, "C", "c", 0, 0)
    # 未知调名回退 C
    assert T.resolve_key("ZZZ")[1] == "C"


def test_resolve_key_table_values():
    # 典型调：主音半音数、LilyPond 音名、MusicXML 调号、移调半音
    assert T.resolve_key("D")[0] == 2
    assert T.resolve_key("D")[2] == "d"
    assert T.resolve_key("bE")[3] == -3
    assert T.resolve_key("G") == (7, "G", "g", 1, -5)   # >6 半音下行
    assert T.resolve_key("bB") == (10, "♭B", "bf", -2, -2)


def test_transpose_note_list():
    notes = [(0.0, 0.5, 62, 0.9, None), (0.0, 0.5, 60, 0.8, None)]
    up = T.transpose_note_list(notes, 2)
    assert up[0][2] == 64 and up[1][2] == 62
    # 零移调返回原表
    assert T.transpose_note_list(notes, 0) is notes
    # 下限钳到 0
    low = T.transpose_note_list([(0.0, 0.5, 1, 0.9, None)], -5)
    assert low[0][2] == 0


def test_transpose_note_list_octave_shift():
    # 全局八度偏移 = transpose_note_list(notes, 12 * shift)
    notes = [(0.0, 0.5, 72, 0.9, None), (0.6, 1.0, 60, 0.8, None)]
    down1 = T.transpose_note_list(notes, 12 * -1)   # C5->C4, C4->C3
    assert down1[0][2] == 60 and down1[1][2] == 48
    down2 = T.transpose_note_list(notes, 12 * -2)   # 再降八度
    assert down2[0][2] == 48 and down2[1][2] == 36
    up1 = T.transpose_note_list(notes, 12 * 1)      # C5->C6
    assert up1[0][2] == 84
    # 极低音下移不出现负数（钳到 0）
    assert T.transpose_note_list([(0.0, 0.5, 5, 0.9, None)], 12 * -1)[0][2] == 0


def test_jianpu_movable_do_header_and_degree():
    # 选调 D：表头应为 1=D，且 D 音(62) 在首调下唱名为 1，E 音(64) 为 2
    notes = [(0.0, 0.5, 62, 0.9, None)]
    jp = T.build_jianpu(notes, 120, 4, key="D")
    assert jp.startswith("1=D")
    assert "1" in T._jianpu_degree(62, 2) and "1" == T._jianpu_degree(62, 2).replace(T._DOT_ABOVE, "").replace(T._DOT_BELOW, "")
    assert T._jianpu_degree(64, 2).replace(T._DOT_ABOVE, "").replace(T._DOT_BELOW, "") == "2"
    # 固定调 1=C 行为不变
    assert T.build_jianpu(notes, 120, 4, key="C").startswith("1=C")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
