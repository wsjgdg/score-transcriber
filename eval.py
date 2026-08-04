#!/usr/bin/env python3.11
"""
准确率评测基准（方案 0）
======================
用“已知真值”的合成音频验证各增强开关的增益，避免盲调。

- 每个参考片段都有人工指定的真值音符（onset, dur, pitch=MIDI）。
- 合成音频时故意加入干扰（鼓点噪声 / 和弦 / 泛音），以暴露误捡。
- 对多组配置分别跑 transcriber.process，按 onset/pitch 对齐算指标：
    音符级 F1  : onset 与 pitch 同时对（核心指标）
    起音级 F1  : 仅看 onset 是否对（起音检测能力）
    音高准确率 : 已对齐音符中音高正确的比例
    时值 MAE   : 对齐音符的时值平均绝对误差(秒)
    误捡数 FP  : 预测里不在真值中的音符
    漏检数 FN  : 真值里没被预测到的音符

用法：
    python3.11 eval.py            # 跑全部片段×配置，打印对比表
    python3.11 eval.py --quick    # 仅 baseline 与 all，快速看趋势
"""
import os
import sys
import json
import tempfile
import argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transcriber as T

SR = T.SR


# ---------------------------------------------------------------------------
# 1. 真值片段定义（onset 秒, dur 秒, pitch MIDI）
# ---------------------------------------------------------------------------
CLIPS = {
    # 干净钢琴旋律（无鼓）：验证基础识别 + 更细量化
    "piano_melody": {
        "notes": [
            (0.0, 0.5, 60), (0.5, 0.5, 64), (1.0, 0.5, 67),
            (1.5, 1.0, 72), (2.5, 0.5, 67), (3.0, 0.5, 64),
        ],
        "drums": False, "kind": "piano",
    },
    # 人声旋律 + 鼓点：验证 HPSS 去鼓 + 人声单旋律模式
    "vocal_with_drums": {
        "notes": [
            (0.0, 0.4, 62), (0.4, 0.4, 64), (0.8, 0.4, 66), (1.2, 0.4, 67),
            (1.6, 0.6, 69), (2.2, 0.4, 67), (2.6, 0.4, 66), (3.0, 0.8, 64),
        ],
        "drums": True, "kind": "vocal",
    },
    # 和弦片段：验证谐波过滤不会误删真实和弦音（C 大调 / G 大调）
    "chords": {
        "notes": [
            (0.0, 1.0, 60), (0.0, 1.0, 64), (0.0, 1.0, 67),   # C
            (1.0, 1.0, 55), (1.0, 1.0, 59), (1.0, 1.0, 62),   # G
            (2.0, 1.0, 62), (2.0, 1.0, 65), (2.0, 1.0, 69),   # F
        ],
        "drums": False, "kind": "piano",
    },
}

# 待评估的配置（对应 UI 里的开关组合）
CONFIGS = {
    "baseline": {},
    "hpss": {"hpss": True},
    "clean": {"harmonic_filter": True, "min_duration": 0.04,
              "gap_merge": 0.03, "onset_confirm": True},
    "melody": {"melody": True},
    "fine": {"fine_quant": True},
    # melody 是“单旋律模式”，会丢弃和弦只保留最强音，与和声/和弦评测目标冲突，
    # 故不并入 all；其单独作为 melody 配置评估（适用于 piano_melody / vocal_with_drums）。
    "all": {"hpss": True, "harmonic_filter": True, "min_duration": 0.04,
            "gap_merge": 0.03, "onset_confirm": True, "fine_quant": True},
}


# ---------------------------------------------------------------------------
# 2. 音频合成
# ---------------------------------------------------------------------------
def synth_note(freq, dur, sr=SR, kind="piano", amp=0.4):
    n = int(sr * dur)
    t = np.arange(n) / sr
    if kind == "vocal":
        # 纯音 + 轻微颤音，模拟人声
        vib = 1 + 0.005 * np.sin(2 * np.pi * 5 * t)
        sig = amp * np.sin(2 * np.pi * freq * vib * t)
    else:
        # 钢琴：基音 + 2、3 次谐波，带快起音慢释放包络
        sig = amp * (np.sin(2 * np.pi * freq * t)
                     + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
                     + 0.25 * np.sin(2 * np.pi * 3 * freq * t))
    env = np.ones(n)
    fa = min(int(0.01 * sr), n // 4)
    fr = min(int(0.15 * sr), n // 2)
    env[:fa] = np.linspace(0, 1, fa)
    env[-fr:] = np.linspace(1, 0, fr)
    return sig * env


def synth_drum(dur=0.12, sr=SR, amp=0.5):
    n = int(sr * dur)
    t = np.arange(n) / sr
    # 带通噪声模拟鼓点
    noise = np.random.randn(n)
    sig = amp * noise * np.exp(-t / (0.04))
    return sig


def make_audio(clip):
    notes = clip["notes"]
    end = max(o + d for (o, d, _p) in notes) + 0.3
    total = int(SR * end)
    y = np.zeros(total)
    for (o, d, p) in notes:
        i0 = int(o * SR)
        seg = synth_note(T._midi_to_hz(p), d, kind=clip["kind"])
        i1 = i0 + len(seg)
        if i1 <= total:
            y[i0:i1] += seg
    if clip["drums"]:
        for beat in np.arange(0.0, end, 0.5):
            i0 = int(beat * SR)
            seg = synth_drum()
            i1 = i0 + len(seg)
            if i1 <= total:
                y[i0:i1] += seg
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.8
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. 指标
# ---------------------------------------------------------------------------
def align(gt, pred, onset_tol=0.05):
    """greedy 对齐：每个真值音找 onset 最近的未用预测音（容差内）。返回对齐对列表。"""
    gt_s = sorted(gt, key=lambda n: n[0])
    pr_s = sorted(pred, key=lambda n: n[0])
    used = [False] * len(pr_s)
    pairs = []
    for g in gt_s:
        best, bestd = -1, 1e9
        for i, p in enumerate(pr_s):
            if used[i]:
                continue
            d = abs(p[0] - g[0])
            if d <= onset_tol and d < bestd:
                bestd, best = d, i
        if best >= 0:
            used[best] = True
            pairs.append((g, pr_s[best]))
    return pairs


def metrics(gt_notes, pred_notes):
    gt = [(o, o + d, p) for (o, d, p) in gt_notes]
    pred = [(o, s, p) for (o, s, p, _v) in pred_notes]
    pairs = align(gt, pred)
    onset_tp = len(pairs)
    onset_fp = len(pred) - onset_tp
    onset_fn = len(gt) - onset_tp
    note_tp = sum(1 for (g, p) in pairs if g[2] == p[2])
    note_fp = len(pred) - note_tp
    note_fn = len(gt) - note_tp

    def f1(tp, fp, fn):
        return 0.0 if (tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)

    pitch_acc = note_tp / onset_tp if onset_tp else 0.0
    dur_mae = (sum(abs(p[1] - p[0] - (g[1] - g[0])) for (g, p) in pairs) / onset_tp
               ) if onset_tp else 0.0
    return {
        "n_gt": len(gt), "n_pred": len(pred),
        "note_F1": round(f1(note_tp, note_fp, note_fn), 3),
        "onset_F1": round(f1(onset_tp, onset_fp, onset_fn), 3),
        "pitch_acc": round(pitch_acc, 3),
        "dur_MAE": round(dur_mae, 3),
        "FP": onset_fp, "FN": note_fn,
    }


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="仅 baseline 与 all")
    args = ap.parse_args()

    configs = {"baseline": CONFIGS["baseline"], "all": CONFIGS["all"]} if args.quick \
        else CONFIGS

    tmp = tempfile.mkdtemp()
    print(f"{'clip':18} {'config':10} {'gt':>3} {'pred':>4} {'noteF1':>7} "
          f"{'onsetF1':>8} {'pitch':>6} {'durMAE':>7} {'FP':>3} {'FN':>3}")
    print("-" * 92)
    for cname, clip in CLIPS.items():
        wav = os.path.join(tmp, cname + ".wav")
        sf.write(wav, make_audio(clip), SR)
        for cfg_name, cfg in configs.items():
            out = os.path.join(tmp, cname + "_" + cfg_name)
            os.makedirs(out, exist_ok=True)
            res = T.process(wav, out, bpm=120, beats=4, separate=False, **cfg)
            if res.get("status") != "ok" or not res.get("note_list"):
                print(f"{cname:18} {cfg_name:10}  (无音符)")
                continue
            m = metrics(clip["notes"], res["note_list"])
            print(f"{cname:18} {cfg_name:10} {m['n_gt']:>3} {m['n_pred']:>4} "
                  f"{m['note_F1']:>7} {m['onset_F1']:>8} {m['pitch_acc']:>6} "
                  f"{m['dur_MAE']:>7} {m['FP']:>3} {m['FN']:>3}")


if __name__ == "__main__":
    run()
