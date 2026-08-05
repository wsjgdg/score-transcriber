"""
音视频转乐谱核心模块
- 用 ffmpeg 从视频/音频提取单声道音轨
- 用 basic-pitch 做多音转录 (Multi-pitch Transcription)
- 输出：MIDI、MusicXML、钢琴卷帘图、五线谱 PNG(经 LilyPond)、简谱文本
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as _fm

# 注册中文字体：让钢琴卷帘图的中文标题/轴标签正常显示，而非方框。
# Noto CJK 是泛中日韩字体，其 JP 字形已包含简体常用汉字。
# 字体文件可能不存在（如未安装 fonts-noto-cjk），此时静默回退到默认字体。
_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    if os.path.exists(_CJK_FONT):
        _fm.fontManager.addfont(_CJK_FONT)
        _cjk_names = sorted({f.name for f in _fm.fontManager.ttflist if "CJK" in f.name})
        if _cjk_names:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [_cjk_names[0]] + ["DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
# basic-pitch / pretty_midi 较重且非所有环境都装；惰性导入，使模块在无这些依赖时
# 仍可 import（便于对纯函数做单元测试），调用到相关功能时若缺失再友好报错。
try:
    from basic_pitch.inference import predict
except Exception:
    predict = None
try:
    import pretty_midi
except Exception:
    pretty_midi = None
# music21 仅在 save_musicxml 使用，惰性导入：加快启动、缺失时仅该功能不可用。

SR = 22050

# 五线谱 / 简谱最多使用的和弦事件数（避免谱面过长、渲染过慢）
MAX_STAFF_EVENTS = 600
MAX_JIANPU_EVENTS = 240
MAX_ROLL_SECONDS = 120

# 选调 / 移调表：以 C 为基准，toff = 目标调主音相对 C 的半音数。
# delta = 实际移调半音（≤6 上行、>6 下行，保持位移最小；降号调整体下移，便于人声降调）。
# label = 简谱调号文字；lily = LilyPond 主音音名；sharps = MusicXML 调号（负=降号）。
_KEY_DEFS = {
    "C":  (0,  "C",  "c",  0),
    "bD": (1,  "♭D", "df", -5),
    "D":  (2,  "D",  "d",  2),
    "bE": (3,  "♭E", "ef", -3),
    "E":  (4,  "E",  "e",  4),
    "F":  (5,  "F",  "f",  -1),
    "#F": (6,  "♯F", "fs", 6),
    "G":  (7,  "G",  "g",  1),
    "bA": (8,  "♭A", "af", -4),
    "A":  (9,  "A",  "a",  3),
    "bB": (10, "♭B", "bf", -2),
    "B":  (11, "B",  "b",  5),
}
DEFAULT_KEY = "C"


def resolve_key(key: str):
    """返回 (toff, label, lily, sharps, delta)。未知调名回退 C。"""
    toff, label, lily, sharps = _KEY_DEFS.get(key, _KEY_DEFS[DEFAULT_KEY])
    delta = toff if toff <= 6 else toff - 12
    return toff, label, lily, sharps, delta


def transpose_note_list(note_list, semis: int):
    """把音符表整体移调 semis 个半音（用于选调到目标调）。下限钳到 0。"""
    if not semis:
        return note_list
    out = []
    for (o, s, p, v, b) in note_list:
        np_ = int(round(p)) + semis
        if np_ < 0:
            np_ = 0
        out.append((o, s, np_, v, b))
    return out


def correct_note_octave_to_f0(pitch_midi: float, f0_midi) -> tuple:
    """纯函数：把单个音符的八度对齐到真实基频 F0。
    仅当音名(pitch class)相同、且比 F0 高 1~2 个八度时下拉，避免误改正确音；
    返回 (new_pitch, shifted: bool)。F0 来自 pYIN/CREPE 等鲁棒基频估计，
    用于自动修正 basic-pitch 系统性“高八度”错误（无需用户手动选旋钮）。
    """
    if pitch_midi < 0 or f0_midi is None:
        return pitch_midi, False
    if (int(round(pitch_midi)) % 12) != (int(round(f0_midi)) % 12):
        return pitch_midi, False  # 音名不同（可能是复调另一声部），不修正
    oct_diff = int(round((pitch_midi - f0_midi) / 12.0))
    if 1 <= oct_diff <= 2:  # 高 1~2 个八度 → 拉回真 F0 的八度
        np_ = int(round(pitch_midi)) - 12 * oct_diff
        if np_ >= 0:
            return np_, True
    return pitch_midi, False


def f0_octave_correct(note_list, wav_path, sr: int = SR, y=None):
    """用 pYIN 从原音频估真实基频轨迹，自动把每个音符的八度对齐到真 F0。
    仅依赖 lazy import 的 librosa/numpy，CI 无 librosa 时静默跳过（返回原表）。
    返回 (corrected_note_list, shifted_count)。
    """
    if not note_list:
        return note_list, 0
    try:
        import librosa
        import numpy as _np
    except Exception:
        return note_list, 0
    try:
        if y is None:
            y, _sr = librosa.load(wav_path, sr=sr, mono=True)
        else:
            _sr = sr
        fmin = librosa.note_to_hz("C2")
        fmax = librosa.note_to_hz("C7")
        f0 = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=_sr,
                          frame_length=2048, hop_length=512)[0]
        times = librosa.times_like(f0, sr=_sr, hop_length=512)
    except Exception:
        return note_list, 0
    voiced = ~_np.isnan(f0)
    if voiced.sum() < 3:
        return note_list, 0
    f0_hz = f0[voiced]
    f0_times = times[voiced]
    out = []
    shifted = 0
    for (o, s, p, v, b) in note_list:
        if p < 0:
            out.append((o, s, p, v, b))
            continue
        lo = max(0.0, float(o) - 0.05)
        hi = float(s) + 0.05
        mask = (f0_times >= lo) & (f0_times <= hi)
        if mask.sum() < 2:
            out.append((o, s, p, v, b))
            continue
        med = float(_np.median(f0_hz[mask]))
        if med <= 0 or _np.isnan(med):
            out.append((o, s, p, v, b))
            continue
        m_f0 = 69.0 + 12.0 * _np.log2(med / 440.0)
        new_p, did = correct_note_octave_to_f0(p, m_f0)
        if did:
            shifted += 1
        out.append((o, s, new_p, v, b))
    return out, shifted


def build_outputs(note_list, out_dir: str, bpm: float = 120, beats: int = 4,
                 key: str = DEFAULT_KEY, fine_quant: bool = False,
                 rms_vel: bool = False, lyrics_tokens=None, beat_denom: int = 4,
                 on_progress=None) -> dict:
    """把已识别的音符表渲染成完整结果（MIDI / 钢琴卷帘 / 简谱 / 五线谱 / MusicXML / 合成音频）。
    被 process()（音视频转乐谱）与 OMR（识谱成曲）共用，避免重复渲染逻辑。
    返回的 result 字典与 process() 同构（不含原音频，OMR 无源音频），可直接喂给前端 renderResult。
    note_list: [(onset_sec, offset_sec, pitch_midi, velocity_0_1, bends), ...]
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    stem = "score"
    toff, key_label, _l, _sh, _d = resolve_key(key)

    if callable(on_progress):
        on_progress("生成 MIDI", 60)
    midi_path = os.path.join(out_dir, stem + ".mid")
    try:
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        for (o, s, p, v, _b) in note_list:
            if p < 0 or s <= o:
                continue
            nv = max(1, min(127, int(round(v * 127))))
            inst.notes.append(pretty_midi.Note(
                velocity=nv, pitch=int(round(p)),
                start=float(o), end=float(s)))
        pm.instruments.append(inst)
        pm.write(midi_path)
    except Exception as e:
        print("MIDI 写入失败:", e)
        midi_path = None
    files = {}
    if midi_path:
        files["midi"] = midi_path

    if callable(on_progress):
        on_progress("生成钢琴卷帘", 72)
    roll_path = os.path.join(out_dir, stem + "_pianoroll.png")
    if save_pianoroll(note_list, roll_path, show_confidence=False):
        files["pianoroll"] = roll_path

    if callable(on_progress):
        on_progress("生成简谱", 80)
    groups = _build_groups(note_list, bpm, fine=fine_quant)
    jianpu = build_jianpu(note_list, bpm, beats, beat_denom=beat_denom,
                          groups=groups, key=key)

    if callable(on_progress):
        on_progress("生成五线谱", 90)
    ly = build_lily(groups, bpm, beats, lyrics=lyrics_tokens,
                    beat_denom=beat_denom, key=key)
    staff_path = os.path.join(out_dir, stem + "_staff.png")
    if render_lily_png(ly, staff_path):
        files["staff"] = staff_path

    if callable(on_progress):
        on_progress("生成 MusicXML", 95)
    xml_path = os.path.join(out_dir, stem + ".musicxml")
    if save_musicxml(note_list, xml_path, bpm, beats, lyrics=lyrics_tokens,
                     beat_denom=beat_denom, groups=groups, key=key):
        files["musicxml"] = xml_path

    # 合成可播放音频（OMR 无原音频，仅合成钢琴谱）
    score_wav = os.path.join(out_dir, stem + "_score.wav")
    synth_ok = False
    if FLUIDSYNTH_OK and midi_path:
        try:
            synth_ok = render_midi_fluidsynth(midi_path, score_wav, SR)
        except Exception as e:
            print("真实音源合成失败，回退基础合成:", e)
    if not synth_ok and midi_path:
        synth_ok = synth_score(note_list, score_wav)
    if synth_ok:
        score_mp3 = os.path.join(out_dir, stem + "_score.mp3")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", score_wav, "-vn", "-ar", "44100",
                            "-b:a", "128k", score_mp3],
                           check=True, capture_output=True, text=True)
            files["score_audio"] = score_mp3
        except Exception as e:
            print("钢琴谱音频转码失败:", e)
        try:
            os.remove(score_wav)
        except OSError:
            pass

    duration = max((s for (_o, s, _p, _v, _b) in note_list), default=0.0)
    stats = {
        "num_notes": len([n for n in note_list if n[2] >= 0]),
        "num_events": len(groups),
        "duration_sec": round(duration, 2),
        "bpm": bpm,
        "bpm_estimated": False,
        "beats_per_bar": beats,
        "beats_estimated": False,
        "beats_denom": beat_denom,
        "staff_truncated": len(groups) > MAX_STAFF_EVENTS,
        "staff_max_events": MAX_STAFF_EVENTS,
        "key": key_label,
        "key_raw": key,
        "source": "omr",
    }
    return {"status": "ok", "stats": stats, "files": files, "jianpu": jianpu,
            "note_list": None}


# ----------------------------------------------------------------------------
# 1. 音频提取（视频抽音轨 / 任意音频转标准 wav）
# ----------------------------------------------------------------------------
def extract_audio(src_path: str, wav_path: str, sr: int = SR):
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", "-f", "wav",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("服务器未安装 ffmpeg，无法处理音视频文件")
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "无法解析该音视频文件（ffmpeg 解码失败），请换格式或重新导出后重试")


# ----------------------------------------------------------------------------
# 2. 转录：返回 (PrettyMIDI, note_list)
#    note_list: [(onset, offset, pitch_int, velocity, bends), ...]
# ----------------------------------------------------------------------------
def transcribe(wav_path: str, backend: str = "basic-pitch",
               onset_threshold: float = 0.5, frame_threshold: float = 0.3):
    """多音转录。backend=basic-pitch 用 Spotify basic-pitch（TFCN）；
    backend=mt3 走可选后端（默认未安装，见 _transcribe_mt3）。
    onset_threshold / frame_threshold 仅在 basic-pitch 下生效，用于自适应降噪。"""
    if backend == "mt3":
        return _transcribe_mt3(wav_path)
    if predict is None:
        raise RuntimeError(
            "转录模型 basic-pitch 未安装：请先 pip install basic-pitch 并安装 tensorflow/jax。")
    _, midi_obj, note_list = predict(
        wav_path, onset_threshold=onset_threshold, frame_threshold=frame_threshold)
    return midi_obj, note_list


# MT3 / 复调转录可选后端（默认未安装）。自托管有 GPU 且装好 mt3 + 权重后可启用，
# 识别上限最高（多乐器 + 力度 + 复调），但依赖极重，免费托管环境无法运行。
class BackendUnavailable(RuntimeError):
    """可选大模型后端（MT3 / Whisper 等）未启用时抛出，供接口层按类型转友好提示，
    避免依赖错误消息字符串匹配。"""


try:
    import mt3  # 仅在显式安装后存在
    MT3_OK = True
except Exception:
    MT3_OK = False


def _transcribe_mt3(wav_path: str):
    # 后端骨架：实际推理需 mt3 + 预训练权重（环境开销大，不在免费托管启用）。
    # 自托管启用步骤见 README「可选模型后端」。此处返回明确错误供上层转友好提示。
    raise BackendUnavailable(
        "MT3 后端未启用：请在有 GPU 的自托管环境中安装 mt3 及预训练权重，"
        "并在 transcriber._transcribe_mt3 中实现推理逻辑。"
    )


# ----------------------------------------------------------------------------
# 2c. Whisper 歌词识别（可选 ASR 后端，仅自托管可用）
# ----------------------------------------------------------------------------
# 说明：本工具核心是 AMT（只识别音高/节奏），不识别歌词文字。要把“识别出的人声”
# 变成谱面下方的歌词，需要 ASR 这一步。Whisper 模型很重，免费托管环境不预装、
# 也跑不动，故做成“自托管可选后端”，与 demucs / MT3 同思路优雅降级。
try:
    import faster_whisper  # 轻量 CTranslate2 实现，优先
    _WHISPER_BACKEND = "faster_whisper"
    WHISPER_OK = True
except Exception:
    try:
        import whisper as _ow_whisper  # OpenAI 官方实现，备选
        _WHISPER_BACKEND = "openai-whisper"
        WHISPER_OK = True
    except Exception:
        _WHISPER_BACKEND = None
        WHISPER_OK = False


def transcribe_lyrics_whisper(vocal_wav: str, model_size: str = "base"):
    """用 Whisper 对人声干声做 ASR，返回 [(词, 起始秒, 结束秒), ...]（词级时间戳）。
    仅在 WHISPER_OK 为 True 时可用；否则抛 RuntimeError 由上层转友好提示。"""
    if not WHISPER_OK:
        raise BackendUnavailable(
            "Whisper 歌词识别未启用：当前为免费托管环境，无法运行该大模型。"
            "请在自有 GPU 服务器上部署并安装 faster-whisper / whisper 后启用。"
        )
    try:
        if _WHISPER_BACKEND == "faster_whisper":
            from faster_whisper import WhisperModel
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            # language=None 由 Whisper 自动检测语种（中英文混排更稳）；
            # 若明确只处理中文歌词，可改为 language="zh"。
            segs, _ = model.transcribe(vocal_wav, word_timestamps=True,
                                       language=None)
            words = []
            for s in segs:
                for w in (getattr(s, "words", None) or []):
                    t = (w.word or "").strip()
                    if t:
                        words.append((t, float(w.start), float(w.end)))
            return words
        else:
            import whisper as _whisper
            model = _whisper.load_model(model_size)
            res = model.transcribe(vocal_wav, word_timestamps=True)
            words = []
            for s in res.get("segments", []):
                for w in s.get("words", []):
                    t = (w.get("word") or "").strip()
                    if t:
                        words.append((t, float(w["start"]), float(w["end"])))
            return words
    except Exception as e:
        raise RuntimeError(f"Whisper 歌词识别失败：{e}")


# ----------------------------------------------------------------------------
# 2d. 歌词分词与对齐工具
# ----------------------------------------------------------------------------
_PUNCT_RE = re.compile(
    r'^[\s，。、！？,.!?;；：:"\'「」“”()（）…—\-_/\\|~+]+$')


def tokenize_lyrics(text: str):
    """把用户填的歌词文本切成“词/字”列表，作为逐个音符的歌词。
    - 含空白：按空白切分（英文词 / 有空格的中文）。
    - 无空白：按字符切分（CJK 一字一音最自然），并剔除纯标点。
    """
    if not text or not text.strip():
        return []
    s = text.strip()
    if re.search(r"\s", s):
        parts = re.split(r"\s+", s)
    else:
        parts = list(s)
    toks = []
    for p in parts:
        p = p.strip().strip('，。、！？,.!?;；：:"\'「」“”()（）…—-_')
        if p and not _PUNCT_RE.fullmatch(p):
            toks.append(p)
    return toks


def align_lyrics_by_order(groups, tokens):
    """按音符先后顺序一对一分配歌词（tokens[i] -> groups[i]）。
    词少于音符时，多余音符无歌词；词多于音符时，多余的词被忽略。"""
    if not tokens:
        return [None] * len(groups)
    return [tokens[i] if i < len(tokens) else None
            for i in range(len(groups))]


def align_lyrics_by_time(groups, words, bpm: float):
    """按时间把词对齐到音符组（ASR 场景更准）：取起始时刻落在词区间内的词，
    否则取时间中心最近的词。groups 的 onset 单位为拍，需乘 60/bpm 转秒。"""
    if not words:
        return [None] * len(groups)
    spp = 60.0 / max(bpm, 1e-6)
    out = []
    for g in groups:
        t = g["onset"] * spp
        best, best_d = None, float("inf")
        for (w, ws, we) in words:
            if ws <= t <= we:
                best = w
                break
            d = min(abs(t - ws), abs(t - we))
            if d < best_d:
                best_d, best = d, w
        out.append(best)
    return out


# ----------------------------------------------------------------------------
# 2b. Demucs 人声/乐器分离（可选预处理，默认开启以减少伴奏干扰）
# ----------------------------------------------------------------------------
_DEMUCS_OK = False
try:
    from demucs.pretrained import get_model as _dm_get_model
    from demucs.apply import apply_model as _dm_apply
    from demucs.audio import AudioFile as _dm_audio
    import torch as _torch
    _DEMUCS_OK = True
except Exception:
    _DEMUCS_OK = False


def separate_sources(wav_path: str, target: str = "vocals", out_dir: str = None,
                     timeout: int = 240):
    """用 Demucs(htdemucs 4-stem) 把音轨分离，返回 (选定声部 wav 路径, 错误信息)。
    target: 'vocals' | 'other' | 'auto'(取能量更高的声部)。失败返回 (None, 错误描述)。"""
    if not _DEMUCS_OK:
        return None, "demucs 不可用（未安装）"
    try:
        import os as _os
        import soundfile as _sf
        model = _dm_get_model("htdemucs")
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        model.to(device)
        wav = _dm_audio(wav_path).read(streams=0, samplerate=model.samplerate,
                                       channels=model.audio_channels)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = _dm_apply(model, wav[None], device=device, progress=False)[0]
        # model.sources 顺序通常为 ['drums','bass','other','vocals']
        if target == "auto":
            energies = {s: float(sources[i].abs().mean())
                        for i, s in enumerate(model.sources)}
            target = "vocals" if energies.get("vocals", 0) >= energies.get("other", 0) else "other"
        idx = model.sources.index(target)
        stem = sources[idx] * ref.std() + ref.mean()      # 反归一化
        out = out_dir or _os.path.dirname(wav_path)
        stem_path = _os.path.join(out, "separated_" + target + ".wav")
        _sf.write(stem_path, stem.cpu().numpy().T, model.samplerate)
        return stem_path, None
    except Exception as e:
        print("Demucs 分离失败，回退原音频:", e)
        return None, str(e)


def estimate_bpm(wav_path: str) -> float:
    """用 librosa 节拍跟踪估计 BPM；失败返回 0（调用方据此回退默认/用户值）。"""
    try:
        import librosa
        import numpy as _np
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        if y is None or len(y) == 0:
            return 0.0
        res = librosa.beat.beat_track(y=y, sr=sr)
        tempo = res[0] if isinstance(res, (tuple, list)) else res
        bpm = float(_np.atleast_1d(tempo)[0])
        if not _np.isfinite(bpm) or bpm <= 0:
            return 0.0
        return float(max(40, min(300, round(bpm))))
    except Exception as e:
        print("BPM 估计失败:", e)
        return 0.0


def _adaptive_frame_threshold(wav_path: str, base: float = 0.3) -> float:
    """按音频信噪比粗略调整 basic-pitch 的 frame_threshold：越噪越严（调高），越干净越松（调低）。"""
    try:
        import librosa
        import numpy as _np
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        if len(y) == 0:
            return base
        rms = librosa.feature.rms(y=y).mean()
        # 用整体能量 vs 低能量分位的差值估计动态范围（嘈杂音频动态范围小）
        centroid = _np.percentile(_np.abs(y), 50)
        dyn = float(centroid / (rms + 1e-6))
        # dyn 越小越平（疑似带噪/持续底噪），抬高阈值
        if dyn < 0.6:
            return min(0.6, base + 0.2)
        if dyn > 2.0:
            return max(0.1, base - 0.15)
        return base
    except Exception:
        return base



# ----------------------------------------------------------------------------
# 2e. 准确率后处理增强（CPU 可跑，免费托管可用，不换模型）
# ----------------------------------------------------------------------------
def _midi_to_hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def apply_hpss(wav_path: str, out_dir: str, margin: float = 3.0,
              y: np.ndarray = None, sr: int = SR) -> str:
    """谐波-打击乐分离（HPSS）：保留谐波(有音高)成分、剔除打击乐，显著减少鼓点/镲
    被 basic-pitch 误识别成音符（流行/带鼓音频最大的误捡源）。返回谐波成分 wav 路径；
    失败则回退原路径。y/sr 可传入已解码音频以避免重复解码。"""
    try:
        import librosa
        if y is None:
            y, sr = librosa.load(wav_path, sr=sr, mono=True)
        y_harm, _y_perc = librosa.effects.hpss(y, margin=margin)
        out = os.path.join(out_dir, "hpss_harm.wav")
        sf.write(out, y_harm.astype(np.float32), sr)
        return out
    except Exception as e:
        print("HPSS 分离失败，回退原音频:", e)
        return wav_path


def filter_harmonic_overlap(note_list, vel_drop: float = 0.0):
    """去掉疑似强音泛音的弱音：若某音 p 与同时段更强的音 q 频率满足 ~2×/3×
    （精确八度/十二度），且 p 力度明显弱于 q，则丢弃 p（泛音误检）。
    真实和弦(如 C-E-G)音频率不是精确整数倍，不会被误删。"""
    evs = [(o, s, p, v, b) for (o, s, p, v, b) in note_list if p >= 0]
    if not evs:
        return note_list
    keep = []
    for (o, s, p, v, b) in evs:
        fp = _midi_to_hz(p)
        bad = False
        for (o2, s2, p2, v2, b2) in evs:
            if (o2, s2, p2, v2, b2) == (o, s, p, v, b):
                continue
            if not (o2 < s and o < s2):   # 时间不重叠则跳过
                continue
            fq = _midi_to_hz(p2)
            for k in (2, 3):
                if abs(fp - k * fq) / (k * fq) < 0.03 or abs(fq - k * fp) / (k * fp) < 0.03:
                    if v2 > v:            # 同时存在更强的基音 -> p 更像泛音
                        bad = True
                        break
            if bad:
                break
        if not bad:
            keep.append((o, s, p, v, b))
    return keep


def filter_min_duration(note_list, min_dur: float = 0.04):
    """丢弃时值小于 min_dur 秒的碎片音（basic-pitch 偶发的超短误检）。"""
    return [(o, s, p, v, b) for (o, s, p, v, b) in note_list
            if p >= 0 and (s - o) >= min_dur]


def merge_gaps(note_list, gap: float = 0.03):
    """同音高、间隔 < gap 秒的相邻音符合并为一个（去颤音/断续碎片）。"""
    evs = sorted([e for e in note_list if e[2] >= 0], key=lambda e: (e[2], e[0]))
    out = []
    for e in evs:
        if out and out[-1][2] == e[2] and e[0] - out[-1][1] <= gap:
            o0, s0, p0, v0, b0 = out[-1]
            out[-1] = (o0, max(s0, e[1]), p0, max(v0, e[3]), b0)
        else:
            out.append(e)
    return out


def confirm_onsets(note_list, wav_path: str, sr: int = SR, tol: float = 0.03,
                   drop_unconfirmed: bool = False, y: np.ndarray = None):
    """用 librosa onset 检测交叉验证 basic-pitch 的起音：附近 tol 内无能量突变支撑的
    起音予以降权(默认)或丢弃(drop_unconfirmed)，减少模型偶发假起音。y 可传入已解码音频。"""
    try:
        import librosa
        if y is None:
            y, _sr = librosa.load(wav_path, sr=sr, mono=True)
        else:
            _sr = sr
        onsets = [float(x) for x in librosa.onset.onset_detect(y=y, sr=_sr, units="time")]
    except Exception as e:
        print("onset 确认失败，跳过:", e)
        return note_list
    out = []
    for (o, s, p, v, b) in note_list:
        if p < 0:
            out.append((o, s, p, v, b))
            continue
        confirmed = any(abs(o - ot) <= tol for ot in onsets)
        if confirmed:
            out.append((o, s, p, v, b))
        elif drop_unconfirmed:
            continue
        else:
            out.append((o, s, p, max(v * 0.5, 0.0), b))   # 降权而非丢弃
    return out


def melody_mode(note_list, fmin_hz: float = 80.0, fmax_hz: float = 1000.0):
    """人声单旋律模式：频段限幅(丢弃范围外音) + 同时段只保留最强音(去和弦)，
    得到干净的单线条旋律，最贴合“人声转歌词”主场景，且与填词对齐最准。"""
    fmin_m = 69 + 12 * np.log2(fmin_hz / 440.0)
    fmax_m = 69 + 12 * np.log2(fmax_hz / 440.0)
    evs = [(o, s, p, v, b) for (o, s, p, v, b) in note_list
           if p >= 0 and fmin_m <= p <= fmax_m]
    evs.sort(key=lambda e: e[0])
    out, cluster, cur_end = [], [], -1.0
    for e in evs:
        if cluster and e[0] < cur_end:      # 与当前簇真正重叠 -> 并入（首尾相接不算）
            cluster.append(e)
            cur_end = max(cur_end, e[1])
        else:                               # 不重叠 -> 结算上一簇，开新簇
            if cluster:
                out.append(max(cluster, key=lambda x: x[3]))
            cluster = [e]
            cur_end = e[1]
    if cluster:
        out.append(max(cluster, key=lambda x: x[3]))
    return out


def compute_rms_velocities(note_list, wav_path: str, sr: int = SR,
                           y: np.ndarray = None):
    """按音符区间 RMS 推算力度(0~1)，用于 MIDI 演奏动态，使强弱更自然；
    不改变模型置信度(置信度仍用于降噪/评估)。返回 {(onset四舍五入, pitch): vel}。y 可传入已解码音频。"""
    try:
        import librosa
        if y is None:
            y, _sr = librosa.load(wav_path, sr=sr, mono=True)
        else:
            _sr = sr
    except Exception:
        return None
    res = {}
    for (o, s, p, v, b) in note_list:
        if p < 0:
            continue
        i0 = max(0, int(o * _sr))
        i1 = min(len(y), max(i0 + 1, int(s * _sr)))
        seg = y[i0:i1]
        rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
        res[(round(o, 4), int(round(p)))] = rms
    if not res:
        return None
    vals = np.array(list(res.values()), dtype=float)
    mx = float(vals.max()) if vals.max() > 0 else 1.0
    return {k: float(min(1.0, vv / mx)) for k, vv in res.items()}


def estimate_beats_per_bar(wav_path: str, bpm: float, sr: int = SR):
    """粗略估计每小节拍数(3/4、4/4、6/8 等)：节拍跟踪后看相邻强拍间隔的拍数中位数。
    仅作提示，置信度不足时返回 0（调用方回退用户值）。"""
    try:
        import librosa
        y, _sr = librosa.load(wav_path, sr=sr, mono=True)
        if len(y) < _sr:  # 太短（<1 秒）无法估计拍号
            return 0
        # 先自动估计拍追踪（不给 start_bpm 约束，避免把复合拍误锁成 4 的倍数），
        # 失败/过弱再回退到用户 BPM 约束。
        try:
            _t, beats = librosa.beat.beat_track(y=y, sr=_sr)
        except Exception:
            _t, beats = librosa.beat.beat_track(y=y, sr=_sr, start_bpm=int(bpm))
        if beats is None or len(beats) < 6:
            return 0
        bts = librosa.frames_to_time(beats, sr=_sr)
        inter = np.diff(bts)
        med = float(np.median(inter))
        if med <= 0:
            return 0
        # 常见小节拍数候选：3/4、4/4、6/8 等，按秒匹配（随 bpm 变化）。
        for cand in (4, 3, 6, 2, 5, 12, 9, 8):
            bar_sec = cand * 60.0 / max(bpm, 1e-6)
            # 放宽容差：复合拍/速度波动下更稳
            if abs(med - bar_sec) <= 0.3 * bar_sec + 0.15:
                return cand
        est = int(np.clip(round(med * bpm / 60.0), 2, 12))
        return est
    except Exception as e:
        print("拍号估计失败:", e)
        return 0


# 更细量化网格（rubato/连续时值保留更多细节）
_FINE_DUR_TABLE = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 3, 4]


def _quantize_duration(dur_beats: float, fine: bool = False) -> float:
    if dur_beats <= 0:
        return (0.125 if fine else 0.25)
    table = _FINE_DUR_TABLE if fine else _DUR_TABLE
    return min(table, key=lambda x: abs(x - dur_beats))


# ----------------------------------------------------------------------------
# 3. 量化与小工具
# ----------------------------------------------------------------------------
_DUR_TABLE = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]


def _dur_to_lily(d: float) -> str:
    m = {0.25: "16", 0.5: "8", 0.75: "8.", 1: "4", 1.5: "4.",
         2: "2", 3: "2.", 4: "1"}
    return m.get(d, "4")


_LILY_NAMES = ["c", "cis", "d", "dis", "e", "f",
               "fis", "g", "gis", "a", "ais", "b"]


def _pitch_to_lily(pitch: int) -> str:
    name = _LILY_NAMES[pitch % 12]
    ticks = pitch // 12 - 4          # C4(60) -> c' ; C3(48) -> c ; C5(72) -> c''
    octv = ("'" * ticks) if ticks > 0 else ("," * (-ticks) if ticks < 0 else "")
    return name + octv


# 标准简谱：固定调（1=C），黑键用升号唱名（#1 #2 #4 #5 #6）。
# 八度用组合点表示：高八度在数字上方加点，低八度在下方加点（UNICODE 组合字符）。
_DOT_ABOVE = "\u0307"   # ◌̇ 组合上点：高八度
_DOT_BELOW = "\u0323"   # ◌̣ 组合下点：低八度
_JIANPU_DEG = {0: "1", 1: "#1", 2: "2", 3: "#2", 4: "3", 5: "4",
               6: "#4", 7: "5", 8: "#5", 9: "6", 10: "#6", 11: "7"}


def _jianpu_degree(pitch: int, toff: int = 0) -> str:
    """返回唱名 + 八度点。toff=0 即固定调(1=C)；toff>0 时按目标调主音做首调(1=主音)。
    例：toff=2(D 调) 时 D4→'1'，G4→'5'，C4→'7'(低八度下方)。"""
    deg = _JIANPU_DEG[((pitch % 12) - toff) % 12]
    octv = (pitch - toff) // 12 - 5   # 以目标调主音所在八度为参考 0
    if octv > 0:
        deg = deg + _DOT_ABOVE * octv
    elif octv < 0:
        deg = deg + _DOT_BELOW * (-octv)
    return deg


def _pitch_to_jianpu(pitch: int) -> str:
    """固定调唱名（1=C），兼容旧调用。"""
    return _jianpu_degree(pitch, 0)


# 时值线：以四分音符(=1拍)为基准。减时线(_)=下划线，每条减半；增时线(-)=右侧横线，每条加一拍；附点用 '.'。
_DUR_MARKS = {
    0.125: (3, 0, False), 0.25: (2, 0, False), 0.5: (1, 0, False),
    0.75: (1, 0, True), 1.0: (0, 0, False), 1.5: (0, 0, True),
    2.0: (0, 1, False), 3.0: (0, 1, True), 4.0: (0, 3, False),
}


def _jianpu_dur_marks(dur: float) -> str:
    """把量化时值(拍)转成减时线/增时线/附点标记串。"""
    if dur in _DUR_MARKS:
        under, over, dot = _DUR_MARKS[dur]
    else:
        best = min(_DUR_MARKS, key=lambda x: abs(x - dur))
        under, over, dot = _DUR_MARKS[best]
    return "_" * under + "-" * over + ("." if dot else "")


def _build_groups(note_list, bpm: float, fine: bool = False, chord_gap: float = 0.05):
    """把音符按量化起始拍分组为和弦事件，便于生成乐谱。
    每个组额外携带 vels（组内各音力度）、conf（组平均置信度 %）与 ref（组内首音原始起音）。
    fine=True 时使用更细量化网格（1/8 等），保留更多 rubato 细节。
    chord_gap：同 1/4 拍网格内、起音间隔 ≤ chord_gap 秒才合并为和弦；间隔更大的快速
    琶音视为依次出现的独立事件，避免被误并成和弦。"""
    spp = 60.0 / bpm
    evs = []
    for onset, offset, pitch, vel, _bends in note_list:
        if pitch < 0:
            continue
        dur = (offset - onset) / spp
        evs.append({"onset": onset / spp, "pitch": int(round(pitch)),
                    "dur": dur, "vel": float(vel)})
    evs.sort(key=lambda e: e["onset"])
    groups = []
    for e in evs:
        ob = round(e["onset"] * 4) / 4           # 量化到 1/4 拍网格
        raw = e["onset"] * spp                    # 原始起音（秒），用于和弦判定
        dq = _quantize_duration(e["dur"], fine)
        found = None
        for g in groups:
            # 同 1/4 拍网格内，且起音时间接近（属同一次击弦/和弦）才合并；
            # 间隔较大的快速琶音视为依次出现的独立事件，避免误并成和弦。
            if abs(g["onset"] - ob) < 0.01 and (raw - g["ref"]) <= chord_gap + 1e-6:
                found = g
                break
        if found is None:
            groups.append({"onset": ob, "pitches": [e["pitch"]],
                           "durs": [dq], "vels": [e["vel"]], "ref": raw})
        else:
            found["pitches"].append(e["pitch"])
            found["durs"].append(dq)
            found["vels"].append(e["vel"])
    for g in groups:
        g["pitches"].sort()
        g["dur"] = max(g["durs"])
        g["conf"] = round(100.0 * sum(g["vels"]) / len(g["vels"]), 1) if g["vels"] else 0.0
    return groups


# ----------------------------------------------------------------------------
# 4. 钢琴卷帘图
# ----------------------------------------------------------------------------
def save_pianoroll(note_list, png_path: str, show_confidence: bool = True,
                  audio_path: str = None):
    notes = [(o, s, int(round(p)), v)
             for (o, s, p, v, _b) in note_list if p >= 0 and o <= MAX_ROLL_SECONDS]
    if not notes:
        return False
    onsets = [n[0] for n in notes]
    ends = [n[1] for n in notes]
    pitches = [n[2] for n in notes]
    t_max = max(ends)
    p_min, p_max = min(pitches), max(pitches)

    # 音符过多时（长片段）关闭文字标注，避免图面糊成一片
    annotate = show_confidence and len(notes) <= 300

    # 可选：叠加原始波形包络（灰色细线，置于最底层），便于人工核对起音对齐。
    wav_xs = wav_ys = None
    if audio_path:
        try:
            import librosa
            y, _sr = librosa.load(audio_path, sr=SR, mono=True)
            y = y[: int(max(t_max, 1) * _sr)]          # 仅取谱面时间范围内
            n = max(2000, int(t_max * 200))
            if len(y) > n:
                y = y[:: len(y) // n][:n]
            y_n = y / (np.max(np.abs(y)) + 1e-9)        # 归一化到 [-1, 1]
            lo, hi = p_min - 1.5, p_max + 1.5
            wav_ys = (lo + (y_n * 0.5 + 0.5) * (hi - lo)).astype(float)
            wav_xs = np.linspace(0, max(t_max, 1), len(wav_ys))
        except Exception as e:
            print("波形叠加失败，跳过:", e)

    fig, ax = plt.subplots(figsize=(max(10, t_max / 3.0), 6))
    if wav_xs is not None:
        ax.plot(wav_xs, wav_ys, color="gray", alpha=0.25,
                linewidth=0.4, zorder=0)
    for onset, offset, pitch, vel in notes:
        dur = max(offset - onset, 0.04)
        color = plt.cm.viridis(min(max(vel, 0.05), 1.0))
        ax.add_patch(mpatches.Rectangle(
            (onset, pitch - 0.45), dur, 0.9,
            facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.9))
        if annotate:
            # 在方块中央标注模型置信度百分比（velocity 即基本置信度代理）
            pct = int(round(vel * 100))
            ax.text(onset + dur / 2, pitch, f"{pct}%",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.35, pad=0.25,
                              edgecolor="none", boxstyle="round,pad=0.2"))

    # 画中央 C 参考线
    for pc in range(p_min // 12 * 12, p_max + 1, 12):
        ax.axhline(pc, color="gray", linewidth=0.4, alpha=0.5, linestyle="--")

    ax.set_xlim(0, max(t_max, 1))
    ax.set_ylim(p_min - 1.5, p_max + 1.5)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("音高 (MIDI)")
    ax.set_title("钢琴卷帘图 (色块颜色/数字 = 模型响应强度，非转录准确度保证)")
    ax.set_yticks(range(p_min, p_max + 1, 1))
    ax.set_yticklabels([_pitch_to_lily(p) for p in range(p_min, p_max + 1, 1)])
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------
# 5. 简谱文本
# ----------------------------------------------------------------------------
def build_jianpu(note_list, bpm: float, beats: int, beat_denom: int = 4,
                groups: list = None, key: str = DEFAULT_KEY) -> str:
    """生成标准简谱：
    数字=唱名(1-7)，数字上方点=高八度、下方点=低八度；
    下划线=减时线(每条减半)，右侧横线=增时线(每条加一拍)，'.'=附点；
    '0'=休止符；'|'=小节线；开头标注调号与拍号。beat_denom 为拍号分母（默认 4）。
    key 为目标调（默认 C，固定调）；选其他调时按该调主音做首调并改写表头 1=X。
    groups 可传入已构建的和弦事件列表以复用（避免重复分组）。"""
    toff, label, _lily, _sharps, _delta = resolve_key(key)
    if groups is None:
        groups = _build_groups(note_list, bpm)
    if not groups:
        return "(未检测到音符)"
    header = f"1={label}   {beats}/{beat_denom}"
    out = []
    beats_accum = 0.0
    bar_count = 0
    prev_end = 0.0
    shown = 0

    def flush_bars():
        nonlocal beats_accum, bar_count
        while beats_accum >= beats - 1e-9:
            out.append("|")
            beats_accum -= beats
            bar_count += 1
            if bar_count % 4 == 0:
                out.append("\n")      # 每 4 小节换行，便于阅读

    for g in groups:
        if shown >= MAX_JIANPU_EVENTS:
            break
        # 前导休止：用休止符 '0' 填充，并可能跨多个小节
        if g["onset"] > prev_end + 1e-6:
            gap = g["onset"] - prev_end
            d = _quantize_duration(gap)
            out.append("0" + _jianpu_dur_marks(d))
            beats_accum += d
            flush_bars()
            prev_end += d
        # 音符（和弦用括号括起，各声部独立标八度点）；按目标调做首调唱名
        toks = [_jianpu_degree(p, toff) for p in g["pitches"]]
        marks = _jianpu_dur_marks(g["dur"])
        token = ("(" + " ".join(toks) + ")") if len(toks) > 1 else toks[0]
        out.append(token + marks)
        beats_accum += g["dur"]
        flush_bars()
        prev_end = max(prev_end, g["onset"] + g["dur"])
        shown += 1

    if out and out[-1] != "\n":
        out.append("|")          # 收尾小节线
    body = " ".join(out)
    note = ""
    if len(groups) > MAX_JIANPU_EVENTS:
        note = f"\n…(简谱已截断，仅显示前 {MAX_JIANPU_EVENTS} 个和弦事件)"
    return header + "\n" + body + note


# ----------------------------------------------------------------------------
# 6. LilyPond 五线谱文本 + 渲染
# ----------------------------------------------------------------------------
def _conf_mark(conf: float) -> str:
    """LilyPond 标记：在音符上方以小号字号标注该音的模型置信度百分比。"""
    return '^\\markup{\\tiny "%d%%"}' % int(round(conf))


# LilyPond 歌词中 $ # % { } \ " 均有特殊含义，未处理会导致渲染失败甚至注入；
# 这里移除这些字符（歌词场景下移除比转义更安全直观），并折叠多余空白。
_LILY_LYRIC_UNSAFE = re.compile(r'[$#%{}"\\]')


def _sanitize_lily_lyric(tok: str) -> str:
    s = _LILY_LYRIC_UNSAFE.sub("", tok)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_lily(groups, bpm: float, beats: int, lyrics=None, beat_denom: int = 4,
               key: str = DEFAULT_KEY) -> str:
    body = []
    cursor = 0.0
    for idx, g in enumerate(groups[:MAX_STAFF_EVENTS]):
        if g["onset"] > cursor + 0.01:
            rd = _quantize_duration(g["onset"] - cursor)
            body.append("r" + _dur_to_lily(rd))
            cursor = g["onset"]
        if len(g["pitches"]) == 1:
            body.append(_pitch_to_lily(g["pitches"][0]) + _dur_to_lily(g["dur"])
                        + _conf_mark(g["conf"]))
        else:
            ch = "<" + " ".join(_pitch_to_lily(p) for p in g["pitches"]) + ">" \
                 + _dur_to_lily(g["dur"]) + _conf_mark(g["conf"])
            body.append(ch)
        cursor = max(cursor, g["onset"] + g["dur"])
    # 歌词：每个事件对应一个音节（与音符顺序一致；休止符自动跳过）。
    # 用 \addlyrics 放在五线谱下方。每个音节用双引号包裹以兼容标点/空格。
    lyr_block = ""
    if lyrics:
        syls = []
        for i in range(min(len(groups), MAX_STAFF_EVENTS)):
            tok = lyrics[i] if i < len(lyrics) else None
            if tok:
                safe = _sanitize_lily_lyric(tok)
                # 转义后为空则输出空音节占位，保持与音符对齐
                syls.append('"' + safe + '"' if safe else '""')
            else:
                syls.append('""')   # 空音节占位：该音符无对应歌词，LilyPond 不绘制
        if syls:
            lyr_block = "\n  \\addlyrics {\n    " + " ".join(syls) + "\n  }"
    toff, _label, lily, _sharps, _delta = resolve_key(key)
    ly = (
        '\\version "2.24.3"\n'
        '\\score {\n'
        "  \\new Staff {\n"
        "    \\numericTimeSignature\n"
        f"    \\key {lily} \\major\n"
        f"    \\time {beats}/{beat_denom}\n"
        f"    \\tempo 4 = {int(bpm)}\n"
        "    " + " ".join(body) + "\n"
        "  }" + lyr_block + "\n"
        "  \\layout { }\n"
        "}\n"
    )
    return ly


def render_lily_png(ly_text: str, png_path: str) -> bool:
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ly", delete=False) as f:
            f.write(ly_text)
            ly_path = f.name
        subprocess.run(
            ["lilypond", "--png", "-dresolution=130", "-o",
             ly_path.replace(".ly", ""), ly_path],
            check=True, capture_output=True, text=True, timeout=120)
        # lilypond 输出可能为 -1.png（多页）或 .png
        base = ly_path.replace(".ly", "")
        cand = Path(base + ".png")
        if not cand.exists():
            cand = Path(base + "-1.png")
        if cand.exists():
            os.replace(cand, png_path)
            for ext in (".png", "-1.png", ".pdf", ".ps", ".eps"):
                extra = Path(base + ext)
                if extra.exists() and extra != Path(png_path):
                    try:
                        os.remove(extra)
                    except OSError:
                        pass
            return True
    except Exception as e:
        print("LilyPond 渲染失败:", e)
    return False


# ----------------------------------------------------------------------------
# 7. MusicXML（用 music21 构建，便于下载与二次编辑）
# ----------------------------------------------------------------------------
def save_musicxml(note_list, xml_path: str, bpm: float, beats: int,
                  lyrics=None, beat_denom: int = 4, groups: list = None,
                  key: str = DEFAULT_KEY) -> bool:
    try:
        from music21 import stream, note as m21note, instrument as m21instr
        from music21 import meter as m21meter, tempo as m21tempo, chord as m21chord
        from music21 import key as m21key
    except Exception:
        print("music21 未安装，跳过 MusicXML 生成")
        return False
    try:
        _toff, _label, _lily, sharps, _delta = resolve_key(key)
        if groups is None:
            groups = _build_groups(note_list, bpm)
        s = stream.Score()
        p = stream.Part()
        p.append(m21instr.Piano())
        p.append(m21meter.TimeSignature(f"{beats}/{beat_denom}"))
        p.append(m21tempo.MetronomeMark(number=int(bpm)))
        p.append(m21key.KeySignature(sharps))
        cursor = 0.0
        for i, g in enumerate(groups):
            if g["onset"] > cursor + 0.01:
                rd = _quantize_duration(g["onset"] - cursor)
                p.append(m21note.Rest(quarterLength=rd))
                cursor = g["onset"]
            if len(g["pitches"]) == 1:
                n = m21note.Note(g["pitches"][0], quarterLength=g["dur"])
            else:
                n = m21chord.Chord(g["pitches"], quarterLength=g["dur"])
            p.append(n)
            # 歌词（verse 1）：用户填词或 ASR 识别结果，置于音符下方。
            tok = (lyrics[i] if lyrics and i < len(lyrics) else None)
            if tok:
                l1 = m21note.Lyric(tok)
                l1.number = 1
                n.lyrics.append(l1)
            # 置信度（verse 2）：一眼可辨哪些音是“猜测”。
            l2 = m21note.Lyric("%d%%" % int(round(g["conf"])))
            l2.number = 2
            n.lyrics.append(l2)
            cursor = max(cursor, g["onset"] + g["dur"])
        s.append(p)
        s.write("musicxml", fp=xml_path)
        return True
    except Exception as e:
        print("MusicXML 生成失败:", e)
        return False


# ----------------------------------------------------------------------------
# 8. 总流程
# ----------------------------------------------------------------------------
# 真实音源合成（fluidsynth + 音源文件）检测；不可用则回退到 numpy 基础合成
_FLUIDSYNTH_BIN = shutil.which("fluidsynth")
_SOUNDFONT_PATHS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/TimGM6mb.sf2",
]
SOUNDFONT = next((p for p in _SOUNDFONT_PATHS if os.path.exists(p)), None)
FLUIDSYNTH_OK = bool(_FLUIDSYNTH_BIN and SOUNDFONT)


def render_midi_fluidsynth(midi_path: str, wav_path: str, sr: int = SR) -> bool:
    """用 fluidsynth + 真实音源把 MIDI 渲染为 WAV（强制钢琴音色）。无环境时返回 False。"""
    if not FLUIDSYNTH_OK:
        return False
    import tempfile
    import pretty_midi
    tmpm = tempfile.mktemp(suffix=".mid")
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        for inst in pm.instruments:
            inst.program = 0          # Acoustic Grand Piano
        pm.write(tmpm)
    except Exception:
        tmpm = midi_path              # 兜底：直接用原 MIDI
    try:
        subprocess.run([_FLUIDSYNTH_BIN, "-ni", "-r", str(sr), "-F", wav_path,
                        SOUNDFONT, tmpm],
                       check=True, capture_output=True, text=True, timeout=120)
        return os.path.exists(wav_path)
    except Exception as e:
        print("fluidsynth 渲染失败:", e)
        return False
    finally:
        if tmpm != midi_path:
            try:
                os.remove(tmpm)
            except OSError:
                pass


def synth_score(note_list, wav_path: str, sr: int = SR) -> bool:
    """把识别出的音符表用简单加性合成渲染为单声道 WAV（无需 soundfont / 外部合成器，纯 numpy）。"""
    notes = [(o, s, int(round(p)), v)
             for (o, s, p, v, _b) in note_list if p >= 0 and s > o]
    if not notes:
        return False
    t_max = max(s for (o, s, p, v) in notes)
    out = np.zeros(int((t_max + 0.6) * sr), dtype=np.float32)
    for onset, offset, pitch, vel in notes:
        f = 440.0 * (2.0 ** ((pitch - 69) / 12.0))   # MIDI 69 = A4 = 440Hz
        dur = offset - onset
        n = int(dur * sr)
        if n <= 0:
            continue
        t = np.arange(n) / sr
        # 基频 + 两个泛音，模拟柔和的乐器音色
        sig = (np.sin(2 * np.pi * f * t)
               + 0.5 * np.sin(2 * np.pi * 2 * f * t)
               + 0.25 * np.sin(2 * np.pi * 3 * f * t))
        # ADSR 包络（attack 10ms / release 60ms），避免爆音
        env = np.ones(n)
        a = min(int(0.01 * sr), n)
        r = min(int(0.06 * sr), n)
        if a:
            env[:a] = np.linspace(0, 1, a)
        if r:
            env[-r:] = np.linspace(1, 0, r)
        sig = sig * env * (0.22 * float(np.clip(vel, 0.0, 1.0)))
        i0 = int(onset * sr)
        if i0 + n <= len(out):
            out[i0:i0 + n] += sig
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out = out / peak * 0.85
    try:
        sf.write(wav_path, out, sr)
        return True
    except Exception as e:
        print("合成音频失败:", e)
        return False


def process(src_path: str, out_dir: str, bpm: float = 120, beats: int = 4,
             min_velocity: float = 0.0,
             separate: bool = False, separate_target: str = "vocals",
             high_conf_only: bool = False, auto_bpm: bool = False,
             smart_denoise: bool = False, backend: str = "basic-pitch",
             lyrics: str = "", asr: bool = False,
             hpss: bool = False, melody: bool = False,
             harmonic_filter: bool = False, min_duration: float = 0.0,
             gap_merge: float = 0.0, onset_confirm: bool = False,
             rms_vel: bool = False, fine_quant: bool = False,
             auto_beats: bool = False, key: str = DEFAULT_KEY,
             octave_shift: int = 0, f0_octave_fix: bool = True,
             on_progress=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    def _prog(stage: str, pct: int):
        if callable(on_progress):
            try:
                on_progress(stage, pct)
            except Exception:
                pass

    _prog("准备音频", 5)
    stem = Path(src_path).stem
    wav_path = os.path.join(out_dir, stem + "_audio.wav")
    extract_audio(src_path, wav_path)
    _prog("已提取音轨", 10)
    original_wav = wav_path

    # 方案①：Demucs 人声/乐器分离（默认开启），用分离后的声部去转录以削减伴奏误捡
    transcribe_wav = wav_path
    separated_used = False
    sep_err = None
    if separate:
        sep, sep_err = separate_sources(wav_path, target=separate_target, out_dir=out_dir)
        if sep and os.path.exists(sep):
            transcribe_wav = sep
            separated_used = True
        _prog("伴奏分离完成", 18)

    # 准确率增强（CPU）：HPSS 谐波-打击乐分离，先剔除鼓点/镲，再喂给转录，
    # 大幅减少打击乐被误识别成音符（流行/带鼓音频最大误捡源）。
    if hpss:
        _hpss_y = None
        try:
            import librosa
            _hpss_y, _ = librosa.load(transcribe_wav, sr=SR, mono=True)
        except Exception:
            _hpss_y = None
        transcribe_wav = apply_hpss(transcribe_wav, out_dir, y=_hpss_y)
        _prog("打击乐分离完成", 25)

    # 方案③：自动估 BPM + 自适应阈值（智能降噪）
    effective_bpm = bpm
    bpm_estimated = False
    if auto_bpm:
        est = estimate_bpm(transcribe_wav)
        if est and est > 0:
            effective_bpm = est
            bpm_estimated = True
    frame_th = 0.3
    if smart_denoise:
        frame_th = _adaptive_frame_threshold(transcribe_wav)

    # 转录（方案④：backend 选择。MT3 未装时抛 RuntimeError，由接口层转友好提示）
    _prog("音频转音符中", 30)
    midi_obj, note_list = transcribe(transcribe_wav, backend=backend,
                                     frame_threshold=frame_th)
    _prog("音符识别完成", 50)

    # 准确率增强（CPU，转录后后处理）：人声单旋律 / 泛音过滤 / 最小音长 / 间隙合并 /
    # onset 二次确认。均在降噪阈值之前做，使降噪作用于已净化的音符。
    if melody:
        note_list = melody_mode(note_list)
    if harmonic_filter:
        note_list = filter_harmonic_overlap(note_list)
    if min_duration and min_duration > 0:
        note_list = filter_min_duration(note_list, min_duration)
    if gap_merge and gap_merge > 0:
        note_list = merge_gaps(note_list, gap_merge)

    # 后处理（onset 确认 / RMS 力度）需要音频信号：解码一次复用，避免重复 librosa.load
    _proc_y = None
    if onset_confirm or rms_vel:
        try:
            import librosa
            _proc_y, _ = librosa.load(transcribe_wav, sr=SR, mono=True)
        except Exception:
            _proc_y = None
    if onset_confirm:
        note_list = confirm_onsets(note_list, transcribe_wav, tol=0.03, y=_proc_y)

    # 降噪（连续阈值）+ 方案②：高置信仅保留（vel>=0.75）
    denoised_count = 0
    if min_velocity > 0.0:
        denoised_count = len([n for n in note_list if n[2] >= 0 and n[3] < min_velocity])
        note_list = [n for n in note_list if n[2] >= 0 and n[3] >= min_velocity]
    else:
        note_list = [n for n in note_list if n[2] >= 0]

    high_conf_dropped = 0
    high_conf_empty = False
    if high_conf_only:
        kept = [n for n in note_list if n[3] >= 0.75]
        high_conf_dropped = len(note_list) - len(kept)
        if kept:
            note_list = kept
        else:
            high_conf_empty = True   # 过滤后无剩余音符，回退保留完整结果以免出现空谱

    # 自动八度校正（F0 重锚定）：用 pYIN 从原音频估真实基频，把 basic-pitch
    # 系统性“高八度”的音符拉回真 F0 八度（无需用户手动选旋钮）。
    # 放在降噪/高置信过滤之后、选调/手动八度旋钮之前，使自动修正优先于手动偏移。
    f0_shifted = 0
    if f0_octave_fix:
        note_list, f0_shifted = f0_octave_correct(note_list, transcribe_wav, y=_proc_y)
        if f0_shifted:
            _prog(f"F0 八度校正（修正 {f0_shifted} 音）", 56)

    # 选调 / 移调：把整段音符表移到目标调（key），使五线谱/简谱/MIDI 一致处于该调。
    # 放在所有音符后处理之后，保证移调作用于最终保留的音符。
    toff, key_label, key_lily, key_sharps, key_delta = resolve_key(key)
    if key_delta:
        note_list = transpose_note_list(note_list, key_delta)
        _prog(f"已移调至 {key_label}", 58)

    # 全局八度偏移（修正 basic-pitch 系统性高/低八度）：整段上下移 12*shift 半音。
    # 放在选调之后，与移调叠加；钳到 >=0 由 transpose_note_list 处理。
    octave_shift = max(-2, min(2, int(octave_shift)))
    if octave_shift:
        note_list = transpose_note_list(note_list, 12 * octave_shift)
        _prog(f"已偏移 {octave_shift:+} 个八度", 59)

    # 准确率增强（CPU）：自动估计每小节拍数（拍号），仅作提示，置信不足回退用户值
    effective_beats = beats
    beats_estimated = False
    if auto_beats:
        eb = estimate_beats_per_bar(transcribe_wav, effective_bpm)
        if eb and eb != beats:
            effective_beats = eb
            beats_estimated = True

    # 复合拍号：auto_beats 估出 6/9/12 拍时按八分音符记谱（6/8、9/8、12/8），
    # 与内部“拍=四分音符”量化网格一致；手动拍号仍按 /4（X/4）。
    effective_denom = 8 if (auto_beats and effective_beats in (6, 9, 12)) else 4

    groups = _build_groups(note_list, effective_bpm, fine=fine_quant)
    _prog("节拍量化完成", 62)
    duration = max((s for (_o, s, _p, _v, _b) in note_list), default=0.0)

    # 歌词：把“人声/填词”放到五线谱每一行（音符）下方。
    # 模式 A（手动填词）：用户给出歌词文本，按音符顺序一对一分配。
    # 模式 B（ASR 自动识别）：用 Whisper 对“人声干声”识别词级时间戳，再按时间对齐。
    #   免费托管环境无 whisper，模式 B 会被接口层转友好提示；自托管装好后可启用。
    lyrics_tokens = None
    lyrics_source = None
    if asr:
        # 优先用人声干声；未分离或分离的不是人声时回退原音频
        lyr_wav = transcribe_wav if (separated_used and separate_target == "vocals") else original_wav
        try:
            words = transcribe_lyrics_whisper(lyr_wav)
            lyrics_tokens = align_lyrics_by_time(groups, words, effective_bpm)
            lyrics_source = "asr"
        except Exception as e:
            # 把 whisper 错误向上抛，让接口层转成友好提示（含“自托管”说明）
            raise RuntimeError(str(e))
    elif lyrics and lyrics.strip():
        lyrics_tokens = align_lyrics_by_order(groups, tokenize_lyrics(lyrics))
        lyrics_source = "manual"

    # 置信度多指标：均值 + 中位数 + 高置信占比
    vel_list = [n[3] for n in note_list]
    confidence = round(100.0 * (sum(vel_list) / len(vel_list)), 1) if vel_list else 0.0
    confidence_median = round(100.0 * float(np.median(vel_list)), 1) if vel_list else 0.0
    high_conf_ratio = round(100.0 * sum(1 for v in vel_list if v >= 0.75) / len(vel_list), 1) if vel_list else 0.0
    weak_ratio = round(100.0 * sum(1 for v in vel_list if v < 0.5) / len(vel_list), 1) if vel_list else 0.0

    result = {
        "status": "ok",
        "stats": {
            "num_notes": len([n for n in note_list if n[2] >= 0]),
            "num_events": len(groups),
            "duration_sec": round(duration, 2),
            "bpm": effective_bpm,
            "bpm_estimated": bpm_estimated,
            "beats_per_bar": effective_beats,
            "beats_estimated": beats_estimated,
            "beats_denom": effective_denom,
            "staff_truncated": len(groups) > MAX_STAFF_EVENTS,
            "staff_max_events": MAX_STAFF_EVENTS,
            "confidence": confidence,
            "confidence_median": confidence_median,
            "high_conf_ratio": high_conf_ratio,
            "weak_ratio": weak_ratio,
            "denoised_count": denoised_count,
            "high_conf_dropped": high_conf_dropped,
            "high_conf_empty": high_conf_empty,
            "denoise_threshold": round(min_velocity, 2),
            "separated": separated_used,
            "separate_target": separate_target if separated_used else None,
            "separate_error": (sep_err[:160] if sep_err else None),
            "high_conf_only": high_conf_only,
            "auto_bpm": auto_bpm,
            "smart_denoise": smart_denoise,
            "backend": backend,
            "hpss": hpss,
            "melody": melody,
            "harmonic_filter": harmonic_filter,
            "min_duration": round(min_duration, 3),
            "gap_merge": round(gap_merge, 3),
            "onset_confirm": onset_confirm,
            "rms_vel": rms_vel,
            "fine_quant": fine_quant,
            "auto_beats": auto_beats,
            "key": key_label,
            "key_raw": key,
            "octave_shift": octave_shift,
            "f0_octave_fix": f0_octave_fix,
            "f0_octave_shifted": f0_shifted,
            "lyrics_source": lyrics_source,
            "lyrics_count": (len([t for t in (lyrics_tokens or []) if t])
                             if lyrics_tokens else 0),
        },
        "files": {},
        # 供评测/调试使用：返回精简后的 note_list（API 层会剥离，不入库、不传前端）
        "note_list": [[round(o, 4), round(s, 4), int(round(p)), round(v, 4)]
                      for (o, s, p, v, _b) in note_list if p >= 0],
        "jianpu": "",
    }

    # MIDI：用后处理后的 note_list 重建，使下载的 .mid 与降噪/高置信过滤、播放音频一致
    # （原先直接写 basic-pitch 原始 midi_obj，会包含被过滤掉的弱音、未量化时值）。
    _prog("生成 MIDI", 70)
    midi_path = os.path.join(out_dir, stem + ".mid")
    try:
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        for (o, s, p, v, _b) in note_list:
            if p < 0 or s <= o:
                continue
            nv = max(1, min(127, int(round(v * 127))))
            inst.notes.append(pretty_midi.Note(
                velocity=nv, pitch=int(round(p)),
                start=float(o), end=float(s)))
        pm.instruments.append(inst)
        if rms_vel:
            # 准确率增强（CPU）：用音符区间 RMS 重算力度，使 MIDI 演奏强弱更自然
            # （不改变模型置信度，置信度仍用于降噪/评估）。
            rms_map = compute_rms_velocities(note_list, transcribe_wav, y=_proc_y)
            if rms_map:
                for nt in inst.notes:
                    rv = rms_map.get((round(nt.start, 4), int(nt.pitch)))
                    if rv is not None:
                        nt.velocity = max(1, min(127, int(round(rv * 127))))
        pm.write(midi_path)
    except Exception as e:
        print("MIDI 重建失败，回退 basic-pitch 原始 MIDI:", e)
        try:
            midi_obj.write(midi_path)
        except Exception:
            pass
    result["files"]["midi"] = midi_path

    # 钢琴卷帘
    _prog("生成钢琴卷帘", 78)
    roll_path = os.path.join(out_dir, stem + "_pianoroll.png")
    if save_pianoroll(note_list, roll_path, audio_path=transcribe_wav):
        result["files"]["pianoroll"] = roll_path

    # 简谱
    _prog("生成简谱", 84)
    result["jianpu"] = build_jianpu(note_list, effective_bpm, effective_beats,
                                    beat_denom=effective_denom, groups=groups,
                                    key=key)

    # 五线谱
    _prog("生成五线谱", 90)
    ly = build_lily(groups, effective_bpm, effective_beats, lyrics=lyrics_tokens,
                    beat_denom=effective_denom, key=key)
    staff_path = os.path.join(out_dir, stem + "_staff.png")
    if render_lily_png(ly, staff_path):
        result["files"]["staff"] = staff_path

    # MusicXML
    _prog("生成 MusicXML", 95)
    xml_path = os.path.join(out_dir, stem + ".musicxml")
    if save_musicxml(note_list, xml_path, effective_bpm, effective_beats,
                     lyrics=lyrics_tokens, beat_denom=effective_denom,
                     groups=groups, key=key):
        result["files"]["musicxml"] = xml_path

    # 原音频（供前端播放）：用原始上传音轨转 mp3（注意：不是分离后的声部）
    source_mp3 = os.path.join(out_dir, stem + "_source.mp3")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", original_wav, "-vn", "-ar", "44100",
                        "-b:a", "128k", source_mp3],
                       check=True, capture_output=True, text=True)
        result["files"]["audio"] = source_mp3
    except Exception as e:
        print("原音频转码失败:", e)

    # 识别钢琴谱合成音频：优先用 fluidsynth + 真实音源；不可用则回退 numpy 基础合成
    score_wav = os.path.join(out_dir, stem + "_score.wav")
    synth_ok = False
    if FLUIDSYNTH_OK:
        try:
            synth_ok = render_midi_fluidsynth(midi_path, score_wav, SR)
        except Exception as e:
            print("真实音源合成失败，回退到基础合成:", e)
    if not synth_ok:
        synth_ok = synth_score(note_list, score_wav)
    # 无论哪种合成方式，统一转 mp3 供前端播放
    if synth_ok:
        score_mp3 = os.path.join(out_dir, stem + "_score.mp3")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", score_wav, "-vn", "-ar", "44100",
                            "-b:a", "128k", score_mp3],
                           check=True, capture_output=True, text=True)
            result["files"]["score_audio"] = score_mp3
        except Exception as e:
            print("钢琴谱音频转码失败:", e)
        try:
            os.remove(score_wav)
        except OSError:
            pass

    # 清理中间 wav（source.mp3 / score.mp3 保留供播放）
    for wp in (original_wav, transcribe_wav):
        if wp != source_mp3 and wp != score_mp3:
            try:
                os.remove(wp)
            except OSError:
                pass
    _prog("完成", 100)
    return result
