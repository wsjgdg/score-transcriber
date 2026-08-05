"""识谱成曲 · 五线谱图片识别（OMR）。

采用开源 OMR 引擎 **Audiveris**（Java，需自行安装）：把五线谱图片转成
MusicXML，再用 music21 解析为音符表。未安装 Audiveris 时返回友好错误，
提示用户如何安装，而不是静默失败。

探测顺序：PATH 中的 `audiveris` / `Audiveris` → 常见安装目录里的 `audiveris.jar`
（用 `java -jar` 启动）。
"""
import os
import shutil
import subprocess

from transcriber import DEFAULT_KEY, resolve_key, BackendUnavailable

# 常见安装位置（jar / 可执行）
_AUDIVERIS_CANDIDATES = [
    "audiveris", "Audiveris", "audiveris.sh",
]
_AUDIVERIS_DIRS = [
    "/opt/audiveris", "/usr/local/opt/audiveris",
    os.path.expanduser("~/audiveris"), r"C:\Program Files\Audiveris",
    r"C:\Audiveris",
]


def find_audiveris():
    """返回 (cmd_list) 可直接传给 subprocess.run；找不到返回 None。"""
    for name in _AUDIVERIS_CANDIDATES:
        p = shutil.which(name)
        if p:
            return [p]
    # 查找 jar
    for d in _AUDIVERIS_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower() == "audiveris.jar":
                    jar = os.path.join(root, f)
                    java = shutil.which("java")
                    if java:
                        return [java, "-jar", jar]
    return None


def _parse_musicxml(xml_path: str, bpm: float):
    """用 music21 把 MusicXML 解析为 note_list（秒为单位）。"""
    import music21 as m21
    score = m21.converter.parse(xml_path)
    tempo = bpm
    try:
        for mm in score.flatten().getElementsByClass(m21.tempo.MetronomeMark):
            if mm.number:
                tempo = float(mm.number)
                break
    except Exception:
        pass
    spp = 60.0 / max(40.0, min(300.0, tempo))  # 每四分音符秒数
    notes = []
    for n in score.flatten().notes:
        ql = float(n.quarterLength)
        start = float(n.offset) * spp
        end = (float(n.offset) + ql) * spp
        if n.isChord:
            for p in n.pitches:
                notes.append((start, end, int(p.midi), 1.0, None))
        else:
            notes.append((start, end, int(n.pitch.midi), 1.0, None))
    notes.sort(key=lambda x: x[0])
    return notes


def recognize_staff(path: str, bpm: float = 120, beats: int = 4,
                   key: str = DEFAULT_KEY):
    """识别五线谱图片 → (note_list, meta)。依赖 Audiveris。"""
    cmd = find_audiveris()
    if not cmd:
        raise BackendUnavailable(
            "未安装 Audiveris（开源五线谱 OMR 引擎），无法识别五线谱图片。\n"
            "请先安装：\n"
            "  · 下载 Audiveris 并解压（需 Java 11+）；把 `audiveris`/`Audiveris` 加到 PATH，\n"
            "    或把 `audiveris.jar` 放到 /opt/audiveris 等目录。\n"
            "  · 也可改用「简谱」识别（本地 OCR 即可，无需额外引擎）。")

    out_dir = os.path.join(os.path.dirname(path), "audiveris_out")
    os.makedirs(out_dir, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd + ["-batch", "-export", "-output", out_dir, path],
            check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        raise BackendUnavailable(
            "Audiveris 运行失败：" + (e.stderr or e.stdout or str(e))[:300])
    except FileNotFoundError:
        raise BackendUnavailable("Audiveris 可执行文件无法启动，请检查 Java 与安装路径。")

    # 查找生成的 MusicXML
    xml = None
    for root, _d, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith((".mxl", ".xml", ".musicxml")):
                xml = os.path.join(root, f)
                break
        if xml:
            break
    if not xml:
        raise BackendUnavailable(
            "Audiveris 未产出 MusicXML（可能图片不含可识别的五线谱）。原始输出：\n"
            + (proc.stdout or "")[:300])

    note_list = _parse_musicxml(xml, bpm)
    toff, key_label, _l, _sh, _d = resolve_key(key)
    return note_list, {
        "xml": xml,
        "note_count": len([n for n in note_list if n[2] >= 0]),
        "notation": "staff",
        "key": key_label,
    }
