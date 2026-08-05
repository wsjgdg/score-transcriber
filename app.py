"""
音视频转乐谱 —— FastAPI 服务
提供上传音频/视频、可选 BPM 与拍号，返回乐谱图片、简谱文本与可下载的 MIDI/MusicXML。
"""
import os
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import threading
import tempfile
import numpy as np
import soundfile as sf
import transcriber
import omr_jianpu
import omr_staff

BASE = Path(__file__).resolve().parent
UPLOAD = BASE / "uploads"
OUT = BASE / "outputs"
UPLOAD.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# 历史记录：把每次成功转谱的结果存入 JSON 文件，便于回看（本运行实例内有效）
HISTORY_FILE = BASE / "history.json"
HISTORY_LOCK = threading.Lock()
MAX_HISTORY = 50

# 转录任务进度：uid -> {status, stage, progress, result, error}
# 长任务（ffmpeg/Demucs/basic-pitch/LilyPond）放后台线程跑，前端轮询拿进度与结果，
# 避免同步请求在预览代理超时（约 60s）内被掐断。
_JOBS = {}
_JOBS_LOCK = threading.Lock()

app = FastAPI(title="音视频转乐谱")
app.mount("/outputs", StaticFiles(directory=str(OUT)), name="outputs")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


# 启动时在后台预加载转录模型，避免首个请求撞上冷启动（tf 模型加载耗时几十秒，
# 预览代理可能等不及而断开连接，导致前端 Failed to fetch）。
def _warmup_model():
    try:
        y = np.zeros(int(transcriber.SR * 0.5), dtype="float32")
        fd, p = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(p, y, transcriber.SR)
            transcriber.transcribe(p)   # 触发 basic-pitch 模型加载并驻留内存
        finally:
            try:
                os.remove(p)
            except OSError:
                pass
    except Exception as e:
        print("模型预热失败（可忽略，首请求会自动加载）:", e)


@app.on_event("startup")
def on_startup():
    threading.Thread(target=_warmup_model, daemon=True).start()
    # 历史记录存于磁盘（BASE/history.json），随运行实例持久保留，重启后自动加载
    try:
        n = len(_load_history())
        print(f"已加载历史记录 {n} 条（位于 {HISTORY_FILE}）")
    except Exception:
        pass

ALLOWED = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma",
           ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}


# ---------------------------------------------------------------------------
# 历史记录（JSON 文件持久化，单进程内用锁保护并发写入）
# ---------------------------------------------------------------------------
def _load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _add_history(record: dict):
    with HISTORY_LOCK:
        hist = _load_history()
        hist.append(record)
        if len(hist) > MAX_HISTORY:
            removed = hist[:-MAX_HISTORY]
            hist = hist[-MAX_HISTORY:]
            # 回收被移除记录的输出目录，避免磁盘随使用无限增长。
            # 输出目录名即为 record["id"]（process 写入 OUT/uid），直接据此回收，
            # 不再依赖 files.midi 路径推断（无 midi 时旧逻辑会漏删）。
            for rec in removed:
                try:
                    d = OUT / rec["id"]
                    if d.is_dir():
                        shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)


def _clear_history():
    with HISTORY_LOCK:
        try:
            HISTORY_FILE.write_text("[]", encoding="utf-8")
        except Exception:
            pass


@app.get("/")
def index():
    # 直接跳到新上传页面（app.html），避免手机缓存旧的 index.html 入口
    return RedirectResponse(url="/app.html", status_code=302)


@app.get("/app.html", response_class=HTMLResponse)
def app_page():
    # no-store：避免手机/浏览器缓存旧版页面
    return HTMLResponse(
        (BASE / "static" / "app.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/view.html", response_class=HTMLResponse)
def view_page():
    # 历史记录查看页（独立页面，按 ?id= 展示单条结果）
    return HTMLResponse(
        (BASE / "static" / "view.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...),
                         bpm: float = Form(120),
                         beats: int = Form(4),
                         denoise_threshold: float = Form(0.0),
                         separate: bool = Form(True),
                         separate_target: str = Form("vocals"),
                         high_conf_only: bool = Form(False),
                         auto_bpm: bool = Form(False),
                         smart_denoise: bool = Form(False),
                         backend: str = Form("basic-pitch"),
                         lyrics: str = Form(""),
                         asr: bool = Form(False),
                         hpss: bool = Form(False),
                         melody: bool = Form(True),
                         clean: bool = Form(True),
                         rms_vel: bool = Form(False),
                         fine_quant: bool = Form(False),
                         auto_beats: bool = Form(False),
                         key: str = Form("C"),
                         octave_shift: int = Form(0),
                         f0_octave_fix: bool = Form(True)):
    ext = Path(file.filename or "x.wav").suffix.lower()
    if ext not in ALLOWED:
        return JSONResponse({"status": "error", "msg": f"不支持的文件格式：{ext}"})

    # 服务端强制大小上限，避免大文件拖垮实例（README 建议 ≤30MB）。
    # 先按 Content-Length 粗筛（部分客户端以分块方式上传不发送该头，则落盘后再查）。
    MAX_UPLOAD = 30 * 1024 * 1024
    _cl = file.headers.get("content-length")
    if _cl and _cl.isdigit() and int(_cl) > MAX_UPLOAD:
        return JSONResponse({"status": "error",
            "msg": "文件过大（上限 30MB），请截取 ≤1 分钟片段后重试。"})

    uid = uuid.uuid4().hex
    src = UPLOAD / f"{uid}{ext}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if src.stat().st_size > MAX_UPLOAD:
        try:
            os.remove(src)
        except OSError:
            pass
        return JSONResponse({"status": "error",
            "msg": "文件过大（上限 30MB），请截取 ≤1 分钟片段后重试。"})

    bpm = max(40.0, min(300.0, float(bpm)))
    beats = max(1, min(12, int(beats)))
    # 降噪阈值：过滤模型置信度（velocity）低于该值的弱音符（疑似伴奏/噪声误捡）。
    # 0 表示关闭降噪，保留全部音符；范围 0~1，默认 0.5。
    min_velocity = max(0.0, min(1.0, float(denoise_threshold)))
    separate_target = separate_target or "vocals"
    backend = backend or "basic-pitch"
    key = key or "C"
    octave_shift = max(-2, min(2, int(octave_shift)))

    # 整理本次优化选项，便于历史回看展示
    opts = {
        "separate": separate, "separate_target": separate_target,
        "high_conf_only": high_conf_only, "auto_bpm": auto_bpm,
        "smart_denoise": smart_denoise, "backend": backend,
        "lyrics": bool(lyrics and lyrics.strip()),
        "asr": asr,
        "hpss": hpss, "melody": melody, "clean": clean,
        "rms_vel": rms_vel, "fine_quant": fine_quant, "auto_beats": auto_beats,
        "key": key, "octave_shift": octave_shift,
        "f0_octave_fix": f0_octave_fix,
    }

    # 长任务放后台线程，立即返回 job_id；前端轮询 /api/progress/{job_id} 拿进度与最终谱面。
    with _JOBS_LOCK:
        # 防止任务字典无限增长：超过上限时回收最早的一批已完成/过期任务
        if len(_JOBS) > 300:
            old_ids = [k for k in _JOBS if _JOBS[k]["status"] in ("done", "error")]
            for k in old_ids[: len(_JOBS) - 300]:
                _JOBS.pop(k, None)
        _JOBS[uid] = {"status": "queued", "stage": "排队中", "progress": 0,
                      "result": None, "error": None}
    threading.Thread(
        target=_run_job,
        args=(uid, str(src), str(OUT / uid), bpm, beats, min_velocity,
              separate, separate_target, high_conf_only, auto_bpm,
              smart_denoise, backend, lyrics, asr, hpss, melody, clean,
              rms_vel, fine_quant, auto_beats, key, octave_shift, f0_octave_fix, opts, file.filename),
        daemon=True,
    ).start()
    return {"status": "queued", "job_id": uid}


def _run_job(uid, src, out_dir, bpm, beats, min_velocity, separate,
             separate_target, high_conf_only, auto_bpm, smart_denoise,
             backend, lyrics, asr, hpss, melody, clean, rms_vel, fine_quant,
             auto_beats, key, octave_shift, f0_octave_fix, opts, filename):
    """后台执行转录并把进度/结果写入 _JOBS[uid]，完成时写历史。"""
    def _report(stage, pct):
        with _JOBS_LOCK:
            j = _JOBS.get(uid)
            if j:
                j["stage"] = stage
                j["progress"] = pct

    def _finish(status, result=None, error=None):
        with _JOBS_LOCK:
            _JOBS[uid] = {"status": status, "stage": "", "progress": 100,
                          "result": result, "error": error}
        try:
            os.remove(src)
        except OSError:
            pass

    try:
        res = transcriber.process(src, out_dir, bpm=bpm, beats=beats,
                                  min_velocity=min_velocity,
                                  separate=separate, separate_target=separate_target,
                                  high_conf_only=high_conf_only, auto_bpm=auto_bpm,
                                  smart_denoise=smart_denoise, backend=backend,
                                  lyrics=lyrics, asr=asr,
                                  hpss=hpss, melody=melody,
                                  harmonic_filter=clean,
                                  min_duration=0.04 if clean else 0.0,
                                  gap_merge=0.03 if clean else 0.0,
                                  onset_confirm=clean,
                                  rms_vel=rms_vel, fine_quant=fine_quant,
                                  auto_beats=auto_beats, key=key,
                                  octave_shift=octave_shift,
                                  f0_octave_fix=f0_octave_fix,
                                  on_progress=_report)
        res.pop("note_list", None)   # 评测用，不入库、不传前端
    except transcriber.BackendUnavailable as e:
        # 可选大模型后端（MT3 / Whisper）未启用：按受控错误消息分支，避免字符串匹配误判。
        msg = str(e)
        if "MT3" in msg:
            _finish("error", error="MT3 模型后端未启用：当前为免费托管环境，无法运行该大模型。"
                     "请在自有 GPU 服务器上部署并安装 mt3 后切换模型后端。")
        elif "Whisper" in msg:
            _finish("error", error="Whisper 歌词识别未启用：当前为免费托管环境，无法运行该大模型。"
                     "请在自有 GPU 服务器上部署并安装 faster-whisper / whisper 后启用「自动识别人声歌词」。")
        else:
            _finish("error", error=f"处理失败：{msg}")
        return
    except Exception as e:
        _finish("error", error=f"处理失败：{e}")
        return

    if res.get("status") != "ok" or not res.get("files"):
        _finish("error", error="未检测到音符，请换一段人声/旋律更清晰、伴奏更弱的音频再试。")
        return

    def to_url(p):
        return "/outputs/" + os.path.relpath(p, str(OUT)).replace(os.sep, "/")

    files = {k: to_url(v) for k, v in res.get("files", {}).items()}

    # 写入历史记录（结构与响应一致，便于前端直接复用 render 回看）
    record = {
        "id": uid,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "bpm": bpm,
        "beats": beats,
        "denoise_threshold": round(min_velocity, 2),
        "opts": opts,
        "stats": res["stats"],
        "files": files,
        "jianpu": res["jianpu"],
    }
    _add_history(record)
    _finish("done", result={"status": "ok", "stats": res["stats"],
                            "files": files, "jianpu": res["jianpu"]})


# ---------------------------------------------------------------------------
# 识谱成曲（OMR）：图片 → 音符表（简谱用本地 OCR，五线谱用 Audiveris）
# 进度轮询端点：前端 runJob 靠它拿 status / stage / progress / result。
# 之前该路由缺失，导致所有长任务（转录 / 识谱）永远停在初始 2%。
@app.get("/api/progress/{job_id}")
def api_progress(job_id: str):
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if not j:
            return {"status": "not_found"}
        return {
            "status": j["status"],
            "stage": j.get("stage", ""),
            "progress": j.get("progress", 0),
            "result": j.get("result"),
            "error": j.get("error"),
        }


# 复用 _JOBS 进度机制：立即返回 job_id，前端轮询 /api/progress/{job_id}。
# ---------------------------------------------------------------------------
@app.post("/api/omr")
async def api_omr(file: UploadFile = File(...),
                  notation: str = Form("jianpu"),
                  bpm: float = Form(120),
                  beats: int = Form(4),
                  key: str = Form("C")):
    ext = Path(file.filename or "x.png").suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"):
        return JSONResponse({"status": "error", "msg": f"不支持的图片格式：{ext}"})
    note_type = "staff" if notation == "staff" else "jianpu"
    uid = uuid.uuid4().hex
    src = UPLOAD / f"{uid}{ext}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    bpm = max(40.0, min(300.0, float(bpm)))
    beats = max(1, min(12, int(beats)))
    key = key or "C"
    opts = {"mode": "omr", "notation": note_type, "bpm": bpm,
            "beats": beats, "key": key}
    with _JOBS_LOCK:
        if len(_JOBS) > 300:
            old_ids = [k for k in _JOBS if _JOBS[k]["status"] in ("done", "error")]
            for k in old_ids[: len(_JOBS) - 300]:
                _JOBS.pop(k, None)
        _JOBS[uid] = {"status": "queued", "stage": "排队中", "progress": 0,
                      "result": None, "error": None}
    threading.Thread(
        target=_run_omr_job,
        args=(uid, str(src), str(OUT / uid), note_type, bpm, beats, key, opts, file.filename),
        daemon=True,
    ).start()
    return {"status": "queued", "job_id": uid}


def _run_omr_job(uid, src, out_dir, note_type, bpm, beats, key, opts, filename):
    def _report(stage, pct):
        with _JOBS_LOCK:
            j = _JOBS.get(uid)
            if j:
                j["stage"] = stage
                j["progress"] = pct

    def _finish(status, result=None, error=None):
        with _JOBS_LOCK:
            _JOBS[uid] = {"status": status, "stage": "", "progress": 100,
                          "result": result, "error": error}
        try:
            os.remove(src)
        except OSError:
            pass

    try:
        _report("识别乐谱图片…", 6)
        if note_type == "staff":
            note_list, meta = omr_staff.recognize_staff(src, bpm=bpm, beats=beats, key=key)
        else:
            note_list, meta = omr_jianpu.recognize_jianpu(src, bpm=bpm, beats=beats, key=key)
        _report("生成乐谱文件…", 20)
        if not note_list:
            _finish("error", error="未从图片中识别出音符。请换一张印刷清晰、排版规整的乐谱再试。")
            return
        res = transcriber.build_outputs(note_list, out_dir, bpm=bpm, beats=beats,
                                        key=key, on_progress=_report)
        res.pop("note_list", None)
    except transcriber.BackendUnavailable as e:
        _finish("error", error=str(e))
        return
    except Exception as e:
        _finish("error", error=f"识谱失败：{e}")
        return

    if res.get("status") != "ok" or not res.get("files"):
        _finish("error", error="识谱完成但未生成任何乐谱文件，请重试。")
        return
    _finish_omr_result(uid, res, filename, opts, bpm, beats)


def _finish_omr_result(uid, res, filename, opts, bpm, beats):
    """把 OMR 结果落库历史并标记 job 完成（图片 / 手动文本共用）。"""
    def to_url(p):
        return "/outputs/" + os.path.relpath(p, str(OUT)).replace(os.sep, "/")
    files = {k: to_url(v) for k, v in res.get("files", {}).items()}
    record = {
        "id": uid,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "bpm": bpm,
        "beats": beats,
        "denoise_threshold": 0,
        "opts": opts,
        "stats": res["stats"],
        "files": files,
        "jianpu": res["jianpu"],
    }
    _add_history(record)
    with _JOBS_LOCK:
        _JOBS[uid] = {"status": "done", "stage": "", "progress": 100,
                      "result": {"status": "ok", "stats": res["stats"],
                                 "files": files, "jianpu": res["jianpu"]},
                      "error": None}



@app.post("/api/omr/text")
async def api_omr_text(text: str = Form(...),
                        notation: str = Form("jianpu"),
                        bpm: float = Form(120),
                        beats: int = Form(4),
                        key: str = Form("C")):
    """手动输入简谱文本 → 直接解析为音符表，跳过 OCR 图片识别。"""
    if notation == "staff":
        return JSONResponse({"status": "error",
                             "msg": "手动输入暂仅支持简谱（jianpu）。"})
    uid = uuid.uuid4().hex
    out_dir = str(OUT / uid)
    bpm = max(40.0, min(300.0, float(bpm)))
    beats = max(1, min(12, int(beats)))
    key = key or "C"
    opts = {"mode": "omr", "notation": "jianpu", "bpm": bpm,
            "beats": beats, "key": key, "source": "manual"}
    with _JOBS_LOCK:
        _JOBS[uid] = {"status": "queued", "stage": "排队中", "progress": 0,
                      "result": None, "error": None}
    threading.Thread(target=_run_omr_text_job,
                    args=(uid, text, out_dir, bpm, beats, key, opts),
                    daemon=True).start()
    return {"status": "queued", "job_id": uid}


def _run_omr_text_job(uid, text, out_dir, bpm, beats, key, opts):
    def _report(stage, pct):
        with _JOBS_LOCK:
            j = _JOBS.get(uid)
            if j:
                j["stage"] = stage
                j["progress"] = pct

    def _finish(status, result=None, error=None):
        with _JOBS_LOCK:
            _JOBS[uid] = {"status": status, "stage": "", "progress": 100,
                          "result": result, "error": error}
    try:
        _report("解析简谱文本…", 8)
        note_list = omr_jianpu.parse_jianpu_text(text, bpm=bpm)
        if not note_list:
            _finish("error", error="文本中未解析到任何音符。请按模板格式输入，例如：1 2 3 4 | 5 -")
            return
        _report("生成乐谱文件…", 25)
        res = transcriber.build_outputs(note_list, out_dir, bpm=bpm, beats=beats,
                                        key=key, on_progress=_report)
        res.pop("note_list", None)
    except Exception as e:
        _finish("error", error=f"识谱失败：{e}")
        return
    if res.get("status") != "ok" or not res.get("files"):
        _finish("error", error="生成乐谱失败，请重试。")
        return
    _finish_omr_result(uid, res, "(手动输入)", opts, bpm, beats)


@app.get("/api/caps")
def api_caps():
    # 运行时能力探测：前端据此自适应默认选项（如环境无 demucs 则默认不分离并提示）
    import importlib.util
    def _has(mod):
        return importlib.util.find_spec(mod) is not None
    omr_jianpu_ok = _has("paddleocr") or _has("pytesseract")
    omr_staff_ok = omr_staff.find_audiveris() is not None
    return {"demucs": transcriber._DEMUCS_OK, "mt3": transcriber.MT3_OK,
            "fluidsynth": transcriber.FLUIDSYNTH_OK, "whisper": transcriber.WHISPER_OK,
            "omr_jianpu": omr_jianpu_ok, "omr_staff": omr_staff_ok}
    # 前端轮询拿转录进度与最终结果（长任务避免同步阻塞被代理掐断）。
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if not j:
            return JSONResponse({"status": "not_found",
                                 "msg": "任务不存在或已过期"}, status_code=404)
        return dict(j)


@app.get("/healthz")
def healthz():
    # 平台（Railway / Render 等）健康检查端点：轻量、无副作用。
    return {"status": "ok"}


@app.get("/download/source")
def download_source():
    # 暴露本项目完整源码 zip，便于自托管/本地用户直接下载。
    # 优先用预生成的 zip（云端部署常见）；缺失时按项目目录现打包（排除运行产物）。
    zip_path = BASE / "score-transcriber-github.zip"
    if not zip_path.exists():
        zip_path = Path("/workspace/score-transcriber-github.zip")
    if not zip_path.exists():
        try:
            import tempfile
            import zipfile
            exclude_dirs = {"outputs", "uploads", "__pycache__", ".git"}
            exclude_files = {"score-transcriber-github.zip", "history.json"}
            fd, tmp = tempfile.mkstemp(suffix=".zip")
            os.close(fd)
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                for root, dirs, files in os.walk(BASE):
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    for f in files:
                        if f in exclude_files:
                            continue
                        fp = Path(root) / f
                        if fp.resolve() == Path(tmp).resolve():
                            continue
                        z.write(fp, fp.relative_to(BASE))
            zip_path = Path(tmp)
        except Exception as e:
            return JSONResponse({"status": "error", "msg": f"源码包生成失败：{e}"},
                                status_code=500)
    return FileResponse(
        zip_path, filename="score-transcriber-github.zip",
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="score-transcriber-github.zip"'})


@app.get("/api/history")
def api_history():
    # 返回时倒序（最新在前）
    return {"status": "ok", "history": list(reversed(_load_history()))}


@app.post("/api/history/clear")
def api_history_clear():
    _clear_history()
    return {"status": "ok"}


@app.get("/api/history/{rec_id}")
def api_history_item(rec_id: str):
    for r in _load_history():
        if r.get("id") == rec_id:
            return {"status": "ok", "record": r}
    return JSONResponse(
        {"status": "error", "msg": "未找到该历史记录（可能已被清空或超出保留上限）"},
        status_code=404)


@app.delete("/api/history/{rec_id}")
def api_history_delete(rec_id: str):
    with HISTORY_LOCK:
        hist = _load_history()
        new = [r for r in hist if r.get("id") != rec_id]
        if len(new) == len(hist):
            return JSONResponse(
                {"status": "error", "msg": "未找到该历史记录"}, status_code=404)
        # 回收被删记录的输出目录，释放空间（目录名即 record["id"]）
        for r in hist:
            if r.get("id") == rec_id:
                try:
                    d = OUT / rec_id
                    if d.is_dir():
                        shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
                break
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(new, f, ensure_ascii=False, indent=2)
    return {"status": "ok"}
