"""识谱成曲 · 简谱（ numbered musical notation ）图片识别。

流程：图片 → OCR（PaddleOCR / Tesseract，自动探测）→ 字符+位置字形 →
按行排序、识别高/低八度点、减时线、附点等 → 顺序铺成音符表 note_list。

注意：简谱 OMR 是启发式实现，针对**印刷清晰、排版规整**的简谱效果最好；
手写、花哨字体、竖排或带复杂装饰的简谱可能识别不准。解析核心
`parse_jianpu_glyphs` 是纯函数（不依赖 OCR），可独立单测。
"""
# 默认调（与 transcriber.DEFAULT_KEY 一致）；transcriber 在运行时惰性导入，避免硬依赖。
DEFAULT_KEY = "C"

# 简谱唱名 → 相对 C 的半音数（C 大调内）
DEGREE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
# 中音 1 的基音 MIDI（这里取 C4 = 60；上方点 +12，下方点 -12）
BASE_MIDI = 60

import logging
logger = logging.getLogger("omr_jianpu")

# PaddleOCR 实例缓存：首次构造会加载检测/识别/分类模型，较重；
# 失败则缓存异常，避免每次请求都重复触发并暴露真实错误。
_PADDLE_OCR = None
_PADDLE_OCR_ERR = None


def _get_paddle_ocr(use_gpu: bool = False):
    """惰性构造并缓存 PaddleOCR 实例；首次失败会记录完整 traceback 并缓存异常。"""
    global _PADDLE_OCR, _PADDLE_OCR_ERR
    if _PADDLE_OCR_ERR is not None:
        raise _PADDLE_OCR_ERR
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu, show_log=False)
        _PADDLE_OCR = ocr
        logger.info("PaddleOCR 初始化成功")
        return ocr
    except Exception as e:
        logger.exception("PaddleOCR 初始化失败")
        _PADDLE_OCR_ERR = e
        raise


def _paddle_to_glyphs(raw):
    """把 PaddleOCR 的 ocr() 返回结构展平成逐字符字形列表。"""
    glyphs = []
    for line in raw:
        if not line:
            continue
        for box, (text, _score) in line:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            w, h = x1 - x0, y1 - y0
            n = len(text)
            if n == 0:
                continue
            char_w = w / n
            for i, ch in enumerate(text):
                glyphs.append({
                    "char": ch,
                    "x": x0 + i * char_w,
                    "y": y0,
                    "w": char_w,
                    "h": h,
                })
    return glyphs


def probe_jianpu():
    """轻量能力探测（供 /api/caps 使用）：只验证模块可 import + paddle 版本，
    不触发重型模型加载。返回 {paddleocr, paddle_version, detail}。"""
    info = {"paddleocr": False, "detail": ""}
    try:
        import paddleocr  # noqa: F401
        info["paddleocr"] = True
    except Exception as e:
        info["detail"] = f"import paddleocr 失败: {type(e).__name__}: {e}"
        return info
    try:
        import paddle
        info["paddle_version"] = getattr(paddle, "__version__", "?")
    except Exception:
        info["paddle_version"] = "unknown"
    return info


def _glyph_center(g):
    return (g["x"] + g["w"] / 2.0, g["y"] + g["h"] / 2.0)


def ocr_image(path: str):
    """返回字形列表：[{char, x, y, w, h}, ...]，坐标以左上角为原点（像素）。
    优先 PaddleOCR（中文更准），其次 Tesseract。两者皆无则抛 BackendUnavailable。

    多字符识别框会被**按宽度均分**成逐字符字形，使 `parse_jianpu_glyphs`
    能依据每个数字的真实水平位置判断附点/减时线的归属。

    注意：PaddleOCR 的任何真实运行错误都会**显式抛出**（带原始异常信息），
    不再被静默吞掉——否则在云端会被伪装成「未安装 OCR 引擎」而难以排查。
    """
    # 1) PaddleOCR（优先）
    try:
        ocr = _get_paddle_ocr()
    except Exception as e:
        logger.warning("PaddleOCR 不可用，跳过：%s", e)
        ocr = None

    if ocr is not None:
        try:
            raw = ocr.ocr(path, cls=True)
            glyphs = _paddle_to_glyphs(raw)
            if glyphs:
                return glyphs
            # 引擎可用但识别为空：尝试 Tesseract 兜底，避免误判为"无内容"
            logger.info("PaddleOCR 未识别出字符，尝试 Tesseract 兜底：%s", path)
        except Exception as e:
            # 真实错误显式透传，不再静默吞掉
            logger.exception("PaddleOCR 识别失败：%s", path)
            raise RuntimeError(f"简谱 OCR（PaddleOCR）识别失败：{e}") from e

    # 2) Tesseract（备选；逐字符框，原点在左下，需翻转 y）
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        try:
            data = pytesseract.image_to_boxes(
                img, config="--psm 6 -c tessedit_char_whitelist=0123456789.|-_><=#")
        except Exception:
            # 某些 Tesseract 版本不接受白名单参数，回退到默认配置
            data = pytesseract.image_to_boxes(img, config="--psm 6")
        W, H = img.size
        glyphs = []
        for row in data.strip().splitlines():
            parts = row.split()
            if len(parts) < 6:
                continue
            ch, x1, y1, x2, y2 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            glyphs.append({"char": ch, "x": x1, "y": H - y2,
                           "w": x2 - x1, "h": y2 - y1})
        if glyphs:
            return glyphs
    except Exception as e:
        logger.warning("Tesseract 不可用，跳过：%s", e)

    from transcriber import BackendUnavailable
    raise BackendUnavailable(
        "OCR 识别失败：PaddleOCR 与 Tesseract 均不可用。\n"
        "请在部署环境安装其一：pip install paddleocr（含 paddlepaddle），"
        "或 apt-get install tesseract-ocr + pip install pytesseract。")


def parse_jianpu_glyphs(glyphs, bpm: float = 120, base_midi: int = BASE_MIDI):
    """把 OCR 字形解析为 note_list：[(onset, offset, pitch, velocity, None), ...]。
    启发式规则（针对横排印刷简谱）：
      · 数字 1-7 = 唱名；0 = 休止符。
      · 数字**上方**相邻小点 = 高八度(+12)，**下方**相邻小点 = 低八度(-12)。
      · 数字**右侧同高**小点 = 附点（+0.5 拍）。
      · 数字**下方**短横线 = 减时线，每条减半（1 条=八分、2 条=十六分）。
      · '|' / '‖' = 小节线，不计时值。
    时值按 bpm 顺序铺排（默认四分音符=1 拍）。

    与数字分离的装饰点（高/低八度点、减时线）在 OCR 中往往落在与数字
    **不同行**的簇里，因此这里的点/线检测直接对**全部字形**做空间检索，
    而非仅限当前行。
    """
    # 过滤无效字形（只保留含可见字符的）
    gs = [g for g in glyphs if g.get("char", "").strip()]
    if not gs:
        return []

    # 按行聚类（纵向中心接近者同属一行），仅用于确定**阅读顺序**：
    # 逐行从上到下、行内从左到右。点/线归属的几何判断使用完整字形表 gs。
    gs_sorted = sorted(gs, key=lambda g: (g["y"], g["x"]))
    lines = []
    cur = [gs_sorted[0]]
    line_h = gs_sorted[0]["h"] or 1
    for g in gs_sorted[1:]:
        if g["y"] - cur[-1]["y"] > max(0.6 * line_h, 6):
            lines.append(cur)
            cur = [g]
            line_h = g["h"] or 1
        else:
            cur.append(g)
            line_h = max(line_h, g["h"] or 1)
    lines.append(cur)

    beat_sec = 60.0 / max(40.0, min(300.0, bpm))
    note_list = []
    cursor = 0.0

    def find_dots(g, all_gs):
        """针对数字 g，从全部字形中找出八度点与附点，返回 (octave_delta, dotted)。

        判定以数字 g 的几何中心 (gcx,gcy) 与自身高宽 (gw,gh) 为基准：
          · 点在中线**显著上方**(dy < -0.5gh) → 高八度 +1（可叠加）
          · 点在中线**显著下方**(dy >  0.5gh) → 低八度 -1（可叠加）
          · 点在中线同高(|dy|<=0.5gh)且**右侧**(dx>0.25gw) → 附点
        """
        gcx, gcy = _glyph_center(g)
        gh = g["h"] or 1
        gw = g["w"] or 1
        octave = 0
        dotted = False
        for d in all_gs:
            if d is g or d["char"] != ".":
                continue
            dcx, dcy = _glyph_center(d)
            dx = dcx - gcx
            dy = dcy - gcy
            # 水平须与数字对齐（覆盖数字本宽 + 右侧附点余量）
            if dx < -1.0 * gw or dx > 1.5 * gw:
                continue
            if dy < -0.5 * gh:          # 显著上方 → 高八度
                octave += 1
            elif dy > 0.5 * gh:         # 显著下方 → 低八度
                octave -= 1
            elif dx > 0.25 * gw:        # 同高且在右侧 → 附点
                dotted = True
        return octave, dotted

    def find_underlines(g, all_gs):
        """数字下方减时线数量（每条减半时值）。"""
        cnt = 0
        gcx, gcy = _glyph_center(g)
        gh = g["h"] or 1
        gw = g["w"] or 1
        for u in all_gs:
            if u is g:
                continue
            if u["char"] not in ("_", "-", "—", "‐"):
                continue
            ucx, ucy = _glyph_center(u)
            # 减时线在数字**下方**、水平覆盖其书写跨度
            if (g["x"] - 0.5 * gw) <= ucx <= (g["x"] + 1.5 * gw) and \
               ucy > gcy + 0.3 * gh:
                cnt += 1
        return cnt

    for line in lines:
        for g in sorted(line, key=lambda x: x["x"]):
            ch = g["char"].strip()
            if ch in ("|", "‖", "│", "!", "∣"):
                continue  # 小节线不计时值
            if ch == "0":
                dur = 1.0
                under = find_underlines(g, gs)
                if under >= 1:
                    dur = 1.0 / (2 ** under)
                if find_dots(g, gs)[1]:
                    dur *= 1.5
                note_list.append((cursor, cursor + dur * beat_sec, -1, 1.0, None))
                cursor += dur * beat_sec
                continue
            if ch.isdigit() and ch in "1234567":
                d = int(ch)
                octave, dotted = find_dots(g, gs)
                pitch = base_midi + DEGREE_SEMITONES[d] + 12 * octave
                dur = 1.0
                under = find_underlines(g, gs)
                if under >= 1:
                    dur = 1.0 / (2 ** under)
                if dotted:
                    dur *= 1.5
                note_list.append((cursor, cursor + dur * beat_sec, pitch, 1.0, None))
                cursor += dur * beat_sec
                continue
            # 其它字符（括号、歌词等）忽略
    return note_list


def recognize_jianpu(path: str, bpm: float = 120, beats: int = 4,
                    key: str = DEFAULT_KEY):
    """识别简谱图片 → (note_list, meta)。"""
    from transcriber import resolve_key
    glyphs = ocr_image(path)
    note_list = parse_jianpu_glyphs(glyphs, bpm=bpm)
    toff, key_label, _l, _sh, _d = resolve_key(key)
    return note_list, {
        "glyph_count": len(glyphs),
        "note_count": len([n for n in note_list if n[2] >= 0]),
        "notation": "jianpu",
        "key": key_label,
    }


# 手动输入用的简谱文本模板（与前端「填充模板」保持一致）
TEMPLATE_JIANPU_TEXT = """# 简谱文本输入（空格分隔每个音；| 小节线；# 开头为注释）
# 唱名: 1 2 3 4 5 6 7   休止: 0
# 高八度加 >  (例 >1)    低八度加 <  (例 <6)
# 八分音符加 _ (例 _1)    十六分加 __ (例 __1)
# 延长拍加 - 每拍一个 (例 1- 2--)   附点加 . (例 3. _1.)
1 2 3 4 | >5 >6 >7 | 0 0 | 3. 2. 1. | _1 _2 _3 _4 | <5 <6 <7 |
"""


def parse_jianpu_text(text, bpm: float = 120, base_midi: int = BASE_MIDI):
    """把简谱文本解析为 note_list：[(onset, offset, pitch, velocity, None), ...]。

    语法（与 parse_jianpu_glyphs 时值规则一致，纯文本版）：
      · 1-7 唱名；0 休止。
      · >数字 高八度(+12)；<数字 低八度(-12)。
      · _数字 八分音符(×0.5)；__数字 十六分(×0.25)。
      · 数字- 延长1拍(可叠加)；数字. 附点(×1.5)。可组合：1-. = 3拍。
      · | 小节线不计时值；# 开头的整行忽略。
    调性（key）不在此处转调，交给下游 build_outputs 统一处理显示，
    与图片 OMR 行为保持一致。
    """
    beat_sec = 60.0 / max(40.0, min(300.0, bpm))
    note_list = []
    cursor = 0.0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for tok in line.replace("|", " ").split():
            octave = 0
            i = 0
            while i < len(tok) and tok[i] in "<>":
                octave += 1 if tok[i] == ">" else -1
                i += 1
            shorten = 0
            while i < len(tok) and tok[i] == "_":
                shorten += 1
                i += 1
            if i >= len(tok):
                continue
            ch = tok[i]
            i += 1
            if ch == "0":
                is_rest = True
            elif ch in "1234567":
                is_rest = False
            else:
                continue
            dur = 1.0
            dotted = False
            sustain = 0
            while i < len(tok):
                c = tok[i]
                if c == "-":
                    sustain += 1
                elif c == ".":
                    dotted = True
                else:
                    break
                i += 1
            if shorten >= 1:
                dur = 1.0 / (2 ** shorten)
            dur += sustain
            if dotted:
                dur *= 1.5
            if is_rest:
                pitch = -1
            else:
                pitch = base_midi + DEGREE_SEMITONES[int(ch)] + 12 * octave
            note_list.append((cursor, cursor + dur * beat_sec, pitch, 1.0, None))
            cursor += dur * beat_sec
    return note_list
