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


def _glyph_center(g):
    return (g["x"] + g["w"] / 2.0, g["y"] + g["h"] / 2.0)


def ocr_image(path: str):
    """返回字形列表：[{char, x, y, w, h}, ...]，坐标以左上角为原点（像素）。
    优先 PaddleOCR（中文更准），其次 Tesseract。两者皆无则抛 BackendUnavailable。

    多字符识别框会被**按宽度均分**成逐字符字形，使 `parse_jianpu_glyphs`
    能依据每个数字的真实水平位置判断附点/减时线的归属。
    """
    # 1) PaddleOCR
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        raw = ocr.ocr(path, cls=True)
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
        if glyphs:
            return glyphs
    except Exception:
        pass

    # 2) Tesseract（逐字符框，原点在左下，需翻转 y）
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        data = pytesseract.image_to_boxes(img)
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
    except Exception:
        pass

    from transcriber import BackendUnavailable
    raise BackendUnavailable(
        "未安装 OCR 引擎（PaddleOCR / Tesseract），无法识别简谱图片。\n"
        "请在部署环境安装其一：pip install paddleocr 或 pip install pytesseract "
        "（后者还需系统装 tesseract-ocr）。")


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
