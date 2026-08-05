FROM python:3.11-slim

# 系统依赖
# - ffmpeg: 音频抽取
# - lilypond: 五线谱渲染
# - fonts-noto-cjk: 钢琴卷帘 / 图表中文
# - fluidsynth + 音源: 真实乐器音色试听（缺失则自动回退基础合成）
# - default-jre-headless: 五线谱 OMR 引擎 Audiveris（Java）运行所需
# - libgomp1 / libgl1 / libsm6 / libxext6 / libxrender1 / libglib2.0-0:
#       paddlepaddle / tensorflow / torch(demucs) / opencv 运行所需的 .so
# - curl / ca-certificates: 下载 Audiveris 发行包
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    lilypond \
    fonts-noto-cjk \
    fluidsynth \
    timgm6mb-soundfont \
    default-jre-headless \
    libgomp1 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 五线谱 OMR 引擎 Audiveris（开源，Java，纯 Java 应用）。
# 取 ubuntu22.04 的 deb（GLIBC 2.35 低于 Debian 12 的 2.36，兼容性更好）。
# 关键：Audiveris 不是单 jar 能跑——必须连同 lib/ 依赖。这里解包出完整应用到
# /opt/audiveris（audiveris.jar + lib/），并动态读取 jar 内 Main-Class 生成带
# lib/* classpath 的启动器到 /usr/local/bin/audiveris（在 PATH 上），确保
# find_audiveris() 直接命中且 classpath 正确（仅拷 jar 会因缺 lib 而崩溃）。
# 版本固定以保证镜像可复现；升级时改 AUDIVERIS_VERSION 即可。
ENV AUDIVERIS_VERSION=5.11.0
RUN set -eux; \
    curl -fSL -o /tmp/audiveris.deb \
      "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu22.04-x86_64.deb"; \
    dpkg-deb -x /tmp/audiveris.deb /tmp/audiveris_extract; \
    rm -f /tmp/audiveris.deb; \
    jar=$(find /tmp/audiveris_extract -name 'audiveris.jar' -print -quit); \
    if [ -z "$jar" ]; then echo "ERROR: audiveris.jar not found in extracted deb"; exit 1; fi; \
    app=$(dirname "$jar"); \
    mkdir -p /opt/audiveris; \
    cp -r "$app"/. /opt/audiveris/; \
    rm -rf /tmp/audiveris_extract; \
    main=$(python3 -c "import zipfile,re; z=zipfile.ZipFile('/opt/audiveris/audiveris.jar'); m=z.read('META-INF/MANIFEST.MF').decode('utf-8','ignore'); mm=re.search(r'Main-Class:\s*(\S+)', m); print(mm.group(1) if mm else 'org.audiveris.omr.Main')"); \
    printf '#!/bin/sh\nexec java -Xmx1200m -cp \"/opt/audiveris/audiveris.jar:/opt/audiveris/lib/*\" %s \"$@\"\n' "$main" > /usr/local/bin/audiveris; \
    chmod +x /usr/local/bin/audiveris; \
    which audiveris; \
    java -version

WORKDIR /app

# 依赖清单：默认整站部署用 requirements.txt；仅做识谱（OMR）可改用轻量的
# requirements-omr.txt（不含 basic-pitch / demucs / torch / tensorflow），镜像更小。
ARG REQ_FILE=requirements.txt
COPY ${REQ_FILE} .
# basic-pitch 会拉 tensorflow；jax 锁 0.4.38 以兼容 numpy 1.26 / tf 2.15
RUN pip install --no-cache-dir -r ${REQ_FILE}

# 预下载简谱 OCR（PaddleOCR）模型，避免云上首次请求卡顿。
# 失败不阻断构建——运行时 PaddleOCR 仍会自动下载。
RUN python - <<'PY' || echo "PaddleOCR 模型预下载跳过（运行时将自动下载）"
from PIL import Image
Image.new('RGB', (32, 32)).save('/tmp/_warm.png')
from paddleocr import PaddleOCR
PaddleOCR(use_angle_cls=True, lang='ch', show_log=False).ocr('/tmp/_warm.png', cls=True)
PY

COPY . .

EXPOSE 8000
CMD ["python", "run.py"]
