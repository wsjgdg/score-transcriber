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

# 五线谱 OMR 引擎 Audiveris（开源，Java）。
# 取官方 .deb 发行包，仅解包取出 audiveris.jar 到 /opt/audiveris
# （omr_staff.find_audiveris() 会自动探测该路径）。
# 版本固定以保证镜像可复现；升级时改 AUDIVERIS_VERSION 即可。
ENV AUDIVERIS_VERSION=5.11.0
RUN set -eux; \
    curl -fSL -o /tmp/audiveris.deb \
      "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb"; \
    dpkg-deb -x /tmp/audiveris.deb /tmp/audiveris_extract; \
    mkdir -p /opt/audiveris; \
    jar=$(find /tmp/audiveris_extract -name 'audiveris.jar' | head -n1); \
    if [ -n "$jar" ]; then cp "$jar" /opt/audiveris/audiveris.jar; fi; \
    rm -rf /tmp/audiveris.deb /tmp/audiveris_extract; \
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
