FROM python:3.11-slim

# 系统依赖
# - ffmpeg: 音频抽取
# - lilypond: 五线谱渲染
# - fonts-noto-cjk: 钢琴卷帘 / 图表中文
# - fluidsynth + 音源: 真实乐器音色试听（缺失则自动回退基础合成）
# - libgomp1 / libgl1 / libsm6 / libxext6 / libxrender1 / libglib2.0-0:
#       paddlepaddle / tensorflow / torch(demucs) / opencv 运行所需的 .so
# - curl / ca-certificates: 下载 Audiveris 发行包 与 Java 25 JRE
# 注意：Debian 自带 default-jre-headless 仅 Java 21（class file version 上限 65），
#       而 Audiveris 5.11 由 Java 25（class file version 69）编译，会报
#       UnsupportedClassVersionError。故改用 Adoptium Eclipse Temurin 25 JRE，
#       在下方「Java 25 JRE」块单独安装并软链到 PATH。
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    lilypond \
    fonts-noto-cjk \
    fluidsynth \
    timgm6mb-soundfont \
    libgomp1 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Java 25 JRE（Eclipse Temurin，经 Adoptium API 直接拉取最新 25 GA 的 JRE tarball）。
# Debian 自带 JRE 太旧（Java 21），无法满足 Audiveris 5.11 的 class file version 69。
# 解包到 /opt/java25，将 bin/java 软链进 /usr/local/bin（PATH 优先级高于 /usr/bin），
# 并设 JAVA_HOME，确保 Audiveris 官方启动器 /usr/bin/audiveris 命中 Java 25。
ENV JAVA_HOME=/opt/java25
ENV PATH="/opt/java25/bin:${PATH}"
RUN set -eux; \
    curl -fSL -o /tmp/java25.tar.gz \
      "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jre/hotspot/normal/eclipse"; \
    mkdir -p /opt/java25; \
    tar -xzf /tmp/java25.tar.gz -C /opt/java25 --strip-components=1; \
    rm -f /tmp/java25.tar.gz; \
    ln -sf /opt/java25/bin/java /usr/local/bin/java; \
    java -version

# 五线谱 OMR 引擎 Audiveris（开源，Java，纯 Java 应用）。
# 取 ubuntu22.04 的 deb（GLIBC 2.35 低于 Debian 12 的 2.36，兼容性更好）。
# 关键做法：直接 `dpkg-deb -x` 解包到根目录——deb 自带的官方启动器（通常
# /usr/bin/audiveris）与完整应用（audiveris.jar + lib/）落到标准绝对路径，
# classpath / 主类由官方启动器正确设置，find_audiveris() 经 PATH 直接命中，
# 无需自己解析主类（曾误读为 "Audiveris" 导致 ClassNotFoundException）。
# 仅当官方启动器不在 PATH 时，才生成 `java -jar`（JVM 自读 MANIFEST）的兜底启动器。
# 版本固定以保证镜像可复现；升级时改 AUDIVERIS_VERSION 即可。
ENV AUDIVERIS_VERSION=5.11.0
RUN set -eux; \
    curl -fSL -o /tmp/audiveris.deb \
      "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu22.04-x86_64.deb"; \
    dpkg-deb -x /tmp/audiveris.deb /; \
    rm -f /tmp/audiveris.deb; \
    if [ ! -x /usr/bin/audiveris ] && [ ! -x /usr/local/bin/audiveris ]; then \
      appjar=$(find /usr /opt -name audiveris.jar 2>/dev/null | head -n1); \
      if [ -n "$appjar" ]; then \
        printf '#!/bin/sh\nexec java -Xmx1200m -jar "%s" "$@"\n' "$appjar" > /usr/local/bin/audiveris; \
        chmod +x /usr/local/bin/audiveris; \
      fi; \
    fi; \
    which audiveris || ls -la /usr/bin/audiveris /usr/local/bin/audiveris 2>/dev/null || true; \
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
