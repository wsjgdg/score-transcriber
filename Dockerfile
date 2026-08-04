FROM python:3.11-slim

# 系统依赖：音频抽取 + 五线谱渲染（必须 apt 装，pip 装不了）
# fonts-noto-cjk 让钢琴卷帘图的中文标题/轴标签正常显示
# fluidsynth + 音源让“识别钢琴谱”用真实乐器音色播放（不可用则自动回退基础合成）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    lilypond \
    fonts-noto-cjk \
    fluidsynth \
    timgm6mb-soundfont \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# basic-pitch 会拉 tensorflow；jax 锁 0.4.38 以兼容 numpy 1.26 / tf 2.15
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "run.py"]
