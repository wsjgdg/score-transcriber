# 部署方案 B：内网穿透（不买服务器、不用 Docker、不跑 WorkBuddy）

> 适用场景：你有台**一直开着的电脑**（Win/Mac/Linux），想临时把"音视频转乐谱"分享给别人。
> 原理：电脑本地跑服务，用 `cloudflared`（或 ngrok）把本地端口映射成一个公网 HTTPS 地址，转发给别人即可。
> 前提：你的电脑开机且服务在跑，链接才有效。

---

## 1. 在本机准备并启动服务

```bash
cd score-transcriber
pip install -r requirements.txt          # 或已装好则跳过
python run.py                           # 默认监听 http://0.0.0.0:8000
```

> 没装 ffmpeg / lilypond？
> - macOS: `brew install ffmpeg lilypond`
> - Ubuntu: `sudo apt-get install -y ffmpeg lilypond`
> - Windows: 从 ffmpeg.org / lilypond.org 下载，加入 PATH

确认本地能打开：浏览器访问 `http://localhost:8000` 。

## 2. 用 cloudflared 映射公网地址（推荐，免费、无需账号）

```bash
# 安装（任选其一）
# macOS:  brew install cloudflared
# 其他:   去 https://github.com/cloudflare/cloudflared/releases 下载对应平台二进制

# 启动隧道，把本地 8000 端口暴露到公网
cloudflared tunnel --url http://localhost:8000
```

启动后终端会打印一行：

```
Your quick Tunnel has been created! ...
https://xxxx.trycloudflare.com
```

把这个 `https://xxxx.trycloudflare.com` 发给别人即可。**根路径 `/` 会自动跳到 `/app.html` 上传页。**

> 备选 ngrok：`ngrok http 8000` → 同样得到一个公网地址（免费版需登录、有会话时限）。

## 3. 注意事项

- **电脑不能关机/休眠**，否则链接失效。
- 免费隧道地址每次重启会变（cloudflared 的 quick tunnel 是临时的）。要固定地址需登录 cloudflared 建命名隧道。
- 上传文件仍受网络带宽 + 前面说的"建议 1 分钟内、<30MB"建议影响（纯粹是处理耗时，与穿透无关）。
- 关闭穿透：终端 `Ctrl+C` 停掉 `cloudflared` 即可，不影响本地服务。

---

# 部署方案 C：平台一键部署（长期公开、不用本机常开）

> 适用场景：要长期给别人用、不想管服务器、不想装 Docker。
> 原理：把代码推到 GitHub，在 Railway / Render 等平台连仓库自动构建运行，平台给公网域名。
> 前提：一个 GitHub 账号 + 一个平台账号（都有免费额度）。

---

## 准备（一次性）

1. 把整个 `score-transcriber/` 目录推到你的 GitHub 仓库。
   关键文件必须包含：`app.py`、`run.py`、`transcriber.py`、`static/`、`requirements.txt`、`Procfile`。
2. 平台会自动：
   - 用 `requirements.txt` 装 Python 依赖
   - 用 `Procfile`（`web: python run.py`）启动
   - 注入 `PORT` 环境变量（已在 `run.py` 中读取并绑定）

> 平台默认 Ubuntu 构建环境，缺系统包（ffmpeg / lilypond）。
> 下面两种办法二选一，推荐**方案 C1（用 Dockerfile 让平台装系统包）**，最稳。

### C1（推荐）：用 Dockerfile 部署到 Railway

`Dockerfile`（已在本目录提供）已做云端开箱即用加固：

- 装 `default-jre-headless` 并自动下载 **Audiveris 5.11.0** 解包出 `audiveris.jar` 到 `/opt/audiveris` → **五线谱 OMR 在云上直接可用**（无需部署者再装 Java/引擎）。
- 补 `libgomp1 / libgl1 / libsm6 / libxext6 / libxrender1 / libglib2.0-0` → PaddleOCR / tensorflow / torch(demucs) / opencv 在 slim 镜像里能正常 import（否则会缺 `.so` 崩溃）。
- 构建期预下载 **PaddleOCR** 中文模型 → 简谱 OCR 首次请求不再卡顿（失败也不阻断构建）。
- 支持 `ARG REQ_FILE`：**整站部署**用默认 `requirements.txt`（含音频转录重型 ML 栈）；**只想做识谱（OMR）**可改用轻量的 `requirements-omr.txt`（镜像显著更小）：
  ```bash
  docker build --build-arg REQ_FILE=requirements-omr.txt -t score-transcriber-omr .
  ```

即：把仓库推到 GitHub，在 Railway/Render 连仓库选 Docker 部署即可，**识别引擎随镜像预装，访客打开网址即用**。

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg lilypond fonts-noto-cjk fluidsynth timgm6mb-soundfont \
    default-jre-headless libgomp1 libgl1 libsm6 libxext6 libxrender1 \
    libglib2.0-0 curl ca-certificates && rm -rf /var/lib/apt/lists/*
# 下载并解包 Audiveris 5.11.0 到 /opt/audiveris（五线谱 OMR 引擎）
ENV AUDIVERIS_VERSION=5.11.0
RUN set -eux; \
    curl -fSL -o /tmp/audiveris.deb \
      "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb"; \
    dpkg-deb -x /tmp/audiveris.deb /tmp/audiveris_extract; \
    mkdir -p /opt/audiveris; \
    jar=$(find /tmp/audiveris_extract -name 'audiveris.jar' | head -n1); \
    [ -n "$jar" ] && cp "$jar" /opt/audiveris/audiveris.jar; \
    rm -rf /tmp/audiveris.deb /tmp/audiveris_extract; \
    java -version
WORKDIR /app
ARG REQ_FILE=requirements.txt
COPY ${REQ_FILE} .
RUN pip install --no-cache-dir -r ${REQ_FILE}
RUN python - <<'PY' || echo "PaddleOCR 模型预下载跳过（运行时将自动下载）"
from PIL import Image
Image.new('RGB', (32, 32)).save('/tmp/_warm.png')
from paddleocr import PaddleOCR
PaddleOCR(use_angle_cls=True, lang='ch', show_log=False).ocr('/tmp/_warm.png', cls=True)
PY
COPY . .
EXPOSE 8000
CMD ["python", "run.py"]
```

Railway 检测到 Dockerfile 会自动用它构建；`run.py` 读取平台注入的 `PORT`，无需额外配置。

### C1b（Railway 实测推荐）：用 OMR-only 镜像，避开重型 ML 构建失败

在 Railway 上直接构建默认 `Dockerfile`（即 `requirements.txt`）曾实测失败：
`pip install` 会同时拉 `demucs`(torch) + `basic-pitch`(tensorflow/jax) + `paddleocr`(paddlepaddle)，
多 GB 轮子把构建容器的内存/磁盘/时长撑爆，pip 以退出码 1 结束（构建跑满 10m+）。

若你只需「识谱成曲」云端开箱即用（最常见的诉求），用本仓库提供的 **OMR-only 配置**即可：

- `Dockerfile.omr`：除系统依赖 / Audiveris 外，依赖默认 `requirements-omr.txt`（**不含** torch / tensorflow / demucs / basic-pitch / librosa）。
- `railway.json`：已把 `build.dockerfilePath` 设为 `Dockerfile.omr`，推上去 Railway 自动用它构建。

推仓库后在 Railway 触发一次重新部署即可，无需任何额外配置（`run.py` 读取平台注入的 `PORT`）。
需要完整「音视频转乐谱」时，再把 `dockerfilePath` 改回 `Dockerfile` 并准备更大构建资源。

### C2：纯 Python 部署到 Render

Render 的 Python 环境**不含 ffmpeg/lilypond**，纯 pip 方案装不上这两个系统包，所以：

- 若平台提供 "Docker" 部署选项 → 同 C1 用 Dockerfile。
- 若只能用 "Python" 运行时 → 需改用不依赖 lilypond 的渲染（如用 music21 直接出图，或只提供 MIDI/MusicXML），或在构建命令里 `apt-get` 安装（Render 的 Build Command 是 shell，可写 `apt-get update && apt-get install -y ffmpeg lilypond && pip install -r requirements.txt`）。

> 一句话：**Render 上最省心也是走 Dockerfile（C1）**。

## 部署步骤（Railway 示例）

1. 打开 https://railway.app → 用 GitHub 登录 → "New Project" → "Deploy from GitHub repo" → 选你的仓库。
2. 平台识别到 Dockerfile 自动构建；或识别到 Procfile 用 Python 构建。
3. 构建完成后，Railway 给一个 `*.railway.app` 域名，点开即是 `/`（自动跳 `/app.html`）。
4. 转发这个域名给别人即可，长期有效，不用你本机开。

## 注意事项

- **超时/大小限制**：Railway/Render 免费版对单次请求有超时（通常 30–60s），长音频/大视频仍可能失败——和之前沙箱限制同源。要彻底解决需付费档或自建服务器。
- **静态资源**：五线谱/卷帘图/下载文件都经 `/outputs` 提供，平台托管无需额外配置。
- 不再依赖 WorkBuddy，应用跑在平台容器里，WorkBuddy 可随时关闭。

---

## 两种方案对比

| | B 内网穿透 | C 平台部署 |
|---|---|---|
| 是否要本机常开 | ✅ 要（你的电脑） | ❌ 不要 |
| 是否要 Docker | ❌ 不要 | 可选（推荐用） |
| 是否要买服务器 | ❌ 不要 | ❌ 不要（免费额度足够） |
| 链接有效期 | 随本机开关变化 | 长期固定 |
| 适合 | 临时分享、测试 | 长期公开给别人用 |
| 配置难度 | 低（装个 cloudflared） | 低（推仓库连平台） |
