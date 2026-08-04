"""
统一启动入口：读取 PORT 环境变量（平台注入，默认 8000），绑定 0.0.0.0。
Railway / Render / 本地裸跑 / 内网穿透都通过本文件启动，命令一致：
    python run.py
"""
import os
from app import app
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=120)
