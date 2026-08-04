#!/usr/bin/env bash
# 从 .env 读取 GITHUB_TOKEN 并推送到 origin（避免 token 出现在命令行/聊天里）。
set -euo pipefail
cd "$(dirname "$0")/.."   # 切到项目根目录（scripts/ -> 上一级）

if [ ! -f .env ]; then
  echo ".env 不存在：请 cp .env.example .env 并填入 GITHUB_TOKEN"
  exit 1
fi

set -a
. ./.env
set +a

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GITHUB_TOKEN 在 .env 中为空"
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "pushing branch '${BRANCH}' to origin (token from .env)..."
GIT_TERMINAL_PROMPT=0 git push "https://${GITHUB_TOKEN}@github.com/wsjgdg/score-transcriber.git" "$BRANCH"
