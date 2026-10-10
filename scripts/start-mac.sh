#!/bin/zsh
# AI Beauty Style Lab — 一键启动本地 Web App（macOS）
#
# 用法：
#   ./scripts/start-mac.sh            # 前台启动，日志直接打印
#   ./scripts/start-mac.sh --open     # 启动后在浏览器打开界面
#
# 启动后访问：http://127.0.0.1:8000/
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
HOST="127.0.0.1"
PORT="${BEAUTY_LAB_PORT:-8000}"
URL="http://${HOST}:${PORT}/"

if [ ! -x "$PYTHON" ]; then
  echo "未找到虚拟环境：$PYTHON"
  echo "请先在仓库根目录创建并安装依赖："
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -e ."
  exit 1
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "端口 $PORT 已被占用：可能软件已经在运行。"
  echo "直接打开：$URL"
  if [ "${1:-}" = "--open" ]; then open "$URL"; fi
  exit 0
fi

echo "正在启动 AI Beauty Style Lab …"
echo "访问地址：$URL"
echo "按 Ctrl+C 停止服务。"
echo

if [ "${1:-}" = "--open" ]; then
  ( sleep 1.5; open "$URL" >/dev/null 2>&1 || true ) &
fi

cd "$REPO_DIR/backend"
exec "$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
