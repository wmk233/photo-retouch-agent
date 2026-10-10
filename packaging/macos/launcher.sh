#!/bin/zsh
# 图片工作台 —— .app 启动器（由 LaunchServices 直接执行，不弹终端窗口）
set -u

CONTENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
REPO_DIR="${BEAUTY_LAB_REPO:-$HOME/AI-beauty-style-lab/repos/photo-retouch-agent}"
PYTHON="$REPO_DIR/.venv/bin/python"
RUNTIME="$RESOURCES_DIR/app_runtime.py"
LOG_DIR="$HOME/Library/Logs/图片工作台"
mkdir -p "$LOG_DIR"
BOOT_LOG="$LOG_DIR/launcher.log"

fail() {
  /usr/bin/osascript -e "display alert \"图片工作台 无法启动\" message \"$1\" as critical" >/dev/null 2>&1
  echo "[$(/bin/date '+%Y-%m-%d %H:%M:%S')] $1" >> "$BOOT_LOG"
  exit 1
}

[ -x "$PYTHON" ] || fail "未找到运行环境：$PYTHON ；请在仓库中创建虚拟环境后再启动。"
[ -f "$RUNTIME" ] || fail "安装包缺少外壳脚本：$RUNTIME"

export BEAUTY_LAB_REPO="$REPO_DIR"
export PYTHONPATH="$REPO_DIR/backend${PYTHONPATH:+:$PYTHONPATH}"
echo "[$(/bin/date '+%Y-%m-%d %H:%M:%S')] launching $RUNTIME" >> "$BOOT_LOG"
exec "$PYTHON" "$RUNTIME"
