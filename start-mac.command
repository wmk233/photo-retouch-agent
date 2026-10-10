#!/bin/zsh
# 双击即可启动「AI 个人形象设计」本地 Web App。
# 关闭窗口 / 按 Ctrl+C 即停止服务。
cd "$(dirname "$0")"
exec ./scripts/start-mac.sh --open
