#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
APP_URL="http://127.0.0.1:8765"
HEALTH_URL="$APP_URL/api/health"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"
ERROR_LOG="$LOG_DIR/server.error.log"

show_alert() {
  local message="$1"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display alert \"NextEcho\" message \"$message\""
  fi
}

mkdir -p "$LOG_DIR"

if [ ! -x "$ROOT_DIR/.venv/bin/python" ]; then
  bash "$ROOT_DIR/scripts/install_mac.sh"
  exit 0
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v whisper-cli >/dev/null 2>&1; then
  bash "$ROOT_DIR/scripts/install_mac.sh"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  show_alert "未检测到 curl，无法检查本地服务状态。请重新运行安装器。"
  exit 1
fi

if ! bash "$ROOT_DIR/scripts/mac_service.sh" start >/dev/null 2>&1; then
  show_alert "NextEcho 服务没有成功启动。请查看日志：$LOG_FILE 和 $ERROR_LOG"
  exit 1
fi

open "$APP_URL"
