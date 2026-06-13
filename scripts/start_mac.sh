#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
APP_URL="http://127.0.0.1:8765"

show_alert() {
  local message="$1"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display alert \"NextEcho\" message \"$message\""
  fi
}

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

if ! curl -sSf "$APP_URL" >/dev/null 2>&1; then
  nohup "$ROOT_DIR/.venv/bin/python" app.py >/tmp/nextecho-open.log 2>&1 &
  sleep 3
fi

open "$APP_URL"
