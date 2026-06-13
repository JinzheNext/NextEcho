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

if ! command -v python3 >/dev/null 2>&1; then
  show_alert "未检测到 Python 3。请先安装 Python 3.10 或更高版本，再重新双击安装器。"
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

missing=()
for bin in ffmpeg whisper-cli; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    missing+=("$bin")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg whisper-cpp
  else
    show_alert "NextEcho 需要 Homebrew 来自动安装 ffmpeg 和 whisper.cpp。请先安装 Homebrew，然后重新双击安装器。"
    exit 1
  fi
fi

if ! command -v yt-dlp >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install yt-dlp || true
  fi
fi

mkdir -p models/whisper.cpp outputs/transcriptions
python -m workbench.cli doctor || true
.venv/bin/python -m workbench.cli download-model base || true

if command -v curl >/dev/null 2>&1 && ! curl -sSf "$APP_URL" >/dev/null 2>&1; then
  nohup "$ROOT_DIR/.venv/bin/python" app.py >/tmp/nextecho-install.log 2>&1 &
  sleep 3
fi

open "$APP_URL"
show_alert "NextEcho 安装完成，浏览器即将打开。以后可以直接双击 Open NextEcho.command 或 NextEcho.app 使用。"
