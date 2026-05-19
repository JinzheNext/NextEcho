#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Install Python 3.10+ first."
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
  echo "Missing required system dependencies: ${missing[*]}"
  if command -v brew >/dev/null 2>&1; then
    echo "Suggested install: brew install ffmpeg whisper-cpp"
  else
    echo "Install Homebrew, then run: brew install ffmpeg whisper-cpp"
  fi
fi

if ! command -v yt-dlp >/dev/null 2>&1; then
  echo "Optional dependency yt-dlp is missing. Webpage URL extraction may fail."
  echo "Suggested install: brew install yt-dlp"
fi

mkdir -p models/whisper.cpp outputs/transcriptions
python -m workbench.cli doctor || true

printf "\nInstall complete. Start the web UI with:\n"
echo "  source .venv/bin/activate && python -m workbench.cli serve"
echo "Then open http://127.0.0.1:8765"
