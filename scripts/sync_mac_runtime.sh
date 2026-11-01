#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT_DIR="$HOME/Library/Application Support/NextEcho"
APP_DIR="$SUPPORT_DIR/app"
STATE_DIR="${NEXTECHO_RUNTIME_DIR:-$SUPPORT_DIR/runtime}"

mkdir -p "$SUPPORT_DIR"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required to sync the macOS runtime." >&2
  exit 1
fi

python3 - "$APP_DIR" "$STATE_DIR" <<'PY'
from pathlib import Path
import sys

app, state = (Path(value).expanduser().resolve() for value in sys.argv[1:])
if app == state or app.is_relative_to(state) or state.is_relative_to(app):
    raise SystemExit("Refusing code sync: application and job-state directories overlap.")
PY

mkdir -p "$APP_DIR"
if [ "$(cd "$ROOT_DIR" && pwd -P)" = "$(cd "$APP_DIR" && pwd -P)" ]; then
  printf "%s\n" "$APP_DIR"
  exit 0
fi

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.data/' \
  --exclude '/jobs.sqlite3*' \
  --exclude '/heavy.lock' \
  --exclude '/jobs/' \
  --exclude '/binding.json' \
  --exclude '/calibration.json' \
  "$ROOT_DIR/" "$APP_DIR/"

printf "%s\n" "$APP_DIR"
