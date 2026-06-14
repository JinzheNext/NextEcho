#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT_DIR="$HOME/Library/Application Support/NextEcho"
RUNTIME_DIR="$SUPPORT_DIR/runtime"

mkdir -p "$SUPPORT_DIR"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required to sync the macOS runtime." >&2
  exit 1
fi

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  "$ROOT_DIR/" "$RUNTIME_DIR/"

printf "%s\n" "$RUNTIME_DIR"
