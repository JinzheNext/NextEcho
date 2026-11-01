#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export NEXTECHO_HOST="${NEXTECHO_HOST:-127.0.0.1}"
export NEXTECHO_PORT="${NEXTECHO_PORT:-8765}"

exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/server_runner.py"
