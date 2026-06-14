#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT_DIR="$HOME/Library/Application Support/NextEcho"
RUNTIME_DIR="$SUPPORT_DIR/runtime"
LABEL="ai.jinzhe.nextecho"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/NextEcho"
STDOUT_LOG="$LOG_DIR/server.log"
STDERR_LOG="$LOG_DIR/server.error.log"
SITE_PACKAGES_PATH="$(printf '%s\n' "$RUNTIME_DIR"/.venv/lib/python*/site-packages | head -n 1)"
PYTHON_BIN="$(python3 - <<'PY'
from pathlib import Path
print(Path.home() / "Library/Application Support/NextEcho/runtime/.venv/bin/python")
PY
)"

mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents" "$SUPPORT_DIR"
touch "$STDOUT_LOG"
touch "$STDERR_LOG"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python runtime not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -d "$SITE_PACKAGES_PATH" ]; then
  echo "Virtualenv site-packages not found: $SITE_PACKAGES_PATH" >&2
  exit 1
fi

write_plist() {
  cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON_BIN</string>
    <string>-m</string>
    <string>workbench.cli</string>
    <string>serve</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8765</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$RUNTIME_DIR</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$STDOUT_LOG</string>
  <key>StandardErrorPath</key>
  <string>$STDERR_LOG</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>NEXTECHO_HOST</key>
    <string>127.0.0.1</string>
    <key>NEXTECHO_PORT</key>
    <string>8765</string>
    <key>PYTHONPATH</key>
    <string>$SITE_PACKAGES_PATH</string>
    <key>VIRTUAL_ENV</key>
    <string>$RUNTIME_DIR/.venv</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
EOF
}

bootstrap_agent() {
  launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
}

case "${1:-install}" in
  install)
    bash "$ROOT_DIR/scripts/sync_mac_runtime.sh" >/dev/null
    write_plist
    bootstrap_agent
    ;;
  start)
    bash "$ROOT_DIR/scripts/sync_mac_runtime.sh" >/dev/null
    write_plist
    bootstrap_agent
    ;;
  stop)
    launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
    ;;
  status)
    launchctl print "gui/$(id -u)/$LABEL"
    ;;
  plist)
    write_plist
    printf "%s\n" "$PLIST_PATH"
    ;;
  *)
    echo "Usage: $0 {install|start|stop|status|plist}" >&2
    exit 1
    ;;
esac
