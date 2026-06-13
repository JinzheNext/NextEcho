#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="NextEcho"
APP_DIR="$ROOT_DIR/dist/$APP_NAME.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"
RESOURCES_DIR="$APP_DIR/Contents/Resources"
LAUNCHER_PATH="$MACOS_DIR/nextecho"
INFO_PLIST="$APP_DIR/Contents/Info.plist"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$LOG_DIR"

cat > "$INFO_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key>
  <string>$APP_NAME</string>
  <key>CFBundleExecutable</key>
  <string>nextecho</string>
  <key>CFBundleIdentifier</key>
  <string>ai.jinzhe.nextecho</string>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
</dict>
</plist>
EOF

cat > "$LAUNCHER_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$ROOT_DIR"
LOG_DIR="$LOG_DIR"
APP_URL="http://127.0.0.1:8765"
LOG_FILE="\$LOG_DIR/app.log"
mkdir -p "\$LOG_DIR"

show_alert() {
  local message="\$1"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display alert \"NextEcho\" message \"\$message\" as critical"
  fi
}

if [ ! -d "\$ROOT_DIR/.venv" ]; then
  echo "[launcher] missing .venv" >> "\$LOG_FILE"
  bash "\$ROOT_DIR/scripts/install_mac.sh"
  exit 0
fi

if [ ! -x "\$ROOT_DIR/.venv/bin/python" ]; then
  echo "[launcher] missing python in virtualenv" >> "\$LOG_FILE"
  bash "\$ROOT_DIR/scripts/install_mac.sh"
  exit 0
fi

cd "\$ROOT_DIR"
"\$ROOT_DIR/.venv/bin/python" -m workbench.cli doctor >> "\$LOG_FILE" 2>&1 || true

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v whisper-cli >/dev/null 2>&1; then
  echo "[launcher] missing ffmpeg or whisper-cli" >> "\$LOG_FILE"
  bash "\$ROOT_DIR/scripts/install_mac.sh"
  exit 0
fi

if ! curl -sSf "\$APP_URL" >/dev/null 2>&1; then
  nohup "\$ROOT_DIR/.venv/bin/python" app.py >> "\$LOG_FILE" 2>&1 &
  sleep 2
fi

open "\$APP_URL"
EOF

chmod +x "$LAUNCHER_PATH"

printf "Built %s\n" "$APP_DIR"
