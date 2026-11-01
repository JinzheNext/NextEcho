#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_URL="http://127.0.0.1:8765"
HEALTH_URL="$APP_URL/api/health"
LOG_DIR="$HOME/Library/Logs/NextEcho"
PID_FILE="$LOG_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"
ERROR_LOG="$LOG_DIR/server.error.log"

mkdir -p "$LOG_DIR"
touch "$LOG_FILE" "$ERROR_LOG"

pid_value() {
  cat "$PID_FILE" 2>/dev/null || true
}

pid_is_alive() {
  local pid
  pid="$(pid_value)"
  [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1
}

health_ok() {
  curl -sSf "$HEALTH_URL" >/dev/null 2>&1
}

cleanup_stale_pid() {
  if [ -f "$PID_FILE" ] && ! pid_is_alive; then
    rm -f "$PID_FILE"
  fi
}

start_service() {
  cleanup_stale_pid

  if health_ok; then
    return 0
  fi

  for _ in {1..20}; do
    bash "$ROOT_DIR/scripts/mac_launch_agent.sh" start >/dev/null 2>&1 || true
    if health_ok; then
      rm -f "$PID_FILE"
      return 0
    fi
    sleep 1
  done

  return 1
}

stop_service() {
  bash "$ROOT_DIR/scripts/mac_launch_agent.sh" stop >/dev/null 2>&1 || true
  if pid_is_alive; then
    kill "$(pid_value)" >/dev/null 2>&1 || true
    for _ in {1..10}; do
      if ! pid_is_alive; then
        break
      fi
      sleep 1
    done
  fi
  rm -f "$PID_FILE"
}

print_status() {
  cleanup_stale_pid
  if health_ok; then
    printf "NextEcho service: running\n"
    printf "URL: %s\n" "$APP_URL"
    printf "Log: %s\n" "$LOG_FILE"
    printf "Error log: %s\n" "$ERROR_LOG"
    return 0
  fi

  printf "NextEcho service: stopped\n"
  printf "Log: %s\n" "$LOG_FILE"
  printf "Error log: %s\n" "$ERROR_LOG"
  return 1
}

case "${1:-start}" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  status)
    print_status
    ;;
  *)
    echo "Usage: $0 {start|stop|status}" >&2
    exit 1
    ;;
esac
