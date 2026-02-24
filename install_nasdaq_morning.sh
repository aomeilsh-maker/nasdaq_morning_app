#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_PY="$APP_DIR/nasdaq_morning_app.py"
VENV_DIR="$APP_DIR/.venv-nasdaq"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
# On macOS system Python 3.9 + LibreSSL, urllib3 v2 emits warnings. Keep a compatible pin.
"$VENV_DIR/bin/pip" install "urllib3<2" requests yfinance pandas lxml >/dev/null

cat <<EOF
Environment ready.

Run once manually:
  $VENV_DIR/bin/python $APP_PY

Scheduling:
  This script NO LONGER installs macOS LaunchAgent timers.
  Please use OpenClaw cron (recommended), e.g.:

  openclaw cron add \
    --name stock-analysis:daily-0800 \
    --schedule 'cron 0 8 * * * @ Asia/Tokyo' \
    --message "Run $VENV_DIR/bin/python $APP_PY in $APP_DIR"
EOF
