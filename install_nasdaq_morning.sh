#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_PY="$APP_DIR/nasdaq_morning_app.py"
VENV_DIR="$APP_DIR/.venv-nasdaq"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$LAUNCH_AGENTS_DIR/com.openclaw.nasdaqmorning.plist"
LOG_OUT="$APP_DIR/nasdaq_morning.out.log"
LOG_ERR="$APP_DIR/nasdaq_morning.err.log"

mkdir -p "$LAUNCH_AGENTS_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
# On macOS system Python 3.9 + LibreSSL, urllib3 v2 emits warnings. Keep a compatible pin.
"$VENV_DIR/bin/pip" install "urllib3<2" requests yfinance pandas lxml >/dev/null

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.openclaw.nasdaqmorning</string>

  <key>ProgramArguments</key>
  <array>
    <string>$VENV_DIR/bin/python</string>
    <string>$APP_PY</string>
  </array>

  <key>WorkingDirectory</key>
  <string>$APP_DIR</string>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>8</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>

  <key>StandardOutPath</key>
  <string>$LOG_OUT</string>
  <key>StandardErrorPath</key>
  <string>$LOG_ERR</string>

  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "Installed. It will run every day at 08:00 local time."
echo "To test now: $VENV_DIR/bin/python $APP_PY"
echo "To inspect logs: tail -n 100 $LOG_ERR ; tail -n 100 $LOG_OUT"
