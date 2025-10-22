#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# prefer project-root venv if available
ROOT="$(cd "$DIR/.." && pwd)"
PY="$ROOT/.venv/bin/python3"
if [ ! -x "$PY" ]; then
  echo "Virtualenv python not found at $PY — please create venv and install requirements." >&2
  exit 1
fi

# prefer main.py but fall back to bubbleplay.py if main.py is missing
ENTRY="$ROOT/main.py"
if [ ! -f "$ENTRY" ]; then
  if [ -f "$ROOT/bubbleplay.py" ]; then
    ENTRY="$ROOT/bubbleplay.py"
  elif [ -f "$ROOT/bubbleplay/bubbleplay.py" ]; then
    ENTRY="$ROOT/bubbleplay/bubbleplay.py"
  elif [ -f "$DIR/bubbleplay.py" ]; then
    ENTRY="$DIR/bubbleplay.py"
  else
    echo "No entry script found (main.py or bubbleplay.py)." >&2
    exit 1
  fi
fi

echo "Running with $PY -> $ENTRY"
"$PY" "$ENTRY"
