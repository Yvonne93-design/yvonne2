#!/usr/bin/env bash
set -euo pipefail
# Run the prototype using the project's virtual environment (activate then run)
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Activating virtualenv at $DIR/.venv"
source "$DIR/.venv/bin/activate"
echo "Running main.py with $(which python)"
python "$DIR/main.py"
