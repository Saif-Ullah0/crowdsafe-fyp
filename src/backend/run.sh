#!/usr/bin/env bash
# CrowdSafe MVP Run Helper Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Python interpreter
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_BIN="../.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

echo "============================================================"
echo " Starting CrowdSafe MVP with $PYTHON_BIN"
echo "============================================================"

exec "$PYTHON_BIN" app.py "$@"
