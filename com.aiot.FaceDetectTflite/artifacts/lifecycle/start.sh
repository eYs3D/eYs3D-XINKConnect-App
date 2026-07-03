#!/bin/bash
set -e
VENV_DIR="${AIOT_APP_WORK_DIR:-$PWD}/venv"
cd "$(dirname "$0")/.."

echo "[$(date)] [start.sh pid=$$] APP_CONFIG_PATH=${APP_CONFIG_PATH:-<unset>}"

# Optional: load high-sensitivity secrets from a user-placed .env (see APP_UPLOAD_SPEC.md §3.4)
[ -f .env ] && set -a && . ./.env && set +a

PYTHON=$([ -x "$VENV_DIR/bin/python3" ] && echo "$VENV_DIR/bin/python3" || echo "python3")
exec $PYTHON backend/main.py
