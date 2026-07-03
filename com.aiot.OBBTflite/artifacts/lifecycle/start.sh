#!/bin/bash
set -e
VENV_DIR="${AIOT_APP_WORK_DIR:-$PWD}/venv"
cd "$(dirname "$0")/.."

echo "[$(date)] [start.sh pid=$$] APP_CONFIG_PATH=${APP_CONFIG_PATH:-<unset>}"

[ -f .env ] && set -a && . ./.env && set +a

PYTHON=$([ -x "$VENV_DIR/bin/python3" ] && echo "$VENV_DIR/bin/python3" || echo "python3")
exec $PYTHON backend/main.py
