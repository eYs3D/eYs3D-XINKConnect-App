#!/bin/bash
set -e
VENV_DIR="${AIOT_APP_WORK_DIR:-$PWD}/venv"
cd "$(dirname "$0")/.."

echo "[$(date)] Installing com.aiot.OBBTflite..."

# venv lives in the writable work dir (VENV_DIR = $AIOT_APP_WORK_DIR/venv,
# injected by the generated recipe); the artifacts dir is root-owned read-only.
# macOS provisioning installs python@3.12 (lambda/device/install.sh) because
# pinned deps (e.g. ai-edge-litert==1.2.0) have no wheels for newer Pythons.
# Linux boards only ship the distro python3 (<= 3.12), so fall back to it.
if command -v python3.12 >/dev/null 2>&1; then PY=python3.12; else PY=python3; fi

# The venv is reused across deploys — if it was built by a different Python
# than the one chosen above (e.g. Homebrew python3 -> 3.14), rebuild it.
if [ -x "$VENV_DIR/bin/python3" ] && \
   [ "$("$VENV_DIR/bin/python3" -c 'import sys; print(sys.version_info[:2])' 2>/dev/null)" != \
     "$("$PY" -c 'import sys; print(sys.version_info[:2])' 2>/dev/null)" ]; then
    echo "[$(date)] Existing venv Python differs from $PY — recreating"
    rm -rf "$VENV_DIR"
fi

mkdir -p "$(dirname "$VENV_DIR")"
if "$PY" -m venv "$VENV_DIR" 2>/dev/null; then
    echo "[$(date)] Using virtual environment at $VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r requirements.txt
else
    echo "[$(date)] venv unavailable, installing globally"
    rm -rf "$VENV_DIR"
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
fi

echo "[$(date)] com.aiot.OBBTflite installed successfully."
