#!/bin/bash
# start.sh — com.aiot.EYS3DCamera
# Export absolute SDK paths, then launch backend/main.py.
set -e
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."

ARTIFACT_ROOT="$(pwd)"
SDK_DIR="${ARTIFACT_ROOT}/sdk"

export PYTHONPATH="${SDK_DIR}:${SDK_DIR}/eys3d:${PYTHONPATH:-}"
# When packaged from Windows, the artifact contains both `sdk/eYs3D/` (SDK
# runtime cfg) and `sdk/eys3d/` (Python package). Greengrass extracts them as
# two distinct directories on Linux, but the cfg files (ModeConfig.db,
# libeYs3D.log.config, SWPP.cfg) may end up only in one of them. Pick whichever
# actually contains cfg/ — without this, `set_preset_mode_config` loads an
# empty database and every mode falls back to 0x0 resolution.
if [ -d "${SDK_DIR}/eYs3D/cfg" ]; then
    export EYS3D_HOME="${SDK_DIR}/eYs3D"
elif [ -d "${SDK_DIR}/eys3d/cfg" ]; then
    export EYS3D_HOME="${SDK_DIR}/eys3d"
else
    echo "[start.sh] FATAL: neither sdk/eYs3D/cfg nor sdk/eys3d/cfg exists" >&2
    exit 1
fi
export EYS3D_SDK_HOME="${EYS3D_HOME}"
export LD_LIBRARY_PATH="${SDK_DIR}/eSPDI:${LD_LIBRARY_PATH:-}"

echo "[$(date)] [start.sh pid=$$] APP_CONFIG_PATH=${APP_CONFIG_PATH:-<unset>}"
echo "[start.sh] PYTHONPATH=$PYTHONPATH"
echo "[start.sh] EYS3D_HOME=$EYS3D_HOME"
echo "[start.sh] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

[ -f .env ] && set -a && . ./.env && set +a

PYTHON=$([ -x .venv/bin/python3 ] && echo ".venv/bin/python3" || echo "python3")
exec $PYTHON backend/main.py
