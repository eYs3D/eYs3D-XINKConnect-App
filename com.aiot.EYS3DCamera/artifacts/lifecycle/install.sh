#!/bin/bash
# install.sh — com.aiot.EYS3DCamera
# Detect arch, symlink the correct libeSPDI native library, and install Python deps.
set -e
cd "$(dirname "$0")/.."

echo "[$(date)] Installing com.aiot.EYS3DCamera..."

ARCH="$(uname -m)"
ESPDI_DIR="sdk/eSPDI"

case "$ARCH" in
    aarch64)
        SRC="libeSPDI_NVIDIA_64.so.5.1.0.2"
        UNVERSIONED="libeSPDI_NVIDIA_64.so"
        ;;
    x86_64)
        SRC="libeSPDI_X86_64.so.5.1.0.2"
        UNVERSIONED="libeSPDI_X86_64.so"
        ;;
    *)
        echo "[install.sh] ERROR: unsupported architecture: $ARCH" >&2
        exit 1
        ;;
esac

if [ ! -f "$ESPDI_DIR/$SRC" ]; then
    echo "[install.sh] ERROR: $ESPDI_DIR/$SRC not found" >&2
    exit 1
fi

# Three symlinks the SDK runtime + eys3dPy.*.so look up via dlopen / SONAME:
#   libeSPDI.so          ← generic name our wrapper / `LD_LIBRARY_PATH` resolution uses
#   libeSPDI_<arch>.so   ← arch-specific unversioned name (matches SONAME used by NVIDIA/X86_64 builds)
ln -sf "$SRC" "$ESPDI_DIR/libeSPDI.so"
ln -sf "$SRC" "$ESPDI_DIR/$UNVERSIONED"
echo "[install.sh] symlinked $ESPDI_DIR/libeSPDI.so + $UNVERSIONED -> $SRC ($ARCH)"

# OpenCL symlinks for the matching arch. The artifact ships only the real
# libOpenCL.so.1.2; .so.1 / .so are recreated here because git-bash on Windows
# corrupts symlinks at packaging time.
OPENCL_LIB_DIR="$ESPDI_DIR/OpenCL/$ARCH/lib"
if [ -f "$OPENCL_LIB_DIR/libOpenCL.so.1.2" ]; then
    ln -sf "libOpenCL.so.1.2" "$OPENCL_LIB_DIR/libOpenCL.so.1"
    ln -sf "libOpenCL.so.1"   "$OPENCL_LIB_DIR/libOpenCL.so"
    echo "[install.sh] symlinked OpenCL libs in $OPENCL_LIB_DIR"
else
    echo "[install.sh] WARN: $OPENCL_LIB_DIR/libOpenCL.so.1.2 missing, OpenCL features may fail"
fi

if [ "${EYS3D_INSTALL_SDK_ONLY:-0}" = "1" ]; then
    echo "[install.sh] EYS3D_INSTALL_SDK_ONLY=1, skipping pip install"
    exit 0
fi

if python3 -m venv .venv 2>/dev/null; then
    echo "[install.sh] using venv"
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
else
    echo "[install.sh] venv unavailable, installing globally"
    rm -rf .venv
    pip3 install -r requirements.txt
fi

# System library probe — non-fatal, just log
echo "[install.sh] checking system libs..."
for lib in libusb-1.0.so.0 liblog4cplus-2.0.so.3 liblog4cplus.so; do
    if ldconfig -p 2>/dev/null | grep -q "$lib"; then
        echo "[install.sh]   found: $lib"
    else
        echo "[install.sh]   WARN: $lib not in ldconfig — install via apt: libusb-1.0-0 liblog4cplus-2.0.0"
    fi
done

echo "[install.sh] com.aiot.EYS3DCamera installed."
