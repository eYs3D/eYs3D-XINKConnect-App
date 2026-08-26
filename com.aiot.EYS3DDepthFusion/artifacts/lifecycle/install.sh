#!/bin/bash
# install.sh — com.aiot.EYS3DDepthFusion
# Detect arch, symlink the correct libeSPDI native library, and install Python deps.
set -e
VENV_DIR="${AIOT_APP_WORK_DIR:-$PWD}/venv"
cd "$(dirname "$0")/.."

echo "[$(date)] Installing com.aiot.EYS3DDepthFusion..."

ARCH="$(uname -m)"
ESPDI_DIR="sdk/eSPDI"

case "$ARCH" in
    aarch64)
        SRC="libeSPDI_NVIDIA_64.so.5.1.0.2"
        ;;
    x86_64)
        SRC="libeSPDI_X86_64.so.5.1.0.2"
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
echo "[install.sh] using $ESPDI_DIR/$SRC ($ARCH)"

# NOTHING in this directory may be a symlink, and none are needed.
#
# Why none are needed: eys3dPy.cpython-*.so records `NEEDED
# libeSPDI_<arch>.so.5.1.0.2` — the fully versioned name, matching the .so's
# SONAME — plus `RUNPATH $ORIGIN:$ORIGIN/eSPDI/`, which points straight at the
# real file ($ORIGIN is sdk/, so $ORIGIN/eSPDI is this directory). The loader
# matches NEEDED entries by exact filename, never by "some 5.1.x", so no
# unversioned alias is ever looked up; nothing in backend/ or sdk/eys3d/
# dlopen()s one either. (start.sh also exports $ESPDI_DIR on LD_LIBRARY_PATH,
# but RUNPATH already resolves it — that export is redundant belt-and-braces.)
#
# libeSPDI itself records `NEEDED libOpenCL.so.1`, satisfied from the system
# (/lib/<triplet>/libOpenCL.so.1). We deliberately do NOT bundle that: only
# libOpenCL.so.1.2 was ever shipped, reachable solely through a symlink, and
# libeSPDI uses DT_RPATH ($ORIGIN/OpenCL/<arch>/lib) which outranks both
# LD_LIBRARY_PATH and ldconfig — so a bundled copy would permanently shadow the
# device's real driver with no way to override it. The bundled file was only an
# ocl_icd dispatch shim (~30KB) anyway, and yields zero platforms on a device
# with no /etc/OpenCL/vendors. The whole OpenCL/ tree was dropped in 0.1.6.
#
# Why none may exist: Greengrass sets artifact permissions by opening every
# entry with O_NOFOLLOW, and Linux returns ELOOP for O_NOFOLLOW on *any*
# symlink — a valid one-hop link, not only a cycle. Greengrass surfaces that as
# "Too many levels of symbolic links" and fails the entire deployment with
# SET_PERMISSION_ERROR, which blocks every component in it, not just this one.
# The walk runs on each deployment, so one symlink here bricks all future
# deployments to the device until it is removed by hand.
#
# This cleanup is what un-bricks devices that already ran version <= 0.1.5.
find "$ESPDI_DIR" -type l -delete 2>/dev/null || true

if [ "${EYS3D_INSTALL_SDK_ONLY:-0}" = "1" ]; then
    echo "[install.sh] EYS3D_INSTALL_SDK_ONLY=1, skipping pip install"
    exit 0
fi

# The venv MUST NOT live in the artifacts dir. `python3 -m venv` creates a
# lib64 -> lib symlink, and Greengrass recursively chmods this directory on
# every later deployment; it follows that symlink and aborts with
# "Too many levels of symbolic links", permanently blocking ALL deployments to
# the device (not just this component). VENV_DIR is the writable work dir
# injected by the generated recipe as AIOT_APP_WORK_DIR.
mkdir -p "$(dirname "$VENV_DIR")"
if python3 -m venv "$VENV_DIR" 2>/dev/null; then
    echo "[install.sh] using venv at $VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r requirements.txt
else
    echo "[install.sh] venv unavailable, installing globally"
    rm -rf "$VENV_DIR"
    pip3 install -r requirements.txt
fi

# Remove the in-artifact venv left behind by versions <= 1.0.1, which is what
# triggers the chmod failure above on devices that already ran the old script.
rm -rf .venv

# System library probe — non-fatal, just log
echo "[install.sh] checking system libs..."
for lib in libusb-1.0.so.0 liblog4cplus-2.0.so.3 liblog4cplus.so; do
    if ldconfig -p 2>/dev/null | grep -q "$lib"; then
        echo "[install.sh]   found: $lib"
    else
        echo "[install.sh]   WARN: $lib not in ldconfig — install via apt: libusb-1.0-0 liblog4cplus-2.0.0"
    fi
done

echo "[install.sh] com.aiot.EYS3DDepthFusion installed."
