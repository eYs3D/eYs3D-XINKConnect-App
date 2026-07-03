"""eYs3D camera auto-detect and Pipeline initialization."""
import time

from eys3d import EYS3DSystem, Device, Config


def _parse_pid(value):
    if not value:
        return None
    if isinstance(value, int):
        return value
    return int(value, 16)


def auto_detect_camera(cfg):
    """Enumerate USB-connected eYs3D cameras and return (device, pid) for the
    first match. If `cfg["module_pid"]` is set, only cameras with that PID are
    considered. Returns (None, None) when nothing matches.
    """
    target_pid = _parse_pid(cfg.get("module_pid", ""))

    sys_obj = EYS3DSystem()
    count = sys_obj.get_camera_device_count()
    if count == 0:
        return None, None

    for idx in range(count):
        device = Device(camera_index=idx)
        info = device.get_device_info()
        detected_pid = info["dev_info"]["PID"]
        if target_pid is None or detected_pid == target_pid:
            return device, detected_pid

    return None, None


def build_sdk_config(device, pid, cfg):
    """Build an SDK Config object using ModeConfig.db preset.

    Note: callback-mode streaming (Device.open_device(...)) consumes this
    Config directly. Do NOT also create a Pipeline — Device's state machine
    forbids mixing callback mode and pipeline mode.
    """
    mode_index = int(cfg.get("mode_index", 1))
    depth_bits = cfg.get("depth_bits")
    # Greengrass config UI may store unset values as "" instead of null,
    # and ints may arrive as numeric strings — normalize both to None.
    if depth_bits in (None, "", "null"):
        depth_bits = None
    usb_type = device.get_usb_type()

    sdk_cfg = Config()
    if depth_bits is None:
        sdk_cfg.set_preset_mode_config(pid, mode_index, usb_type)
    else:
        sdk_cfg.set_preset_mode_config(pid, mode_index, usb_type, depth_bits=int(depth_bits))

    return sdk_cfg


def detect_with_retry(cfg, retry_interval=5.0, max_attempts=None):
    """Poll auto_detect_camera until a camera is found.

    `max_attempts=None` means loop forever (production use); tests pass an int.
    """
    attempts = 0
    while True:
        attempts += 1
        device, pid = auto_detect_camera(cfg)
        if device is not None:
            return device, pid
        if max_attempts is not None and attempts >= max_attempts:
            return None, None
        time.sleep(retry_interval)
