"""Config loader for com.aiot.EYS3DCamera.

Reads JSON from APP_CONFIG_PATH and merges over DEFAULTS.
"""
import json
import os

DEFAULTS = {
    "stream_mode": "rgb",
    "module_pid": "",
    "mode_index": 1,
    "depth_bits": None,
    "socket_host": "0.0.0.0",
    "socket_port": 9999,
    "video_fps": 15,
    "h264_bitrate": 500000,
    "gop_size": 10,
    "max_capture_fps": 15,
    "det_threshold": 0.5,
    "det_every_n_frames": 1,
    "npu_delegate_path": "/usr/lib/libvx_delegate.so",
    "tflite_model": {},
    # Depth-stream post-processing (no effect when stream_mode == "rgb").
    "depth_filter_enabled": True,
    "depth_filter_edge_preserve": True,
    "depth_filter_edge_level": 5,        # 1..10
    "depth_filter_hole_fill": True,
    "depth_filter_hole_fill_level": 3,   # 1..3
    "depth_filter_hole_fill_kernel": 1,
    "depth_filter_temporal": True,
    "depth_filter_temporal_alpha": 0.4,
}


def load_app_config():
    cfg = dict(DEFAULTS)
    path = os.environ.get("APP_CONFIG_PATH")
    if not path:
        return cfg, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f) or {}
    except (OSError, ValueError):
        return cfg, path
    cfg.update(loaded)
    return cfg, path
