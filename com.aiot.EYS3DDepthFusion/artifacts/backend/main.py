#!/usr/bin/env python3
"""
com.aiot.EYS3DDepthFusion — eYs3D stereo depth fused with MiDaS monocular depth.

Why this is a separate component from com.aiot.EYS3DCamera rather than a mode
inside it: fusion needs metric millimetres from `Frame.get_depth_ZD_value()`,
and the only thing EYS3DCamera puts on the wire is a colorized heatmap through
a lossy H264 encoder. Millimetres cannot be recovered downstream, so the fusion
has to happen in the process that owns the camera. Keeping it here leaves
EYS3DCamera and DepthTflite as the minimal single-purpose examples they are.

Pipeline:
  C++ SDK producer
    ├─ color_callback ─┐
    │                  ├─ FramePairer (match by frame serial number)
    └─ depth_callback ─┘        └─ queue → FusionThread
                                             ├─ MiDaS (TFLite, NPU or CPU)
                                             ├─ DepthFuser (align → fill)
                                             └─ colorize/compose → last_frame

  StreamServer (TCP, length-prefixed H264 packets)
    └─ per-client handler thread → libx264 encoder → sendall()

APP_CONFIG_PATH JSON keys: see config.py DEFAULTS.
"""
import json
import os
import queue
import signal
import sys
import threading
import time

from config import load_app_config
import camera as camera_mod
import capture as capture_mod
import depth_filter as depth_filter_mod
import ir_control
from depth_fusion import DepthFuser
from fusion_worker import FusionThread
from stream_server import StreamServer
from shutdown import shutdown as graceful_shutdown

# Fallback working range when the device reports none. 300mm is below the
# closest reading observed on the ecv board (338mm); 8000mm is well past where
# fills stay trustworthy, and raising it to the sensor's 16383mm saturation
# ceiling moved the valid-pixel fraction by 0.002, so it buys nothing.
DEFAULT_Z_MIN_MM = 300.0
DEFAULT_Z_MAX_MM = 8000.0


def _resolve_z_range(device, cfg):
    """Working depth range in mm for the validity mask.

    Asks the device first, since a module reports the range it is actually
    calibrated for. In practice this is usually unusable: the ecv board
    (PID 0x181, SDK 5.1.0.2) returns `{'Near': 0, 'Far': 0}`, so config is the
    only real source. The device path is kept because a populated response is
    strictly better than a constant, but the fallback is the normal case, not
    the exception — hence the plain log rather than a warning.

    Key spelling varies across SDK builds; 'Near'/'Far' is what this one uses.
    """
    z_min = cfg.get("z_min_mm")
    z_max = cfg.get("z_max_mm")
    if z_min and z_max:
        return float(z_min), float(z_max)

    fallback = (float(z_min or DEFAULT_Z_MIN_MM), float(z_max or DEFAULT_Z_MAX_MM))
    try:
        rng = device.get_z_range() or {}
    except Exception as exc:
        print(f"[main] get_z_range unavailable ({exc}); using {fallback} mm", flush=True)
        return fallback

    def _pick(*names):
        for n in names:
            v = rng.get(n)
            if v:  # 0 means "not reported", which is what this SDK returns
                return float(v)
        return None

    lo = _pick("Near", "zNear", "z_near", "near", "min")
    hi = _pick("Far", "zFar", "z_far", "far", "max")
    if lo is None or hi is None or hi <= lo:
        print(f"[main] device z range {rng} unusable; using {fallback} mm", flush=True)
        return fallback
    return lo, hi


def _create_estimator(cfg):
    """MiDaS TFLite estimator, or None to run raw-depth passthrough.

    Returning None rather than raising keeps the component useful as a plain
    depth streamer on a device with no TFLite runtime, which is also how the
    raw-depth baseline is captured for comparison.
    """
    model = cfg.get("tflite_model") or {}
    path = model.get("path") if isinstance(model, dict) else model
    if not path:
        print("[main] no tflite_model configured; raw depth only", flush=True)
        return None
    if not os.path.isabs(str(path)):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), str(path))
    if not os.path.isfile(str(path)):
        print(f"[main] model not found at {path}; raw depth only", flush=True)
        return None

    try:
        from depth_estimator import DepthEstimator
        return DepthEstimator(
            path,
            delegate_path=cfg.get("npu_delegate_path"),
            input_size=int(cfg.get("midas_input_size", 256)),
        )
    except Exception as exc:
        print(f"[main] MiDaS init failed ({exc}); raw depth only", flush=True)
        return None


def main():
    cfg, cfg_path = load_app_config()
    print(f"[main] pid={os.getpid()} config={cfg_path!r}", flush=True)
    print(f"[main] params={json.dumps({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}, sort_keys=True)}", flush=True)

    # ── 1. Auto-detect camera (with retry) ────────────────────────────
    print("[main] detecting eYs3D camera...", flush=True)
    device, pid = camera_mod.detect_with_retry(cfg, retry_interval=5.0)
    if device is None:
        print("[main] no camera detected, aborting", flush=True)
        sys.exit(1)
    print(f"[main] camera detected: PID=0x{pid:04x}", flush=True)

    # ── 2. Build SDK config (callback-mode; not a Pipeline) ───────────
    sdk_cfg = camera_mod.build_sdk_config(device, pid, cfg)
    color_h, color_w = sdk_cfg.get_color_stream_resolution()
    depth_h, depth_w = sdk_cfg.get_depth_stream_resolution()
    print(f"[main] color={color_w}x{color_h} depth={depth_w}x{depth_h}", flush=True)
    if (color_w, color_h) != (depth_w, depth_h):
        # Fusion is per-pixel, so the two maps must share a geometry. MiDaS
        # output is resized to the depth frame, which is correct only if the
        # views are also registered — see README "Registration".
        print("[main] WARN: color and depth resolutions differ; "
              "fusion assumes registered views", flush=True)

    z_min, z_max = _resolve_z_range(device, cfg)
    print(f"[main] depth working range: {z_min:.0f}..{z_max:.0f} mm", flush=True)

    # ── 3. Shared state ───────────────────────────────────────────────
    last_frame = capture_mod.LastFrameHolder()
    # maxsize 1: only the newest pair is worth processing. A deeper queue would
    # buy latency, not throughput, because fusion is the bottleneck.
    pair_q = queue.Queue(maxsize=1)

    def _emit(rgb, z_mm, heat):
        try:
            pair_q.put_nowait((rgb, z_mm, heat))
        except queue.Full:
            pass  # fusion is behind; drop rather than queue stale frames

    pairer = capture_mod.FramePairer(
        emit=_emit,
        max_sn_delta=int(cfg.get("max_serial_delta", 1)),
        log=lambda m: print(m, flush=True),
    )

    max_fps = int(cfg.get("max_capture_fps", 15))
    color_cb = capture_mod.make_color_callback(
        state=capture_mod.CallbackState(), pairer=pairer,
        max_capture_fps=max_fps, height=color_h, width=color_w)
    depth_cb = capture_mod.make_depth_callback(
        state=capture_mod.CallbackState(), pairer=pairer,
        max_capture_fps=max_fps, height=depth_h, width=depth_w)

    # ── 4. Open device with BOTH callbacks ────────────────────────────
    device.open_device(sdk_cfg, colorFrameCallback=color_cb, depthFrameCallback=depth_cb)
    device.enable_color_depth_stream()

    # ── 4b. IR projector ──────────────────────────────────────────────
    # Do this first: every hole IR closes is a real measurement, which beats
    # anything fusion can estimate for the same pixel.
    ir_control.configure(device, cfg)

    # ── 4c. SDK depth post-processing ─────────────────────────────────
    # Runs before fusion sees the data, so hole-fill/temporal settings change
    # what fusion has left to do. Turn them off to measure fusion alone.
    depth_filter_mod.configure(device, cfg)

    # ── 5. Fusion thread ──────────────────────────────────────────────
    estimator = _create_estimator(cfg)
    fuser = DepthFuser(
        z_min=z_min, z_max=z_max,
        ema_alpha=float(cfg.get("fusion_ema_alpha", 0.3)),
        trim_sigma=float(cfg.get("fusion_trim_sigma", 2.5)),
        trim_rounds=int(cfg.get("fusion_trim_rounds", 2)),
        min_valid_ratio=float(cfg.get("fusion_min_valid_ratio", 0.05)),
        correction_kernel=int(cfg.get("fusion_correction_kernel", 101)),
        subsample=int(cfg.get("fusion_subsample", 4)),
        local_correction=bool(cfg.get("fusion_local_correction", True)),
        max_fill_mm=float(cfg.get("fusion_max_fill_mm", 2500)),
    )
    worker = FusionThread(
        in_queue=pair_q,
        last_frame_holder=last_frame,
        estimator=estimator,
        fuser=fuser,
        display_mode=str(cfg.get("display_mode", "fused")),
        colormap=str(cfg.get("colormap", "eys3d")),
        overlay_alpha=float(cfg.get("overlay_alpha", 0.5)),
        show_hud=bool(cfg.get("show_hud", True)),
        midas_every_n=int(cfg.get("midas_every_n", 4)),
        display_min_mm=float(cfg.get("display_min_mm", 200)),
        display_max_mm=float(cfg.get("display_max_mm", 1000)),
        log=lambda m: print(m, flush=True),
    )
    worker.start()

    # ── 6. TCP stream server ──────────────────────────────────────────
    server = StreamServer(
        host=str(cfg.get("socket_host", "0.0.0.0")),
        port=int(cfg.get("socket_port", 9999)),
        video_fps=int(cfg.get("video_fps", 15)),
        h264_bitrate=int(cfg.get("h264_bitrate", 500000)),
        gop_size=int(cfg.get("gop_size", 10)),
        last_frame_holder=last_frame,
    )
    server.start()
    print(f"[main] streaming on {cfg['socket_host']}:{cfg['socket_port']} "
          f"mode={cfg.get('display_mode', 'fused')}", flush=True)

    # ── 7. Signal handlers ────────────────────────────────────────────
    stopped = threading.Event()

    def _on_signal(sig, _frame):
        print(f"[main] signal {sig}, shutting down...", flush=True)
        graceful_shutdown(server=server, worker=worker, device=device)
        stopped.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # ── 8. Main idle loop ─────────────────────────────────────────────
    while not stopped.is_set():
        time.sleep(0.5)

    worker.join(timeout=2.0)
    print("[main] exit 0", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
