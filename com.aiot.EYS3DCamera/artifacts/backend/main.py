#!/usr/bin/env python3
"""
com.aiot.EYS3DCamera — eYs3D 3D stereo camera RGB/Depth H264 streaming component.

Pipeline:
  C++ SDK producer
    ├─ color_callback (throttled to max_capture_fps)
    │     └─ RGB mode: enqueue → InferenceThread → annotate → last_frame
    └─ depth_callback (throttled to max_capture_fps)
          └─ Depth mode: write heatmap directly to last_frame

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
from inference import InferenceThread
from stream_server import StreamServer
from shutdown import shutdown as graceful_shutdown


def _create_detector(cfg):
    """Best-available TFLite detector; None means passthrough."""
    try:
        import tflite_runtime.interpreter as _tflite_mod  # noqa: F401
    except ImportError:
        try:
            import tensorflow.lite as _tflite_mod  # type: ignore  # noqa: F401
        except ImportError:
            return None

    model = cfg.get("tflite_model") or {}
    path = model.get("path") if isinstance(model, dict) else model
    if not (path and os.path.isfile(str(path))):
        return None

    # Defer import; FaceDetectTflite has the SCRFD class — for V1 we only
    # passthrough unless an explicit detector is plugged in by the deployment.
    try:
        from detector_tflite import SCRFDDetectorTFLite  # type: ignore
    except ImportError:
        return None

    return SCRFDDetectorTFLite(
        path,
        input_size=cfg.get("det_input_size", 640),
        threshold=cfg.get("det_threshold", 0.5),
        nms_iou=cfg.get("det_nms_iou", 0.4),
        delegate_path=cfg.get("npu_delegate_path", "/usr/lib/libvx_delegate.so"),
    )


def _annotate_passthrough(frame, boxes, scores, kps):
    """No-op annotator (V1 ships without SCRFD wired in)."""
    return


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

    # ── 3. Shared state ───────────────────────────────────────────────
    last_frame = capture_mod.LastFrameHolder()
    inference_q = queue.Queue(maxsize=2)
    color_state = capture_mod.CallbackState()
    depth_state = capture_mod.CallbackState()

    stream_mode = {"v": str(cfg.get("stream_mode", "rgb"))}

    color_cb = capture_mod.make_color_callback(
        state=color_state,
        inference_queue=inference_q,
        last_frame_holder=last_frame,
        max_capture_fps=int(cfg.get("max_capture_fps", 15)),
        stream_mode_getter=lambda: stream_mode["v"],
        height=color_h,
        width=color_w,
    )
    depth_cb = capture_mod.make_depth_callback(
        state=depth_state,
        last_frame_holder=last_frame,
        max_capture_fps=int(cfg.get("max_capture_fps", 15)),
        stream_mode_getter=lambda: stream_mode["v"],
        height=depth_h,
        width=depth_w,
    )

    # ── 4. Open device with callbacks ─────────────────────────────────
    device.open_device(sdk_cfg, colorFrameCallback=color_cb, depthFrameCallback=depth_cb)
    device.enable_color_depth_stream()

    # ── 4b. Depth post-processing (no-op for RGB stream) ──────────────
    if stream_mode["v"] == "depth":
        depth_filter_mod.configure(device, cfg)

    # ── 5. Inference thread ───────────────────────────────────────────
    detector = _create_detector(cfg)
    inference = InferenceThread(
        in_queue=inference_q,
        last_frame_holder=last_frame,
        detector=detector,
        annotate_fn=_annotate_passthrough,
        det_every_n_frames=int(cfg.get("det_every_n_frames", 1)),
    )
    inference.start()

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
    print(f"[main] streaming on {cfg['socket_host']}:{cfg['socket_port']} mode={stream_mode['v']}", flush=True)

    # ── 7. Signal handlers ────────────────────────────────────────────
    stopped = threading.Event()

    def _on_signal(sig, _frame):
        print(f"[main] signal {sig}, shutting down...", flush=True)
        graceful_shutdown(server=server, inference=inference, device=device)
        stopped.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # ── 8. Main idle loop ─────────────────────────────────────────────
    while not stopped.is_set():
        time.sleep(0.5)

    inference.join(timeout=2.0)
    print("[main] exit 0", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
