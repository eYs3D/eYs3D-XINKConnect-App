#!/usr/bin/env python3
"""
Face Detection RTSP Stream Server (com.aiot.FaceDetect)

Reads video (RTSP, local files, or local camera), runs SCRFD face detection,
annotates frames with bounding boxes + 5 facial keypoints,
then streams H264-encoded output over TCP socket.

Inference backends (selected automatically in priority order):
  1. TFLite + VX delegate  → Vivante NPU hardware acceleration (fastest)
  2. TFLite + CPU          → TFLite without NPU delegate
  3. ONNX Runtime + CPU    → Legacy CPU fallback (requires onnxruntime installed)
  No detector              → Stream passthrough without annotation

APP_CONFIG_PATH keys:
    video_source      (string)  video input selector:
                        "rtsp"  → use rtsp_url (default)
                        "local" → loop video files in artifacts/video/
                        "0"     → built-in laptop camera
                        "1"     → USB/external camera
    rtsp_url          (string)  RTSP source URL (used when video_source is "rtsp")
    socket_host       (string)  TCP bind address              ["0.0.0.0"]
    socket_port       (number)  TCP port                      [9999]
    video_fps         (number)  output FPS cap                [15]
    h264_bitrate      (number)  bits/s                        [500000]
    gop_size          (number)  H264 GOP size                 [10]
    det_threshold     (number)  face score cutoff             [0.5]
    det_nms_iou       (number)  NMS IoU threshold             [0.4]
    det_input_size    (number)  SCRFD input resolution        [640]
    det_every_n_frames (number) run inference every N frames  [1]
                                (higher = less load; in-between frames reuse last result)
    npu_delegate_path (string)  path to VX delegate .so       ["/usr/lib/libvx_delegate.so"]

Model bindings (platform-injected):
    tflite_model    (object)  {"model_version_id": "...", "path": "/abs/path.tflite"}
                              Primary model — enables NPU/TFLite inference path
    detector_model  (object)  {"model_version_id": "...", "path": "/abs/path.onnx"}
                              Legacy ONNX model — used only if tflite_model absent/missing

Dependencies:
    pip install tflite-runtime opencv-python-headless av numpy
    # Optional: pip install onnxruntime  (for ONNX CPU fallback only)
"""

import av
import cv2
import json
import numpy as np
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
from fractions import Fraction

# ── TFLite runtime (primary inference backend) ────────────────────────────────
# Board (aarch64 NPU) ships tflite_runtime — first and only branch taken there.
# x86_64 desktops have no tflite-runtime wheel; they install ai-edge-litert (the
# official lightweight LiteRT runtime) which exposes the SAME interpreter.Interpreter
# / load_delegate API. tensorflow.lite is a last-resort fallback.
try:
    import tflite_runtime.interpreter as _tflite_mod
    _TFLITE_AVAILABLE = True
except ImportError:
    try:
        import ai_edge_litert.interpreter as _tflite_mod  # type: ignore
        _TFLITE_AVAILABLE = True
    except ImportError:
        try:
            import tensorflow.lite as _tflite_mod          # type: ignore
            _TFLITE_AVAILABLE = True
        except ImportError:
            _tflite_mod = None
            _TFLITE_AVAILABLE = False

# ── ONNX Runtime (optional legacy fallback) ───────────────────────────────────
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    ort = None                                          # type: ignore
    _ORT_AVAILABLE = False


DEFAULTS = {
    "video_source": "rtsp",                            # "rtsp" | "local" | "0"/"1" (local cam index)
    "rtsp_url": "rtsp://192.168.110.123/stream2.264",
    "socket_host": "0.0.0.0",
    "socket_port": 9999,
    "video_fps": 15,
    "h264_bitrate": 500000,
    "gop_size": 10,
    "det_threshold": 0.5,
    "det_nms_iou": 0.4,
    "det_input_size": 640,
    "det_every_n_frames": 1,                           # run inference every N frames (1 = every frame)
    "npu_delegate_path": "/usr/lib/libvx_delegate.so", # VeriSilicon VX delegate path
    "tflite_model": {},                                # platform-injected TFLite model
    "detector_model": {},                              # platform-injected ONNX model (legacy)
}

RTSP_OPTIONS = {
    'rtsp_transport': 'tcp',
    'stimeout': '5000000',
    'max_delay': '500000',
    'reorder_queue_size': '0',
}


def load_app_config():
    cfg = dict(DEFAULTS)
    path = os.environ.get("APP_CONFIG_PATH")
    if not path:
        print("[main] APP_CONFIG_PATH not set, using defaults", flush=True)
        return cfg, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f) or {}
    except (OSError, ValueError) as e:
        print(f"[main] cannot load config {path!r}: {e}, using defaults", flush=True)
        return cfg, path
    cfg.update(loaded)
    return cfg, path


# ── SCRFD decoder (shared postprocessing for both ONNX and TFLite backends) ───
# Decode math lives in postprocess.py (pure numpy, unit-tested).
from postprocess import (
    _classify_scrfd_outputs,
    _build_anchor_levels,
    _decode_scrfd_outputs,
)


# ── Backend 1: TFLite + VX delegate (NPU) ────────────────────────────────────

class SCRFDDetectorTFLite:
    """
    SCRFD face detector backed by TFLite runtime.

    On construction, attempts to load the VX delegate for Vivante NPU
    acceleration. If the delegate fails or is unavailable, falls back to
    TFLite CPU automatically.

    Input: any SCRFD .tflite converted from the insightface ONNX models
    (e.g. via onnx2tf). Supports both NHWC (onnx2tf default) and NCHW layouts.
    """

    def __init__(self, model_path, input_size=640, threshold=0.5, nms_iou=0.4,
                 delegate_path="/usr/lib/libvx_delegate.so"):
        assert _TFLITE_AVAILABLE, "tflite_runtime not installed"
        self.input_size = int(input_size)
        self.threshold  = float(threshold)
        self.nms_iou    = float(nms_iou)

        # ── Load VX delegate (NPU) ────────────────────────────────────────
        delegates = []
        self._npu_enabled = False
        if delegate_path and os.path.exists(delegate_path):
            try:
                delegate = _tflite_mod.load_delegate(delegate_path)
                delegates = [delegate]
                self._npu_enabled = True
                print(f"[SCRFD-TFLite] NPU delegate loaded: {delegate_path}", flush=True)
            except Exception as e:
                print(f"[SCRFD-TFLite] NPU delegate failed ({e}), falling back to CPU", flush=True)
        else:
            print(f"[SCRFD-TFLite] delegate not found at {delegate_path!r}, using CPU", flush=True)

        # ── Create interpreter ────────────────────────────────────────────
        self._interp = _tflite_mod.Interpreter(
            model_path=model_path,
            experimental_delegates=delegates,
        )
        self._interp.allocate_tensors()

        inp_details = self._interp.get_input_details()
        out_details = self._interp.get_output_details()
        self._inp_detail  = inp_details[0]
        self._out_details = out_details

        # Check and handle dynamic/placeholder shapes in TFLite
        inp_shape = list(self._inp_detail["shape"])
        self._input_nchw = (len(inp_shape) == 4 and inp_shape[1] == 3)
        expected_shape = [1, 3, self.input_size, self.input_size] if self._input_nchw else [1, self.input_size, self.input_size, 3]
        if inp_shape != expected_shape:
            print(f"[SCRFD-TFLite] Resizing model input tensor from {inp_shape} to {expected_shape}...", flush=True)
            try:
                self._interp.resize_tensor_input(self._inp_detail["index"], expected_shape)
                self._interp.allocate_tensors()
                inp_details = self._interp.get_input_details()
                out_details = self._interp.get_output_details()
                self._inp_detail  = inp_details[0]
                self._out_details = out_details
            except Exception as e:
                print(f"[SCRFD-TFLite] WARNING: Failed to resize input tensor: {e}", flush=True)

        backend = "NPU (VX delegate)" if self._npu_enabled else "CPU"
        print(
            f"[SCRFD-TFLite] model={os.path.basename(model_path)!r} "
            f"backend={backend} "
            f"inputs={len(inp_details)} outputs={len(out_details)} "
            f"input_details={self._inp_detail}",
            flush=True,
        )

        self._frame_count = 0
        self._total_preprocess = 0.0
        self._total_invoke = 0.0
        self._total_postprocess = 0.0

        self._build_anchors()

    def _build_anchors(self):
        """Probe with a dummy forward pass; build anchor grids and detect sigmoid need."""
        s   = self.input_size
        inp = self._inp_detail

        self._input_nchw = (len(inp["shape"]) == 4 and inp["shape"][1] == 3)

        if self._input_nchw:
            dummy = np.zeros((1, 3, s, s), dtype=inp["dtype"])
        else:
            dummy = np.zeros((1, s, s, 3), dtype=inp["dtype"])

        self._interp.set_tensor(inp["index"], dummy)
        self._interp.invoke()

        outs = [self._interp.get_tensor(d["index"]) for d in self._out_details]

        score_ii, bbox_ii, kps_ii = _classify_scrfd_outputs(outs)

        if not score_ii or not bbox_ii:
            raise RuntimeError(
                f"[SCRFD-TFLite] Cannot classify model outputs. "
                f"Shapes: {[o.shape for o in outs]}. "
                f"Expected 6 or 9 tensors with last-dim 1/4/10."
            )

        self._score_ii = score_ii
        self._bbox_ii  = bbox_ii
        self._kps_ii   = kps_ii if kps_ii else None
        self.has_kps   = bool(kps_ii)
        self._levels   = _build_anchor_levels(score_ii, outs, self.input_size)

        # Dequantize before checking sigmoid range
        raw = outs[score_ii[0]]
        sd = self._out_details[score_ii[0]]
        if sd["dtype"] != np.float32:
            sc_q, sc_zp = sd["quantization"]
            raw = (raw.astype(np.float32) - sc_zp) * sc_q
        raw = raw.reshape(-1, 1)[:, 0]
        self._apply_sigmoid = bool(raw.min() < -0.05 or raw.max() > 1.05)
        print(
            f"[SCRFD-TFLite] has_kps={self.has_kps}  "
            f"apply_sigmoid={self._apply_sigmoid}  "
            f"input_layout={'NCHW' if self._input_nchw else 'NHWC'}",
            flush=True,
        )

    def detect(self, img_bgr):
        """
        Detect faces in a BGR uint8 image.
        Returns: boxes (N,4), scores (N,), kps (N,5,2) or None
        """
        t0 = time.time()
        h0, w0 = img_bgr.shape[:2]
        s = self.input_size

        resized = cv2.resize(img_bgr, (s, s))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob    = (rgb.astype(np.float32) - 127.5) * (1.0 / 128.0)

        inp = self._inp_detail
        if self._input_nchw:
            blob_in = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis])  # (1,3,s,s)
        else:
            blob_in = blob[np.newaxis]  # (1,s,s,3)

        if inp["dtype"] != np.float32:
            scale, zp = inp["quantization"]
            blob_in = (blob_in / scale + zp).astype(inp["dtype"])

        self._interp.set_tensor(inp["index"], blob_in)

        t1 = time.time()
        self._interp.invoke()
        t2 = time.time()

        outs = [self._interp.get_tensor(d["index"]) for d in self._out_details]

        # Dequantize if INT8 model
        dequant_outs = []
        for i, o in enumerate(outs):
            d = self._out_details[i]
            if d["dtype"] != np.float32:
                scale, zp = d["quantization"]
                o = (o.astype(np.float32) - zp) * scale
            dequant_outs.append(o)

        res = _decode_scrfd_outputs(
            dequant_outs,
            self._score_ii, self._bbox_ii, self._kps_ii,
            self._levels, self._apply_sigmoid,
            self.threshold, self.nms_iou, w0, h0, s,
        )
        t3 = time.time()

        self._frame_count += 1
        self._total_preprocess += (t1 - t0)
        self._total_invoke += (t2 - t1)
        self._total_postprocess += (t3 - t2)

        if self._frame_count % 1 == 0:
            avg_pre  = (self._total_preprocess  / 1) * 1000
            avg_inv  = (self._total_invoke       / 1) * 1000
            avg_post = (self._total_postprocess  / 1) * 1000
            avg_tot  = avg_pre + avg_inv + avg_post
            print(
                f"[SCRFD-TFLite Profile] frame={self._frame_count} "
                f"Preprocess={avg_pre:.1f}ms  "
                f"Invoke={avg_inv:.1f}ms  "
                f"Postprocess={avg_post:.1f}ms  "
                f"Total={avg_tot:.1f}ms ({1000/avg_tot:.2f} FPS)",
                flush=True,
            )
            self._total_preprocess = 0.0
            self._total_invoke = 0.0
            self._total_postprocess = 0.0

        return res


# ── Backend 2: ONNX Runtime CPU (legacy fallback) ────────────────────────────

class SCRFDDetector:
    """
    Lightweight SCRFD face detector backed by ONNX Runtime (CPU fallback).

    Used only when tflite_model is absent and onnxruntime is installed.
    Auto-detects CUDA if available; otherwise falls back to CPU.
    """

    def __init__(self, model_path, input_size=640, threshold=0.5, nms_iou=0.4):
        assert _ORT_AVAILABLE, "onnxruntime not installed"
        self.input_size = int(input_size)
        self.threshold  = float(threshold)
        self.nms_iou    = float(nms_iou)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        avail = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail else []
        ) + ["CPUExecutionProvider"]

        self.session    = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        print(
            f"[SCRFD] model={os.path.basename(model_path)!r} "
            f"providers={self.session.get_providers()}",
            flush=True,
        )
        self._build_anchors()

    def _build_anchors(self):
        s     = self.input_size
        dummy = np.zeros((1, 3, s, s), dtype=np.float32)
        outs  = self.session.run(None, {self.input_name: dummy})

        score_ii, bbox_ii, kps_ii = _classify_scrfd_outputs(outs)

        self.score_ii = score_ii
        self.bbox_ii  = bbox_ii
        self.kps_ii   = kps_ii if kps_ii else None
        self.has_kps  = bool(kps_ii)
        self._levels  = _build_anchor_levels(score_ii, outs, self.input_size)

        raw = outs[score_ii[0]].reshape(-1, 1)[:, 0]
        self._apply_sigmoid = bool(raw.min() < -0.05 or raw.max() > 1.05)
        print(
            f"[SCRFD] has_kps={self.has_kps}  apply_sigmoid={self._apply_sigmoid}",
            flush=True,
        )

    def detect(self, img_bgr):
        h0, w0 = img_bgr.shape[:2]
        s = self.input_size

        resized = cv2.resize(img_bgr, (s, s))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob    = (rgb.astype(np.float32) - 127.5) * (1.0 / 128.0)
        blob    = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis])

        outs = self.session.run(None, {self.input_name: blob})

        return _decode_scrfd_outputs(
            outs,
            self.score_ii, self.bbox_ii, self.kps_ii,
            self._levels, self._apply_sigmoid,
            self.threshold, self.nms_iou, w0, h0, s,
        )


# ── Detector factory ──────────────────────────────────────────────────────────

def _resolve_model_path(cfg_value):
    """Extract file path from a model config value (dict or string)."""
    if isinstance(cfg_value, dict):
        return cfg_value.get("path")
    if isinstance(cfg_value, str) and cfg_value:
        return cfg_value
    return None


def create_detector(cfg):
    """
    Create the best available detector for the current environment.

    Priority:
      1. TFLite + VX delegate  (NPU, fastest)
      2. TFLite + CPU          (TFLite without NPU)
      3. ONNX Runtime CPU      (legacy fallback)
      4. None                  (passthrough, no detection)
    """
    threshold  = cfg["det_threshold"]
    nms_iou    = cfg["det_nms_iou"]
    input_size = cfg["det_input_size"]
    npu_path   = str(cfg.get("npu_delegate_path", "/usr/lib/libvx_delegate.so"))

    # ── 1st/2nd choice: TFLite (NPU or CPU) ──────────────────────────────
    if _TFLITE_AVAILABLE:
        tflite_path = _resolve_model_path(cfg.get("tflite_model"))
        if tflite_path and os.path.isfile(tflite_path):
            try:
                det = SCRFDDetectorTFLite(
                    tflite_path,
                    input_size=input_size,
                    threshold=threshold,
                    nms_iou=nms_iou,
                    delegate_path=npu_path,
                )
                mode = "NPU (VX delegate)" if det._npu_enabled else "TFLite CPU"
                print(f"[main] Detector ready: TFLite backend — {mode}", flush=True)
                return det
            except Exception as exc:
                print(f"[main] TFLite detector init failed: {exc}", flush=True)
        else:
            print(
                f"[main] tflite_model path={tflite_path!r} not found "
                f"(convert det_500m.onnx first — see tools/convert_to_tflite.py)",
                flush=True,
            )
    else:
        print("[main] tflite_runtime not available", flush=True)

    # ── 3rd choice: ONNX Runtime CPU ─────────────────────────────────────
    if _ORT_AVAILABLE:
        onnx_path = _resolve_model_path(cfg.get("detector_model"))
        if onnx_path and os.path.isfile(str(onnx_path)):
            try:
                det = SCRFDDetector(
                    onnx_path,
                    input_size=input_size,
                    threshold=threshold,
                    nms_iou=nms_iou,
                )
                print("[main] Detector ready: ONNX Runtime CPU (fallback)", flush=True)
                return det
            except Exception as exc:
                print(f"[main] ONNX detector init failed: {exc}", flush=True)
        else:
            print(f"[main] detector_model path={onnx_path!r} not found", flush=True)
    else:
        print("[main] onnxruntime not installed (expected on this device)", flush=True)

    # ── Passthrough ───────────────────────────────────────────────────────
    print("[main] No usable detector — streaming without face detection", flush=True)
    return None


# ── Drawing ───────────────────────────────────────────────────────────────────

_ACCENT     = (250, 200,  50)   # BGR
_LABEL_FG   = ( 20,  20,  20)
_KPT_COLOR  = (255, 255, 255)


def annotate(img, boxes, scores, kps):
    """Draw face detections in corner-bracket style."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(np.int32)
        w, h = x2 - x1, y2 - y1
        L = max(6, min(20, min(w, h) // 5))

        for (cx, cy, dx, dy) in (
            (x1, y1,  L,  L),
            (x2, y1, -L,  L),
            (x1, y2,  L, -L),
            (x2, y2, -L, -L),
        ):
            cv2.line(img, (cx, cy), (cx + dx, cy), _ACCENT, 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy), _ACCENT, 2, cv2.LINE_AA)

        label = f"{scores[i] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        pad_x, pad_y = 6, 4
        lx2 = x1 + tw + pad_x * 2
        ly1 = max(0, y1 - (th + pad_y * 2))
        ly2 = ly1 + th + pad_y * 2
        cv2.rectangle(img, (x1, ly1), (lx2, ly2), _ACCENT, -1, cv2.LINE_AA)
        cv2.putText(
            img, label, (x1 + pad_x, ly2 - pad_y),
            font, 0.45, _LABEL_FG, 1, cv2.LINE_AA,
        )

        if kps is not None:
            for j in range(kps.shape[1]):
                kx, ky = int(kps[i, j, 0]), int(kps[i, j, 1])
                cv2.circle(img, (kx, ky), 2, _KPT_COLOR, -1, cv2.LINE_AA)


# ── Stream server ─────────────────────────────────────────────────────────────

_LOCAL_VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts")


def _parse_video_source(cfg):
    """Return (cap_arg, is_local_cam, is_local_files).
    video_source values:
      "rtsp"  → use rtsp_url  → (url,  False, False)
      "local" → scan video/   → (None, False, True)
      "0"/"1" → camera index  → (int,  True,  False)
    """
    src_raw = str(cfg.get("video_source", "rtsp")).strip()
    src = src_raw.lower()
    if src == "local":
        return None, False, True
    if src == "rtsp":
        return str(cfg["rtsp_url"]), False, False
    try:
        return int(src), True, False
    except ValueError:
        return src_raw, False, False


class FaceDetectServer:
    def __init__(self, cfg, detector):
        self._cap_arg, self._is_local_cam, self._is_local_files = _parse_video_source(cfg)
        # artifacts/video/ is one level up from backend/
        self._video_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video")
        self.socket_host  = str(cfg["socket_host"])
        self.socket_port  = int(cfg["socket_port"])
        self.video_fps    = int(cfg["video_fps"])
        self.h264_bitrate = int(cfg["h264_bitrate"])
        self.gop_size     = int(cfg["gop_size"])
        self.detector     = detector
        self.det_every_n  = max(1, int(cfg.get("det_every_n_frames", 1)))

        self.frame_time = 1.0 / self.video_fps
        self.running    = True
        self._lock      = threading.Lock()
        self.last_frame = None
        self._srv_sock  = None
        self._infer_q   = queue.Queue(maxsize=2)  # bounded: capture never blocks on slow NPU

        self._fps_count = 0
        self._fps_t0    = time.time()
        self._last_fps  = 0.0

    def _interruptible_sleep(self, secs):
        end = time.time() + secs
        while self.running and time.time() < end:
            time.sleep(0.1)

    # ── inference thread ──────────────────────────────────────────────────────
    # Runs detector.detect() in its own thread so NPU invoke() never blocks
    # the capture loop. Annotates the frame and writes to last_frame.

    def _inference_loop(self):
        last_boxes  = np.empty((0, 4), np.float32)
        last_scores = np.empty(0, np.float32)
        last_kps    = None
        while self.running:
            try:
                frame = self._infer_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.detector is not None:
                try:
                    last_boxes, last_scores, last_kps = self.detector.detect(frame)
                except Exception as exc:
                    print(f"[det] inference error: {exc}", flush=True)
            if len(last_boxes):
                annotate(frame, last_boxes, last_scores, last_kps)
            with self._lock:
                self.last_frame = frame
        print("[inf] thread exited", flush=True)

    # ── capture thread ────────────────────────────────────────────────────────

    def _capture_loop(self):
        if self._is_local_files:
            self._local_capture_loop()
            return
        if self._is_local_cam:
            self._cam_capture_loop()
            return
        self._rtsp_capture_loop()

    def _rtsp_capture_loop(self):
        """RTSP capture via PyAV — robust tcp transport with configurable timeouts."""
        cap_frame_idx = 0
        container = None
        stream = None

        while self.running:
            if container is None:
                try:
                    container = av.open(self._cap_arg, options=RTSP_OPTIONS, timeout=10)
                    stream = container.streams.video[0]
                    stream.thread_type = 'AUTO'
                    print(
                        f"[cap] RTSP opened: {stream.width}x{stream.height} "
                        f"codec={stream.codec_context.name}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[cap] RTSP open failed: {e}, retry in 5s", flush=True)
                    container = None
                    self._interruptible_sleep(5)
                    continue

            try:
                for packet in container.demux(stream):
                    if not self.running:
                        break
                    for frame_av in packet.decode():
                        if not self.running:
                            break
                        frame = frame_av.to_ndarray(format='bgr24')
                        cap_frame_idx += 1

                        self._fps_count += 1
                        now = time.time()
                        dt  = now - self._fps_t0
                        if dt >= 1.0:
                            self._last_fps  = self._fps_count / dt
                            self._fps_count = 0
                            self._fps_t0    = now
                            print(f"[cap] FPS: {self._last_fps:.1f}  (det every {self.det_every_n})", flush=True)

                        if cap_frame_idx % self.det_every_n == 0:
                            try:
                                self._infer_q.put_nowait(frame)
                            except queue.Full:
                                pass

            except av.EOFError:
                print("[cap] RTSP stream ended, reconnecting...", flush=True)
            except Exception as exc:
                print(f"[cap] RTSP error: {exc}, reconnecting...", flush=True)

            try:
                container.close()
            except Exception:
                pass
            container = None
            self._interruptible_sleep(1)

        if container:
            try:
                container.close()
            except Exception:
                pass
        print("[cap] thread exited", flush=True)

    def _cam_capture_loop(self):
        """Local camera capture via OpenCV."""
        cap_frame_idx = 0
        cap = None
        while self.running:
            if cap is None:
                cap = cv2.VideoCapture(self._cap_arg)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    print(f"[cap] cannot open camera {self._cap_arg}, retry in 5s", flush=True)
                    cap.release()
                    cap = None
                    self._interruptible_sleep(5)
                    continue
                print(f"[cap] opened camera {self._cap_arg}", flush=True)

            ret, frame = cap.read()
            if not ret:
                print("[cap] camera read failed, reconnecting...", flush=True)
                cap.release()
                cap = None
                self._interruptible_sleep(1)
                continue

            cap_frame_idx += 1
            self._fps_count += 1
            now = time.time()
            dt  = now - self._fps_t0
            if dt >= 1.0:
                self._last_fps  = self._fps_count / dt
                self._fps_count = 0
                self._fps_t0    = now
                print(f"[cap] FPS: {self._last_fps:.1f}  (det every {self.det_every_n})", flush=True)

            if cap_frame_idx % self.det_every_n == 0:
                try:
                    self._infer_q.put_nowait(frame)
                except queue.Full:
                    pass

        if cap:
            cap.release()
        print("[cap] thread exited", flush=True)

    # ── local-file capture ────────────────────────────────────────────────────

    def _list_local_videos(self):
        try:
            names = sorted(
                f for f in os.listdir(self._video_dir)
                if not f.startswith(".") and f.lower().endswith(_LOCAL_VIDEO_EXTS)
            )
        except FileNotFoundError:
            return []
        return [os.path.join(self._video_dir, n) for n in names]

    def _local_capture_loop(self):
        """Loop video files from artifacts/video/ via PyAV.
        Rescans the folder between files so newly-dropped videos play next."""
        cap_frame_idx = 0
        played = set()
        print(f"[cap] local mode: watching {self._video_dir}", flush=True)

        while self.running:
            files = self._list_local_videos()
            if not files:
                print(f"[cap] no local videos in {self._video_dir}, retry in 5s", flush=True)
                self._interruptible_sleep(5)
                continue

            unplayed = [p for p in files if p not in played]
            if not unplayed:
                played.clear()
                unplayed = files

            target = unplayed[0]
            played.add(target)
            cap_frame_idx = self._play_local_file(target, cap_frame_idx)

        print("[cap] thread exited", flush=True)

    def _play_local_file(self, path, cap_frame_idx):
        """Decode one video file at native FPS via PyAV, push frames to inference queue."""
        try:
            container = av.open(path)
        except Exception as e:
            print(f"[cap] local open failed {os.path.basename(path)}: {e}", flush=True)
            self._interruptible_sleep(1)
            return cap_frame_idx

        try:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            rate = stream.average_rate or stream.base_rate
            fps = float(rate) if rate else 30.0
            if fps <= 0:
                fps = 30.0
            frame_interval = 1.0 / fps
            print(
                f"[cap] local file: {os.path.basename(path)} "
                f"{stream.width}x{stream.height} @ {fps:.1f}fps",
                flush=True,
            )

            next_t = time.time()
            for packet in container.demux(stream):
                if not self.running:
                    break
                for frame_av in packet.decode():
                    if not self.running:
                        break
                    frame = frame_av.to_ndarray(format='bgr24')
                    cap_frame_idx += 1

                    self._fps_count += 1
                    now = time.time()
                    dt = now - self._fps_t0
                    if dt >= 1.0:
                        self._last_fps = self._fps_count / dt
                        self._fps_count = 0
                        self._fps_t0 = now
                        print(
                            f"[cap] FPS: {self._last_fps:.1f}  (det every {self.det_every_n})",
                            flush=True,
                        )

                    if cap_frame_idx % self.det_every_n == 0:
                        try:
                            self._infer_q.put_nowait(frame)
                        except queue.Full:
                            pass

                    # Pace playback at native FPS
                    next_t += frame_interval
                    sleep = next_t - time.time()
                    if sleep > 0:
                        time.sleep(sleep)
                    elif sleep < -1.0:
                        next_t = time.time()  # fallen behind, reset

        except av.EOFError:
            pass
        except Exception as exc:
            print(f"[cap] local error {os.path.basename(path)}: {exc}", flush=True)
        finally:
            try:
                container.close()
            except Exception:
                pass
        return cap_frame_idx

    # ── TCP client handler ────────────────────────────────────────────────────

    def _handle_client(self, cs, addr):
        print(f"[srv] client {addr} connected", flush=True)
        encoder  = None
        enc_w    = enc_h = -1
        frame_no = 0
        no_frame_count = 0

        try:
            while self.running:
                t0 = time.time()

                with self._lock:
                    frame = self.last_frame

                if frame is None:
                    no_frame_count += 1
                    if no_frame_count > 50:
                        print(f"[srv] no frame for {addr}, closing", flush=True)
                        break
                    time.sleep(0.1)
                    continue
                no_frame_count = 0

                h, w = frame.shape[:2]
                if encoder is None or w != enc_w or h != enc_h:
                    if encoder:
                        try:
                            encoder.close()
                        except Exception:
                            pass
                    enc_w, enc_h = w, h
                    encoder = av.CodecContext.create("libx264", "w")
                    encoder.width     = enc_w
                    encoder.height    = enc_h
                    encoder.pix_fmt   = "yuv420p"
                    encoder.time_base = Fraction(1, self.video_fps)
                    encoder.framerate = Fraction(self.video_fps, 1)
                    encoder.bit_rate  = self.h264_bitrate
                    encoder.gop_size  = self.gop_size
                    encoder.max_b_frames = 0
                    encoder.options = {
                        "preset": "ultrafast",
                        "tune": "zerolatency",
                        "profile": "baseline",
                        "level": "3.1",
                        "forced-idr": "1",
                        "repeat-headers": "1",
                    }
                    encoder.open()
                    print(f"[srv] encoder {enc_w}x{enc_h} ready for {addr}", flush=True)

                av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                av_frame.pts       = frame_no
                av_frame.time_base = Fraction(1, self.video_fps)

                try:
                    for pkt in encoder.encode(av_frame):
                        data = bytes(pkt)
                        cs.sendall(struct.pack(">I", len(data)) + data)
                except (BrokenPipeError, ConnectionResetError):
                    print(f"[srv] client {addr} disconnected", flush=True)
                    break
                except Exception as exc:
                    print(f"[srv] send error {addr}: {exc}", flush=True)
                    break

                frame_no += 1
                sleep = self.frame_time - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        finally:
            if encoder:
                try:
                    for pkt in encoder.encode(None):
                        try:
                            cs.sendall(bytes(pkt))
                        except Exception:
                            pass
                    encoder.close()
                except Exception:
                    pass
            try:
                cs.close()
            except Exception:
                pass
            print(f"[srv] client {addr} handler done", flush=True)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        threading.Thread(target=self._inference_loop, daemon=True).start()
        threading.Thread(target=self._capture_loop,   daemon=True).start()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)
        srv.bind((self.socket_host, self.socket_port))
        srv.listen(5)
        self._srv_sock = srv
        print(f"[srv] TCP listening on {self.socket_host}:{self.socket_port}", flush=True)

        while self.running:
            try:
                cs, addr = srv.accept()
                threading.Thread(
                    target=self._handle_client, args=(cs, addr), daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception as exc:
                if self.running:
                    print(f"[srv] accept error: {exc}", flush=True)

        try:
            srv.close()
        except Exception:
            pass
        print("[srv] stopped", flush=True)

    def stop(self):
        self.running = False
        try:
            if self._srv_sock:
                self._srv_sock.close()
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg, cfg_path = load_app_config()
    print(
        f"[main] pid={os.getpid()} config={cfg_path!r} "
        f"params={json.dumps({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}, sort_keys=True)}",
        flush=True,
    )
    print(
        f"[main] tflite_runtime={'available' if _TFLITE_AVAILABLE else 'NOT available'}  "
        f"onnxruntime={'available' if _ORT_AVAILABLE else 'NOT available'}",
        flush=True,
    )

    detector = create_detector(cfg)

    server = FaceDetectServer(cfg, detector)

    def _on_signal(sig, _frame):
        print(f"[main] signal {sig}, shutting down...", flush=True)
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    server.run()
    sys.exit(0)
