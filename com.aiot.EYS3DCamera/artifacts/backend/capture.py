"""Frame callbacks with FPS throttling for com.aiot.EYS3DCamera.

Throttle is applied at the earliest entry point inside the callback so that
dropped frames never trigger `get_rgb_data()` (which is a C++ memcpy).

The SDK returns RGB as a 1D byte buffer; we reshape to (H, W, 3) and convert
to BGR so downstream PyAV `from_ndarray(..., format="bgr24")` consumes the
correct layout. Without this, the encoder emits garbage H264 and the WebRTC
client disconnects with EPIPE.
"""
import queue
import threading
import time

import numpy as np


class CallbackState:
    """Per-callback last-accepted timestamp (monotonic seconds).

    Initialized to -inf so the first frame always passes the throttle check.
    """
    def __init__(self):
        self.last_accept_ts = float("-inf")
        self.first_logged = False


class LastFrameHolder:
    """Thread-safe holder for the most recent frame (consumed by client handlers)."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def set(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


def _interval(max_fps):
    return 1.0 / float(max_fps) if max_fps and max_fps > 0 else 0.0


def _to_bgr(frame, height, width):
    """SDK Frame -> contiguous (H, W, 3) BGR uint8 ndarray.

    Despite the SDK method being named `get_rgb_data()`, the underlying
    eys3dPy.System is initialized with COLOR_RGB24 byte ordering which on this
    build returns BGR-ordered bytes. Empirically verified: applying RGB→BGR
    here turns skin tones blue, so we pass the buffer through unchanged.
    """
    buf = frame.get_rgb_data()
    arr = np.asarray(buf, dtype=np.uint8)
    if height and width:
        arr = arr.reshape(height, width, 3)
    return arr


def make_color_callback(state, inference_queue, last_frame_holder,
                        max_capture_fps, stream_mode_getter, height=None, width=None):
    """Build a callback for `device.open_device(colorFrameCallback=...)`.

    RGB mode: enqueue a safe-copy into the inference queue.
    Depth mode: skip — depth callback handles its own frames.
    """
    min_interval = _interval(max_capture_fps)

    def _cb(frame):
        if stream_mode_getter() != "rgb":
            return
        now = time.monotonic()
        if now - state.last_accept_ts < min_interval:
            return  # drop without copying
        state.last_accept_ts = now

        img = _to_bgr(frame, height, width)
        if not state.first_logged:
            print(f"[capture] first color frame shape={img.shape} dtype={img.dtype}", flush=True)
            state.first_logged = True
        try:
            inference_queue.put_nowait(img)
        except queue.Full:
            pass  # silently drop when inference can't keep up

    return _cb


def make_depth_callback(state, last_frame_holder,
                        max_capture_fps, stream_mode_getter, height=None, width=None):
    """Build a callback for `device.open_device(depthFrameCallback=...)`.

    Depth mode: write the safe-copy heatmap directly to last_frame (no inference).
    """
    min_interval = _interval(max_capture_fps)

    def _cb(frame):
        if stream_mode_getter() != "depth":
            return
        now = time.monotonic()
        if now - state.last_accept_ts < min_interval:
            return
        state.last_accept_ts = now

        img = _to_bgr(frame, height, width)
        if not state.first_logged:
            print(f"[capture] first depth frame shape={img.shape} dtype={img.dtype}", flush=True)
            state.first_logged = True
        last_frame_holder.set(img)

    return _cb
