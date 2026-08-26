"""Frame callbacks and color/depth pairing for com.aiot.EYS3DDepthFusion.

Unlike com.aiot.EYS3DCamera, which serves one stream at a time and lets the
unused callback return early, this component needs BOTH streams for the same
instant: MiDaS runs on the color image and its output is aligned against the
stereo depth of the same scene. A color frame paired with a stale depth frame
would be aligned against geometry that has already moved.

The SDK gives us the means to do this properly — `Device.open_device` passes
`CONTROL_MODE.IMAGE_SN_SYNC` and every frame carries `get_serial_number()` — so
pairing is by serial number, not by arrival time. Arrival order is not reliable:
the two callbacks run on separate C++ threads.

Throttling stays at the top of each callback so dropped frames never trigger
`get_rgb_data()` / `get_depth_ZD_value()`, both of which are C++ memcpys.
"""
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

    On a DEPTH frame the same method returns the SDK's colorized heatmap, not
    distances — `_to_zd` is what carries millimetres.
    """
    buf = frame.get_rgb_data()
    arr = np.asarray(buf, dtype=np.uint8)
    if height and width:
        arr = arr.reshape(height, width, 3)
    return arr


def _to_zd(frame, height, width):
    """Depth SDK Frame -> (H, W) uint16 metric millimetres, 0 where unmeasured.

    This is the whole reason fusion has to live in-process. The heatmap that
    com.aiot.EYS3DCamera streams is a colorization of these values through a
    lossy H264 encoder; millimetres cannot be recovered from it downstream, so
    they have to be read here, at the source.

    Returns None when the build does not expose ZD values, which makes the
    caller fall back to passthrough rather than fail.
    """
    try:
        buf = frame.get_depth_ZD_value()
    except Exception:
        return None
    arr = np.asarray(buf, dtype=np.uint16)
    if height and width:
        arr = arr.reshape(height, width)
    return arr


class FramePairer:
    """Match the latest color and depth frames by SDK serial number.

    Only the newest frame of each kind is kept. A pair that cannot be completed
    before the next frame of the same kind arrives is dropped rather than
    queued: for live video a late pair is worthless, and holding a backlog only
    adds latency.

    `max_sn_delta` exists because the two streams do not always carry identical
    serial numbers — the SDK's ILMFrameRouter separates interleaved color and
    depth by serial-number PARITY on G120 (eSP936) modules, so a matched pair
    can legitimately differ by one. The observed delta is logged on the first
    few pairs so the real relationship can be confirmed on hardware instead of
    assumed.
    """

    def __init__(self, emit, max_sn_delta=1, log=print, log_first=5):
        self._emit = emit
        self._max_delta = int(max_sn_delta)
        self._log = log
        self._log_remaining = int(log_first)
        self._lock = threading.Lock()
        self._color = None  # (sn, rgb)
        self._depth = None  # (sn, z_mm, heatmap)
        self.pairs = 0
        self.unmatched = 0

    def put_color(self, sn, rgb):
        with self._lock:
            if self._color is not None:
                self.unmatched += 1
            self._color = (sn, rgb)
            pair = self._take_locked()
        if pair is not None:
            self._emit(*pair)

    def put_depth(self, sn, z_mm, heatmap):
        with self._lock:
            if self._depth is not None:
                self.unmatched += 1
            self._depth = (sn, z_mm, heatmap)
            pair = self._take_locked()
        if pair is not None:
            self._emit(*pair)

    def _take_locked(self):
        if self._color is None or self._depth is None:
            return None
        c_sn, rgb = self._color
        d_sn, z_mm, heat = self._depth
        # Serial numbers wrap; compare as a signed short-ish delta so a wrap
        # does not look like an enormous mismatch and stall pairing forever.
        delta = int(c_sn) - int(d_sn)
        if abs(delta) > self._max_delta:
            # Drop whichever is older so the streams can re-converge instead of
            # deadlocking on a permanently mismatched pair.
            if delta > 0:
                self._depth = None
            else:
                self._color = None
            return None
        if self._log_remaining > 0:
            self._log(f"[pair] color_sn={c_sn} depth_sn={d_sn} delta={delta}")
            self._log_remaining -= 1
        self._color = None
        self._depth = None
        self.pairs += 1
        return rgb, z_mm, heat


def make_color_callback(state, pairer, max_capture_fps, height=None, width=None):
    """Build a callback for `device.open_device(colorFrameCallback=...)`."""
    min_interval = _interval(max_capture_fps)

    def _cb(frame):
        now = time.monotonic()
        if now - state.last_accept_ts < min_interval:
            return  # drop without copying
        state.last_accept_ts = now

        img = _to_bgr(frame, height, width)
        if not state.first_logged:
            print(f"[capture] first color frame shape={img.shape} dtype={img.dtype}", flush=True)
            state.first_logged = True
        pairer.put_color(frame.get_serial_number(), img)

    return _cb


def make_depth_callback(state, pairer, max_capture_fps, height=None, width=None):
    """Build a callback for `device.open_device(depthFrameCallback=...)`.

    Carries BOTH the metric millimetres (for fusion) and the SDK's own colorized
    heatmap (so raw-depth display mode needs no second colorization pass and
    stays a like-for-like baseline against com.aiot.EYS3DCamera).
    """
    min_interval = _interval(max_capture_fps)

    def _cb(frame):
        now = time.monotonic()
        if now - state.last_accept_ts < min_interval:
            return
        state.last_accept_ts = now

        heat = _to_bgr(frame, height, width)
        z_mm = _to_zd(frame, height, width)
        if not state.first_logged:
            zd = "unavailable" if z_mm is None else f"{z_mm.shape} {z_mm.dtype}"
            print(f"[capture] first depth frame heat={heat.shape} zd={zd}", flush=True)
            state.first_logged = True
        pairer.put_depth(frame.get_serial_number(), z_mm, heat)

    return _cb
