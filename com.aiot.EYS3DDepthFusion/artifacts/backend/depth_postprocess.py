"""Depth postprocess for com.aiot.DepthTflite: MiDaS output -> colorized frame.

Pure numpy + cv2, no TFLite import, so this is unit-testable off-device.

Two things here are specific to MiDaS and easy to get wrong:

1. The output is INVERSE relative depth — LARGER value means CLOSER. So feeding
   it straight to a colormap already gives near=bright; it is the far=bright
   convention that requires flipping (see `colorize`). It is also
   scale-and-shift invariant, so the absolute numbers carry no metric meaning
   and only the per-frame ordering does.

2. Because absolute values are meaningless and drift frame to frame,
   normalization MUST be per-frame min/max. A fixed range would make the display
   flicker or wash out as the scene changes. `_ema` then smooths the min/max
   across frames so the colormap does not pulse on every small change.
"""

import cv2
import numpy as np

# MiDaS-small was trained on 256x256 inputs; anything else changes the effective
# receptive field and degrades output.
DEFAULT_INPUT_SIZE = 256

# ImageNet normalization is BAKED INTO the ONNX graph (Sub 0.485/0.456/0.406 then
# Div 0.229/0.224/0.225, in RGB order) and survives into the TFLite conversion.
# So preprocessing here must ONLY scale to [0,1] — applying ImageNet norm a
# second time here would double-normalize and quietly wreck the output.
COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "gray": None,
}


def resolve_colormap(name):
    """Map a config string to a cv2 colormap constant. None means grayscale."""
    return COLORMAPS.get(str(name).lower(), cv2.COLORMAP_INFERNO)


def preprocess(frame_bgr, input_size=DEFAULT_INPUT_SIZE):
    """BGR uint8 HxWx3 -> (1, size, size, 3) float32 RGB in [0,1], NHWC.

    NHWC (not NCHW) because onnx2tf transposes the graph to TFLite's native
    layout during conversion — the .tflite input is [1,H,W,3] even though the
    source ONNX was [1,3,H,W].

    BGR->RGB matters: the baked-in ImageNet Sub/Div is per-channel in RGB order,
    so feeding BGR applies the wrong constant to each channel.

    Getting this wrong is NOT detectable after the fact. Measured against FP32
    ONNX for the shipped INT8 model, rank error (1 - spearman) from a swapped
    channel order (0.0101 aloe / 0.0401 lena) is the same order of magnitude as
    from INT8 quantization itself (0.0201 / 0.0355) — so it hides inside the
    quantization noise while still producing a plausible-looking depth map. The
    correctness has to come from this function, not from a downstream check.
    """
    resized = cv2.resize(frame_bgr, (input_size, input_size),
                         interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb[np.newaxis, ...]


def _ema(prev, value, alpha):
    """Exponential moving average; `prev` None means first sample."""
    if prev is None:
        return float(value)
    return float(alpha * value + (1.0 - alpha) * prev)


class DepthNormalizer:
    """Per-frame min/max normalizer with EMA-smoothed bounds.

    Absolute MiDaS values are meaningless (scale-and-shift invariant), so the
    display range has to be derived per frame. Raw per-frame min/max alone makes
    the colormap pulse, so the bounds are smoothed. Percentile clipping drops
    outlier pixels that would otherwise compress the whole useful range.
    """

    def __init__(self, alpha=0.2, clip_percentile=1.0):
        self._lo = None
        self._hi = None
        self.alpha = float(alpha)
        self.clip_percentile = float(clip_percentile)

    def __call__(self, depth):
        p = self.clip_percentile
        if p > 0:
            lo = float(np.percentile(depth, p))
            hi = float(np.percentile(depth, 100.0 - p))
        else:
            lo = float(depth.min())
            hi = float(depth.max())
        if hi <= lo:
            hi = lo + 1e-6
        self._lo = _ema(self._lo, lo, self.alpha)
        self._hi = _ema(self._hi, hi, self.alpha)
        span = self._hi - self._lo
        if span < 1e-6:
            span = 1e-6
        return np.clip((depth - self._lo) / span, 0.0, 1.0)


def colorize(depth_norm, colormap=cv2.COLORMAP_INFERNO, invert=True):
    """Normalized [0,1] depth -> BGR uint8 heatmap.

    `invert` selects the DISPLAY convention, not an operation on the array:
        invert=True  -> near is BRIGHT (matches the hardware depth stream in
                        com.aiot.EYS3DCamera)
        invert=False -> near is DARK
    Because MiDaS already emits INVERSE depth (large value = close), near-bright
    is the pass-through case and it is far-bright that needs the 1-d flip. Doing
    the arithmetic the other way round yields a map that still looks like
    plausible depth but reads inside-out.
    """
    d = depth_norm
    if not invert:
        d = 1.0 - d
    u8 = np.clip(d * 255.0, 0, 255).astype(np.uint8)
    if colormap is None:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(u8, colormap)


def compose(frame_bgr, depth_color, mode="side_by_side", alpha=0.5):
    """Build the displayed frame.

    mode: "depth" (depth only) | "side_by_side" (RGB | depth) |
          "overlay" (depth blended onto RGB)
    Depth is resized to the source frame size first, so output dimensions always
    follow the camera, not the model's 256x256.
    """
    h, w = frame_bgr.shape[:2]
    if depth_color.shape[:2] != (h, w):
        depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_LINEAR)

    if mode == "depth":
        return depth_color
    if mode == "overlay":
        return cv2.addWeighted(frame_bgr, 1.0 - alpha, depth_color, alpha, 0.0)
    return np.hstack([frame_bgr, depth_color])


def draw_hud(frame, lines, origin=(8, 20), line_h=18):
    """Draw diagnostic text (backend, FPS, latency) with a dark outline so it
    stays readable over both bright and dark colormap regions."""
    x, y = origin
    for i, text in enumerate(lines):
        pos = (x, y + i * line_h)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame
