"""
Shared NPU/CPU TFLite backend helper for com.beta.release apps.

NPU-first, CPU-automatic-fallback inference on the Verisilicon VX delegate
(ecv5546 / libvx_delegate.so). Mirrors the inference-loading behavior proven in
com.beta.prepare/com.aiot.FaceDetectTflite, factored out so every *Tflite app
shares an identical implementation.

This file is COPIED verbatim into each app's backend/ directory — Greengrass
components are deployed independently and cannot import across components.

Only numpy is a hard import. tflite_runtime and cv2 are imported lazily so the
pure helpers (dequantize / letterbox math / path resolution) remain importable
and unit-testable on machines without those native deps.
"""

from pathlib import Path

import numpy as np

DEFAULT_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

# ── tflite_runtime (lazy / swappable for tests) ───────────────────────────────
# Board (aarch64 NPU) ships tflite_runtime — first and only branch taken there.
# x86_64 desktops have no tflite-runtime wheel; they install ai-edge-litert (the
# official lightweight LiteRT runtime, ~17 MB, inference-only) which exposes the
# SAME interpreter.Interpreter / load_delegate API, so make_interpreter below is
# unchanged. tensorflow.lite is a last-resort fallback for environments that only
# have full TensorFlow.
try:
    import tflite_runtime.interpreter as _tflite  # type: ignore
except ImportError:  # pragma: no cover - device has tflite_runtime; dev may not
    try:
        import ai_edge_litert.interpreter as _tflite  # type: ignore
    except ImportError:
        try:
            import tensorflow.lite as _tflite  # type: ignore
        except ImportError:
            _tflite = None


# ── NBG cache sentinel ────────────────────────────────────────────────────────
def _prepare_nbg_cache(model_path):
    """Cache hygiene for the VX delegate's NBG cache (sentinel protocol).

    A run killed mid-compile (Restart Service SIGTERM, power loss) leaves a
    truncated <model>.nb; the delegate loads it on the next start and
    segfaults at invoke ("Create node[0] NBG fail") — unrecoverable from
    Python. `<model>.nb.ok` is written only after a warm-up inference
    succeeds, so a .nb without its marker is unverified: delete it and
    recompile cold. A stale marker without a .nb is cleared too, so a kill
    during the upcoming compile is still detected next start.
    """
    cache_file = Path(model_path).with_suffix(".nb")
    ok_marker = Path(str(cache_file) + ".ok")
    if cache_file.exists() and not ok_marker.exists():
        print(f"[tflite] unverified NBG cache (no .ok marker) — removing: {cache_file}",
              flush=True)
        try:
            cache_file.unlink()
        except OSError:
            pass
    if not cache_file.exists():
        try:
            ok_marker.unlink()
        except OSError:
            pass
    return cache_file, ok_marker


def _mark_nbg_cache_verified(cache_file, ok_marker):
    """Marker write is best-effort — a read-only dir just means every start
    recompiles cold, which is slow but correct."""
    if cache_file.exists() and not ok_marker.exists():
        try:
            ok_marker.touch()
        except OSError:
            pass


# ── Interpreter creation: NPU first, CPU fallback ─────────────────────────────
def make_interpreter(model_path, delegate_path=DEFAULT_DELEGATE_PATH,
                     num_threads=None):
    """Build a TFLite Interpreter, preferring the VX delegate (NPU).

    Returns (interpreter, backend_label) where backend_label is
    "NPU (VX delegate)" or "CPU". Falls back to CPU automatically when the
    delegate path is missing or load_delegate raises.
    """
    if _tflite is None:
        raise RuntimeError(
            "no TFLite runtime — install tflite-runtime (aarch64 board) or "
            "ai-edge-litert (x86_64 / macOS)"
        )

    delegates = []
    backend = "CPU"
    cache_file = ok_marker = None
    dpath = Path(delegate_path) if delegate_path else None
    if dpath and dpath.exists():
        try:
            cache_file, ok_marker = _prepare_nbg_cache(model_path)
            cached = cache_file.exists()
            delegates = [_tflite.load_delegate(str(dpath), options={
                "allowed_cache_mode": "true",
                "cache_file_path": str(cache_file),
            })]
            backend = "NPU (VX delegate)"
            label = "hit" if cached else "cold"
            print(f"[tflite] VX delegate loaded: {dpath} (NBG cache {label}: {cache_file})", flush=True)
        except Exception as exc:
            print(f"[tflite] VX delegate failed ({exc}); falling back to CPU",
                  flush=True)
            delegates = []
            backend = "CPU"
    elif delegate_path:
        print(f"[tflite] delegate not found at {delegate_path!r}; using CPU",
              flush=True)

    kwargs: dict = {"model_path": str(model_path)}
    if delegates:
        kwargs["experimental_delegates"] = delegates
    if num_threads:
        kwargs["num_threads"] = num_threads
    interp = _tflite.Interpreter(**kwargs)
    interp.allocate_tensors()
    if delegates and cache_file is not None:
        # Warm-up inference exercises the NBG graph load — a corrupt cache
        # dies HERE (native segfault) with the marker still absent, so the
        # next start deletes it. Only a cache that survived this run of the
        # graph is marked verified.
        interp.invoke()
        _mark_nbg_cache_verified(cache_file, ok_marker)
    return interp, backend


def log_model_details(interp, backend, model_path):
    """Print the active backend plus every input/output tensor's
    shape/dtype/quantization — diagnostic for delegate op-fallback cases."""
    print(f"[tflite] backend={backend} model={Path(model_path).name}", flush=True)
    for kind, details in (("input", interp.get_input_details()),
                          ("output", interp.get_output_details())):
        for d in details:
            print(
                f"[tflite]   {kind}[{d['index']}] shape={list(d['shape'])} "
                f"dtype={d['dtype'].__name__} quant={d.get('quantization')}",
                flush=True,
            )


# ── Quantization ──────────────────────────────────────────────────────────────
def dequantize(arr, quantization):
    """Dequantize an output tensor. quantization is (scale, zero_point).
    scale == 0 means the tensor is already float (no quantization)."""
    scale, zero_point = quantization
    arr = np.asarray(arr)
    if not scale:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - float(zero_point)) * float(scale)


def quantize(arr, quantization, dtype):
    """Quantize a float input tensor for an int-typed model input.
    scale == 0 means the input is float (return as the model dtype)."""
    scale, zero_point = quantization
    arr = np.asarray(arr, dtype=np.float32)
    if not scale:
        return arr.astype(dtype)
    q = np.round(arr / float(scale) + float(zero_point))
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


# ── Letterbox (aspect-preserving resize) ──────────────────────────────────────
def compute_letterbox(src_h, src_w, dst):
    """Compute aspect-preserving resize params for a square dst.

    Returns dict with new_w, new_h, pad_x, pad_y, scale. Pure math — no image
    op — so it is testable without cv2. Padding is centered (YOLO default).
    """
    scale = min(dst / src_w, dst / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    pad_x = (dst - new_w) / 2.0
    pad_y = (dst - new_h) / 2.0
    return {"new_w": new_w, "new_h": new_h,
            "pad_x": pad_x, "pad_y": pad_y, "scale": scale}


def letterbox(img, dst, pad_value=114):
    """Resize+pad an HxWxC image to dst x dst, preserving aspect ratio.
    Uses cv2 (imported lazily). Returns (padded_img, params)."""
    import cv2

    src_h, src_w = img.shape[:2]
    p = compute_letterbox(src_h, src_w, dst)
    resized = cv2.resize(img, (p["new_w"], p["new_h"]), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((dst, dst, img.shape[2]), pad_value, dtype=img.dtype)
    top = int(round(p["pad_y"] - 0.1))
    left = int(round(p["pad_x"] - 0.1))
    canvas[top:top + p["new_h"], left:left + p["new_w"]] = resized
    return canvas, p


def scale_coords_back(x, y, params, src_w, src_h):
    """Map a point from letterboxed space back to original image coords,
    clipped to image bounds."""
    ox = (x - params["pad_x"]) / params["scale"]
    oy = (y - params["pad_y"]) / params["scale"]
    ox = min(max(ox, 0.0), float(src_w))
    oy = min(max(oy, 0.0), float(src_h))
    return ox, oy


# ── Layout detection ──────────────────────────────────────────────────────────
def detect_layout(input_shape):
    """Return 'NCHW' or 'NHWC' from a 4-D input shape (channels = 3)."""
    s = list(input_shape)
    if len(s) == 4 and s[1] == 3:
        return "NCHW"
    return "NHWC"


# ── Model path resolution (pathlib, cross-platform) ───────────────────────────
def resolve_model_path(model_cfg, base_dir=None):
    """Resolve a model path from config to an absolute Path via pathlib.

    Accepts a dict ({"path": ...}) or a plain string. Relative paths are joined
    onto base_dir (defaults to cwd). Returns None when no path is configured.
    Existence is the caller's responsibility (so it can log a clear error).
    """
    if isinstance(model_cfg, dict):
        raw = model_cfg.get("path")
    else:
        raw = model_cfg
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        p = base / p
    return p.resolve()
