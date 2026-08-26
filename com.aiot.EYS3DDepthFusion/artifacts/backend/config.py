"""Config loader for com.aiot.EYS3DDepthFusion.

Reads JSON from APP_CONFIG_PATH and merges over DEFAULTS.
"""
import json
import os

DEFAULTS = {
    # ── Camera ────────────────────────────────────────────────────────
    "module_pid": "",
    "mode_index": 1,
    "depth_bits": None,
    "max_capture_fps": 15,
    # IR projector. null leaves the camera untouched; an int sets the emitter
    # level. This is the single most effective lever on hole count, because
    # every hole it closes becomes a REAL measurement rather than a fusion
    # estimate. Measured on the ecv board (static scene, settled):
    #
    #   IR    holes    valid px
    #    0    51.5%     445673
    #    6    41.4%     540207   <- device default
    #   11    32.4%     622078
    #   15    29.4%     650859
    #
    # On the pixels valid at BOTH 6 and 15, noise also drops (1.97 -> 1.35mm);
    # the pixels IR newly recovers are grainier (16.6mm) but are real readings
    # where there was nothing. See ir_control.py for why the aggregate noise
    # figure moves the other way.
    #
    # Left unset by default because it drives an emitter on the user's
    # hardware and the right level is scene-dependent: it does nothing beyond
    # the projector's throw, and reflective surfaces can bloom.
    # ir_extended unlocks 7..15; without it the maximum is 6.
    "ir_value": None,
    "ir_extended": True,
    # Color and depth frames of one instant may differ by a serial number on
    # interleaved modules; see capture.FramePairer.
    "max_serial_delta": 1,
    # Working range for the validity mask. Empty means "ask the device", which
    # is preferred — these are only the fallback.
    "z_min_mm": None,
    "z_max_mm": None,

    # ── Stream out ────────────────────────────────────────────────────
    "socket_host": "0.0.0.0",
    "socket_port": 9999,
    "video_fps": 15,
    "h264_bitrate": 500000,
    "gop_size": 10,

    # ── Display ───────────────────────────────────────────────────────
    # "fused" | "compare" | "raw_depth" | "rgb" | "side_by_side" | "overlay"
    #
    # "compare" is the mode for judging this component: raw depth on the left,
    # fused on the right, both colorized through the SAME palette and range, so
    # the only visible difference is the holes that were filled.
    # "raw_depth" shows the SDK's own heatmap — the like-for-like baseline
    # against com.aiot.EYS3DCamera.
    # "side_by_side" is RGB on the left, fused on the right.
    "display_mode": "fused",
    # "eys3d" reproduces the SDK's own palette (hue sweep, red near -> violet
    # far) so `fused` and `raw_depth` look like the same instrument. cv2 names
    # (jet, turbo, inferno, ...) also work.
    "colormap": "eys3d",
    # DISPLAY range, deliberately separate from the z_min_mm/z_max_mm validity
    # range: colour resolution is spent entirely inside this window. These
    # match the SDK's own colorizer, measured from a capture — it sweeps hue
    # 0..240 deg across 200..1000mm. Widening this to the full validity range
    # is what made everything render one shade of red: a scene occupying
    # 350..1200mm spread over 300..8000mm lands in a sliver of the palette.
    # Raise display_max_mm to see far fills in colour, at the cost of contrast
    # where the data actually is.
    "display_min_mm": 200,
    "display_max_mm": 1000,
    "overlay_alpha": 0.5,
    "show_hud": True,

    # ── MiDaS ─────────────────────────────────────────────────────────
    "tflite_model": {"path": "backend/midas_small_256_npu.tflite"},
    "midas_input_size": 256,
    "npu_delegate_path": "/usr/lib/libvx_delegate.so",
    # Run MiDaS every Nth pair and reuse the previous map in between.
    # 4 is measured, not guessed: on the ecv board MiDaS-small costs 226ms via
    # the VX delegate (4.4/sec; CPU fallback is 1304ms, so the NPU is working —
    # the model is simply expensive). At video_fps 15 that is 3.75 inferences
    # per second, just inside budget. Lower it only if inference gets faster;
    # a reused map is stale geometry and shows up as drag on moving objects.
    "midas_every_n": 4,

    # ── Fusion ────────────────────────────────────────────────────────
    "fusion_ema_alpha": 0.3,
    "fusion_trim_sigma": 2.5,
    "fusion_trim_rounds": 2,
    "fusion_min_valid_ratio": 0.05,
    "fusion_correction_kernel": 101,
    "fusion_subsample": 4,
    "fusion_local_correction": True,
    # Drop fills beyond this distance instead of publishing them. Measured on
    # real board frames: median fill error is 8-47mm inside 1.2m but 1107mm
    # past 2.9m, because the alignment is affine in inverse depth. 0 disables.
    "fusion_max_fill_mm": 2500,

    # ── SDK depth post-processing (runs BEFORE fusion) ────────────────
    "depth_filter_enabled": True,
    "depth_filter_edge_preserve": True,
    "depth_filter_edge_level": 5,        # 1..10
    # OFF by default: the SDK's hole fill interpolates from neighbouring
    # pixels, so it pre-fills the very holes fusion is here to fill — with a
    # weaker estimate, and while hiding both the true hole structure and what
    # fusion contributed. Turn it back on to compare the two approaches.
    "depth_filter_hole_fill": False,
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
