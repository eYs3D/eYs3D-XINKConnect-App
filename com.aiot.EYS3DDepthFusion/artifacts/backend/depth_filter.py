"""Apply eYs3D SDK's built-in depth post-processing.

Only takes effect when stream_mode == "depth"; the SDK runs the filters in the
producer thread, so RGB stream is never touched. Failures are logged and
swallowed — depth still streams unfiltered.
"""


def configure(device, cfg):
    """Set up DepthFilterOptions on `device` from the app config dict."""
    if not cfg.get("depth_filter_enabled", True):
        return

    try:
        opts = device.get_depthFilter_options()
    except Exception as exc:
        print(f"[depth_filter] get_depthFilter_options unavailable: {exc}", flush=True)
        return

    enabled = []

    if cfg.get("depth_filter_edge_preserve", True):
        try:
            opts.edgePreServingFilter.enable()
            opts.edgePreServingFilter.set_edge_level(int(cfg.get("depth_filter_edge_level", 5)))
            enabled.append(f"edgePreserve(level={cfg.get('depth_filter_edge_level', 5)})")
        except Exception as exc:
            print(f"[depth_filter] edgePreserve failed: {exc}", flush=True)

    if cfg.get("depth_filter_hole_fill", True):
        try:
            opts.holeFill.enable()
            opts.holeFill.set(
                kernel_size=int(cfg.get("depth_filter_hole_fill_kernel", 1)),
                level=int(cfg.get("depth_filter_hole_fill_level", 3)),
            )
            enabled.append(
                f"holeFill(kernel={cfg.get('depth_filter_hole_fill_kernel', 1)},"
                f"level={cfg.get('depth_filter_hole_fill_level', 3)})"
            )
        except Exception as exc:
            print(f"[depth_filter] holeFill failed: {exc}", flush=True)

    if cfg.get("depth_filter_temporal", True):
        try:
            opts.temporalFilter.enable()
            opts.temporalFilter.set_alpha(float(cfg.get("depth_filter_temporal_alpha", 0.4)))
            enabled.append(f"temporal(alpha={cfg.get('depth_filter_temporal_alpha', 0.4)})")
        except Exception as exc:
            print(f"[depth_filter] temporal failed: {exc}", flush=True)

    try:
        opts.enable()
    except Exception as exc:
        print(f"[depth_filter] master enable failed: {exc}", flush=True)
        return

    print(f"[depth_filter] enabled: {', '.join(enabled) if enabled else '(none)'}", flush=True)
