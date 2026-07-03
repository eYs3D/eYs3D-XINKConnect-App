#!/usr/bin/env python3
# =============================================================================
# main.py — your app's entry point. start.sh runs this file.
#
# Keep main.py SMALL: it only (1) reads config, (2) wires graceful shutdown,
# and (3) hands off to your code in another module. The actual RTSP work lives
# in rtsp_server.py — import it and call it. Put new business logic in its own
# .py file too, then import it here.
#
#   - DEFAULTS / load_app_config(): reads settings from APP_CONFIG_PATH.
#       👉 Edit values in the "config.json" panel, NOT here. Add keys there,
#          read them via cfg["your_key"].
#   - sigterm_handler / sys.exit(0): graceful shutdown. Greengrass stops the
#       app with SIGTERM — without this the app is force-killed and marked
#       BROKEN. Do NOT remove.
# =============================================================================
import json
import os
import signal
import sys

from rtsp_server import RTSPStreamServer

# Default values for your settings. Real values come from the config.json panel
# (injected at runtime via APP_CONFIG_PATH) and override these.
DEFAULTS = {
    "rtsp_url": "rtsp://192.168.110.123/stream2.264",
    "socket_host": "0.0.0.0",
    "socket_port": 9999,
    "video_width": 640,
    "video_height": 360,
    "video_fps": 24,
    "h264_bitrate": 500000,
    "gop_size": 10,
}


def load_app_config():
    """Read APP_CONFIG_PATH JSON, overlay on DEFAULTS. Returns (cfg, path)."""
    cfg = dict(DEFAULTS)
    path = os.environ.get("APP_CONFIG_PATH")
    if not path:
        print("[main] APP_CONFIG_PATH not set; using DEFAULTS", flush=True)
        return cfg, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f) or {}
    except (OSError, ValueError) as e:
        print(f"[main] failed to load APP_CONFIG_PATH={path!r}: {e}", flush=True)
        return cfg, path
    for k, v in loaded.items():
        # Skip nested model-binding objects — this app uses only flat scalars.
        if isinstance(v, dict) and "model_version_id" in v:
            continue
        if k in DEFAULTS:
            cfg[k] = v
    return cfg, path


def main():
    cfg, path = load_app_config()
    print(f"[main] starting with config from {path!r}: {cfg}", flush=True)

    server = RTSPStreamServer(cfg)

    def sigterm_handler(sig, frame):
        print("[main] received shutdown signal, stopping...", flush=True)
        server.running = False
        server.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    server.run()


if __name__ == "__main__":
    main()
