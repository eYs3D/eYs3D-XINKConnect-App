"""Graceful shutdown helper — release server, inference thread, device in order."""


def shutdown(server=None, inference=None, device=None):
    if server is not None:
        try:
            server.stop()
        except Exception:
            pass
    if inference is not None:
        try:
            inference.stop()
        except Exception:
            pass
    if device is not None:
        try:
            device.close_stream()
        except Exception:
            pass
