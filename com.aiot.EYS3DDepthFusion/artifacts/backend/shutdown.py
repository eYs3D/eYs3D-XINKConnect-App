"""Graceful shutdown helper — release server, fusion thread, device in order."""


def shutdown(server=None, worker=None, device=None):
    if server is not None:
        try:
            server.stop()
        except Exception:
            pass
    if worker is not None:
        try:
            worker.stop()
        except Exception:
            pass
    if device is not None:
        try:
            device.close_stream()
        except Exception:
            pass
