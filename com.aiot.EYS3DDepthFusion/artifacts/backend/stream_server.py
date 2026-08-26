"""TCP H264 stream server (length-prefixed packets).

Each accepted client gets its own handler thread + libx264 encoder.
A client disconnect cleans up its encoder but does not affect other clients
or the server.
"""
import socket
import struct
import threading
import time
from fractions import Fraction

try:
    import av  # type: ignore
except ImportError:
    av = None  # tests inject a fake


def pack_packet(payload: bytes) -> bytes:
    """Length-prefixed packet: 4-byte big-endian length + payload."""
    return struct.pack(">I", len(payload)) + payload


def _to_bytes(packet) -> bytes:
    return bytes(packet)


def create_encoder(width, height, video_fps, h264_bitrate, gop_size):
    enc = av.CodecContext.create("libx264", "w")
    enc.width = width
    enc.height = height
    enc.pix_fmt = "yuv420p"
    enc.time_base = Fraction(1, int(video_fps))
    enc.framerate = Fraction(int(video_fps), 1)
    enc.bit_rate = int(h264_bitrate)
    enc.gop_size = int(gop_size)
    enc.max_b_frames = 0
    enc.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "profile": "baseline",
        "level": "3.1",
        "forced-idr": "1",
        "repeat-headers": "1",
    }
    enc.open()
    return enc


class StreamServer:
    def __init__(self, host, port, video_fps, h264_bitrate, gop_size,
                 last_frame_holder):
        self.host = host
        self.port = int(port)
        self.video_fps = int(video_fps)
        self.h264_bitrate = int(h264_bitrate)
        self.gop_size = int(gop_size)
        self.frame_time = 1.0 / max(1, self.video_fps)
        self._holder = last_frame_holder
        self.running = False
        self._srv = None
        self._accept_thread = None
        self._bound_port = 0

    def bound_port(self):
        return self._bound_port

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(0.5)
        srv.bind((self.host, self.port))
        srv.listen(8)
        self._bound_port = srv.getsockname()[1]
        self._srv = srv
        self.running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def stop(self):
        self.running = False
        try:
            if self._srv:
                self._srv.close()
        except Exception:
            pass
        if self._accept_thread:
            self._accept_thread.join(timeout=2.0)

    def _accept_loop(self):
        while self.running:
            try:
                cs, addr = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            print(f"[srv] client {addr} connected", flush=True)
            threading.Thread(target=self._handle_client, args=(cs, addr), daemon=True).start()

    def _handle_client(self, cs, addr):
        encoder = None
        enc_w = enc_h = -1
        frame_no = 0
        no_frame_count = 0
        try:
            while self.running:
                t0 = time.time()
                frame = self._holder.get()
                if frame is None:
                    no_frame_count += 1
                    if no_frame_count == 50:  # ~2.5s with 0.05s sleep
                        print(f"[srv] no frame for {addr} after 2.5s — callbacks may not be firing", flush=True)
                    time.sleep(0.05)
                    continue
                no_frame_count = 0
                h, w = frame.shape[:2]
                if encoder is None or w != enc_w or h != enc_h:
                    if encoder is not None:
                        try:
                            encoder.close()
                        except Exception:
                            pass
                    enc_w, enc_h = w, h
                    encoder = create_encoder(w, h, self.video_fps, self.h264_bitrate, self.gop_size)
                    print(f"[srv] encoder {enc_w}x{enc_h} ready for {addr}", flush=True)

                av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                av_frame.pts = frame_no
                av_frame.time_base = Fraction(1, self.video_fps)
                try:
                    for pkt in encoder.encode(av_frame):
                        data = _to_bytes(pkt)
                        cs.sendall(pack_packet(data))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
                frame_no += 1
                sleep = self.frame_time - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            if encoder is not None:
                try:
                    encoder.close()
                except Exception:
                    pass
            try:
                cs.close()
            except Exception:
                pass
