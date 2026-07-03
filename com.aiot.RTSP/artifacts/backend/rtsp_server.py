"""RTSP socket server — the app's actual work, kept out of main.py.

Captures video from an RTSP stream using PyAV, encodes to H264, and streams it
to connected clients over a length-prefixed TCP socket. main.py builds an
RTSPStreamServer from the runtime config and calls run().

Dependencies:
    pip install av numpy
"""

import av
import socket
import struct
import threading
import time
from fractions import Fraction


# PyAV RTSP options — internal, not user-tunable.
RTSP_OPTIONS = {
    'rtsp_transport': 'tcp',
    'stimeout': '5000000',
    'max_delay': '500000',
    'reorder_queue_size': '0',
}


class RTSPStreamServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rtsp_url = str(cfg["rtsp_url"])
        self.socket_host = str(cfg["socket_host"])
        self.socket_port = int(cfg["socket_port"])
        self.video_width = int(cfg["video_width"])
        self.video_height = int(cfg["video_height"])
        self.video_fps = int(cfg["video_fps"])
        self.h264_bitrate = int(cfg["h264_bitrate"])
        self.gop_size = int(cfg["gop_size"])

        self.frame_time = 1.0 / self.video_fps
        self.stats_interval = self.video_fps
        self.stats_reset_interval = self.video_fps * 10

        self.container = None
        self.video_stream = None
        self.server_socket = None
        self.running = True
        self.camera_lock = threading.Lock()
        self.keepalive_thread = None
        self.last_frame = None
        self.last_frame_time = 0

    def _interruptible_sleep(self, seconds):
        """Sleep in small increments so SIGTERM can interrupt promptly."""
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(0.1)

    def initialize_camera(self, retry_interval=5):
        """Initialize RTSP stream using PyAV, retrying until connected or shutdown."""
        while self.running:
            print(f"Initializing RTSP stream from {self.rtsp_url}...")
            try:
                self.container = av.open(self.rtsp_url, options=RTSP_OPTIONS, timeout=10)
                self.video_stream = self.container.streams.video[0]
                self.video_stream.thread_type = 'AUTO'

                print(f"RTSP stream opened: {self.video_stream.width}x{self.video_stream.height}, "
                      f"codec={self.video_stream.codec_context.name}")
                print(f"Output: {self.video_width}x{self.video_height} @ {self.video_fps}fps")
                print("Using TCP transport for RTSP stream")
                return self.video_width, self.video_height
            except Exception as e:
                print(f"Failed to connect to RTSP stream: {e}. Retrying in {retry_interval}s...")
                self._interruptible_sleep(retry_interval)
        return None, None

    def start_server(self):
        """Start socket server"""
        print(f"Starting socket server on {self.socket_host}:{self.socket_port}...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(1.0)
        self.server_socket.bind((self.socket_host, self.socket_port))
        self.server_socket.listen(5)
        print(f"Server listening on port {self.socket_port}")

    def handle_client(self, client_socket, client_address):
        """Handle individual client connection"""
        print(f"Client connected: {client_address}")

        encoder = None
        try:
            # Create H264 encoder using PyAV
            encoder = av.CodecContext.create('libx264', 'w')
            encoder.width = self.video_width
            encoder.height = self.video_height
            encoder.pix_fmt = 'yuv420p'
            encoder.time_base = Fraction(1, self.video_fps)
            encoder.framerate = Fraction(self.video_fps, 1)
            encoder.bit_rate = self.h264_bitrate
            encoder.gop_size = self.gop_size
            encoder.max_b_frames = 0  # No B-frames (critical for iOS)
            encoder.options = {
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'profile': 'baseline',
                'level': '3.1',
                'forced-idr': '1',
                'repeat-headers': '1',  # Repeat SPS/PPS with every keyframe (iOS needs this)
            }
            encoder.open()
            print(f"H264 encoder started for client {client_address}")

            frame_count = 0
            start_time = time.time()
            consecutive_failures = 0
            max_failures = 5

            while self.running:
                loop_start = time.time()

                # Get frame from keep-alive thread (shared read)
                with self.camera_lock:
                    if self.last_frame is not None:
                        frame = self.last_frame
                        ret = True
                    else:
                        ret = False

                if not ret:
                    consecutive_failures += 1
                    print(f"No frame available from keep-alive thread "
                          f"(attempt {consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        print("Too many consecutive failures, closing connection")
                        break
                    else:
                        time.sleep(0.1)
                        continue

                consecutive_failures = 0

                # Convert numpy frame to av.VideoFrame
                av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
                av_frame = av_frame.reformat(width=self.video_width, height=self.video_height, format='yuv420p')
                av_frame.pts = frame_count
                av_frame.time_base = Fraction(1, self.video_fps)

                # Encode to H264 packets, send each frame as length-prefixed message
                # Format: [4 bytes big-endian length][H264 data]
                # This allows the JS client to split frames and assign per-frame RTP timestamps
                try:
                    packets = encoder.encode(av_frame)
                    for packet in packets:
                        h264_data = bytes(packet)
                        length_header = struct.pack('>I', len(h264_data))
                        client_socket.sendall(length_header + h264_data)
                except (BrokenPipeError, ConnectionResetError):
                    print(f"Client {client_address} disconnected")
                    break
                except Exception as e:
                    print(f"Encode/send error for {client_address}: {e}")
                    break

                frame_count += 1

                # Print FPS at regular intervals
                if frame_count % self.stats_interval == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"[{client_address}] FPS: {fps:.2f}")

                    if frame_count >= self.stats_reset_interval:
                        frame_count = 0
                        start_time = time.time()

                # Frame rate control
                elapsed = time.time() - loop_start
                sleep_time = self.frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"Error handling client {client_address}: {e}")

        finally:
            print(f"Client disconnected: {client_address}")
            if encoder:
                try:
                    # Flush encoder
                    packets = encoder.encode(None)
                    for packet in packets:
                        try:
                            client_socket.sendall(bytes(packet))
                        except Exception:
                            pass
                    encoder.close()
                except Exception:
                    pass
            try:
                client_socket.close()
            except Exception:
                pass

    def start_keepalive(self):
        """Start background thread to keep RTSP stream alive"""
        def keepalive_loop():
            print("Starting RTSP keep-alive thread...")

            while self.running:
                try:
                    if self.container:
                        for packet in self.container.demux(self.video_stream):
                            if not self.running:
                                break
                            for frame in packet.decode():
                                # Convert to numpy array (BGR24 for compatibility)
                                np_frame = frame.to_ndarray(format='bgr24')

                                with self.camera_lock:
                                    self.last_frame = np_frame
                                    self.last_frame_time = time.time()
                    else:
                        print("Keep-alive: No container available")
                        time.sleep(1)
                except av.EOFError:
                    print("Keep-alive: RTSP stream ended, reconnecting...")
                    self._reconnect()
                except Exception as e:
                    print(f"Keep-alive error: {e}")
                    self._reconnect()
            print("Keep-alive thread stopped")

        self.keepalive_thread = threading.Thread(target=keepalive_loop, daemon=True)
        self.keepalive_thread.start()

    def _reconnect(self):
        """Reconnect to RTSP stream"""
        self._interruptible_sleep(1)
        if not self.running:
            return
        try:
            if self.container:
                self.container.close()
        except Exception:
            pass
        try:
            print(f"Reconnecting to {self.rtsp_url}...")
            self.container = av.open(self.rtsp_url, options=RTSP_OPTIONS, timeout=10)
            self.video_stream = self.container.streams.video[0]
            self.video_stream.thread_type = 'AUTO'
            print("Reconnected to RTSP stream")
        except Exception as e:
            print(f"Reconnect failed: {e}")

    def run(self):
        """Main server loop"""
        try:
            w, h = self.initialize_camera()
            if not self.running:
                return
            self.start_server()
            self.start_keepalive()

            print("\nRTSP Stream Server running...")
            print("Waiting for clients to connect...")

            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    if self.running:
                        print(f"Error accepting client: {e}")

        except Exception as e:
            print(f"Server error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("\nShutting down server...")
        self.running = False

        if self.container:
            try:
                self.container.close()
            except Exception:
                pass

        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

        print("Server stopped")
