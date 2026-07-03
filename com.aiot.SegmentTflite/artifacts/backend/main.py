#!/usr/bin/env python3
"""
Instance Segmentation RTSP Stream Server (com.aiot.SegmentTflite)

NPU-accelerated variant of com.aiot.Segment. Runs YOLO-seg via TFLite on the
Verisilicon VX delegate (NPU) with automatic CPU fallback. Overlays per-instance
colored masks with bounding boxes + labels, then streams H264 over a TCP socket.

APP_CONFIG_PATH keys:
    video_source    (string)  "rtsp" | "local" | camera index
    rtsp_url        (string)  RTSP source URL (when video_source is "rtsp")
    socket_host     (string)  TCP bind address          ["0.0.0.0"]
    socket_port     (number)  TCP port                  [9999]
    video_fps       (number)  output FPS cap            [15]
    h264_bitrate    (number)  bits/s                    [500000]
    gop_size        (number)  H264 GOP size             [10]
    det_threshold   (number)  confidence cutoff         [0.5]
    det_nms_iou     (number)  NMS IoU threshold         [0.45]
    det_input_size  (number)  model input resolution    [640]
    det_every_n_frames (number) run inference every N frames [1]
    mask_alpha      (number)  mask overlay opacity      [0.5]
    npu_delegate_path (string) VX delegate .so path     ["/usr/lib/libvx_delegate.so"]

Model binding:
    tflite_model    (object)  {"path": "backend/yolov8n-seg_npu.tflite"}

Dependencies:
    pip install tflite-runtime opencv-python-headless av numpy
"""

import av
import cv2
import json
import numpy as np
import queue
import os
import signal
import socket
import struct
import sys
import threading
import time
from fractions import Fraction

import tflite_backend as tb
import postprocess as pp


DEFAULTS = {
    "video_source": "rtsp",
    "rtsp_url": "rtsp://192.168.110.123/stream2.264",
    "socket_host": "0.0.0.0",
    "socket_port": 9999,
    "video_fps": 15,
    "h264_bitrate": 500000,
    "gop_size": 10,
    "det_threshold": 0.5,
    "det_nms_iou": 0.45,
    "det_input_size": 640,
    "det_every_n_frames": 1,
    "mask_alpha": 0.5,
    "npu_delegate_path": "/usr/lib/libvx_delegate.so",
    "tflite_model": {},
}

RTSP_OPTIONS = {
    'rtsp_transport': 'tcp',
    'stimeout': '5000000',
    'max_delay': '500000',
    'reorder_queue_size': '0',
}


def load_app_config():
    cfg = dict(DEFAULTS)
    path = os.environ.get("APP_CONFIG_PATH")
    if not path:
        print("[main] APP_CONFIG_PATH not set, using defaults", flush=True)
        return cfg, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f) or {}
    except (OSError, ValueError) as e:
        print(f"[main] cannot load config {path!r}: {e}, using defaults", flush=True)
        return cfg, path
    cfg.update(loaded)
    return cfg, path


def load_class_names():
    """Parse coco_80_classes.txt (lines of '<id>  <name>') bundled beside this
    file (artifacts/backend/) so it ships inside the upload zip."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "coco_80_classes.txt")
    names = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    names[int(parts[0])] = parts[1].strip()
    except OSError:
        print(f"[main] class names file not found at {path!r}", flush=True)
    return names


def mask_to_frame(mask_lowres, lb, input_size, w, h):
    """Resize a low-res proto mask to the original frame, removing letterbox pad."""
    full = cv2.resize(mask_lowres, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, int(round(lb["pad_x"])))
    y0 = max(0, int(round(lb["pad_y"])))
    x1 = input_size - x0
    y1 = input_size - y0
    crop = full[y0:y1, x0:x1]
    if crop.size == 0:
        crop = full
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


# ── TFLite seg detector (NPU first, CPU fallback) ─────────────────────────────

class TFLiteDetector:
    """YOLO-seg head on TFLite. Returns (boxes_xyxy, scores, classes, masks, lb)."""

    def __init__(self, model_path, delegate_path, conf, iou, input_size=640):
        self.interp, self.backend = tb.make_interpreter(model_path, delegate_path)
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        self.layout = tb.detect_layout(self.inp["shape"])
        shp = list(self.inp["shape"])
        if self.layout == "NCHW":
            self.input_size = int(shp[2]) if shp[2] > 0 else input_size
        else:
            self.input_size = int(shp[1]) if shp[1] > 0 else input_size
        self.conf = float(conf)
        self.iou = float(iou)
        tb.log_model_details(self.interp, self.backend, model_path)

    def infer(self, frame):
        h, w = frame.shape[:2]
        img, lbp = tb.letterbox(frame, self.input_size)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None] if self.layout == "NCHW" else rgb[None]

        if np.issubdtype(self.inp["dtype"], np.integer):
            x = tb.quantize(blob, self.inp["quantization"], self.inp["dtype"])
        else:
            x = blob.astype(self.inp["dtype"])

        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()

        det = protos = None
        for d in self.outs:
            arr = tb.dequantize(self.interp.get_tensor(d["index"]), d["quantization"])
            if arr.ndim == 4:
                protos = arr
            else:
                det = arr
        if det is None or protos is None:
            return (np.empty((0, 4), np.float32), np.empty(0, np.float32),
                    np.empty(0, np.int32), np.empty((0, 1, 1), np.float32), lbp)

        boxes, scores, classes, masks = pp.decode_seg(
            det, protos, self.conf, self.iou, self.input_size, lbp, w, h)
        return boxes, scores, classes, masks, lbp


# ── Drawing ───────────────────────────────────────────────────────────────────

_COLORS = [
    (85, 85, 255), (85, 145, 255), (70, 200, 255), (70, 225, 175),
    (125, 215, 75), (195, 205, 65), (255, 180, 75), (255, 140, 105),
    (255, 105, 165), (255, 85, 220), (185, 85, 255), (155, 115, 255),
    (175, 225, 115), (95, 225, 185), (235, 190, 85), (80, 165, 255),
    (255, 95, 130), (125, 75, 255), (165, 235, 85), (95, 195, 200),
]
_LABEL_FG = (20, 20, 20)


def annotate_segmentation(img, dets, names, input_size, alpha):
    """Overlay per-instance colored masks, draw boxes + labels.
    dets is (boxes, scores, classes, masks_lowres, lb)."""
    if dets is None:
        return
    boxes, scores, classes, masks, lb = dets
    if len(boxes) == 0:
        return
    h, w = img.shape[:2]

    overlay = img.copy()
    for i in range(len(masks)):
        cls_id = int(classes[i])
        m = mask_to_frame(masks[i], lb, input_size, w, h) > 0.5
        overlay[m] = _COLORS[cls_id % len(_COLORS)]
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        cls_id = int(classes[i])
        conf = float(scores[i])
        color = _COLORS[cls_id % len(_COLORS)]
        x1, y1, x2, y2 = boxes[i].astype(int)
        bw, bh = x2 - x1, y2 - y1
        L = max(6, min(20, min(bw, bh) // 5))
        for (cx, cy, dx, dy) in (
            (x1, y1,  L,  L),
            (x2, y1, -L,  L),
            (x1, y2,  L, -L),
            (x2, y2, -L, -L),
        ):
            cv2.line(img, (cx, cy), (cx + dx, cy), color, 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy), color, 2, cv2.LINE_AA)

        name = names.get(cls_id, str(cls_id))
        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        pad_x, pad_y = 6, 3
        ly1 = max(0, y1 - th - pad_y * 2)
        ly2 = ly1 + th + pad_y * 2
        cv2.rectangle(img, (x1, ly1), (x1 + tw + pad_x * 2, ly2), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1 + pad_x, ly2 - pad_y), font, 0.45, _LABEL_FG, 1, cv2.LINE_AA)


# ── Stream server ─────────────────────────────────────────────────────────────


_LOCAL_VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts")


def _parse_video_source(cfg):
    """Return (cap_arg, is_local_cam, is_local_files)."""
    src_raw = str(cfg.get("video_source", "rtsp")).strip()
    src = src_raw.lower()
    if src == "local":
        return None, False, True
    if src == "rtsp":
        return str(cfg["rtsp_url"]), False, False
    try:
        return int(src), True, False
    except ValueError:
        return src_raw, False, False  # treat as a direct URL string (preserve case)


class SegmentServer:
    def __init__(self, cfg, detector, names):
        self._cap_arg, self._is_local_cam, self._is_local_files = _parse_video_source(cfg)
        self._video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video")
        self.socket_host  = str(cfg["socket_host"])
        self.socket_port  = int(cfg["socket_port"])
        self.video_fps    = int(cfg["video_fps"])
        self.h264_bitrate = int(cfg["h264_bitrate"])
        self.gop_size     = int(cfg["gop_size"])
        self.detector     = detector
        self.names        = names
        self.mask_alpha   = float(cfg.get("mask_alpha", 0.5))
        self.input_size   = int(cfg.get("det_input_size", 640))
        self.det_threshold = float(cfg["det_threshold"])
        self.det_every_n  = max(1, int(cfg.get("det_every_n_frames", 1)))

        self.frame_time  = 1.0 / self.video_fps
        self.running     = True
        self._lock       = threading.Lock()
        self.last_frame  = None
        self._srv_sock   = None
        self._infer_q    = queue.Queue(maxsize=2)

        self._fps_count = 0
        self._fps_t0    = time.time()
        self._last_fps  = 0.0

    def _interruptible_sleep(self, secs):
        end = time.time() + secs
        while self.running and time.time() < end:
            time.sleep(0.1)

    def _open_rtsp(self):
        try:
            container = av.open(self._cap_arg, options=RTSP_OPTIONS, timeout=10)
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            print(f"[cap] RTSP opened: {stream.width}x{stream.height} codec={stream.codec_context.name}", flush=True)
            return container, stream
        except Exception as e:
            print(f"[cap] RTSP open failed: {e}", flush=True)
            return None, None

    def _inference_loop(self):
        last_dets = None
        while self.running:
            try:
                frame = self._infer_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.detector is not None:
                try:
                    last_dets = self.detector.infer(frame)
                except Exception as exc:
                    print(f"[seg] inference error: {exc}", flush=True)
            annotate_segmentation(frame, last_dets, self.names, self.input_size, self.mask_alpha)
            with self._lock:
                self.last_frame = frame
        print("[inf] thread exited", flush=True)

    def _capture_loop(self):
        if self._is_local_files:
            self._local_capture_loop()
            return

        cap_frame_idx = 0
        container = None
        stream = None

        while self.running:
            if container is None:
                if self._is_local_cam:
                    cap = cv2.VideoCapture(self._cap_arg)
                    if not cap.isOpened():
                        print(f"[cap] cannot open camera {self._cap_arg}, retry in 5s", flush=True)
                        self._interruptible_sleep(5)
                        continue
                    print(f"[cap] opened camera {self._cap_arg}", flush=True)
                    while self.running:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        cap_frame_idx += 1
                        if cap_frame_idx % self.det_every_n == 0:
                            try:
                                self._infer_q.put_nowait(frame)
                            except queue.Full:
                                pass
                    cap.release()
                    continue

                container, stream = self._open_rtsp()
                if container is None:
                    self._interruptible_sleep(5)
                    continue

            try:
                for packet in container.demux(stream):
                    if not self.running:
                        break
                    for frame_av in packet.decode():
                        if not self.running:
                            break
                        frame = frame_av.to_ndarray(format='bgr24')
                        cap_frame_idx += 1

                        self._fps_count += 1
                        now = time.time()
                        dt  = now - self._fps_t0
                        if dt >= 1.0:
                            self._last_fps  = self._fps_count / dt
                            self._fps_count = 0
                            self._fps_t0    = now
                            print(f"[cap] FPS: {self._last_fps:.1f}  (det every {self.det_every_n})", flush=True)

                        if cap_frame_idx % self.det_every_n == 0:
                            try:
                                self._infer_q.put_nowait(frame)
                            except queue.Full:
                                pass

            except av.EOFError:
                print("[cap] RTSP stream ended, reconnecting...", flush=True)
            except Exception as exc:
                print(f"[cap] RTSP error: {exc}, reconnecting...", flush=True)

            try:
                container.close()
            except Exception:
                pass
            container = None
            self._interruptible_sleep(1)

        if container:
            try:
                container.close()
            except Exception:
                pass
        print("[cap] thread exited", flush=True)

    def _list_local_videos(self):
        try:
            names = sorted(
                f for f in os.listdir(self._video_dir)
                if not f.startswith(".") and f.lower().endswith(_LOCAL_VIDEO_EXTS)
            )
        except FileNotFoundError:
            return []
        return [os.path.join(self._video_dir, n) for n in names]

    def _local_capture_loop(self):
        """Loop video files from artifacts/video/ at native FPS via PyAV.
        Rescans the folder between files so newly-dropped videos play next."""
        cap_frame_idx = 0
        played = set()
        print(f"[cap] local mode: watching {self._video_dir}", flush=True)

        while self.running:
            files = self._list_local_videos()
            if not files:
                print(f"[cap] no local videos in {self._video_dir}, retry in 5s", flush=True)
                self._interruptible_sleep(5)
                continue

            unplayed = [p for p in files if p not in played]
            if not unplayed:
                played.clear()
                unplayed = files

            target = unplayed[0]
            played.add(target)
            cap_frame_idx = self._play_local_file(target, cap_frame_idx)

        print("[cap] thread exited", flush=True)

    def _play_local_file(self, path, cap_frame_idx):
        """Decode one file at native FPS, push frames to inference queue."""
        try:
            container = av.open(path)
        except Exception as e:
            print(f"[cap] local open failed {os.path.basename(path)}: {e}", flush=True)
            self._interruptible_sleep(1)
            return cap_frame_idx

        try:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            rate = stream.average_rate or stream.base_rate
            fps = float(rate) if rate else 30.0
            if fps <= 0:
                fps = 30.0
            frame_interval = 1.0 / fps
            print(
                f"[cap] local file: {os.path.basename(path)} "
                f"{stream.width}x{stream.height} @ {fps:.1f}fps",
                flush=True,
            )

            next_t = time.time()
            for packet in container.demux(stream):
                if not self.running:
                    break
                for frame_av in packet.decode():
                    if not self.running:
                        break
                    frame = frame_av.to_ndarray(format='bgr24')
                    cap_frame_idx += 1

                    self._fps_count += 1
                    now = time.time()
                    dt = now - self._fps_t0
                    if dt >= 1.0:
                        self._last_fps = self._fps_count / dt
                        self._fps_count = 0
                        self._fps_t0 = now
                        print(
                            f"[cap] FPS: {self._last_fps:.1f}  (det every {self.det_every_n})",
                            flush=True,
                        )

                    if cap_frame_idx % self.det_every_n == 0:
                        try:
                            self._infer_q.put_nowait(frame)
                        except queue.Full:
                            pass

                    next_t += frame_interval
                    sleep = next_t - time.time()
                    if sleep > 0:
                        time.sleep(sleep)
                    elif sleep < -1.0:
                        next_t = time.time()
        except av.EOFError:
            pass
        except Exception as exc:
            print(f"[cap] local error {os.path.basename(path)}: {exc}", flush=True)
        finally:
            try:
                container.close()
            except Exception:
                pass
        return cap_frame_idx

    def _handle_client(self, cs, addr):
        print(f"[srv] client {addr} connected", flush=True)
        encoder  = None
        enc_w    = enc_h = -1
        frame_no = 0
        no_frame_count = 0

        try:
            while self.running:
                t0 = time.time()

                with self._lock:
                    frame = self.last_frame

                if frame is None:
                    no_frame_count += 1
                    if no_frame_count > 50:
                        print(f"[srv] no frame for {addr}, closing", flush=True)
                        break
                    time.sleep(0.1)
                    continue
                no_frame_count = 0

                h, w = frame.shape[:2]
                if encoder is None or w != enc_w or h != enc_h:
                    if encoder:
                        try:
                            encoder.close()
                        except Exception:
                            pass
                    enc_w, enc_h = w, h
                    encoder = av.CodecContext.create("libx264", "w")
                    encoder.width     = enc_w
                    encoder.height    = enc_h
                    encoder.pix_fmt   = "yuv420p"
                    encoder.time_base = Fraction(1, self.video_fps)
                    encoder.framerate = Fraction(self.video_fps, 1)
                    encoder.bit_rate  = self.h264_bitrate
                    encoder.gop_size  = self.gop_size
                    encoder.max_b_frames = 0
                    encoder.options = {
                        "preset": "ultrafast",
                        "tune": "zerolatency",
                        "profile": "baseline",
                        "level": "3.1",
                        "forced-idr": "1",
                        "repeat-headers": "1",
                    }
                    encoder.open()
                    print(f"[srv] encoder {enc_w}x{enc_h} ready for {addr}", flush=True)

                av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                av_frame.pts       = frame_no
                av_frame.time_base = Fraction(1, self.video_fps)

                try:
                    for pkt in encoder.encode(av_frame):
                        data = bytes(pkt)
                        cs.sendall(struct.pack(">I", len(data)) + data)
                except (BrokenPipeError, ConnectionResetError):
                    print(f"[srv] client {addr} disconnected", flush=True)
                    break
                except Exception as exc:
                    print(f"[srv] send error {addr}: {exc}", flush=True)
                    break

                frame_no += 1
                sleep = self.frame_time - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        finally:
            if encoder:
                try:
                    for pkt in encoder.encode(None):
                        try:
                            cs.sendall(bytes(pkt))
                        except Exception:
                            pass
                    encoder.close()
                except Exception:
                    pass
            try:
                cs.close()
            except Exception:
                pass
            print(f"[srv] client {addr} handler done", flush=True)

    def run(self):
        threading.Thread(target=self._inference_loop, daemon=True).start()
        threading.Thread(target=self._capture_loop, daemon=True).start()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)
        for _attempt in range(10):
            try:
                srv.bind((self.socket_host, self.socket_port))
                break
            except OSError as _exc:
                if _exc.errno != 98 or _attempt == 9:
                    raise
                print(f"[srv] port {self.socket_port} in use, retrying ({_attempt + 1}/10)...", flush=True)
                time.sleep(1.0)
        srv.listen(5)
        self._srv_sock = srv
        print(f"[srv] TCP listening on {self.socket_host}:{self.socket_port}", flush=True)

        while self.running:
            try:
                cs, addr = srv.accept()
                threading.Thread(
                    target=self._handle_client, args=(cs, addr), daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception as exc:
                if self.running:
                    print(f"[srv] accept error: {exc}", flush=True)

        try:
            srv.close()
        except Exception:
            pass
        print("[srv] stopped", flush=True)

    def stop(self):
        self.running = False
        try:
            if self._srv_sock:
                self._srv_sock.close()
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg, cfg_path = load_app_config()
    print(
        f"[main] pid={os.getpid()} config={cfg_path!r} "
        f"params={json.dumps({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}, sort_keys=True)}",
        flush=True,
    )

    names = load_class_names()
    detector = None
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    model_path = tb.resolve_model_path(cfg.get("tflite_model"), base_dir=base_dir)
    if model_path is None:
        print("[main] no tflite_model configured — streaming without segmentation", flush=True)
    elif not model_path.is_file():
        print(f"[main] tflite model not found at {model_path} — streaming without segmentation", flush=True)
    else:
        try:
            detector = TFLiteDetector(
                str(model_path),
                delegate_path=str(cfg.get("npu_delegate_path", "")),
                conf=cfg["det_threshold"],
                iou=cfg.get("det_nms_iou", 0.45),
                input_size=int(cfg.get("det_input_size", 640)),
            )
            print(f"[main] detector ready: {detector.backend}", flush=True)
        except Exception as exc:
            print(f"[main] detector init failed: {exc} — streaming without segmentation", flush=True)

    server = SegmentServer(cfg, detector, names)

    def _on_signal(sig, _frame):
        print(f"[main] signal {sig}, shutting down...", flush=True)
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    server.run()
    sys.exit(0)
