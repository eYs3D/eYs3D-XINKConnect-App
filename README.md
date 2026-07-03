# AIoT Edge AI Apps — NPU-Accelerated Vision Apps for AWS Greengrass

A collection of real-time computer vision apps for edge devices equipped with a **hardware NPU**. Each app runs as an AWS Greengrass component, accepts video from RTSP, a local camera, or a local video file, and streams the annotated H264 output over a TCP socket.

---

## Apps at a Glance

| Component | Task | Model | Tracking |
|---|---|---|---|
| `com.aiot.DetectTflite` | Object detection | YOLO11 (80 COCO classes) | BYTETracker |
| `com.aiot.CarClassifyTflite` | Vehicle detection + fine classification | YOLO11 + YOLO11-cls | BYTETracker |
| `com.aiot.PoseTflite` | Human pose estimation | YOLO11-pose (17 keypoints) | BYTETracker |
| `com.aiot.OBBTflite` | Oriented bounding box detection | YOLO11-obb (15 DOTAv1 classes) | — |
| `com.aiot.SegmentTflite` | Instance segmentation | YOLOv8-seg (80 COCO classes) | — |
| `com.aiot.SemanticTflite` | Semantic segmentation | YOLO11-seg (80 COCO classes) | — |
| `com.aiot.FaceDetectTflite` | Face detection + keypoints | SCRFD-500M (5 keypoints) | — |
| `com.aiot.RTSP` | RTSP relay + H264 encode | — | — |
| `com.aiot.EYS3DCamera` | eYs3D stereo camera RGB/Depth stream + optional object detection | — | — |

---

## Hardware & Runtime Requirements

| Requirement | Detail |
|---|---|
| Target arch | `aarch64` Linux (compatible NPU board) |
| NPU delegate | `libvx_delegate.so` — provided by the BSP, default path `/usr/lib/libvx_delegate.so` |
| TFLite runtime | `tflite-runtime >= 2.13.0` (aarch64) |
| x86_64 dev machine | `ai-edge-litert==1.2.0` (CPU only, no NPU) |
| Python | 3.8+ |

All inference apps **automatically fall back to CPU** when the NPU delegate is absent or fails to load, so they can be developed and tested on any x86_64 machine.

---

## Architecture Overview

```
Video source (RTSP / local camera / local video files)
        │
        │  [capture thread]  PyAV / OpenCV
        ▼
  frame queue (bounded)
        │
        │  [inference thread]  TFLite → NPU delegate or CPU
        ▼
  annotated frame (in-place)
        │
        │  [TCP server]  libx264 H264 encode via PyAV
        ▼
Connected clients (any number)
```

Key design points:

- **Capture and inference run in separate threads** so a slow NPU invoke never drops capture frames.
- **The output frame buffer always holds the latest annotated frame.** TCP clients encode and transmit at the configured FPS cap; if a client is slow it reuses the same frame.
- **`det_every_n_frames`** skips inference on intermediate frames (reusing the previous detection result) to reduce CPU/NPU load without lowering the output frame rate.
- **Graceful shutdown** — all apps handle `SIGTERM` (sent by Greengrass on stop/deploy) and exit with code `0`.

---

## Per-App Details

### com.aiot.DetectTflite

Object detection and multi-object tracking.

- **Model**: YOLO11 detect head, 80 COCO classes (`coco_80_classes.txt` bundled)
- **Tracking**: BYTETracker — each object receives a persistent track ID displayed as `#N`
- **Annotation**: corner-bracket bounding boxes with class label, confidence, and track ID
- **Default model file**: `backend/yolo11n_npu.tflite`

**Key config parameters**:

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | `0.5` | Detection confidence cutoff |
| `det_nms_iou` | `0.45` | NMS IoU threshold |
| `det_input_size` | `640` | Model input resolution |
| `det_every_n_frames` | `1` | Run inference every N frames |
| `tracker` | `"bytetrack.yaml"` | Enable BYTETracker (set to `""` to disable) |
| `track_classes` | `null` | Array of COCO class IDs to keep; `null` = all |
| `tflite_model` | `{}` | `{"path": "<path to .tflite>"}` |

---

### com.aiot.CarClassifyTflite

Two-stage pipeline: YOLO11 detects vehicles (COCO classes: car, motorcycle, bus, truck), then a YOLO11-cls classifier runs on each detected crop to output a fine vehicle-type label (e.g., *pickup*, *sports_car*, *school_bus*).

- **Detector model**: YOLO11 detect, filtered to COCO vehicle IDs `{2, 3, 5, 7}`
- **Classifier model**: YOLO11-cls (1000-class ImageNet), whitelist-filtered to vehicle-type names
- **Tracking**: BYTETracker on the detector output
- **Annotation**: two-tier label pill — fine class name (top) + COCO class + confidence (sub)

**Key config parameters**:

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | `0.5` | Detection confidence cutoff |
| `cls_threshold` | `0.3` | Classifier confidence cutoff |
| `det_every_n_frames` | `1` | Run inference every N frames |
| `tracker` | `"bytetrack.yaml"` | Enable BYTETracker |
| `detector_model` | `{}` | `{"path": "<yolo11n_npu.tflite>"}` |
| `classify_model` | `{}` | `{"path": "<yolo11n-cls_npu.tflite>"}` |
| `classify_labels` | `""` | Path to label file (defaults to bundled `imagenet_classes.txt`) |

---

### com.aiot.PoseTflite

Human pose estimation with multi-person tracking.

- **Model**: YOLO11-pose, 17 COCO keypoints per person
- **Tracking**: BYTETracker — each person gets a persistent track ID
- **Annotation**: skeleton lines, keypoint dots, bounding boxes, and track IDs
- **Default model file**: `backend/yolo11n-pose_npu.tflite`

**Key config parameters**:

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | `0.5` | Detection confidence cutoff |
| `det_every_n_frames` | `1` | Run inference every N frames |
| `tracker` | `"bytetrack.yaml"` | Enable BYTETracker |
| `tflite_model` | `{}` | `{"path": "<path to .tflite>"}` |

---

### com.aiot.OBBTflite

Oriented Bounding Box (OBB) detection for rotated objects — useful for top-down/aerial scenes such as parking lots, satellite imagery, and document scanning.

- **Model**: YOLO11-obb trained on DOTAv1
- **Classes (15)**: plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool
- **Annotation**: rotated quadrilateral polygon + label
- **Default model file**: `backend/yolo11n-obb_npu.tflite`

**Key config parameters**:

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | `0.5` | Detection confidence cutoff |
| `det_nms_iou` | `0.45` | Rotated NMS IoU threshold |
| `det_every_n_frames` | `1` | Run inference every N frames |
| `tflite_model` | `{}` | `{"path": "<path to .tflite>"}` |

---

### com.aiot.SegmentTflite

Instance segmentation — each detected object gets its own per-pixel mask.

- **Model**: YOLOv8-seg, 80 COCO classes
- **Annotation**: per-instance colored mask overlay + bounding box + label
- **Default model file**: `backend/yolov8n-seg_npu.tflite`

**Key config parameters**:

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | `0.5` | Detection confidence cutoff |
| `det_every_n_frames` | `1` | Run inference every N frames |
| `seg_alpha` | `0.4` | Mask overlay opacity (0 = hidden, 1 = opaque) |
| `tflite_model` | `{}` | `{"path": "<path to .tflite>"}` |

> **Note**: `YOLO_OFFLINE=1` is set in `start.sh` to prevent Ultralytics from attempting model downloads at runtime. The `.tflite` must be pre-supplied via platform model binding.

---

### com.aiot.SemanticTflite

Semantic segmentation — all instances of the same class share a single merged mask (no instance distinction).

- **Model**: YOLO11-seg, 80 COCO classes
- **Annotation**: per-class colored mask + legend in the top-left corner
- **Default model file**: `backend/yolo11n-seg_npu.tflite`

Compared to `com.aiot.SegmentTflite`: this app does not distinguish between individuals of the same class — all *person* pixels merge into one region, all *car* pixels merge into another.

**Key config parameters**:

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | `0.5` | Detection confidence cutoff |
| `det_every_n_frames` | `1` | Run inference every N frames |
| `seg_alpha` | `0.5` | Mask overlay opacity |
| `tflite_model` | `{}` | `{"path": "<path to .tflite>"}` |

---

### com.aiot.FaceDetectTflite

Face detection with 5-point facial keypoints (eyes, nose, mouth corners).

- **Model**: SCRFD-500M (InsightFace), converted from ONNX to TFLite
- **Annotation**: corner-bracket boxes + keypoint dots + confidence score
- **Default model file**: `backend/det_500m_npu.tflite`

**Backend priority** (auto-selected at startup):

| Priority | Backend | Condition |
|---|---|---|
| 1 | TFLite + NPU delegate | `tflite_model` path set + delegate `.so` found |
| 2 | TFLite CPU | `tflite_model` path set, delegate absent/failed |
| 3 | ONNX Runtime CPU | `detector_model` (`.onnx`) path set, `onnxruntime` installed |
| 4 | Passthrough | No model available — streams video without detection |

**Key config parameters**:

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | `0.5` | Face confidence cutoff |
| `det_nms_iou` | `0.4` | NMS IoU threshold |
| `det_input_size` | `640` | SCRFD input resolution |
| `det_every_n_frames` | `1` | Run inference every N frames |
| `tflite_model` | `{}` | `{"path": "<det_500m_npu.tflite>"}` |
| `detector_model` | `{}` | `{"path": "<det_500m.onnx>"}` (ONNX fallback) |

---

### com.aiot.RTSP

A pure RTSP relay and H264 encoder — no inference. Useful as a passthrough video source for other apps or as a reference implementation for `main.py` structure.

- Connects to an RTSP source via PyAV (TCP transport, 5 s timeout)
- Re-encodes to H264 (`ultrafast` / `zerolatency` preset) at the configured resolution and bitrate
- Streams to all connected TCP clients simultaneously

**Key config parameters**:

| Parameter | Default | Description |
|---|---|---|
| `rtsp_url` | `"rtsp://192.168.110.123/stream2.264"` | RTSP source URL |
| `socket_host` | `"0.0.0.0"` | TCP bind address |
| `socket_port` | `9999` | TCP port |
| `video_width` / `video_height` | `640` / `360` | Output resolution |
| `video_fps` | `24` | Output frame rate |
| `h264_bitrate` | `500000` | H264 bit rate (bits/s) |
| `gop_size` | `10` | H264 GOP size |

---

### com.aiot.EYS3DCamera

RGB / Depth streaming component for eYs3D 3D stereo cameras (G100+, G120, and other compatible models). It captures frames from the camera, encodes them with libx264, and streams them over a length-prefixed TCP socket. Typically, `com.aiot.WebrtcProxy` connects to this socket to relay the video track to a web browser.

It also supports optional object detection (inference) on the RGB stream, and various hardware-accelerated depth filtering options when depth mode is enabled.

- **Model**: — (supports optional TFLite model)
- **Tracking**: —
- **Annotation**: —

**Key config parameters**:

| Parameter | Default | Description |
|---|---|---|
| `stream_mode` | `"rgb"` | Streaming content: `"rgb"` (color image) or `"depth"` (SDK-colorized depth heatmap, H×W×3 BGR) |
| `module_pid` | `""` | USB PID filter. Empty string for auto-detecting the first unit. Hex string, e.g., `"0x0181"` (G100+) or `"0x0202"` (G120C) |
| `mode_index` | `1` | Mode index from `ModeConfig.db` (0-based). If invalid/out-of-range, falls back to `0` with a warning |
| `depth_bits` | `null` | Depth precision bits (G120 series only). Options: `11` or `14`. `null` or empty uses the database default |
| `socket_host` | `"0.0.0.0"` | TCP server bind address |
| `socket_port` | `9999` | TCP server port |
| `video_fps` | `15` | Target encoder FPS. Should match `max_capture_fps` |
| `h264_bitrate` | `500000` | H264 bit rate (bits/s) |
| `gop_size` | `10` | H264 GOP size |
| `max_capture_fps` | `15` | Hard limit on capture FPS to prevent encoder/queue overflow |
| `tflite_model` | `{}` | `{"path": "<path to .tflite>"}`. Empty for passthrough (no inference) |
| `det_threshold` | `0.5` | Detection confidence cutoff |
| `det_every_n_frames` | `1` | Run inference every N frames |
| `npu_delegate_path` | `"/usr/lib/libvx_delegate.so"` | Path to NPU delegate |

**Depth Post-Processing (Only effective when `stream_mode` is `"depth"`)**:
Filters are executed by the SDK in the producer thread to avoid loading the CPU.

| Parameter | Default | Options | Description |
|---|---|---|---|
| `depth_filter_enabled` | `true` | `true`/`false` | Master switch. `false` bypasses all filters to stream raw depth |
| `depth_filter_edge_preserve` | `true` | `true`/`false` | Edge-preserving smoothing to reduce noise while keeping boundaries sharp |
| `depth_filter_edge_level` | `5` | `1` to `10` | Smoothing strength (higher values are smoother but may blur details) |
| `depth_filter_hole_fill` | `true` | `true`/`false` | Fills depth holes using neighboring pixels |
| `depth_filter_hole_fill_level`| `3` | `1` to `3` | Hole-filling strength (3 is strongest, filling larger holes) |
| `depth_filter_hole_fill_kernel`| `1` | `≥1` | Kernel size for hole filling. Usually `1` is sufficient |
| `depth_filter_temporal` | `true` | `true`/`false` | Inter-frame temporal smoothing to suppress flickering |
| `depth_filter_temporal_alpha` | `0.4` | `0.0` to `1.0` | Exponential moving average factor. Lower is smoother but has more motion blur |

**Filter Tuning Tips**:
- **Static / Slow Scenes**: Default values are recommended; the output will be noticeably cleaner.
- **Fast-Moving Objects**: Set `depth_filter_temporal_alpha` to `0.7~0.9` to minimize ghosting/motion blur.
- **Raw Depth Inspection**: Set `depth_filter_enabled` to `false`.
- **Individual Filter Bypass**: Disable specific filter flags individually if needed rather than disabling all filtering.

#### Example Config: Depth Stream + Enhanced Smoothing
```json
{
  "stream_mode": "depth",
  "depth_filter_temporal_alpha": 0.3,
  "depth_filter_edge_level": 7
}
```

#### Example Config: Low-Bandwidth RGB Stream
```json
{
  "stream_mode": "rgb",
  "video_fps": 10,
  "max_capture_fps": 10,
  "h264_bitrate": 300000,
  "gop_size": 20
}
```

#### Verification on Device
Verify logs by running:
```bash
sudo tail -n 200 /greengrass/v2/logs/com.aiot.app.<APP_ID>.log | grep -E "main|srv|capture|depth_filter"
```
Expected output sequence:
```
[main] camera detected: PID=0x...
[main] color=WxH depth=WxH        ← non-zero dimensions
[main] streaming on 0.0.0.0:9999 mode=...
[depth_filter] enabled: edgePreserve(...), holeFill(...), temporal(...)   ← (depth mode only)
[capture] first color/depth frame shape=(H, W, 3) ...
[srv] client (...) connected
```

---

## Common Configuration Reference

All inference apps share these parameters. Values are loaded from the JSON file at `APP_CONFIG_PATH`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_source` | string | `"rtsp"` | `"rtsp"` · `"local"` · `"0"` / `"1"` (camera index) |
| `rtsp_url` | string | — | RTSP source URL (required when `video_source` is `"rtsp"`) |
| `socket_host` | string | `"0.0.0.0"` | TCP server bind address |
| `socket_port` | number | `9999` | TCP server port |
| `video_fps` | number | `15` | Output frame rate cap |
| `h264_bitrate` | number | `500000` | H264 bit rate (bits/s) |
| `gop_size` | number | `10` | H264 GOP size |
| `det_every_n_frames` | number | `1` | Run inference every N frames; intermediate frames reuse the last detection result |
| `npu_delegate_path` | string | `"/usr/lib/libvx_delegate.so"` | Path to the NPU delegate `.so` |

### `video_source` values

| Value | Behavior |
|---|---|
| `"rtsp"` | Pull stream from `rtsp_url` via PyAV (TCP transport, auto-reconnect) |
| `"local"` | Loop video files from `artifacts/video/` in alphabetical order; rescans between files |
| `"0"` / `"1"` / … | Open `/dev/videoN` local camera via OpenCV |

### Model binding

All apps accept model paths as JSON objects inside the config:

```json
{
  "tflite_model": { "path": "backend/yolo11n_npu.tflite" }
}
```

The `path` field is resolved relative to the `artifacts/` directory if not absolute.

---

## About Ultralytics and BYTETracker

The YOLO models used by these apps (`YOLO11`, `YOLOv8`) originate from [Ultralytics](https://github.com/ultralytics/ultralytics) and are converted to TFLite format for NPU deployment.

Multi-object tracking (`com.aiot.DetectTflite`, `com.aiot.CarClassifyTflite`, `com.aiot.PoseTflite`) is powered by **BYTETracker** (AGPL-3.0).

---

## File Structure (per app)

```
com.aiot.<Name>/
├── artifacts/
│   ├── backend/
│   │   ├── main.py            # entry point: config loading, server, SIGTERM handler
│   │   ├── tflite_backend.py  # TFLite interpreter + NPU delegate + letterbox helpers
│   │   ├── postprocess.py     # decode_detect / decode_obb / decode_seg / decode_pose
│   │   ├── tracker/           # vendored BYTETracker (tracking apps only)
│   │   └── *.tflite           # pre-converted NPU model
│   ├── lifecycle/
│   │   ├── install.sh         # create venv + pip install requirements.txt
│   │   └── start.sh           # activate venv + exec main.py
│   ├── requirements.txt
│   └── video/
│       └── sample.mp4         # bundled demo clip for local mode
├── sample_local_config.json   # example config for local testing
└── icon.png
```

---

## Local Testing

Each app can be run on any machine (including x86_64) without Greengrass. The TFLite runtime automatically falls back to CPU when the NPU delegate is absent.

```bash
# Install Python dependencies
cd com.aiot.<Name>/artifacts
pip install -r requirements.txt

# Run with the sample config
APP_CONFIG_PATH=../sample_local_config.json python3 backend/main.py
```

`sample_local_config.json` sets `video_source` to `"local"` so the app loops over `artifacts/video/sample.mp4` without needing a camera or RTSP stream.

To switch to a live camera:

```json
{
  "video_source": "0"
}
```

To switch to an RTSP stream:

```json
{
  "video_source": "rtsp",
  "rtsp_url": "rtsp://192.168.1.50:8554/stream.264"
}
```

The H264 TCP stream is served on `0.0.0.0:9999` by default. Any video player that can receive a raw H264 TCP stream can connect (e.g., `ffplay tcp://localhost:9999`).

---

## Python Dependencies

| Package | Purpose |
|---|---|
| `tflite-runtime` (aarch64) / `ai-edge-litert` (x86_64) | TFLite inference runtime |
| `opencv-python-headless` | Frame decode, resize, annotation drawing |
| `av` (PyAV) | RTSP/video file capture + H264 encoding |
| `numpy` | Tensor preprocessing and postprocessing |
| `scipy` | Kalman filter + linear assignment in BYTETracker (tracking apps only) |
| `onnxruntime` | Optional ONNX CPU fallback (FaceDetectTflite only) |

---

## License

This project includes third-party libraries from [Ultralytics](https://github.com/ultralytics/ultralytics) and [InsightFace](https://github.com/deepinsight/insightface). Please refer to their respective repositories for license terms.

All other source code in this repository is proprietary to eYs3D.

This project is developed based on [Ultralytics](https://github.com/ultralytics/ultralytics) and is licensed under the [AGPL-3.0 License](LICENSE).
