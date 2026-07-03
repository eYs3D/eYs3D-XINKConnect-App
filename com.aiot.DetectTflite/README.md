# com.aiot.Detect

Object Detection + Tracking App using the YOLO11 detection model with ByteTrack tracker.

---

## Features

- YOLO11 object detection (80 COCO classes)
- ByteTrack multi-object tracking (each object has a unique Track ID)
- Bounding box + class label + confidence + track ID annotation
- Supports both RTSP streaming and local camera input

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | 0.5 | Detection confidence threshold |
| `det_every_n_frames` | 1 | Run inference every N frames (reduces CPU load) |
| `tracker` | `bytetrack.yaml` | Tracker configuration (built-in to ultralytics) |

---

Local testing:

```bash
cd artifacts
APP_CONFIG_PATH=../sample_local_config.json python3 backend/main.py
```
