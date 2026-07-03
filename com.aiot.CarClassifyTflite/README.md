# com.aiot.CarClassify

Two-stage vehicle detection + fine classification app. YOLO11 detects vehicles (COCO classes: car, motorcycle, bus, truck), then a YOLO11-cls classifier runs on each detected crop to output a fine-grained vehicle type label (e.g., *pickup*, *sports_car*, *school_bus*).

---

## Features

- YOLO11 vehicle detection (COCO classes: car `2`, motorcycle `3`, bus `5`, truck `7`)
- YOLO11-cls fine classification on each detected crop (1000-class ImageNet, filtered to vehicle-type names)
- ByteTrack multi-object tracking (each vehicle has a unique Track ID)
- Two-tier label annotation: fine class name (top) + COCO class + confidence (sub)
- Supports both RTSP streaming and local camera input

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | 0.5 | Detection confidence threshold |
| `cls_threshold` | 0.3 | Classifier confidence threshold |
| `det_every_n_frames` | 1 | Run inference every N frames (reduces CPU load) |
| `tracker` | `bytetrack.yaml` | Tracker configuration (built-in to ultralytics) |
| `detector_model` | `{}` | `{"path": "<yolo11n_npu.tflite>"}` |
| `classify_model` | `{}` | `{"path": "<yolo11n-cls_npu.tflite>"}` |
| `classify_labels` | `""` | Path to label file (defaults to bundled `imagenet_classes.txt`) |

---

Local testing:

```bash
cd artifacts
APP_CONFIG_PATH=../sample_local_config.json python3 backend/main.py
```
