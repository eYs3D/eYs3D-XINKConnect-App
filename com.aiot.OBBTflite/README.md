# com.aiot.OBB

Oriented Bounding Box (OBB) Detection App using the YOLO11-obb model for detecting rotated objects.

---

## Features

- YOLO11 oriented bounding box detection
- Suitable for top-down views, satellite imagery, and tilted object scenes (e.g., parking lot vehicles, documents, ships)
- Rotated quadrilateral + class label + confidence annotation
- Supports both RTSP streaming and local camera input

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | 0.5 | Detection confidence threshold |
| `det_every_n_frames` | 1 | Run inference every N frames (reduces CPU load) |

---

## Supported Classes

The OBB model is trained on the DOTAv1 dataset and supports 15 classes:
plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court,
ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter,
roundabout, soccer-ball-field, swimming-pool

---

Local testing:

```bash
cd artifacts
APP_CONFIG_PATH=../sample_local_config.json python3 backend/main.py
```
