# com.aiot.Semantic

Semantic Segmentation App using the YOLO11-seg model, merging all instance masks of the same class into a single region.

---

## Features

- YOLO11 semantic segmentation (all instance masks of the same class are merged into one region)
- One color per class, with a legend in the top-left corner
- Difference from `com.aiot.Segment` (instance segmentation): this module does not distinguish between individual instances of the same class
- Supports both RTSP streaming and local camera input

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | 0.5 | Detection confidence threshold |
| `det_every_n_frames` | 1 | Run inference every N frames (reduces CPU load) |
| `seg_alpha` | 0.5 | Mask overlay opacity (0 = hidden, 1 = fully opaque) |

---

Local testing:

```bash
cd artifacts
APP_CONFIG_PATH=../sample_local_config.json python3 backend/main.py
```
