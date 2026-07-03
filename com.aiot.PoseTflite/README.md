# com.aiot.Pose

Pose Estimation + Tracking App using the YOLO11-pose model with ByteTrack tracker.

---

## Features

- YOLO11 human pose estimation (17 COCO keypoints)
- ByteTrack multi-person tracking (each person has a unique Track ID)
- Skeleton lines + keypoint dots + bounding box annotation
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
