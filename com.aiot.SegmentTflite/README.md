# com.aiot.Segment

Instance Segmentation App using the YOLOv8-seg model, supporting RTSP streaming and local camera as input sources.

---

## Upload and Deployment Flow

### 1. Upload the Model First

Upload a YOLOv8-seg TFLite model (e.g., `yolov8n-seg_npu.tflite`) on the **AI Models** page in the Hub to get a Model Version.

### 2. Package and Upload the App

```bash
cd com.aiot.SegmentTflite
zip -r upload.zip artifacts \
    -x "artifacts/.venv/*" "artifacts/**/__pycache__/*" \
       "artifacts/**/.DS_Store" "__MACOSX/*"
```

Go to the Hub **Apps** page and click `+ New App`, then follow the wizard steps:

- **Step 2** — Link Models: bind `tflite_model` to the Model Version just uploaded
- **Step 3** — Default Configs: adjust `video_source`, `det_threshold`, `seg_alpha`, and other defaults as needed
- **Step 4** — Upload `upload.zip`

### 3. Modify Settings After Deployment

1. Find the target device on the **Devices** page in the Hub and click **Edit Config**
2. Modify `video_source` and the corresponding fields
3. Adjust `det_every_n_frames` to reduce inference frequency
4. After saving, the platform will automatically update the config on the device and restart the component

---

## video_source Switching

| Value | Description | Required Settings |
|---|---|---|
| `"rtsp"` | Pull stream from RTSP URL | `rtsp_url` |
| `"0"` / `"1"` / `"2"` ... | Local camera index | — |

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `det_threshold` | 0.5 | Detection confidence threshold |
| `det_every_n_frames` | 1 | Run inference every N frames (reduces CPU load) |
| `seg_alpha` | 0.4 | Mask overlay opacity (0 = hidden, 1 = fully opaque) |

---

## Permissions and Offline Mode

This app sets `YOLO_OFFLINE=1` in `start.sh` to prevent ultralytics from attempting model downloads at runtime. The `.tflite` model must be pre-supplied via the Hub and have its path injected by the platform.

---

Local testing:

```bash
cd artifacts
APP_CONFIG_PATH=../sample_local_config.json python3 backend/main.py
```
