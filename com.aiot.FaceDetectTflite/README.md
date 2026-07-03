# com.aiot.FaceDetect

Face Detection App using the SCRFD TFLite model, supporting both RTSP streaming and local camera input sources.

---

## Upload and Deployment Flow

### 1. Upload the Model First

Upload the SCRFD TFLite model (e.g., `det_500m_npu.tflite`) on the **AI Models** page in the Hub to get a Model Version, which will be used for binding in App wizard Step 2.

### 2. Package and Upload the App

```bash
cd com.aiot.FaceDetectTflite
zip -r upload.zip artifacts \
    -x "artifacts/.venv/*" "artifacts/**/__pycache__/*" \
       "artifacts/**/.DS_Store" "__MACOSX/*"
```

Go to the Hub **Apps** page and click `+ New App`, then follow the wizard steps:

- **Step 2** — Link Models: bind `tflite_model` to the Model Version just uploaded
- **Step 3** — Default Configs: adjust `video_source`, `video_fps`, `det_threshold`, and other defaults as needed
- **Step 4** — Upload `upload.zip`

### 3. Modify Settings After Deployment

1. Find the target device on the **Devices** page in the Hub and click **Edit Config**
2. Modify `video_source` and the corresponding fields (see below)
3. Adjust `det_every_n_frames` to reduce inference frequency (see below)
4. After saving, the platform will automatically update the config on the device and restart the component

---

## video_source Switching

`video_source` controls the video input source and accepts two types of values:

| Value | Description | Required Settings |
|---|---|---|
| `"rtsp"` | Pull stream from RTSP URL | `rtsp_url` |
| `"0"` / `"1"` / `"2"` ... | Local camera index, corresponding to `/dev/videoN` | — |

Which number to use depends on the actual device nodes. Use `ls /dev/video*` to check, or `v4l2-ctl --list-devices` to see the mapping. USB camera drivers sometimes occupy two nodes, so the usable index does not necessarily start at 0.

**Example: Switch to local camera**

```json
{
  "video_source": "0"
}
```

**Example: Switch back to RTSP**

```json
{
  "video_source": "rtsp",
  "rtsp_url": "rtsp://192.168.1.50:8554/stream1.264"
}
```

---

## det_every_n_frames (Inference Frequency Control)

Devices with lower compute can increase this value to reduce the number of inferences per second:

| Value | Description |
|---|---|
| `1` | Run detection on every frame (default) |
| `3` | Run detection every 3 frames; intermediate frames reuse the last result |
| `5` | Run detection every 5 frames; lowest CPU load but higher tracking latency |

```json
{
  "det_every_n_frames": 3
}
```

---

For local testing, use `sample_local_config.json` as a reference, pointing to it via an environment variable:

```bash
cd artifacts
APP_CONFIG_PATH=../sample_local_config.json python3 backend/main.py
```
