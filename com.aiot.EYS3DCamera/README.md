# com.aiot.EYS3DCamera

RGB / Depth streaming component for eYs3D 3D stereo cameras (G100+, G120, and other compatible models). It captures frames from the camera, encodes them with libx264, and streams them over a length-prefixed TCP socket. Typically, `com.aiot.WebrtcProxy` connects to this socket to relay the video track to a web browser.

## Config Parameters

All parameters can be overridden in the Greengrass deployment component configuration. Default values are defined in `DefaultConfiguration` in [`recipe.yaml`](recipe.yaml).

### Streaming Mode

| Key | Type | Default | Description |
|---|---|---|---|
| `stream_mode` | string | `"rgb"` | Streaming content. `"rgb"` = color image; `"depth"` = SDK-colorized depth heatmap (H×W×3 BGR) |

### Camera Selection

| Key | Type | Default | Description |
|---|---|---|---|
| `module_pid` | string | `""` | USB PID filter. Empty string = auto-select the first unit. Specify as a hex string, e.g., `"0x0181"` (G100+) or `"0x0202"` (G120C) |
| `mode_index` | int | `1` | Mode index from `ModeConfig.db` (0-based). Valid indices differ by PID; if not in the database, falls back to index 0 with a warning |
| `depth_bits` | int / null | `null` | Depth precision bits (G120 series only; ignored on other cameras). Options: `11` or `14`; `null` or empty string uses the `ModeConfig.db` default |

### TCP Server

| Key | Type | Default | Description |
|---|---|---|---|
| `socket_host` | string | `"0.0.0.0"` | Bind address. Usually keep as `0.0.0.0` in container/Greengrass environments |
| `socket_port` | int | `9999` | TCP port. WebrtcProxy connects to 9999 by default; changing this is not recommended |

### H264 Encoding (libx264)

| Key | Type | Default | Description |
|---|---|---|---|
| `video_fps` | int | `15` | Target encoder FPS; affects PTS/framerate headers. Should match `max_capture_fps` |
| `h264_bitrate` | int | `500000` | bits/sec. 500K~2M recommended for 720p; lower if network conditions are poor |
| `gop_size` | int | `10` | GOP length (number of frames between keyframes). Smaller = lower latency, higher bandwidth |
| `max_capture_fps` | int | `15` | Hard limit on capture FPS. The SDK may produce up to 30/60 fps; excess frames are dropped to prevent encoder/queue overflow |

### Inference (Optional)

| Key | Type | Default | Description |
|---|---|---|---|
| `tflite_model` | object / string | `{}` / `""` | TFLite model path. Empty = passthrough, no object detection |
| `det_threshold` | float | `0.5` | Detection confidence threshold |
| `det_every_n_frames` | int | `1` | Run inference every N frames (previous result is reused). `1` = run on every frame |
| `npu_delegate_path` | string | `"/usr/lib/libvx_delegate.so"` | Path to NPU delegate shared library. No effect if NPU is absent — `_create_detector()` will fall back to passthrough if no model is found |

### Depth Post-Processing (Only effective when `stream_mode` is `"depth"`)

Filters are executed by the SDK in the producer thread to avoid loading the Python-side CPU. Each filter failure is logged only and does not interrupt streaming.

| Key | Type | Default | Range / Options | Description |
|---|---|---|---|---|
| `depth_filter_enabled` | bool | `true` | true / false | Master switch. `false` = fully bypass all filters, stream raw depth |
| `depth_filter_edge_preserve` | bool | `true` | true / false | Edge-preserving smoothing. Reduces noise while keeping object boundaries sharp |
| `depth_filter_edge_level` | int | `5` | 1..10 | Smoothing strength. Higher values produce a smoother result but may blur fine details |
| `depth_filter_hole_fill` | bool | `true` | true / false | Fill depth holes using neighboring known pixels |
| `depth_filter_hole_fill_level` | int | `3` | 1..3 | Hole-filling strength. 3 is strongest and fills larger holes |
| `depth_filter_hole_fill_kernel` | int | `1` | ≥1 | Kernel size for hole filling. Usually `1` is sufficient |
| `depth_filter_temporal` | bool | `true` | true / false | Inter-frame temporal smoothing to suppress flickering |
| `depth_filter_temporal_alpha` | float | `0.4` | 0.0..1.0 | Temporal EMA coefficient. Lower = smoother (but more motion blur on dynamic scenes); `1.0` ≈ no temporal smoothing |

#### Tuning Tips

- **Static / Slow Scenes** — Default values are recommended; the output will be noticeably cleaner.
- **Fast-Moving Objects** — Set `depth_filter_temporal_alpha` to `0.7~0.9` to minimize ghosting/motion blur.
- **Raw Depth Inspection** — Set `depth_filter_enabled` to `false`.
- **Individual Filter Issue** — Disable that specific filter flag rather than disabling all filtering.

## Example Config: Depth Stream + Enhanced Smoothing

```json
{
  "stream_mode": "depth",
  "depth_filter_temporal_alpha": 0.3,
  "depth_filter_edge_level": 7
}
```

## Example Config: Low-Bandwidth RGB Stream

```json
{
  "stream_mode": "rgb",
  "video_fps": 10,
  "max_capture_fps": 10,
  "h264_bitrate": 300000,
  "gop_size": 20
}
```

## Verification on Device

```bash
sudo tail -n 200 /greengrass/v2/logs/com.aiot.app.<APP_ID>.log | grep -E "main|srv|capture|depth_filter"
```

Expected output sequence:

```
[main] camera detected: PID=0x...
[main] color=WxH depth=WxH        ← non-zero dimensions
[main] streaming on 0.0.0.0:9999 mode=...
[depth_filter] enabled: edgePreserve(...), holeFill(...), temporal(...)   ← depth mode only
[capture] first color/depth frame shape=(H, W, 3) ...
[srv] client (...) connected
```

## Related Components

- `com.aiot.WebrtcProxy` — Receives H264 length-prefixed packets from port 9999 and wraps them into RTP for a PeerConnection
- `com.aiot.DetectTflite` — Reference implementation using the same TCP streaming protocol
