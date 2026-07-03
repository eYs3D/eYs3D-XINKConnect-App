# com.aiot.RTSP

RTSP Stream Server that captures video from an RTSP source, encodes it as H264, and streams it to connected clients over a TCP socket.

This app also serves as a **reference implementation for the standard main.py structure**: `main.py` is responsible only for reading configuration and handling graceful shutdown, then delegating the actual work to `rtsp_server.py`. Other features should follow the same pattern — write them as independent `.py` files and import them into `main.py`.

---

## File Structure

```
artifacts/
├── backend/
│   ├── main.py          # Entry point: read config + handle SIGTERM + delegate to rtsp_server
│   └── rtsp_server.py    # RTSP capture / H264 encoding / socket server
├── lifecycle/
│   ├── install.sh        # Create venv and install requirements
│   └── start.sh          # Launch main.py
└── requirements.txt
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `rtsp_url` | `rtsp://192.168.110.123/stream2.264` | RTSP source URL |
| `socket_host` | `0.0.0.0` | Socket server bind address |
| `socket_port` | `9999` | Socket server port |
| `video_width` / `video_height` | `640` / `360` | Output resolution |
| `video_fps` | `24` | Output frame rate |
| `h264_bitrate` | `500000` | H264 bit rate |
| `gop_size` | `10` | GOP size |

Configuration values are set in the config panel of the deployment wizard (injected at runtime via `APP_CONFIG_PATH`) and override the `DEFAULTS` in `main.py`.

---

Local testing:

```bash
cd artifacts
APP_CONFIG_PATH=../sample_local_config.json python3 backend/main.py
```
