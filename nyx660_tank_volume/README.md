# NYX660 Tank Volume

A mostly-Python project for measuring **irregular contents** inside a tank or container using a **Vzense NYX660** on an **NVIDIA Jetson Nano**, with a remote HTTP API for calibration and measurements.

## What this project does

- Uses a fixed overhead depth camera view.
- Captures an **empty-tank baseline** so you do **not** have to hand-measure the tank contents during operation.
- Estimates fill volume by comparing the current depth map to the empty-tank baseline.
- Exposes a small **FastAPI** service so you can run it on the Jetson and control it remotely.
- Keeps nearly everything in Python, with only the vendor SDK bridge expected to need NYX660-specific patching.

## Architecture

1. Camera captures a depth frame.
2. `/calibrate` records an **empty tank** reference over multiple frames.
3. Each `/measure` call compares the current frame to the baseline.
4. Per-pixel height differences are integrated into an estimated volume.

That means the empty geometry becomes your ruler. Very lazy. Very efficient.

## Important assumptions

This project is designed for:

- A **fixed camera mount** above the container.
- The camera staying in the **same pose after calibration**.
- **Irregular solid contents** or uneven piles.
- A reasonably stable scene with minimal vibration.

This is **not** SLAM, freehand scanning, or “just yeet the camera somewhere and hope geometry happens.”

## Project layout

```text
nyx660_tank_volume/
  src/nyx660_tank_volume/
    api/
    camera/
    core/
    utils/
    config.py
    main.py
    cli.py
  scripts/
  systemd/
  config.example.yaml
  requirements.txt
  pyproject.toml
```

## Quick start on the Jetson Nano

### 1. Copy the project to the Jetson

Example target path:

```bash
/opt/nyx660_tank_volume
```

### 2. Install dependencies

On the Jetson:

```bash
cd /opt/nyx660_tank_volume
./scripts/install_jetson.sh
```

### 3. Configure it

Copy and edit:

```bash
cp config.example.yaml config.yaml
```

Update at least:

- `server.api_token`
- `camera.backend`
- `camera.intrinsics`
- `camera.crop` if you want to ignore areas outside the tank
- `measurement.known_volume_liters` if you want liters + fill ratio

## First run with the mock backend

Before touching the real camera, validate the stack:

```bash
cd /opt/nyx660_tank_volume
source .venv/bin/activate
cp config.example.yaml config.yaml
# leave camera.backend as mock
./scripts/test_mock.sh
```

If that works, the API and baseline math are fine.

## Running the API server

On the Jetson:

```bash
cd /opt/nyx660_tank_volume
source .venv/bin/activate
nyx660-server --config config.yaml
```

Default API:

- `GET /health`
- `POST /calibrate`
- `POST /measure`
- `GET /frame/depth.png`
- `GET /state`

## Remote usage from Windows PowerShell

Replace `JETSON_IP` and `YOUR_TOKEN`.

### Check health

```powershell
Invoke-RestMethod -Uri "http://JETSON_IP:8080/health" -Method Get
```

### Calibrate to an empty tank

Make sure the tank is empty and the camera is mounted exactly how it will stay.

```powershell
$headers = @{ "x-api-key" = "YOUR_TOKEN" }
$body = @{ frames = 30 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://JETSON_IP:8080/calibrate" -Method Post -Headers $headers -Body $body -ContentType "application/json"
```

### Trigger a measurement

```powershell
$headers = @{ "x-api-key" = "YOUR_TOKEN" }
Invoke-RestMethod -Uri "http://JETSON_IP:8080/measure" -Method Post -Headers $headers
```

### Save a depth preview image locally

```powershell
$headers = @{ "x-api-key" = "YOUR_TOKEN" }
Invoke-WebRequest -Uri "http://JETSON_IP:8080/frame/depth.png" -Headers $headers -OutFile ".\depth.png"
```

## Calibration workflow

1. Mount the NYX660 in its final position.
2. Ensure the container is **empty**.
3. Call `/calibrate`.
4. The service saves an `empty_tank_calibration.npz` baseline.
5. Later measurements compare against that baseline.

Because the baseline is the empty vessel shape, you do **not** need to measure every pile or liquid level manually.

## How volume estimation works

For each pixel:

- baseline depth = distance to the empty tank surface/floor
- current depth = distance to the current top surface of the contents
- fill height = `baseline_depth - current_depth`
- volume contribution = `fill_height * pixel_area`

Those contributions are summed across the valid tank footprint.

## About absolute liters vs baseline-only geometry

There are two modes:

### A. Baseline-only mode

If `measurement.known_volume_liters` is `null`, the API returns:

- estimated cubic meters from the baseline comparison
- occupied area
- average and max fill height

This works without entering a nominal tank capacity.

### B. Absolute liters + fill ratio

If you know the container’s usable capacity, set:

```yaml
measurement:
  known_volume_liters: 850
```

Then `/measure` also returns liters and relative fill ratio.

## Wiring in the real NYX660 SDK

The only intentionally incomplete file is:

```text
src/nyx660_tank_volume/camera/scepter_camera.py
```

That file is the vendor bridge. Everything else is already usable.

### What you need to patch

In `ScepterNyx660Camera.open()`:

- import the installed Scepter/Vzense Python SDK
- enumerate devices
- open the NYX660
- configure the depth stream if required

In `ScepterNyx660Camera.get_frame()`:

- acquire a depth frame from the SDK
- convert it to a `numpy.ndarray` with shape `(height, width)`
- convert depth units to meters
- return `DepthFrame(depth_m=..., timestamp_utc=...)`

### Why I left that seam explicit

Vzense has public SDKs and Python wrappers, but the exact module path and frame API can differ by SDK release. The repo is built so you only patch one file instead of setting your weekend on fire across the whole codebase.

## Getting better accuracy

- Crop to the tank interior.
- Use a rigid mount.
- Recalibrate if the camera moves.
- Increase `baseline_frames` and `smooth_frames`.
- Mask out walls and rim geometry with the crop box.
- Tune `fill_threshold_m` to suppress sensor noise.
- Add a known-capacity value if you want liters and fill ratio.

## Running as a service

A sample systemd unit is included:

```text
systemd/nyx660-tank-volume.service
```

Install example:

```bash
sudo cp systemd/nyx660-tank-volume.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nyx660-tank-volume
sudo systemctl start nyx660-tank-volume
sudo systemctl status nyx660-tank-volume
```

## Notes for Jetson Nano

- Keep resolution modest, like **320x240** or **640x480**, before trying to go full anime boss fight.
- Start with low FPS.
- The current algorithm uses NumPy only, which is Jetson-friendly compared with heavier 3D stacks.

## Known limitations

- The NYX660 SDK bridge still needs to be filled in on the Jetson.
- No pose correction is included; moving the camera after calibration breaks the baseline.
- Very reflective, transparent, or highly absorptive materials can hurt depth quality.
- Complex undercuts and overhangs are approximated from the visible top surface.

## Suggested next upgrades

- Add a web dashboard.
- Persist measurement history to SQLite.
- Add alarm thresholds.
- Add MQTT publishing.
- Add a tank polygon mask instead of simple rectangular cropping.
- Add temperature-compensated revalidation and drift detection.
