# TraceAPI

TraceAPI is a FastAPI-based service for tracing neurons from 3D image volumes. It accepts an uploaded volume, runs local tracing binaries such as neuTube or Vaa3D, and returns the generated SWC file.

The service currently exposes three tracing endpoints:

- `POST /trace_neutube`: runs `algorithms/neuTube`.
- `POST /trace_vaa3d_app2`: runs Vaa3D `vn2/app2` with iterative tracing, candidate seed fallback, SWC merging, and coordinate post-processing.
- `POST /trace_vaa3d_smartTrace`: runs Vaa3D `smartTrace` with iterative tracing, foreground denoising, SWC merging, and coordinate post-processing.

## Project Layout

```text
TraceAPI/
├── app.py                         # FastAPI application and endpoint definitions
├── vaa3d_utils.py                 # Vaa3D helpers, timeout handling, seed generation, SWC merging/post-processing
├── command.sh                     # Startup and manual testing command notes
├── algorithms/
│   ├── neuTube                    # neuTube executable
│   ├── Vaa3D-x.1.1.4_Ubuntu/
│   │   └── Vaa3D-x                # Vaa3D executable
│   └── lib*.so                    # Shared libraries required by Vaa3D
└── runs/                          # Temporary per-request work directories, created at runtime
```

## Requirements

The Python dependencies below are inferred from the current imports:

```bash
pip install fastapi uvicorn python-multipart numpy scipy tifffile

git clone https://github.com/yang-ze-kang/swclib.git
cd swclib
pip install -e .
```

When running Vaa3D on a headless server, Xvfb and Qt/OpenGL libraries are usually required:

```bash
sudo apt-get install -y xvfb libglu1-mesa \
  libqt5widgets5 libqt5gui5 libqt5xml5 libqt5network5 \
  libqt5core5a libqt5concurrent5
```

If Vaa3D cannot find its shared libraries, add `algorithms/` to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$(pwd)/algorithms:$LD_LIBRARY_PATH
```

## Start the Service

If you plan to use the Vaa3D endpoints, start a virtual display first:

```bash
Xvfb :99 -screen 0 1920x1080x24 -ac &
export DISPLAY=:99
```

Start the FastAPI server:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 32
```

After startup, the API is available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## API Usage

All endpoints accept `multipart/form-data` uploads with the field name `file`. Supported file extensions are:

- `.tif`
- `.tiff`
- `.png`
- `.jpg`
- `.jpeg`

The Vaa3D endpoints read the input with `tifffile`, normalize it to `uint8 [0, 255]`, and expect a 3D volume. A non-3D input returns a 400 error.

### 1. neuTube Tracing

```bash
curl -X POST "http://127.0.0.1:8000/trace_neutube" \
  -F "file=@/path/to/volume.tif" \
  --output output_neutube.swc
```

Internal command pattern:

```bash
algorithms/neuTube --command vol.tiff --trace -o output.swc --level 0
```

### 2. Vaa3D APP2 Tracing

```bash
curl -X POST "http://127.0.0.1:8000/trace_vaa3d_app2" \
  -F "file=@/path/to/volume.tif" \
  --output output_app2.swc
```

This endpoint:

1. Normalizes the input image to `uint8`.
2. Runs Vaa3D `vn2/app2`.
3. Falls back to candidate seeds when automatic tracing produces no result.
4. Masks out traced regions after each iteration and continues tracing the remaining foreground.
5. Merges all accepted SWC fragments and fixes the Vaa3D Y-axis coordinate direction.

Key parameters in `app.py`:

```python
max_iters = 32
timeout_sec = 3000
min_nodes_to_accept = 3
max_seed_tries_per_iter = 8
```

### 3. Vaa3D smartTrace Tracing

```bash
curl -X POST "http://127.0.0.1:8000/trace_vaa3d_smartTrace" \
  -F "file=@/path/to/volume.tif" \
  --output output_smartTrace.swc
```

This endpoint:

1. Normalizes the input image to `uint8`.
2. Runs Vaa3D `smartTrace/smartTrace`.
3. Retries after removing small compact foreground noise if the first run produces no trace.
4. Masks out traced regions after each iteration and continues tracing the remaining foreground.
5. Merges all accepted SWC fragments and fixes the Vaa3D Y-axis coordinate direction.

Key parameters in `app.py`:

```python
max_iters = 16
timeout_sec = 3000
min_nodes_to_accept = 3
```

## Temporary Files and Cleanup

Each request creates a timestamped work directory under `runs/`, for example:

```text
runs/20260511_143012_123/
```

A work directory may contain:

- `vol.tiff`: uploaded input file.
- `vol_uint8.tiff`: normalized input file.
- `output_*.swc`: per-iteration tracing results.
- `output.swc`: final result returned by the API.
- `*.marker`: marker files used by APP2 seeded tracing.
- `log.txt`: stdout and stderr from external tracing commands.

On startup, the service creates a background cleanup thread. By default, it scans every 0.5 hours and removes run directories older than 1 hour.

## Troubleshooting

### 500: Vaa3D not found or neuTube not found

Check that the executables exist:

```bash
ls -l algorithms/neuTube
ls -l algorithms/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x
```

If they are not executable:

```bash
chmod +x algorithms/neuTube
chmod +x algorithms/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x
```

### Vaa3D fails with Qt/OpenGL/display errors

Make sure a virtual display is available:

```bash
Xvfb :99 -screen 0 1920x1080x24 -ac &
export DISPLAY=:99
```

Also verify that the Qt/OpenGL dependencies and `LD_LIBRARY_PATH` are configured correctly.

### 400: Expected 3D volume

The Vaa3D endpoints require `tifffile.imread()` to return a 3D array. Check that the input file is a 3D TIFF volume rather than a single 2D image.

### 422: APP2 failed for auto and candidate seeds

APP2 could not produce a valid SWC from either automatic tracing or candidate seed tracing. Inspect the corresponding `runs/<timestamp>/log.txt`, or tune the thresholds and iteration settings in `app.py`.

## Development Notes

- `NEUTUBE_BIN` and `VAA3D_BIN` in `app.py` are resolved relative to the `TraceAPI` directory.
- External command output is written to `log.txt` in each request work directory.
- `run_cmd()` uses a default per-command timeout of 3600 seconds. The Vaa3D iterative endpoints also use a total iteration time budget.
- `postprocess_vaa3d_result()` writes node type as `0` and converts Y coordinates with `maxy - 1 - y`.
