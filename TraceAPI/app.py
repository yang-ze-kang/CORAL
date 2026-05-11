from __future__ import annotations
import os
import signal
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable, Iterable, List, Optional
import tifffile as tiff
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import asyncio
import shutil
from datetime import datetime

from vaa3d_utils import *



# -----------------------------
# Config
# -----------------------------
ALLOWED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"

NEUTUBE_BIN = BASE_DIR / "algorithms" / "neuTube"
VAA3D_BIN = BASE_DIR / "algorithms" / "Vaa3D-x.1.1.4_Ubuntu" / "Vaa3D-x"


# -----------------------------
# Helpers
# -----------------------------
def ensure_executable(path: Path, name: str = "binary") -> None:
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"{name} not found: {path}")
    if not os.access(path, os.X_OK):
        raise HTTPException(status_code=500, detail=f"{name} not executable: {path}")


def cleanup_runs_worker(interval_hours: int = 0.5, max_age_hours: int = 1) -> None:
    """Periodically delete old run directories under RUNS_DIR."""
    while True:
        now = time.time()
        if RUNS_DIR.exists():
            for d in RUNS_DIR.iterdir():
                if not d.is_dir():
                    continue
                age_hours = (now - d.stat().st_mtime) / 3600
                if age_hours > max_age_hours:
                    print(f"[cleanup] removing {d} (age={age_hours:.2f}h)")
                    shutil.rmtree(d, ignore_errors=True)
        time.sleep(int(interval_hours * 3600))


@asynccontextmanager
async def lifespan(app: FastAPI):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    t = threading.Thread(target=cleanup_runs_worker, daemon=True)
    t.start()
    yield


app = FastAPI(lifespan=lifespan)


def make_workdir(max_retries: int = 1000) -> Path:
    base_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 到毫秒

    for i in range(max_retries):
        name = base_name if i == 0 else f"{base_name}_{i}"
        workdir = RUNS_DIR / name
        try:
            workdir.mkdir(parents=True, exist_ok=False)
            return workdir
        except FileExistsError:
            continue

    raise RuntimeError(f"Failed to create unique workdir after {max_retries} retries.")


async def save_upload_to(upload: UploadFile, dst: Path) -> None:
    """Save UploadFile to dst, then close upload."""
    try:
        with dst.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    finally:
        await upload.close()


def check_suffix(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")


def run_cmd(cmd: List[str], log_path: Path, workdir: Path, timeout_sec: int = 3600) -> None:
    """Run a command, writing stdout+stderr to log_path."""
    proc = None
    try:
        with log_path.open("wb") as logf:
            print("[run]", " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=str(workdir),
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return False

            if proc.returncode != 0:
                try:
                    text = log_path.read_text(errors="ignore")[-5000:]
                except Exception:
                    text = "(failed to read log)"
                raise HTTPException(status_code=500, detail=f"Command failed. Tail log:\n{text}")
    except subprocess.TimeoutExpired:
        # raise HTTPException(status_code=504, detail="Command timed out")
        return False
    except BaseException:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise
    return True


def file_response_or_500(path: Path, filename: str = "output.swc") -> FileResponse:
    if not path.exists() or path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail=f"{filename} not generated or empty")
    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=filename,
    )


async def prepare_trace_volume(file: UploadFile) -> tuple[Path, Path, Path, np.ndarray, int]:
    """Common preprocessing for tracing routes."""
    check_suffix(file.filename)
    ensure_executable(VAA3D_BIN, name="Vaa3D")

    workdir = make_workdir()
    local_input = workdir / "vol.tiff"
    local_log = workdir / "log.txt"
    await save_upload_to(file, local_input)

    img = tiff.imread(local_input)
    img_u8 = to_uint8_0_255(img)
    if img_u8.ndim != 3:
        raise HTTPException(status_code=400, detail=f"Expected 3D volume, got shape={img_u8.shape}")

    _, H, _ = img_u8.shape
    tif_file = workdir / "vol_uint8.tiff"
    tiff.imwrite(tif_file, img_u8)
    return workdir, local_input, local_log, img_u8, H


# -----------------------------
# Routes
# -----------------------------
@app.post("/trace_neutube")
async def trace_neutube(file: UploadFile = File(...)):
    check_suffix(file.filename)

    ensure_executable(NEUTUBE_BIN, name="neuTube")

    workdir = make_workdir()
    local_input = workdir / "vol.tiff"
    local_output = workdir / "output.swc"
    local_log = workdir / "log.txt"

    await save_upload_to(file, local_input)

    cmd = [
        str(NEUTUBE_BIN),
        "--command",
        str(local_input),
        "--trace",
        "-o",
        str(local_output),
        "--level",
        "0",
    ]
    run_cmd(cmd, local_log, workdir)

    return file_response_or_500(local_output, filename="output.swc")


@app.post("/trace_vaa3d_app2")
async def trace_vaa3d_app2(file: UploadFile = File(...)):
    workdir, _, local_log, img_u8, H = await prepare_trace_volume(file)
    tif_file = workdir / "vol_uint8.tiff"

    # Iterative tracing settings
    max_iters = 32                  # hard stop to avoid infinite loops
    timeout_sec = 3000              # total time budget for the iterative loop
    min_nodes_to_accept = 3         # too tiny outputs are often noise; tune as needed
    max_seed_tries_per_iter = 8

    def _run_app2(out_swc: Path, marker_file: Optional[Path] = None, timeout_sec: int = 3600) -> bool:
        marker_arg = str(marker_file) if marker_file is not None else "None"
        cmd = [
            str(VAA3D_BIN),
            "-x", "vn2",
            "-f", "app2",
            "-i", str(tif_file),
            "-o", str(out_swc),
            "-p",
            marker_arg,        # inmarker_file
            "0",               # channel
            "10",              # bkg_thresh
            "0",               # b_256cube
            "1",               # b_radiusFrom2D
            "0",               # is_gsdt
            "0",               # is_gap
            "5",               # length_thresh
            "1",               # is_resample
            "0",               # is_brightfield
            "0",               # is_high_intensity
        ]
        try:
            run_cmd(cmd, local_log, workdir, timeout_sec)
            if not out_swc.exists() or out_swc.stat().st_size == 0:
                return False
        except TraceTimeoutError:
            raise
        except:
            return False
        postprocess_vaa3d_result(out_swc, maxy=H)
        return True

    merged_swc = run_trace_iterative_with_seed_fallback(
        img_u8=img_u8,
        tif_file=tif_file,
        workdir=workdir,
        max_iters=max_iters,
        timeout_sec=timeout_sec,
        min_nodes_to_accept=min_nodes_to_accept,
        max_seed_tries_per_iter=max_seed_tries_per_iter,
        run_once=_run_app2,
        seed_prefix="app2_seed",
        error_label="APP2",
    )

    # merged_swc = run_trace_iterative_with_noise_mask(
    #     img_u8=img_u8,
    #     tif_file=tif_file,
    #     workdir=workdir,
    #     max_iters=max_iters,
    #     min_nodes_to_accept=min_nodes_to_accept,
    #     run_once=_run_app2,
    #     error_label="APP2",
    # )

    return file_response_or_500(merged_swc, filename="output.swc")


@app.post("/trace_vaa3d_smartTrace")
async def trace_vaa3d_smartTrace(file: UploadFile = File(...)):
    workdir, _, local_log, img_u8, H = await prepare_trace_volume(file)
    tif_file = workdir / "vol_uint8.tiff"

    # Iterative tracing settings
    max_iters = 16                  # hard stop to avoid infinite loops
    timeout_sec = 3000              # total time budget for the iterative loop
    min_nodes_to_accept = 3         # too tiny outputs are often noise; tune as needed
    cmd_swc_file = Path(str(tif_file) + "_smartTracing.swc")

    def _run_smarttrace(out_swc: Path, marker_file: Optional[Path] = None, timeout_sec: int = 2400) -> bool:
        cmd = [
            str(VAA3D_BIN),
            "-x", "smartTrace",
            "-f", "smartTrace",
            "-i", str(tif_file),
        ]
        try:
            ok = run_cmd(cmd, local_log, workdir, timeout_sec)
            if not ok:
                swc = Swc()
                swc.save_to_swc(cmd_swc_file)
                return False
            if not cmd_swc_file.exists() or cmd_swc_file.stat().st_size == 0:
                return False
        except TraceTimeoutError:
            raise
        except:
            return False
        cmd_swc_file.rename(out_swc)
        postprocess_vaa3d_result(out_swc, maxy=H)
        return True

    merged_swc = run_trace_iterative_with_noise_mask(
        img_u8=img_u8,
        tif_file=tif_file,
        workdir=workdir,
        max_iters=max_iters,
        timeout_sec=timeout_sec,
        min_nodes_to_accept=min_nodes_to_accept,
        run_once=_run_smarttrace,
        error_label="smartTrace",
    )

    return file_response_or_500(merged_swc, filename="output.swc")