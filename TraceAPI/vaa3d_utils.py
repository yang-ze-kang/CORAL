import numpy as np
import subprocess
import tifffile as tiff
import time
import signal
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional, Callable
from fastapi import HTTPException
import scipy.ndimage as ndi

from swclib.data.swc import Swc, merge_swcs
from swclib.image.swc2mask import swc_to_mask_sphere_cone


class TraceTimeoutError(TimeoutError):
    pass


@contextmanager
def timeout(timeout_sec: int):
    deadline = time.monotonic() + timeout_sec

    def remaining_timeout() -> int:
        return max(1, int(deadline - time.monotonic()))

    if threading.current_thread() is not threading.main_thread():
        yield remaining_timeout
        return

    def handle_timeout(signum, frame):
        raise TraceTimeoutError(f"iterative tracing reached timeout_sec={timeout_sec}")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        yield remaining_timeout
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def to_uint8_0_255(arr: np.ndarray) -> np.ndarray:
    """
    Convert input array to uint8 in [0, 255].
    """
    arr = np.asarray(arr)

    # General min-max normalization
    arr_f = arr.astype(np.float32, copy=False)
    mn = float(np.min(arr_f))
    mx = float(np.max(arr_f))
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr_f - mn) * (255.0 / (mx - mn))
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out

def build_neutube_like_seed_pool(cur_img_u8: np.ndarray) -> List[tuple[float, float, float]]:
    """
    Build multiple candidate seeds from local maxima of distance transform,
    approximating neuTube's "multiple candidate seed" behavior.
    """
    seeds: List[tuple[float, float, float]] = []
    D, H, W = cur_img_u8.shape
    try:
        import scipy.ndimage as ndi  # type: ignore
        nz = cur_img_u8[cur_img_u8 > 0]
        if nz.size == 0:
            return seeds

        # Use a lower threshold so neurites are included, then remove small blob noise.
        th = float(np.percentile(nz, 85.0))
        th = max(th, 2.0)
        fg = cur_img_u8 >= th
        if np.count_nonzero(fg) == 0:
            fg = cur_img_u8 > 0

        # Connected-component filtering: remove tiny compact blobs (typical triangle noise).
        lbl, num = ndi.label(fg)
        if num > 0:
            obj_slices = ndi.find_objects(lbl)
            clean = np.zeros_like(fg, dtype=bool)
            for cid in range(1, num + 1):
                slc = obj_slices[cid - 1]
                if slc is None:
                    continue
                comp = (lbl[slc] == cid)
                vox = int(comp.sum())
                dz = slc[0].stop - slc[0].start
                dy = slc[1].stop - slc[1].start
                dx = slc[2].stop - slc[2].start
                # Keep if component is not tiny, or has elongated geometry.
                # This suppresses repeated small bright triangles.
                longest = max(dx, dy, dz)
                shortest = max(1, min(dx, dy, dz))
                elong = float(longest) / float(shortest)
                if vox >= 30 or (vox >= 10 and elong >= 3.0):
                    clean[slc] |= comp
            fg = clean

        if np.count_nonzero(fg) == 0:
            fg = cur_img_u8 >= max(2, int(np.percentile(nz, 95.0)))

        dist = ndi.distance_transform_edt(fg)
        local_max = dist == ndi.maximum_filter(dist, size=(5, 5, 5))
        cand = np.argwhere(local_max & (dist >= 1.0))
        use_intensity_fallback = False
        if cand.shape[0] == 0:
            # Thin fibers can have very small DT; fallback to local maxima on intensity.
            use_intensity_fallback = True
            imax = cur_img_u8 == ndi.maximum_filter(cur_img_u8, size=(5, 5, 5))
            cand = np.argwhere(imax & fg & (cur_img_u8 >= th))
        if cand.shape[0] > 0:
            # Prefer maxima that are center-like (DT) and bright enough locally.
            scores = []
            for z, y, x in cand:
                z0, z1 = max(0, z - 1), min(D, z + 2)
                y0, y1 = max(0, y - 2), min(H, y + 3)
                x0, x1 = max(0, x - 2), min(W, x + 3)
                local_mean = float(cur_img_u8[z0:z1, y0:y1, x0:x1].mean())
                if use_intensity_fallback:
                    s = (float(cur_img_u8[z, y, x]) / 255.0) + 0.2 * (local_mean / 255.0)
                else:
                    s = float(dist[z, y, x]) * (1.0 + local_mean / 255.0)
                scores.append(s)
            scores = np.asarray(scores, dtype=np.float32)
            order = np.argsort(scores)[::-1]
            for i in order[:1024]:
                z, y, x = cand[i]
                seeds.append((float(x), float(y), float(z)))
    except Exception:
        # Fallback: intensity peaks if scipy is unavailable.
        coords = np.argwhere(cur_img_u8 >= np.percentile(cur_img_u8, 99.5))
        if coords.shape[0] > 0:
            vals = cur_img_u8[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
            order = np.argsort(vals)[::-1]
            for i in order[:1024]:
                z, y, x = coords[i]
                seeds.append((float(x), float(y), float(z)))

    # Deduplicate by coarse voxel bin to avoid too dense nearby seeds.
    dedup: List[tuple[float, float, float]] = []
    seen = set()
    for x, y, z in seeds:
        key = (int(x // 3), int(y // 3), int(z // 2))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((x, y, z))
        if len(dedup) >= 512:
            break
    return dedup

def write_marker_file(marker_path: Path, seed_xyz: tuple[float, float, float], D: float, H: float, W: float) -> None:
    x, y, z = seed_xyz
    x1 = min(max(x + 1.0, 1.0), float(W))
    y1 = min(max(float(H) - y, 1.0), float(H))
    z1 = min(max(z + 1.0, 1.0), float(D))
    with marker_path.open("w") as f:
        f.write(f"{x1:.3f},{y1:.3f},{z1:.3f},1,1,seed,0,255,0,0\n")


def denoise_foreground_components_inplace(img_u8: np.ndarray) -> bool:
    """
    Remove tiny compact connected components from non-zero foreground in-place.
    Return True if denoising is applied successfully.
    """
    fg = img_u8 > 0

    lbl, num = ndi.label(fg)
    if num <= 0:
        return False

    obj_slices = ndi.find_objects(lbl)
    clean = np.zeros_like(fg, dtype=bool)
    for cid in range(1, num + 1):
        slc = obj_slices[cid - 1]
        if slc is None:
            continue
        comp = (lbl[slc] == cid)
        vox = int(comp.sum())
        dz = slc[0].stop - slc[0].start
        dy = slc[1].stop - slc[1].start
        dx = slc[2].stop - slc[2].start
        longest = max(dx, dy, dz)
        shortest = max(1, min(dx, dy, dz))
        elong = float(longest) / float(shortest)
        if vox >= 50 or (vox >= 15 and elong >= 3.0 and longest >= 5):
            clean[slc] |= comp

    img_u8[~clean] = np.uint8(0)
    fg_after = np.count_nonzero(img_u8 > 0)
    if fg_after == 0 or np.sum(fg) == np.sum(fg_after):
        return False
    return True


def run_trace_iterative_with_seed_fallback(
    *,
    img_u8: np.ndarray,
    tif_file: Path,
    workdir: Path,
    max_iters: int,
    timeout_sec: int,
    min_nodes_to_accept: int,
    max_seed_tries_per_iter: int,
    run_once: Callable[[Path, Optional[Path], int], bool],
    seed_prefix: str,
    error_label: str,
) -> Path:
    D, H, W = img_u8.shape
    swcs: List[Swc] = []

    seed_mode = False
    try:
        with timeout(timeout_sec) as remaining_timeout:
            for it in range(max_iters):
                tiff.imwrite(tif_file, img_u8)
                swc_file = workdir / f"output_{it:03d}.swc"
                node_num = 0

                if not seed_mode:
                    ok = run_once(swc_file, None, remaining_timeout())
                    swc = Swc(swc_file) if ok else Swc()
                    node_num = len(swc.nodes)

                    if node_num == 0 and int(np.count_nonzero(img_u8 > 0)) > 20:
                        seed_mode = True
                        seed_pool = build_neutube_like_seed_pool(img_u8)

                if seed_mode and seed_pool:
                    seed_iter = 0
                    while len(seed_pool)!=0 and seed_iter < max_seed_tries_per_iter:
                        seed_iter += 1
                        seed_xyz = seed_pool.pop(0)
                        marker_file = workdir / f"{seed_prefix}_{it:03d}.marker"
                        swc_file = workdir / f"output_{it:03d}_seeded.swc"
                        write_marker_file(marker_file, seed_xyz, D, H, W)
                        ok = run_once(swc_file, marker_file, remaining_timeout())
                        swc = Swc(swc_file) if ok else Swc()
                        node_num = len(swc.nodes)
                        if node_num > 0:
                            break

                if node_num==0:
                    break

                if node_num >= min_nodes_to_accept and swc.length >= 3.0:
                    swcs.append(swc)

                # Mask traced tree out, then continue.
                if node_num > 0:
                    mask = swc_to_mask_sphere_cone(
                        swc_file,
                        shape=(D, H, W),
                        foreground_value=1,
                        r_scale=3.0,
                    )
                    img_u8[mask > 0] = np.uint8(0)

                # Remove candidate seeds already covered by traced region.
                if seed_mode:
                    kept = []
                    for x, y, z in seed_pool:
                        xi = int(round(x))
                        yi = int(round(y))
                        zi = int(round(z))
                        if 0 <= zi < D and 0 <= yi < H and 0 <= xi < W and mask[zi, yi, xi] > 0:
                            continue
                        kept.append((x, y, z))
                    seed_pool = kept
    except TraceTimeoutError:
        print(f"{error_label}: iterative tracing reached timeout_sec={timeout_sec}, stopping.")

    if not swcs:
        raise HTTPException(status_code=422, detail=f"{error_label} failed for auto and candidate seeds.")

    merged_swc = workdir / "output.swc"
    swc_merged = merge_swcs(swcs)
    swc_merged.save_to_swc(merged_swc)
    return merged_swc


def run_trace_iterative_with_noise_mask(
    *,
    img_u8: np.ndarray,
    tif_file: Path,
    workdir: Path,
    max_iters: int,
    timeout_sec: int,
    min_nodes_to_accept: int,
    run_once: Callable[[Path, Optional[Path], int], bool],
    error_label: str,
) -> Path:
    D, H, W = img_u8.shape
    swcs: List[Swc] = []

    denoise_flag = False
    try:
        with timeout(timeout_sec) as remaining_timeout:
            for it in range(max_iters):
                tiff.imwrite(tif_file, img_u8)
                swc_file = workdir / f"output_{it:03d}.swc"

                ok = run_once(swc_file, None, remaining_timeout())
                swc = Swc(swc_file) if ok else Swc()
                node_num = len(swc.nodes)

                # If still no trace but foreground is non-trivial, denoise and retry next iteration.
                if node_num == 0:
                    if not denoise_flag and int(np.count_nonzero(img_u8 > 0)) > 10:
                        denoise_flag=True
                        denoise_foreground_components_inplace(img_u8)
                        print(f"Iteration {it}: Denoised foreground components, retrying.")
                        tiff.imwrite(tif_file, img_u8)
                        ok = run_once(swc_file, None, remaining_timeout())
                        swc = Swc(swc_file) if ok else Swc()
                        node_num = len(swc.nodes)
                    else:
                        break

                if node_num == 0 or not ok:
                    break

                if node_num >= min_nodes_to_accept and swc.length > 3.0:
                    swcs.append(swc)

                # Mask traced tree out, then continue.
                mask = swc_to_mask_sphere_cone(
                    swc_file,
                    shape=(D, H, W),
                    foreground_value=1,
                    r_scale=3.0,
                )
                img_u8[mask > 0] = np.uint8(0)
    except TraceTimeoutError:
        print(f"{error_label}: iterative tracing reached timeout_sec={timeout_sec}, stopping.")

    if len(swcs) == 0:
        swc_merged = Swc()
    else:
        swc_merged = merge_swcs(swcs)
    merged_swc = workdir / "output.swc"
    swc_merged.save_to_swc(merged_swc)
    return merged_swc


def postprocess_vaa3d_result(swc_path, maxy=300):
    res = []
    with open(swc_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                res.append(line)
                continue
            nid, ntype, x, y ,z, r, pid = line.strip().split()
            res.append([
                nid,"0",
                "{:.6f}".format(float(x)),
                "{:.6f}".format(maxy - 1 - float(y)),
                "{:.6f}".format(float(z)),
                "{:.6f}".format(float(r)),
                pid
            ])
    with open(swc_path, "w") as f:
        for line in res:
            if isinstance(line, str):
                f.write(line)
            else:
                f.write(" ".join(line) + "\n")
