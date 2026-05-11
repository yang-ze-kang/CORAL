#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from os import path
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os

import click
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Optional dependencies:
# - tifffile for TIFF
# - imageio for PNG/JPG, etc.
# - scipy for gaussian blur (fallback included)
try:
    import tifffile
except Exception:
    tifffile = None  # type: ignore

try:
    import imageio.v3 as iio
except Exception:
    iio = None  # type: ignore

from scipy.ndimage import gaussian_filter


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def robust_percentile_clip_to_float01(x: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """
    Clip by percentiles and normalize to [0, 1] in float32.
    Works for 2D or 3D arrays.
    """
    x = x.astype(np.float32, copy=False)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate image, return zeros
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x


def mad_sigma(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Robust noise scale estimate via MAD:
    sigma ~= 1.4826 * median(|x - median(x)|)
    """
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * float(mad)
    return max(sigma, eps)


def read_image(path: Path) -> np.ndarray:
    """
    Read an image/volume.
    - .npy: any shape
    - .tif/.tiff: supports 2D or 3D stacks if tifffile is available
    - other common image formats: via imageio
    Returns a numpy array.
    """
    ext = path.suffix.lower()

    if ext == ".npy":
        return np.load(path)

    if ext in {".tif", ".tiff"}:
        if tifffile is None:
            raise RuntimeError("tifffile is required to read TIFF files. Please `pip install tifffile`.")
        return tifffile.imread(str(path))

    if iio is None:
        raise RuntimeError("imageio is required to read common image formats. Please `pip install imageio`.")
    return iio.imread(str(path))


def compute_snr_proxy(
    img: np.ndarray,
    clip_low: float,
    clip_high: float,
    blur_sigma: float,
    signal_p_low: float,
    signal_p_high: float,
) -> Tuple[float, float, float]:
    """
    Compute SNR proxy for one image/volume.

    Steps:
    1) Robust clip + normalize to [0,1]
    2) Blur => low-frequency signal estimate
    3) Residual = img - blur(img) => noise estimate (MAD->sigma)
    4) Signal amplitude = p_high - p_low (on blurred signal)
    5) SNR = amplitude / sigma_noise

    Returns:
      (snr, signal_amp, sigma_noise)
    """
    x = np.asarray(img)
    x01 = robust_percentile_clip_to_float01(x, clip_low, clip_high)

    # For huge 3D volumes, you might want to sample slices; here we compute on full array.
    xs = gaussian_filter(x01, sigma=blur_sigma)
    residual = x01 - xs

    sigma_n = mad_sigma(residual)
    sig_amp = float(np.percentile(xs, signal_p_high) - np.percentile(xs, signal_p_low))
    snr = sig_amp / sigma_n
    return snr, sig_amp, sigma_n


@dataclass(frozen=True)
class Item:
    path: Path
    split: str  # "train" or "val"


def worker_compute(path_str: str) -> Tuple[float, float, float]:
    # Read image inside subprocess to avoid pickling huge arrays.
    img = read_image(Path(path_str))
    snr, sig_amp, sigma_n = compute_snr_proxy(
        img,
        clip_low=1.0,
        clip_high=99.0,
        blur_sigma=1.0,
        signal_p_low=5.0,
        signal_p_high=95.0,
    )
    return snr, sig_amp, sigma_n


def main() -> None:
    """
    Compute SNR proxy for a folder of images and plot a histogram.
    Outputs:
      - snr_results.csv
      - snr_hist.png
    """
    dataset_dir = '/data2/C2/cubes1937'
    path1 = "/data1/yangzekang/neuron/neuron-trace/data_split/guolab-ch1-annos1723_tvt/train.txt"
    # path1 = "/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/data_split/C2-cubes1937_tvt/train.txt"
    ds_train = np.genfromtxt(path1, dtype=str, delimiter=",")
    train_paths = [Path(os.path.join(dataset_dir, p)) for p in ds_train[:, 0]]

    path2 = "/data1/yangzekang/neuron/neuron-trace/data_split/guolab-ch1-annos1723_tvt/val.txt"
    # path2 = "/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/data_split/C2-cubes1937_tvt/val.txt"
    ds_val = np.genfromtxt(path2, dtype=str, delimiter=",")
    val_paths = [Path(p) for p in ds_val[:, 0]]

    items: List[Item] = [Item(p, "train") for p in train_paths] + [Item(p, "val") for p in val_paths]

    results = []  # (path_str, split, snr, sig_amp, sigma_n)

    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_item = {
            executor.submit(worker_compute, str(it.path)): it
            for it in items
        }

        for fut in tqdm(as_completed(future_to_item), total=len(future_to_item)):
            it = future_to_item[fut]
            snr, sig_amp, sigma_n = fut.result()
            results.append((str(it.path), it.split, snr, sig_amp, sigma_n))

    if len(results) == 0:
        raise click.ClickException("All files failed to process.")

    # ---------- Choose threshold by median ----------
    snr_vals = np.array([r[2] for r in results], dtype=np.float32)
    thr = float(np.median(snr_vals))

    # Classify and count per split
    # low: snr < thr, high: snr >= thr
    counts = {
        "train": {"low": 0, "high": 0},
        "val": {"low": 0, "high": 0},
        "all": {"low": 0, "high": 0},
    }

    labeled_results = []
    for path_str, split, snr, sig_amp, sigma_n in results:
        cls = "low" if snr < thr else "high"
        counts[split][cls] += 1
        counts["all"][cls] += 1
        labeled_results.append((path_str, split, cls, snr, sig_amp, sigma_n))

    # ---------- Save CSV (with split + class) ----------
    csv_path = "snr_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("path,split,class,snr_proxy,signal_amp,sigma_noise\n")
        for path_str, split, cls, snr, sig_amp, sigma_n in labeled_results:
            f.write(f"\"{path_str}\",{split},{cls},{snr:.10f},{sig_amp:.10f},{sigma_n:.10f}\n")

    # ---------- Plot histogram + threshold line ----------
    plt.figure()
    plt.hist(snr_vals, bins=30)
    plt.axvline(thr, linestyle="--")
    plt.xlabel("SNR proxy")
    plt.ylabel("Count")
    plt.tight_layout()
    fig_path = "snr_hist.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # ---------- Print summary ----------
    click.echo(f"\nMedian threshold (thr) = {thr:.6f}")
    click.echo(f"Train: low={counts['train']['low']}  high={counts['train']['high']}  total={len(train_paths)}")
    click.echo(f"Val:   low={counts['val']['low']}    high={counts['val']['high']}    total={len(val_paths)}")
    click.echo(f"All:   low={counts['all']['low']}    high={counts['all']['high']}    total={len(items)}")

    click.echo(f"\nSaved CSV: {csv_path}")
    click.echo(f"Saved histogram: {fig_path}")
    click.echo(f"Processed: {len(results)} / {len(items)} files")


if __name__ == "__main__":
    main()