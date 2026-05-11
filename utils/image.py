from scipy.ndimage import gaussian_filter
import numpy as np
from typing import Tuple


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


def compute_snr_proxy(
    img: np.ndarray,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    blur_sigma: float = 1.0,
    signal_p_low: float = 5.0,
    signal_p_high: float = 95.0,
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
    return snr