# src/mtf_plus.py
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# use your vendored file
from third_party import mtf as mtf_orig

@dataclass
class MtfMetrics:
    freq_cyc_per_pix: np.ndarray
    mtf: np.ndarray
    mtf50_cyc_per_pix: float
    mtf10_cyc_per_pix: float
    mtf50_lp_per_mm: float | None
    mtf10_lp_per_mm: float | None
    nyquist_percent: float
    transition_width_pix: float

def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    y = np.clip(y, 0.0, 1.1)
    for i in range(1, len(y)):
        if y[i-1] >= level and y[i] <= level:
            t = (level - y[i-1]) / ((y[i] - y[i-1]) + 1e-12)
            return float(x[i-1] + t * (x[i] - x[i-1]))
    return float("nan")

def calculate_from_array(arr01: np.ndarray, pixel_pitch_um: float | None = None,
                         gamma: float | None = None, verbose: int = 0) -> MtfMetrics:
    a = np.asarray(arr01, dtype=float)
    a = np.clip(a, 0.0, 1.0)
    if gamma:
        a = np.clip(a, 1e-8, 1.0) ** float(gamma)

    res = mtf_orig.MTF.CalculateMtf(
        a,
        verbose=(mtf_orig.Verbosity.NONE if verbose == 0
                 else mtf_orig.Verbosity.BRIEF if verbose == 1
                 else mtf_orig.Verbosity.DETAIL)
    )

    x = np.array(res.x, dtype=float)         # 0..1 cycles/pixel (Nyquist=0.5)
    y = np.array(res.y, dtype=float)
    mtf50 = _crossing(x, y, 0.5)
    mtf10 = _crossing(x, y, 0.1)

    conv = (1000.0 / float(pixel_pitch_um)) if pixel_pitch_um else None
    mtf50_lpmm = (mtf50 * conv) if conv and np.isfinite(mtf50) else None
    mtf10_lpmm = (mtf10 * conv) if conv and np.isfinite(mtf10) else None

    return MtfMetrics(
        freq_cyc_per_pix=x,
        mtf=y,
        mtf50_cyc_per_pix=mtf50,
        mtf10_cyc_per_pix=mtf10,
        mtf50_lp_per_mm=mtf50_lpmm,
        mtf10_lp_per_mm=mtf10_lpmm,
        nyquist_percent=float(res.mtfAtNyquist),
        transition_width_pix=float(res.width),
    )

def calculate_from_path(path: str | Path, **kwargs) -> MtfMetrics:
    arr = mtf_orig.Helper.LoadImageAsArray(str(path))
    # Helper returns 0..1 already; pass through
    return calculate_from_array(arr, **kwargs)

