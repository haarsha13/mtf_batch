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
    
# ========================= New plotting + batch utilities =========================
import math
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

def _to_gray01(arr: np.ndarray) -> np.ndarray:
    """Convert input image array to grayscale float64 in [0,1]."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 3 and a.shape[2] >= 3:
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        a = 0.2126 * r + 0.7152 * g + 0.0722 * b
    a = np.clip(a, 0.0, 1.0)
    return a.astype(np.float64, copy=False)

def _edge_angle_deg(gray: np.ndarray) -> float:
    """Estimate dominant edge angle (degrees, 0=+x axis) using PCA of strong gradients."""
    # Simple Sobel-like kernels
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)
    ky = kx.T
    # Convolve (valid-ish via padding)
    pad = 1
    gpad = np.pad(gray, pad, mode="edge")
    H, W = gray.shape
    gx = np.empty_like(gray)
    gy = np.empty_like(gray)
    for y in range(H):
        ys = slice(y, y+3)
        for x in range(W):
            xs = slice(x, x+3)
            tile = gpad[ys, xs]
            gx[y, x] = float((tile * kx).sum())
            gy[y, x] = float((tile * ky).sum())
    mag = np.hypot(gx, gy)
    # pick top 2% gradient pixels (at least 200)
    flat = mag.ravel()
    k = max(200, int(flat.size * 0.02))
    if k >= flat.size:
        k = max(100, flat.size // 10)
    thresh = np.partition(flat, -k)[-k]
    ys, xs = np.nonzero(mag >= thresh)
    if xs.size < 5:
        return 0.0
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    pts -= pts.mean(axis=0, keepdims=True)
    cov = (pts.T @ pts) / max(1, pts.shape[0]-1)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, np.argmax(evals)]  # along-edge direction
    angle = math.degrees(math.atan2(v[1], v[0]))
    return float(angle)

def _otsu_threshold(gray: np.ndarray) -> float:
    """Otsu threshold on [0,1] grayscale."""
    g = np.clip(gray, 0.0, 1.0)
    hist, bin_edges = np.histogram(g.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(float)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.0))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1.0 - omega) + 1e-12)
    idx = np.nanargmax(sigma_b2)
    return float(bin_edges[idx])

def _michelson_contrast(gray: np.ndarray) -> float:
    t = _otsu_threshold(gray)
    dark = gray[gray <= t]
    bright = gray[gray > t]
    if dark.size < 5 or bright.size < 5:
        return 0.0
    Imin = float(dark.mean())
    Imax = float(bright.mean())
    if Imax + Imin == 0:
        return 0.0
    return float((Imax - Imin) / (Imax + Imin))

def _vertical_profile(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pixels, linear edge profile) averaged across x."""
    prof = gray.mean(axis=1)  # average along columns -> function of y (vertical)
    ypix = np.arange(len(prof), dtype=float)
    # Normalize profile to [0,1] for plotting only
    p_min, p_max = float(prof.min()), float(prof.max())
    if p_max > p_min:
        prof_n = (prof - p_min) / (p_max - p_min)
    else:
        prof_n = prof * 0.0
    return ypix, prof_n

def analyze_and_plot(image_path: str | Path,
                     out_dir: str | Path | None = None,
                     pixel_pitch_um: float | None = None,
                     gamma: float | None = None,
                     dpi: int = 160) -> Path:
    """Create a 3-panel report for a PNG (MTF, Edge Profile, Thumbnail).

    Saves a PNG named '<stem>_report.png' into out_dir (or alongside input).
    Returns the output path.
    """
    image_path = Path(image_path)
    arr01 = mtf_orig.Helper.LoadImageAsArray(str(image_path))
    gray = _to_gray01(arr01)
    H, W = gray.shape

    # Metrics
    metrics = calculate_from_array(arr01, pixel_pitch_um=pixel_pitch_um, gamma=gamma, verbose=0)
    angle_deg = _edge_angle_deg(gray)
    contrast = _michelson_contrast(gray)

    # Edge profile (vertical)
    ypix, prof = _vertical_profile(gray)

    # Figure
    fig = plt.figure(figsize=(9, 6), dpi=dpi)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1.0], height_ratios=[1.0, 1.2])
    ax_profile = fig.add_subplot(gs[0, 0])
    ax_thumb   = fig.add_subplot(gs[0, 1])
    ax_mtf     = fig.add_subplot(gs[1, 0])

    # Subplot 1: MTF vs frequency (cycles/pixel)
    ax_mtf.plot(metrics.freq_cyc_per_pix, metrics.mtf, linewidth=1.5)
    ax_mtf.axvline(0.5, linestyle="--", linewidth=1.0)
    ax_mtf.set_xlim(0.0, 1.0)
    ax_mtf.set_ylim(0.0, 1.05)
    ax_mtf.set_xlabel("Frequency, Cycles/pixel")
    ax_mtf.set_ylabel("MTF")
    ax_mtf.set_title("MTF")

    # Subplot 2: Edge profile (linear) by Pixels (Ver)
    ax_profile.plot(ypix, prof, linewidth=1.5)
    ax_profile.set_xlabel("Pixels (Ver)")
    ax_profile.set_ylabel("Edge profile (linear)")
    ax_profile.set_title("Edge profile: Vertical")

    # Subplot 3: Thumbnail (actual image)
    ax_thumb.imshow(np.squeeze(arr01), origin="upper", interpolation="nearest")
    ax_thumb.set_xticks([]); ax_thumb.set_yticks([])
    ax_thumb.set_title("ROI / Image")

    # Info panel (printed facts)
    info = [
        f"Edge angle: {angle_deg:.2f}°",
        f"Est. chart contrast: {contrast*100:.1f}%",
        f"Image width: {W} px",
        f"Image height: {H} px",
    ]
    fig.suptitle(image_path.name, y=0.98)
    fig.text(0.70, 0.50, "\n".join(info), ha="left", va="top")

    # Clean empty axes if any
    for (i, j) in [(0, 2), (1, 1), (1, 2)]:
        ax = fig.add_subplot(gs[i, j])
        ax.axis("off")

    if out_dir is None:
        out_dir = image_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_report.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def batch_analyze(input_path: str | Path,
                  out_dir: str | Path | None = None,
                  pattern: str = "*.png",
                  **kwargs) -> list[Path]:
    """Process a single file or all PNGs in a folder and save report images."""
    input_path = Path(input_path)
    outputs: list[Path] = []
    if input_path.is_dir():
        files = sorted(input_path.glob(pattern))
    else:
        files = [input_path]
    if out_dir is None:
        out_dir = (input_path if input_path.is_dir() else input_path.parent) / "reports"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            outputs.append(analyze_and_plot(f, out_dir=out_dir, **kwargs))
        except Exception as e:
            # still continue
            err_path = out_dir / f"{f.stem}_ERROR.txt"
            err_path.write_text(str(e))
    return outputs
# ======================= End plotting + batch utilities =======================

