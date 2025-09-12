# slice_and_mtf.py
# End-to-end: find slant-edge patches (HyperTarget) -> save patches -> run MTF on each patch.
from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import cv2
import json
import sys
import traceback

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

# ---------------- CONFIG (EDIT THESE) ----------------
# The folder that contains images to be sliced, works with subfolders too:
INPUT = r"/Users/haarshakrishna/Documents/PHYS3810/SN006_ThroughFocusImages_Pt1"

# Folder to write patches and results to:
PATCH_OUT_BASE = r"/Users/haarshakrishna/Documents/GitHub/mtf_batch/outputs"

# Your local paths to the MTF and HyperTarget modules:
MTF_MODULE_PATH = r"/Users/haarshakrishna/Documents/GitHub/mtf_batch/src/mtf_batch/MTF_HD.py"
HYPER_MODULE_PATH = r"/Users/haarshakrishna/Documents/GitHub/mtf_batch/src/third_party/hypertarget.py"

# Uploaded images must be in .PNG format
FILENAME_GLOB = "*.png"

# Keep sub-folder hierarchy in output? If INPUT directory contains subfolders, sort by subfolder name?
ORGANIZE_BY_SUB = True

# Patch controls
PATCH_SIZE = 256          # square crop (pixels) around each slant center
MAX_PATCHES = None        # e.g., 12 to limit per image; None = all

# MTF controls
FRACTION = 0.5            # 0.5 = MTF50 (what fraction of MTF to report)
WRITE_FIGURES = True      # save MTF plot per patch? 
SUMMARY_CSV = "mtf_summary_all.csv"  # written into PATCH_OUT_BASE, and is the summary of all data. 
# -------------- END CONFIG --------------------------


# ---------- helpers ----------
def _load_module_from_path(path_str: str, required=()):
    """Dynamic import of a module by full path."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"MODULE_PATH not found: {p}")
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)   # type: ignore[attr-defined]
    for name in required:
        if not hasattr(mod, name):
            raise AttributeError(f"Loaded '{p.name}' but missing attribute: {name}")
    return mod

def _collect_images(root: Path, pattern: str):
    """Recursively collect images by glob pattern (e.g., '*.png')."""
    return sorted(root.rglob(pattern))

def _read_gray01(path: Path) -> np.ndarray:
    """Read grayscale float image in [0,1]."""
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    im = im.astype(np.float32) / 255.0
    return im

def _safe_patch(arr: np.ndarray, center_xy: np.ndarray, size: int) -> np.ndarray:
    """Square crop centered at (x, y); pad if near borders to keep exact size."""
    h, w = arr.shape[:2]
    x_c, y_c = int(center_xy[0]), int(center_xy[1])   # ht.slant_ifieldsXY is [x, y]
    half = size // 2
    x0, x1 = max(0, x_c - half), min(w, x_c + half)
    y0, y1 = max(0, y_c - half), min(h, y_c + half)
    patch = arr[y0:y1, x0:x1]
    # pad to exact size if truncated by edges
    pad_y = size - patch.shape[0]
    pad_x = size - patch.shape[1]
    if pad_y > 0 or pad_x > 0:
        patch = np.pad(patch, ((0, max(0, pad_y)), (0, max(0, pad_x))), mode="edge")
        patch = patch[:size, :size]
    return patch

# ---------- main processing ----------
def process_image(rs_mtf, rs_hyp, src_path: Path, base_out: Path) -> list[dict]:
    """Slice one image into patches, save them, run MTF on each. Return list of CSV rows."""
    rows: list[dict] = []
    src_stem = src_path.stem

    # keep SNxxx hierarchy if enabled
    if ORGANIZE_BY_SUB:
        parent_name = src_path.parent.name  # e.g., SN005
        img_out_dir = base_out / parent_name / src_stem
    else:
        img_out_dir = base_out / src_stem

    img_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load grayscale
    arr = _read_gray01(src_path)

    # 2) detect slant-edge patch centers
    ht = rs_hyp.HyperTarget(arr, plot_slant=False, plot_patches=False)
    centers = getattr(ht, "slant_ifieldsXY", None)

    # keep some metadata for sanity
    meta = {
        "source_file": src_path.name,
        "image_shape": list(arr.shape),
        "slant_angle_deg": float(getattr(ht, "slant_angle_deg", np.nan)),
        "n_slant_centers": 0 if centers is None else int(len(centers)),
        "image_square_npix": int(getattr(ht, "image_square_npix", 0)),
    }
    with open(img_out_dir / "hypertarget_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if centers is None or len(centers) == 0:
        print(f"[WARN] No slant centers found in {src_path.name} — running MTF on full frame.")
        centers = np.array([[arr.shape[1]//2, arr.shape[0]//2]], dtype=int)

    # 3) optionally limit number of patches
    n = len(centers) if MAX_PATCHES is None else min(len(centers), MAX_PATCHES)

    # 4) per-image manifest of patches found
    manifest = []

    for i in range(n):
        cxy = centers[i]
        patch = _safe_patch(arr, cxy, PATCH_SIZE)
        patch_name = f"{src_stem}_patch{i:02d}.png"
        patch_path = img_out_dir / patch_name

        # save patch as 8-bit PNG
        cv2.imwrite(str(patch_path), (np.clip(patch, 0, 1) * 255).astype(np.uint8))

        manifest.append({
            "patch_idx": i,
            "x_pix": int(cxy[0]),
            "y_pix": int(cxy[1]),
            "patch_file": patch_name
        })

        # 5) run MTF on this patch
        try:
            verbosity = getattr(getattr(rs_mtf, "Verbosity", None), "DETAIL", 0)
        except Exception:
            verbosity = 0

        try:
            rep, fig = rs_mtf.MTF.run(
                patch, FRACTION,
                plot=WRITE_FIGURES,
                verbose=verbosity,
                filename=patch_name
            )
        except TypeError:
            # in case your MTF.run signature differs
            rep, fig = rs_mtf.MTF.run(patch, FRACTION)

        # save figure if present
        fig_path = None
        if fig is not None and WRITE_FIGURES:
            fig_path = img_out_dir / f"{src_stem}_patch{i:02d}_mtf.png"
            fig.savefig(fig_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

        # build CSV row safely
        def _get(obj, name, default=None):
            return getattr(obj, name, default)

        rows.append({
            "source_image": src_path.name,
            "depth_um": _get(rep, "depth_um"),  
            "x_pix": int(cxy[0]),
            "y_pix": int(cxy[1]),
            "image_w": _get(rep, "image_w"),
            "image_h": _get(rep, "image_h"),
            "edge_profile": _get(rep, "edge_profile"),
            "angle_deg": _get(rep, "angle_deg"),
            "width_px": _get(rep, "width_px"),
            "threshold": _get(rep, "threshold"),
            "contrast": _get(rep, "contrast"),
            "mtf50_freq": _get(rep, "mtf50_freq"),
            "mtf_at_nyquist": _get(rep, "mtf_at_nyquist"),
        })

    return rows

def main():
    # load modules
    rs_mtf = _load_module_from_path(MTF_MODULE_PATH, required=("MTF",))
    rs_hyp = _load_module_from_path(HYPER_MODULE_PATH, required=("HyperTarget",))

    in_path = Path(INPUT)
    out_base = Path(PATCH_OUT_BASE)
    out_base.mkdir(parents=True, exist_ok=True)

    files = _collect_images(in_path, FILENAME_GLOB)
    if not files:
        print(f"No files matched {FILENAME_GLOB!r} under: {in_path}")
        sys.exit(0)

    print(f"Found {len(files)} image(s). Writing patches & results under: {out_base}")

    all_rows: list[dict] = []
    for f in files:
        try:
            print(f" → {f}")
            rows = process_image(rs_mtf, rs_hyp, f, out_base)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
            traceback.print_exc()

    if all_rows:
        summary_csv = out_base / SUMMARY_CSV
        pd.DataFrame(all_rows).to_csv(summary_csv, index=False)
        print(f"\nSummary CSV → {summary_csv}")

if __name__ == "__main__":
    main()
