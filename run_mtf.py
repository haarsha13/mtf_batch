# slice_and_mtf.py
# End-to-end: find slant-edge patches (HyperTarget) -> save patches -> run MTF on each patch.
from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import seaborn as sns
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
# The folder that contains images to be sliced, ... subfolders:
INPUT = r"/Users/haarshakrishna/Documents/PHYS3810/SN010_420mm"

# Where to write patches and results:
PATCH_OUT_BASE = r"/Users/haarshakrishna/Documents/GitHub/mtf_batch/outputs/SN006_final_results"

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
SAVE_PER_IMAGE_MANIFEST = True  # save CSV of each patch centers per image?

# MTF controls
FRACTION = 0.5            # 0.5 = MTF50 (what fraction of MTF to report)
WRITE_FIGURES = True      # save MTF plot per patch? 
SUMMARY_CSV = "mtf_summary_all.csv"  # written into PATCH_OUT_BASE, and is the summary of all data. 

# --- quick runner toggle ---
USE_EXISTING_CSV = True # True => read CSV and only make plots; False => run full pipeline
CSV_OVERRIDE_PATH =  None # e.g. r"/path/to/mtf_summary_all.csv" (leave None to use PATCH_OUT_BASE/SUMMARY_CSV)
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
        if not hasattr(mod, name): #Must have the required attribute/class in the code e.g. class MTF, class HyperTarget
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

    # keep name hierarchy if enabled
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

    name = src_path.name
    split_path = name.split("_") #Assumes filename format is (SN001_10x_100_um.png or SN001_10x_100um.png)
    print(split_path[2])
    depth = split_path[2]
  

    with open(img_out_dir / "hypertarget_meta.json", "w") as f:
        json.dump(meta, f, indent=2) #Save metadata for each image as another json file

    if centers is None or len(centers) == 0: #In the case that no slant centers are found, run MTF on the full image
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

        try: #Try to run MTF, if it fails, catch the error and continue
            rep, fig = rs_mtf.MTF.run(
                patch, FRACTION, beta=BETA,
                plot=WRITE_FIGURES,
                verbose=verbosity,
                filename=patch_name
            )

        except Exception as e: #Fail gracefully and give nan values for failed patches
            print(f"[WARN] MTF failed on {patch_name}: {e}")
            rep, fig = None, None

        # save figure if present
        fig_path = None
        if fig is not None and WRITE_FIGURES:
            fig_path = img_out_dir / f"{src_stem}_patch{i:02d}_mtf.png"
            fig.savefig(fig_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

        # build CSV row safely
        def _get(obj, name, default=None):
            return getattr(obj, name, default)

        rows.append({ # one row per patch
            "source_image": src_path.name,
            "depth_um": depth,
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


    if SAVE_PER_IMAGE_MANIFEST: #If enabled, save the patch centers for each image as a csv file
      pd.DataFrame(manifest).to_csv(img_out_dir / "patch_index.csv", index=False)

    return rows
def _clean_plot_data(df, depth_col='depth_um', mtf_col='mtf50_freq', y_lo=0.0, y_hi=1.0, edge_profile_col='edge_profile'):
    """Minimal cleaning for plots."""
    d = df.dropna(subset=[depth_col, mtf_col, edge_profile_col]).copy()
    d = d[(d[mtf_col] > 0) & (d[mtf_col] < 1)]
    d = d[d[mtf_col].between(y_lo, y_hi)]
    return d


def plot_mtf50_vs_depth(df: pd.DataFrame, out_base: Path,
                        depth_col: str = 'depth_um',
                        mtf_col: str = 'mtf50_freq',
                        y_lo: float = 0.0, y_hi: float = 1.0,
                        outfile: str = "mtf50_vs_depth.png"):
    """Scatter: all points (after simple clean/filter). Uses default figsize."""
    d = _clean_plot_data(df, depth_col, mtf_col, y_lo, y_hi)
    if d.empty:
        print("No data to plot for scatter.")
        return

    # keep first-seen order for category axis
    order = d[depth_col].drop_duplicates().tolist()
    x_map = {k: i for i, k in enumerate(order)}
    xs = d[depth_col].map(x_map).values

    plt.figure()  # <-- default size
    plt.scatter(xs, d[mtf_col].values, s=10, alpha=0.6, edgecolors='none')
    ax = plt.gca()

    # order = the exact list of x categories you passed to seaborn (e.g., [-100, -90, ..., 0, 10, ...])
    xtick_indices = [i for i, v in enumerate(order) if abs(v % 20) < 1e-9]
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels([str(int(v)) for v in order if abs(v % 20) < 1e-9], rotation=90)

   
    plt.xlabel('Depth (µm)')
    plt.ylabel('MTF50 (cyc/pixel)')
    plt.title('MTF50 vs Depth')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_base / outfile, dpi=800, bbox_inches="tight")
    plt.close()


def plot_mtf50_violins_with_topmedian_zoom(
    df: pd.DataFrame, out_base: Path,
    depth_col: str = "depth_um",
    mtf_col: str = "mtf50_freq",
    y_lo: float = 0.0, y_hi: float = 1.0,
    all_outfile: str = "mtf50_violin_all.png",
    zoom_outfile: str = "mtf50_violin_zoom_topmedian.png",
    depth_window: float = None
):
    # quick clean (assumes your _clean_plot_data exists; otherwise inline similar)
    d = _clean_plot_data(df, depth_col, mtf_col, y_lo, y_hi)
    if d.empty:
        print("No data to plot.")
        return

    # --- (1) all depths ---
    # Define order for x-axis based on unique depth values
    order = d[depth_col].drop_duplicates().tolist()
    plt.figure(figsize=(max(6, len(np.unique(d[depth_col]))/2), 8), dpi=200) # Dynamic width based on number of unique depths
    sns.violinplot(data=d, x=depth_col, y=mtf_col, order=order, cut=0, inner="box", density_norm="width", bw_method=0.2, width=0.9, linewidth=1)
    ax = plt.gca()
    xtick_indices = [i for i, v in enumerate(order) if abs(v % 20) < 1e-9]
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels([str(int(v)) for v in order if abs(v % 20) < 1e-9], rotation=90)
    
    plt.tight_layout()
    plt.grid(True, alpha=0.4)
    plt.savefig(out_base / all_outfile, dpi=800, bbox_inches="tight")
    plt.close()

    # --- top-median depth ---
    med = d.groupby(depth_col, sort=False)[mtf_col].median()
    top_depth = med.idxmax()
    top_median = float(med.loc[top_depth])

    # --- (2) zoom selection ---
    # Set depth_window as 0.2 * top_depth if not provided and top_depth is numeric
    if depth_window is None:
        try:
            depth_window = 0.2 * float(top_depth)
        except Exception:
            depth_window = 50  # fallback

    if np.issubdtype(d[depth_col].dtype, np.number):
        d_zoom = d[(d[depth_col] - float(top_depth)).abs() <= depth_window]
        order = sorted(d_zoom[depth_col].unique())
    else:
        d_zoom = d[d[depth_col].astype(str) == str(top_depth)]
        order = None  # single depth label

    if d_zoom.empty:
        d_zoom = d[d[depth_col].astype(str) == str(top_depth)]
        order = None

    plt.figure(figsize=(max(6, len(np.unique(d_zoom[depth_col]))/2), 8), dpi=200) # Dynamic width based on number of unique depths
    sns.violinplot(data=d_zoom, x=depth_col, y=mtf_col, order=order, cut=0)
    plt.axhline(top_median, ls="--", lw=1)
    plt.ylim(y_lo, y_hi)
    plt.xticks(rotation=90 if (order and len(order) > 12) else 0)
    plt.grid(True,alpha=0.4)
    plt.title(f"Zoom near top median depth: {top_depth} (MTF50={top_median:.3f})")
    plt.tight_layout()
    plt.savefig(out_base / zoom_outfile, dpi=800, bbox_inches="tight")
    plt.close()


def main():
    in_path = Path(INPUT)
    out_base = Path(PATCH_OUT_BASE)
    out_base.mkdir(parents=True, exist_ok=True) #Make sure an output base folder exists

    # Where to read/write the summary CSV
    summary_csv = Path(CSV_OVERRIDE_PATH) if CSV_OVERRIDE_PATH else (out_base / SUMMARY_CSV)

    # ---------------- PLOTS-ONLY FAST PATH ----------------
    if USE_EXISTING_CSV:
        if not summary_csv.exists():
            print(f"[WARN] Requested plots-only path but CSV not found: {summary_csv}")
            print("Falling back to full pipeline…")
        else:
            print(f"[PLOTS-ONLY] Reading: {summary_csv}")
            dat_all = pd.read_csv(summary_csv)
            # Make the figures and stop
            plot_mtf50_vs_depth(
                dat_all, out_base,
                depth_col='depth_um', mtf_col='mtf50_freq',
                y_lo=0.0, y_hi=0.5, outfile="mtf50_vs_depth.png"
            )
            plot_mtf50_violins_with_topmedian_zoom(
                dat_all, out_base,
                depth_col='depth_um', mtf_col='mtf50_freq',
                y_lo=0.0, y_hi=0.5,
                all_outfile="mtf50_violin_all.png",
                zoom_outfile="mtf50_violin_zoom_topmedian.png"
            )

            # Plot a heatmap showing the average MTF50 value at each (x_pix, y_pix) position
            pivot = dat_all.pivot_table(columns ="x_pix", index="y_pix", values="mtf50_freq")
            cmap = plt.cm.get_cmap("Paired").copy()
            cmap.set_bad(cmap(0.0))  # NaNs appear as the same color as value 0

            sns.heatmap(pivot, annot=False, cmap=cmap,
            cbar_kws={'label': 'MTF50'}, vmin=0, vmax=0.5)
            plt.xlabel("x_pix")
            plt.ylabel("y_pix")
            plt.title("MTF50 vs x_pix and y_pix")
            plt.tight_layout()
            plt.savefig(out_base / "mtf50_x_pix_y_pix_heatmap2.png", dpi=800, bbox_inches="tight")
            plt.close()
            return  # done

    # ---------------- FULL PIPELINE ----------------
    # Load modules only if we actually need to process images
    rs_mtf = _load_module_from_path(MTF_MODULE_PATH, required=("MTF",))
    rs_hyp = _load_module_from_path(HYPER_MODULE_PATH, required=("HyperTarget",))

    files = _collect_images(in_path, FILENAME_GLOB)
    if not files:
        print(f"No files matched {FILENAME_GLOB!r} under: {in_path}")
        sys.exit(0)

    print(f"Found {len(files)} image(s). Writing patches & results under: {out_base}") # Says how many images found in folder sub-folder

    all_rows: list[dict] = []
    for f in files:
        try:
            print(f" → {f}") #Print the file being processed
            rows = process_image(rs_mtf, rs_hyp, f, out_base) #Process the image and get list of dictionaries for each patch
            all_rows.extend(rows) #Add the dictionaries to the all_rows list
        except Exception as e:
            print(f"[ERROR] {Path(f).name}: {e}")
            traceback.print_exc()

    if all_rows:
        summary_csv = out_base / SUMMARY_CSV
        dat_all = pd.DataFrame(all_rows)
        dat_all.to_csv(summary_csv, index=False)
        print(f"\nSummary CSV → {summary_csv}")

    dat_all = pd.DataFrame(all_rows)
    dat_all.to_csv(summary_csv, index=False)
    print(f"\nSummary CSV → {summary_csv}")

    # Make plots from the DataFrame we just created
    plot_mtf50_vs_depth(
        dat_all, out_base,
        depth_col='depth_um', mtf_col='mtf50_freq',
        y_lo=0.0, y_hi=0.5, outfile="mtf50_vs_depth.png"
    )
    
    plot_mtf50_violins_with_topmedian_zoom(
    dat_all, out_base,
    depth_col="depth_um", mtf_col="mtf50_freq",
    y_lo=0.0, y_hi=0.5, depth_window=50.0
    )




if __name__ == "__main__":
    main()

