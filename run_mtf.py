# run_mtf.py — headless batch runner (single CSV, optional figures)

# ---------- CONFIG ----------
INPUT = r"/Users/haarshakrishna/Documents/PHYS3810/slant_edge_japan_best_192"   # folder or single image
OUT_DIR = r"/Users/haarshakrishna/Documents/GitHub/mtf_batch/outputs"
EXTENSIONS = [".png"]                               # add ".jpg", ".jpeg", ".tif", ".tiff" if needed
RECURSIVE = True                                    # True=rglob, False=glob
MODULE_PATH = r"/Users/haarshakrishna/Documents/GitHub/mtf_batch/src/mtf_batch/mtf_plus.py"

FRACTION = 0.5                                      # 0.5 = MTF50
WRITE_FIGURES = True                                # set False to skip PNG plots
SUMMARY_CSV_NAME = "mtf_summary.csv"
# -------- END CONFIG --------

from pathlib import Path
import importlib.util
import json
import pandas as pd

# Headless matplotlib (and no-op show)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None


def _load_module_from_path(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"MODULE_PATH not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    # quick sanity checks
    for name in ("Transform", "MTF", "Verbosity"):
        if not hasattr(mod, name):
            raise AttributeError(f"Loaded module '{path.name}' missing: {name}")
    if not hasattr(mod.MTF, "run"):
        raise AttributeError("Your mtf_plus.py must define MTF.run(...).")
    return mod


def _collect_files(root: Path):
    exts = tuple(EXTENSIONS + [e.upper() for e in EXTENSIONS])
    if root.is_dir():
        it = root.rglob("*") if RECURSIVE else root.glob("*")
        files = [p for p in it if p.is_file() and p.suffix in exts]
        files.sort()
        return files
    return [root]


def _process_file(rs, ipath: Path, out_dir: Path | None):
    # Load → arrayify
    gsimg, _, _ = rs.Transform.LoadImg(str(ipath))
    arr = rs.Transform.Arrayify(gsimg)

    # One-liner: analysis (+ optional plot)
    rep, fig = rs.MTF.run(arr, FRACTION, plot=WRITE_FIGURES,
                          verbose=rs.Verbosity.DETAIL, filename=ipath.name)

    # Save figure if requested
    plot_path = None
    if fig is not None:
        out_dir = out_dir or ipath.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / f"{ipath.stem}_mtf.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    # Build one CSV row (arrays → JSON to keep one-row-per-image)
    row = {
        "filename": rep.filename,
        "image_w": rep.image_w,
        "image_h": rep.image_h,
        "edge_profile": rep.edge_profile,
        "angle_deg": rep.angle_deg,
        "width_px": rep.width_px,
        "threshold": rep.threshold,
        "contrast": rep.contrast,                 # 0..1
        "mtf_fraction": rep.mtf_fraction,         # e.g., 0.5
        "mtf50_freq": rep.mtf50_freq,             # normalized freq at fraction
        "nyquist_frequency": rep.nyquist_frequency,  # 0.5
        "mtf_at_nyquist": rep.mtf_at_nyquist,     # 0..1
        # "mtf_x_json": json.dumps(rep.mtf_x.tolist()),
        # "mtf_y_json": json.dumps(rep.mtf_y.tolist()),
        # "lsf_x_json": json.dumps(rep.lsf_x.tolist()),
        # "lsf_y_json": json.dumps(rep.lsf_y.tolist()),
        # # convenience “coordinates”
        # "mtf50_point_x": rep.mtf50_freq,
        # "mtf50_point_y": (rep.mtf_fraction if rep.mtf50_freq is not None else None),
        # "nyquist_point_x": rep.nyquist_frequency,
        # "nyquist_point_y": rep.mtf_at_nyquist,
    }
    return plot_path, row


def main():
    rs = _load_module_from_path(MODULE_PATH)
    ipath = Path(INPUT)
    out_dir = Path(OUT_DIR) if OUT_DIR else None

    files = _collect_files(ipath)
    print(f"Processing {len(files)} file(s) from: {ipath}")

    rows, outputs = [], []
    for f in files:
        try:
            try:
                rel = f.relative_to(ipath)
            except Exception:
                rel = f
            print(" -", rel)
            plot_path, row = _process_file(rs, f, out_dir)
            rows.append(row)
            if plot_path:
                outputs.append(plot_path)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    # ONE consolidated CSV
    if rows:
        out_dir = out_dir or ipath.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = out_dir / SUMMARY_CSV_NAME
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        print(f"\nSummary CSV -> {summary_csv}")

    if outputs:
        print(f"\nWrote {len(outputs)} figure(s).")

    try:
        input("\nPress Enter to close...")
    except Exception:
        pass


if __name__ == "__main__":
    main()
