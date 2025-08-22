# run_resolution_mtf.py — double‑click friendly runner for resolution_and_sharpness_of_images.py

# ---------- CONFIG ----------
INPUT = r"C:\PHYS3810\slant_edge_japan_best_192"   # folder of PNGs OR a single PNG file
OUT_DIR = r"C:\code\mtf_batch\outputs"             # where reports go (or set to None to save beside inputs)
PATTERN = "*.png"
# -------- END CONFIG --------

import sys
from pathlib import Path
import os

def _import_rs_module():
    """
    Import the user's processing module.
    Tries the two common filenames:
      - resolution_and_sharpness_of_images.py
      - resolution_and_sharpness_of_images_with_orientation.py
    Assumes this runner sits in the same folder as the module or one level above (.\src).
    """
    import importlib
    here = Path(__file__).resolve().parent
    candidates = [
        here / "resolution_and_sharpness_of_images.py",
        here / "resolution_and_sharpness_of_images_with_orientation.py",
        here / "src" / "resolution_and_sharpness_of_images.py",
        here / "src" / "resolution_and_sharpness_of_images_with_orientation.py",
    ]
    # Put both here and here/src on sys.path
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here / "src"))
    for c in candidates:
        if c.exists():
            modname = c.stem  # file name without .py
            try:
                return importlib.import_module(modname)
            except Exception as e:
                print(f"[WARN] Found {c.name} but failed to import: {e}")
    # last attempt by module name only
    try:
        return importlib.import_module("resolution_and_sharpness_of_images")
    except Exception as e:
        raise ImportError(
            "Could not import resolution_and_sharpness_of_images module. "
            "Place this runner next to the module, or adjust sys.path."
        ) from e

def _run_on_file(rs, ipath: Path, out_dir: Path | None) -> Path:
    """
    Runs MTF_Full on a single image and saves a PNG beside OUT_DIR (or beside file).
    The processing/figure generation is handled by the module; we only set paths and save.
    """
    from datetime import datetime
    # Load and convert
    img = rs.Transform.LoadImg(str(ipath))
    arr = rs.Transform.Arrayify(img)
    # Run full pipeline (DETAIL to produce figure)
    rs.MTF.MTF_Full(arr, verbose=rs.Verbosity.DETAIL)
    # choose output path
    if out_dir is None:
        out_dir = ipath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ipath.stem}_mtf.png"
    import matplotlib.pyplot as plt
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    return out_path

def main():
    rs = _import_rs_module()
    ipath = Path(INPUT)
    out_dir = Path(OUT_DIR) if OUT_DIR else None

    if ipath.is_dir():
        print(f"Processing folder: {ipath} (pattern={PATTERN})")
        files = sorted(ipath.glob(PATTERN))
        outputs = []
        for f in files:
            try:
                print(" -", f.name)
                outputs.append(_run_on_file(rs, f, out_dir))
            except Exception as e:
                print(f"[ERROR] {f.name}: {e}")
    else:
        print(f"Processing file: {ipath}")
        outputs = []
        try:
            outputs.append(_run_on_file(rs, ipath, out_dir))
        except Exception as e:
            print(f"[ERROR] {ipath.name}: {e}")

    print(f"\nDone. Wrote {len(outputs)} file(s):")
    for o in outputs:
        print(" -", o)
    try:
        input("\nPress Enter to close...")
    except Exception:
        pass

if __name__ == "__main__":
    main()
