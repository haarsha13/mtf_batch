# run_mtf.py — runner for resolution_and_sharpness_of_images.py

# ---------- CONFIG ----------
INPUT = r"C:\PHYS3810\slant_edge_japan_best_192"   # folder or single PNG
OUT_DIR = r"C:\code\mtf_batch\outputs"             # or None
PATTERN = "*.png"
MODULE_PATH = r"C:\Users\HAARSHA KRISHNA\OneDrive\Documents\GitHub\mtf_batch\src\mtf_batch\resolution_and_sharpness_of_images.py"
# -------- END CONFIG --------

import sys
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt

def _load_module_from_path(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"MODULE_PATH not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _run_on_file(rs, ipath: Path, out_dir: Path | None):
    img = rs.Transform.LoadImg(str(ipath))
    arr = rs.Transform.Arrayify(img)
    rs.MTF.MTF_Full(arr, verbose=rs.Verbosity.DETAIL)
    out_dir = out_dir or ipath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ipath.stem}_mtf.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    return out_path

def main():
    rs = _load_module_from_path(MODULE_PATH)
    ipath = Path(INPUT)
    out_dir = Path(OUT_DIR) if OUT_DIR else None
    outputs = []
    if ipath.is_dir():
        print(f"Processing folder: {ipath} (pattern={PATTERN})")
        for f in sorted(ipath.glob(PATTERN)):
            try:
                print(" -", f.name)
                outputs.append(_run_on_file(rs, f, out_dir))
            except Exception as e:
                print(f"[ERROR] {f.name}: {e}")
    else:
        print(f"Processing file: {ipath}")
        outputs.append(_run_on_file(rs, ipath, out_dir))

    print(f"\nDone. Wrote {len(outputs)} file(s):")
    for o in outputs:
        print(" -", o)
    try: input("\nPress Enter to close...")
    except: pass

if __name__ == "__main__":
    main()

