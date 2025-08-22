# run_mtf.py — headless batch runner for resolution_and_sharpness_of_images.py

# ---------- CONFIG ----------
INPUT = r"C:\PHYS3810\slant_edge_japan_best_192"   # folder or single image
OUT_DIR = r"C:\code\mtf_batch\outputs"             # or None
EXTENSIONS = [".png"]                               # add ".jpg", ".jpeg", ".tif", ".tiff" if needed
RECURSIVE = True                                    # True=rglob, False=glob
MODULE_PATH = r"C:\Users\HAARSHA KRISHNA\OneDrive\Documents\GitHub\mtf_batch\src\mtf_batch\resolution_and_sharpness_of_images.py"
# -------- END CONFIG --------

import sys
from pathlib import Path
import importlib.util

# Use headless backend to prevent plt.show() from blocking
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Ensure any plt.show() inside the module doesn't block
plt.show = lambda *a, **k: None

def _load_module_from_path(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"MODULE_PATH not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _collect_files(root: Path):
    exts = tuple(EXTENSIONS + [e.upper() for e in EXTENSIONS])
    if root.is_dir():
        it = root.rglob("*") if RECURSIVE else root.glob("*")
        files = [p for p in it if p.is_file() and p.suffix in exts]
        files.sort()
        return files
    else:
        return [root]

def _run_on_file(rs, ipath: Path, out_dir: Path | None):
    img = rs.Transform.LoadImg(str(ipath))
    arr = rs.Transform.Arrayify(img)
    # Produce the full figure (DETAIL) but it won't try to display because of Agg/no-op show
    rs.MTF.MTF_Full(arr, verbose=rs.Verbosity.DETAIL, filename=ipath.name)
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

    files = _collect_files(ipath)
    print(f"Processing {len(files)} file(s) from: {ipath}")
    outputs = []
    for f in files:
        try:
            print(" -", f if not f.is_relative_to(ipath) else f.relative_to(ipath))
        except Exception:
            print(" -", f)
        try:
            outputs.append(_run_on_file(rs, f, out_dir))
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    print(f"\nDone. Wrote {len(outputs)} file(s):")
    for o in outputs:
        print(" -", o)
    try:
        input("\nPress Enter to close...")
    except Exception:
        pass

if __name__ == "__main__":
    main()