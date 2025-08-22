# run_mtf.py — runner for resolution_and_sharpness_of_images.py (batch-friendly)

# ---------- CONFIG ----------
INPUT = r"C:\PHYS3810\slant_edge_japan_best_192"   # folder or single image
OUT_DIR = r"C:\code\mtf_batch\outputs"             # or None
# If you want to keep using a glob, you still can (only used when EXTENSIONS is None)
PATTERN = "*.png"
# Set EXTENSIONS to process multiple image types (case-insensitive). If None, uses PATTERN.
EXTENSIONS = [".png"]  # e.g. [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
RECURSIVE = True       # scan subfolders
MODULE_PATH = r"C:\Users\HAARSHA KRISHNA\OneDrive\Documents\GitHub\mtf_batch\src\mtf_batch\resolution_and_sharpness_of_images.py"
# -------- END CONFIG --------

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
    rs.MTF.MTF_Full(arr, verbose=rs.Verbosity.DETAIL, filename=ipath.name)
    out_dir = out_dir or ipath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ipath.stem}_mtf.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    return out_path

def _gather_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    if EXTENSIONS:
        exts = tuple(e.lower() for e in EXTENSIONS)
        it = root.rglob("*") if RECURSIVE else root.glob("*")
        for p in it:
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    else:
        # fall back to pattern
        it = root.rglob(PATTERN) if RECURSIVE else root.glob(PATTERN)
        files = [p for p in it if p.is_file()]
    return sorted(files)

def main():
    rs = _load_module_from_path(MODULE_PATH)
    ipath = Path(INPUT)
    out_dir = Path(OUT_DIR) if OUT_DIR else None

    files = _gather_files(ipath)
    print(f"Processing {len(files)} file(s) from: {ipath}")
    if not files:
        print("No input images found. Check INPUT, EXTENSIONS/PATTERN, or OneDrive 'Always keep on this device'.")
        try: input("\nPress Enter to close...")
        except: pass
        return

    outputs = []
    for f in files:
        try:
            print(" -", f if f == ipath else f.relative_to(ipath) if ipath.is_dir() else f.name)
            outputs.append(_run_on_file(rs, f, out_dir))
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    print(f"\nDone. Wrote {len(outputs)} file(s):")
    for o in outputs:
        print(" -", o)
    try: input("\nPress Enter to close...")
    except: pass

if __name__ == "__main__":
    main()

