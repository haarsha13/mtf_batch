# run_mtf.py — double-click friendly runner for mtf_plus

# ---------- CONFIG ----------
INPUT = r"C:\PHYS3810\slant_edge_japan_best_192"   # folder of PNGs OR a single PNG file
OUT_DIR = r"C:\code\mtf_batch\outputs"             # where reports go (or set to None)
PATTERN = "*.png"
PIXEL_PITCH_UM = None
GAMMA = None
# -------- END CONFIG --------

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))       # so we can "import mtf_plus" from src/

try:
    import mtf_plus as mp
except Exception as e:
    print("\n[ERROR] Could not import 'mtf_plus' from", SRC)
    print("Make sure mtf_plus.py is in src/, and third_party/mtf.py is src/third_party/ if required.")
    print("Original error:", repr(e))
    input("\nPress Enter to exit...")
    raise SystemExit(1)

def main():
    ipath = Path(INPUT)
    out_dir = Path(OUT_DIR) if OUT_DIR else None

    if ipath.is_dir():
        print(f"Processing folder: {ipath} (pattern={PATTERN})")
        outputs = mp.batch_analyze(ipath, out_dir=out_dir, pattern=PATTERN,
                                   pixel_pitch_um=PIXEL_PITCH_UM, gamma=GAMMA)
    else:
        print(f"Processing file: {ipath}")
        outputs = [mp.analyze_and_plot(ipath, out_dir=out_dir,
                                       pixel_pitch_um=PIXEL_PITCH_UM, gamma=GAMMA)]
    print(f"\nDone. Wrote {len(outputs)} file(s):")
    for o in outputs:
        print(" -", o)
    input("\nPress Enter to close...")

if __name__ == "__main__":
    main()
