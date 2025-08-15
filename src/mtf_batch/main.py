import glob, yaml, pandas as pd
from pathlib import Path

def ensure_dirs(root: Path):
    for d in ["csv","overlays","plots","report"]:
        (root / d).mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
outdir = Path(cfg["output_dir"])
ensure_dirs(outdir)

files = sorted(glob.glob(cfg["input_glob"], recursive=True))
print(f"Found {len(files)} input files.")

# stub summary so pipeline 'runs'
pd.DataFrame({"file": files}).to_csv(outdir / "report" / "MTF_summary.csv", index=False)
print(f"Wrote: {outdir / 'report' / 'MTF_summary.csv'}")
