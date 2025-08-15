# src/main.py
import yaml, csv
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from mtf_plus import calculate_from_path

def ensure_dirs(root: Path):
    for d in ["csv", "plots", "report"]:
        (root / d).mkdir(parents=True, exist_ok=True)

def main():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    in_root = Path(cfg["input_dir"])
    out_root = Path(cfg["output_dir"])
    ensure_dirs(out_root)

    files = sorted(in_root.glob(cfg.get("input_glob", "**/*.png")))
    print(f"Found {len(files)} input files.")

    rows = []
    for f in files:
        try:
            m = calculate_from_path(
                f,
                pixel_pitch_um=cfg.get("pixel_pitch_um"),
                gamma=cfg.get("gamma"),
                verbose=0
            )
            rows.append({
                "file": str(f),
                "mtf50_cyc_per_pix": m.mtf50_cyc_per_pix,
                "mtf10_cyc_per_pix": m.mtf10_cyc_per_pix,
                "mtf50_lp_per_mm": m.mtf50_lp_per_mm,
                "mtf10_lp_per_mm": m.mtf10_lp_per_mm,
                "nyquist_percent": m.nyquist_percent,
                "transition_width_pix": m.transition_width_pix,
            })

            if cfg.get("save_plots", True):
                fig = plt.figure()
                plt.plot(m.freq_cyc_per_pix, m.mtf)
                plt.xlabel("Spatial frequency (cycles/pixel) [Nyquist=0.5]")
                plt.ylabel("MTF")
                plt.title(f"{f.name} | MTF50={m.mtf50_cyc_per_pix:.4f} c/p")
                plt.grid(True, alpha=0.3)
                (out_root / "plots").mkdir(parents=True, exist_ok=True)
                fig.savefig(out_root / "plots" / f"{f.stem}_mtf.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

            print(f"[OK] {f.name}  MTF50={m.mtf50_cyc_per_pix:.4f} c/p")

        except Exception as e:
            print(f"[FAIL] {f}: {e}")

    df = pd.DataFrame(rows)
    (out_root / "csv").mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "csv" / "mtf_results.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(out_root / "report" / "MTF_summary.csv", index=False)
    print(f"Wrote: {csv_path}")

if __name__ == "__main__":
    main()
