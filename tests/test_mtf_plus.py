import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from mtf_batch.mtf_plus import calculate_from_path
import matplotlib.pyplot as plt
from pathlib import Path

# change this to a real file you have
img = r"/Users/haarshakrishna/Documents/PHYS3810/slant_edge_japan_best_192/patchX1057Y5455_depth-600um.png"

m = calculate_from_path(img, pixel_pitch_um=4.6)  # set your pitch or None

print(f"MTF50: {m.mtf50_cyc_per_pix:.4f} c/p"
      + (f"  |  {m.mtf50_lp_per_mm:.2f} lp/mm" if m.mtf50_lp_per_mm else ""))
print(f"MTF10: {m.mtf10_cyc_per_pix:.4f} c/p"
      + (f"  |  {m.mtf10_lp_per_mm:.2f} lp/mm" if m.mtf10_lp_per_mm else ""))
print(f"Nyquist (@0.5 c/p): {m.nyquist_percent:.3f}")
print(f"Transition width (px): {m.transition_width_pix:.3f}")

# Save a quick plot
out = Path("outputs/plots"); out.mkdir(parents=True, exist_ok=True)
plt.figure()
plt.plot(m.freq_cyc_per_pix, m.mtf)
plt.xlabel("Spatial frequency (cycles/pixel) [Nyquist=0.5]")
plt.ylabel("MTF")
plt.title(f"{Path(img).name} | MTF50={m.mtf50_cyc_per_pix:.4f} c/p")
plt.grid(True, alpha=0.3)
plt.savefig(out / "test_one_mtf.png", dpi=150, bbox_inches="tight")
