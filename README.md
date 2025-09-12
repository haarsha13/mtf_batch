# Slant-Edge MTF Analysis Pipeline

This project evaluates lens sharpness using **slanted-edge MTF analysis**.  
The workflow extracts **patches** from input images, runs an MTF calculation per patch, and consolidates results (e.g., **MTF50**, **MTF@Nyquist**, **contrast**).

---

## Repository Layout

```
repo/
├─ MTF_HD.py              # Core ESF → LSF → MTF analysis
├─ hypertarget.py         # Finds slant-edge patch centers
├─ run_mtf.py             # Runner: slices images, calls hypertarget + MTF, writes CSV
└─ outputs/               # Created at runtime (patches, plots, summary CSV)
```


## Installation

```bash
# (recommended) create & activate a virtual env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# install all dependencies
pip install -r requirements.txt


## Workflow

1. **Patch extraction**  
   `hypertarget.py` locates slant-edge fields in each image (or defaults to center if none found).

2. **Patch cropping**  
   Patches (e.g., 256×256) are cropped and saved under `outputs/…`.

3. **MTF analysis**  
   - ESF is built and **centered at 50% level**.  
   - Differentiated → LSF, smoothed, windowed (Kaiser).  
   - FFT → MTF curve, reporting:  
     - **MTF50 frequency**  
     - **MTF@Nyquist (0.5 normalized)**  
     - Edge angle, width, contrast.

4. **Results**  
   - Per-image: MTF plots: for each patch, a figure is saved showing the ESF, LSF, and MTF curve (if WRITE_FIGURES = True).
   - Cropped patch images themselves (saved as PNG).
   - Global: `mtf_summary_all.csv` (all patches across all images).

---

## Configuration

Open **`run_mtf.py`** and edit the config block at the top:

```python
# ---------------- CONFIG (EDIT THESE) ----------------
INPUT = r"/path/to/images"      
PATCH_OUT_BASE = r"/path/to/outputs"

MTF_MODULE_PATH = r"/path/to/MTF_HD.py"
HYPER_MODULE_PATH = r"/path/to/hypertarget.py"

FILENAME_GLOB = "*.png"         # image pattern
ORGANIZE_BY_PARENT = True       # keep SNxxx folder hierarchy

PATCH_SIZE = 256                # patch size (pixels)
MAX_PATCHES = None              # limit patches per image; None = all

FRACTION = 0.5                  # 0.5 = MTF50
WRITE_FIGURES = True
SUMMARY_CSV = "mtf_summary_all.csv"
# -----------------------------------------------------
```

---

## Running the Pipeline

From the repo root:

```bash
# Activate your venv first
python run_mtf.py
```

The script will:

- Recursively collect images from `INPUT` matching `FILENAME_GLOB`.
- For each image:
  - Detect slant-edge centers with **HyperTarget**.
  - Crop patches (`PATCH_SIZE`).
  - Run **MTF.run** from `MTF_HD.py`.
  - Save patches, plots, and CSV outputs.
- Write a consolidated summary CSV at:
  ```
  /path/to/outputs/mtf_summary_all.csv
  ```

---

## Output Example

**Summary CSV columns**:

| source_image | x_pix | y_pix | image_w | image_h | edge_profile | angle_deg | width_px | threshold | contrast | mtf50_freq | mtf_at_nyquist |
|--------------|------:|------:|--------:|--------:|--------------|----------:|---------:|----------:|---------:|-----------:|---------------:|

---

## Notes

- If no slant edges are found, the runner falls back to analyzing the **full image center**.  
- Figures are disabled by setting `WRITE_FIGURES = False` in config.  
- You can limit patches for debugging by setting `MAX_PATCHES = 12`.  
- `FRACTION` can be adjusted to report MTF20, MTF50, etc.  
