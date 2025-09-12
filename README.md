# Slant-Edge MTF Analysis Pipeline

This project evaluates lens sharpness using **slanted-edge MTF analysis**.  
The workflow extracts **patches** from input images, runs an MTF calculation per patch, and consolidates results (e.g., **MTF50**, **MTF@Nyquist**, **contrast**).

---

## Repository Layout

```
repo/
├─ hypertarget.py         # Finds slant-edge patch centers
├─ MTF_HD.py              # Core ESF → LSF → MTF analysis
├─ run_mtf.py             # Runner: calls hypertarget and slices images, calls MTF_HD for analysis, writes CSV
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
```

## Workflow

1. **Patch extraction**  
   `hypertarget.py` locates slant-edge fields in each image (or defaults to center if none found).

2. **Patch cropping**  
   Patches (e.g., 256×256 .png file) are cropped and saved under `outputs/…`.

3. **MTF analysis**  
   - ESF is built and **centered at 50% level**.  
   - Differentiated → LSF, smoothed, windowed (Kaiser).  
   - FFT → MTF curve, reporting:  
     - **MTF(fraction) frequency**  
     - **MTF@Nyquist (0.5 normalized)**
     - Edge angle and profile, width, contrast, image size, angle.

4. **Results**  
   - Per-image: MTF plots: for each patch, a figure is saved showing the ESF, LSF, and MTF curve (if WRITE_FIGURES = True).
   - Edge angle and profile, width, contrast, image size, angle, and angle are printed within the plots
   - Cropped patch images themselves (saved as PNG).
   - Global: `mtf_summary_all.csv` (all information for all patches across all images).

---

## Configuration

Firstly, have everything necessary installed to your environment.
Have the run_mtf, MTF_HD, and Hypertargeting all downloaded. 
Open **`run_mtf.py`** and edit the config block at the top:

```python
# ---------------- CONFIG (EDIT THESE) ----------------
INPUT = r"/path/to/images"      
PATCH_OUT_BASE = r"/path/to/outputs"

MTF_MODULE_PATH = r"/path/to/MTF_HD.py"
HYPER_MODULE_PATH = r"/path/to/hypertarget.py"

# Uploaded images must be in .PNG format
FILENAME_GLOB = "*.png"

# Keep sub-folder hierarchy in output? If INPUT directory contains subfolders, sort by subfolder name?
ORGANIZE_BY_SUB = True

# Patch controls
PATCH_SIZE = 256          # square crop (pixels) around each slant center
MAX_PATCHES = None        # e.g., 12 to limit per image; None = all

# MTF controls
FRACTION = 0.5            # 0.5 = MTF50 (what fraction of MTF to report)
WRITE_FIGURES = True      # save MTF plot per patch? 
SUMMARY_CSV = "mtf_summary_all.csv"  # written into PATCH_OUT_BASE, and is the summary of all data. 
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
