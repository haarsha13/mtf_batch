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
0. **Patch Name Format**
   Your image MUST be in the format '..._..._DEPTH_...'
   For example 'exportimage_1658869656624_10_um', or 'hello_kitty_10_uhuh'

2. **Patch extraction**  
   `hypertarget.py` locates slant-edge fields in each image (or defaults to center if none found).

3. **Patch cropping**  
   Patches (e.g., 256×256 .png file) are cropped and saved under `outputs/…`.

4. **MTF analysis**  
   - ESF is built and **centered at 50% level**.  
   - Differentiated → LSF, smoothed, windowed (Kaiser).  
   - FFT → MTF curve, reporting:  
     - **MTF(fraction) frequency**  
     - **MTF@Nyquist (0.5 normalized)**
     - Edge angle and profile, width, contrast, image size, angle.

5. **Results**  
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
# The folder that contains images to be sliced, ... subfolders:
INPUT = ""

# Where to write patches and results:
PATCH_OUT_BASE = ""

# Your local paths to the MTF and HyperTarget modules:
MTF_MODULE_PATH = "dir\\MTF_HD.py"
HYPER_MODULE_PATH = "dir\\hypertarget.py"

# Uploaded images must be in .PNG format
FILENAME_GLOB = "*.png"

# Keep sub-folder hierarchy in output? If INPUT directory contains subfolders, sort by subfolder name?
ORGANIZE_BY_SUB = True

# Patch controls
PATCH_SIZE = 255          # square crop (pixels) around each slant center
MAX_PATCHES = None        # e.g., 12 to limit per image; None = all
SAVE_PER_IMAGE_MANIFEST = False  # save CSV of each patch centers per image?
LENS_FOCAL_LEN = int(420)         # focal length of lens in mm
COLLIMATOR_FOCAL_LEN = int(2000)        # focal length of collimator in mm
SENSOR_PIX_SIZE = float(3.2e-3) # sensor pixel size in mm (e.g., 3.2e-3 for 3.2µm pixels)

# MTF controls
FRACTION = 0.5            # 0.5 = MTF50 (what fraction of MTF to report)
BETA = 14                 # smoothing; higher = smoother, less noisy, but may miss detail (Kaiser window)
WRITE_FIGURES = True      # save MTF plot per patch? 
SUMMARY_CSV = "mtf_summary_all.csv"  # written into PATCH_OUT_BASE, and is the summary of all data. 

# --- quick runner toggle ---
TOGGLE_RUN_MTF = True      # True => run MTF on each patch; False => skip MTF (just save patches and metadata)
TOGGLE_OUTPUT_LENS_MTF = True # True => output theoretical lens MTF curve based on diffraction and pixel sampling
USE_EXISTING_CSV = False   # True => read existing CSV in INPUT and only make plots; False => run full pipeline (For TRUE you must have the CSV_OVERRIDE_PATH go to your csv.)
CSV_OVERRIDE_PATH =  None (leave None to use PATCH_OUT_BASE/SUMMARY_CSV) 
```

---

## Running the Pipeline

From the repo root:

```bash
# Activate your venv first optionally
python run_mtf.py
```

The script will:

- Recursively collect images from `INPUT` that end with something matching `FILENAME_GLOB`.
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

## Output Table Example

**Summary CSV columns**:

| source_image | x_pix | y_pix | image_w | image_h | edge_profile | angle_deg | width_px | threshold | contrast | mtf50_freq | mtf_at_nyquist |
|--------------|------:|------:|--------:|--------:|--------------|----------:|---------:|----------:|---------:|-----------:|---------------:|


---

## License
This repository is licensed under the [MIT License](./LICENSE). © 2025 Damian Howe and Haarsha Krishna Moorthy.

## Third-Party Licenses
License file: `third_party/licence.txt`

---

## Notes

- If no slant edges are found, the runner falls back to analyzing the **full image center**.
- In summary of the runner options, you can change the values and inputs of many things to analyze the patches in the desired way (fraction, beta, etc).
- You can choose to only run the hypertarget without the MTF. 
- You can choose to run the hypertarget and MTF without plotting. 
- You can choose to only run the MTF without the hypertarget, using an existing csv with the same data. 
- You must have the images in the format listed above in the Workflow section step 0. 

