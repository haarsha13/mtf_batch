import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def violin_mtf_by_depth(
    df: pd.DataFrame,
    mtf_col: str = "MTF",
    depth_col: str = "depth_um",
    # identify a "location" (adjust to whatever uniquely identifies a patch/location)
    location_cols = ("x_pix", "y_pix"),
    mtf_min: float = 0.0,
    mtf_max: float = 1.0,
    title: str = "Per-location mean MTF by depth (violin plot)"
):
    """
    df needs columns: depth_col, mtf_col, and location_cols (e.g., patch_x, patch_y).
    - Clips MTF values to [mtf_min, mtf_max]
    - Averages MTF per (depth, location)
    - Plots one violin per depth of those per-location means
    """
    # 1) clip outliers (hard clamp so extreme values don’t leak into violins)
    df = df.copy()
    df[mtf_col] = df[mtf_col].clip(lower=mtf_min, upper=mtf_max)

    # 2) per-location mean within each depth
    group_keys = [depth_col, *location_cols]
    per_loc_mean = (
        df.groupby(group_keys, dropna=False)[mtf_col]
          .mean()
          .reset_index()
    )

    # 3) assemble data for each depth as a list of arrays
    # (keeps violin widths meaningful & avoids implicit sorting surprises)
    depths = (
        per_loc_mean[depth_col]
        .dropna()
        .unique()
    )
    # sort numerically if they’re numbers inside strings:
    try:
        depths_sorted = sorted(depths, key=lambda d: float(str(d).replace("µm","").replace("um","")))
    except Exception:
        depths_sorted = sorted(depths)

    data_by_depth = [per_loc_mean.loc[per_loc_mean[depth_col] == d, mtf_col].values
                     for d in depths_sorted]

    # 4) plot the violins
    fig, ax = plt.subplots(figsize=(9, 5))
    vp = ax.violinplot(
        data_by_depth,
        showmeans=True,      # show a line at the mean of each violin
        showextrema=False,   # whiskers/boxes not needed; we’re showing means only
        showmedians=False
    )

    # Set x-axis ticks/labels to depths
    ax.set_xticks(np.arange(1, len(depths_sorted) + 1))
    ax.set_xticklabels([str(d) for d in depths_sorted], rotation=0)
    ax.set_xlabel("Depth (µm)")
    ax.set_ylabel("Per-location mean MTF")
    ax.set_title(title)

    # 5) enforce visible y-limits to the same clip band
    ax.set_ylim(mtf_min, mtf_max)

    # Optional: grid for readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def one_violin_depth_medians_with_minmax(
    df: pd.DataFrame,
    mtf_col: str = "mtf_at_nyquist",   # or "MTF50", etc.
    depth_col: str = "depth_um",
    location_cols = ("patch_x","patch_y"),
    mtf_min: float = 0.0,
    mtf_max: float = 1.0,
    title: str = "MTF (per-depth medians) with per-depth min–max; outliers clipped"
):
    """
    - Clips MTF to [mtf_min, mtf_max]
    - Averages per (depth, location) first (so each location contributes once)
    - For each depth, computes median + min + max across locations
    - Plots ONE violin of the per-depth medians, and overlays a jittered
      vertical line for each depth showing that depth's [min, max]
    """
    d = df.copy()
    d[mtf_col] = d[mtf_col].clip(lower=mtf_min, upper=mtf_max)

    # Per-location mean within each depth (so multiple rows per location collapse)
    per_loc = (
        d.groupby([depth_col, *location_cols], dropna=False)[mtf_col]
         .mean()
         .reset_index()
    )

    # Per-depth stats across locations
    agg = (
        per_loc.groupby(depth_col, dropna=False)[mtf_col]
        .agg(median="median", min="min", max="max")
        .reset_index()
    )

    # Sort depths numerically if they look like "20um"/"140 µm"
    def _to_num(x):
        s = str(x).lower().replace("µm","").replace("um","").strip()
        try: return float(s)
        except: return np.inf
    agg["__depth_num"] = agg[depth_col].apply(_to_num)
    agg = agg.sort_values("__depth_num").drop(columns="__depth_num")

    vals = agg["median"].values       # distribution we put into ONE violin
    mins = agg["min"].values          # per-depth min (after clipping)
    maxs = agg["max"].values          # per-depth max (after clipping)

    # Plot
    fig, ax = plt.subplots(figsize=(6,5))

    # Single violin of all per-depth medians
    ax.violinplot([vals], showmeans=False, showmedians=True, showextrema=False)

    # Jittered markers for each depth's median + a min–max bar
    x = np.full_like(vals, 1, dtype=float) + np.random.uniform(-0.05, 0.05, size=vals.size)
    ax.scatter(x, vals, s=18, alpha=0.9, label="Per-depth median")
    for xi, lo, hi in zip(x, mins, maxs):
        ax.vlines(xi, lo, hi, alpha=0.35, linewidth=2)  # depth's min–max span

    ax.set_xticks([1])
    ax.set_xticklabels([f"Depth medians (n={len(vals)})"])
    ax.set_ylabel(f"{mtf_col} (clipped to [{mtf_min}, {mtf_max}])")
    ax.set_ylim(mtf_min, mtf_max)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

# Example:
df = pd.read_csv(r"/Users/haarshakrishna/Documents/GitHub/mtf_batch/outputs/SN006_ThroughFocus/mtf_summary_all.csv")
one_violin_depth_medians_with_minmax(
    df,
    mtf_col="mtf50_freq",
    depth_col="depth_um",
    location_cols=("x_pix","y_pix"),
    mtf_min=0,
    mtf_max=1
)



# # ---- Example usage ----
# df = pd.read_csv(r"/Users/haarshakrishna/Documents/GitHub/mtf_batch/outputs/SN006_ThroughFocus/mtf_summary_all.csv")
# violin_mtf_by_depth(
#     df,
#     mtf_col="mtf50_freq",  
#     depth_col="depth_um",
#     location_cols=("x_pix","y_pix"),
#     mtf_min=0,               # set your lower bound
#     mtf_max=1                 # set your upper bound
# )

 if all_rows:
        summary_csv = out_base / SUMMARY_CSV
        dat_all = pd.DataFrame(all_rows)
        dat_all.to_csv(summary_csv, index=False)
        print(f"\nSummary CSV → {summary_csv}")


        # --- Clean + filter band to avoid extremes ---
        mtf_min, mtf_max = 0.05, 0.90  # adjust as needed
        clean = dat_all.dropna(subset=['depth_um', 'mtf50_freq']).copy()
        clean = clean[(clean['mtf50_freq'] > 0) & (clean['mtf50_freq'] < 1)]
        clean = clean[clean['mtf50_freq'].between(mtf_min, mtf_max)]

        # Ensure depth is numeric (no harm if already numeric)
        clean['depth'] = pd.to_numeric(clean['depth_um'], errors='coerce')
        clean = clean.dropna(subset=['depth_um'])

        # --- Sorted unique depths ---
        depths_sorted = np.sort(clean['depth_um'].unique())

        # --- Build distributions: ALL rows per depth (no per-location averaging) ---
        data_by_depth = [
            clean.loc[clean['depth'] == d, 'mtf50_freq'].values
            for d in depths_sorted
        ]
        # Filter out empty arrays and their corresponding depths
        nonempty = [(d, arr) for d, arr in zip(depths_sorted, data_by_depth) if len(arr) > 0]
        if not nonempty:
            print("No data to plot for violin plot.")
            return
        depths_sorted, data_by_depth = zip(*nonempty)

        # --- Per-depth stats (median, min, max) for overlays ---
        stats = (
            clean.groupby('depth_um')['mtf50_freq']
                .agg(median='median', min='min', max='max', n='count')
                .reindex(depths_sorted)
                .reset_index()
        )

        # --- Plot: violins + median marker + min–max bars ---
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.violinplot(
            data_by_depth,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        xs = np.arange(1, len(depths_sorted) + 1)
        for i, (_, row) in enumerate(stats.iterrows(), start=1):
            ax.scatter(i, row['median'], marker='D', s=28, zorder=3, label=None)
            ax.vlines(i, row['min'], row['max'], linewidth=2, alpha=0.5)

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{d:g}" for d in depths_sorted])
        ax.set_xlabel('Depth (µm)')
        ax.set_ylabel('MTF50 (cyc/pixel)')
        ax.set_ylim(mtf_min, mtf_max)
        ax.set_title('MTF50 vs Depth — violins of all rows; median + min–max per depth')

        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_base / "mtf50_vs_depth_violin.png", dpi=300, bbox_inches="tight")
        plt.close()
