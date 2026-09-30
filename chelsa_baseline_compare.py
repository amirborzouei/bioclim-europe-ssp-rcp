"""
chelsa_baseline_compare.py

1. Set the two paths below (PIPELINE_BASELINE_DIR and CHELSA_BASELINE_DIR).
2. Set OUTPUT_DIR to where you want results saved.

@author: amirrezaborzoueinia
"""


import sys
import csv
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray as rxr

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# CONFIGURATION — set these paths to match your system


# Where your pipeline's baseline BIO outputs live.
# Files are named: BIO01_1km.nc, BIO02_1km.nc, ..., BIO19_1km.nc
from config import BASELINE_DIR_1KM, CHELSA_BIOCLIM_DIR, BASELINE_COMPARE_DIR

PIPELINE_BASELINE_DIR = BASELINE_DIR_1KM

# Where CHELSA's distributed V2.1 baseline bioclim GeoTIFFs live.
CHELSA_BASELINE_DIR = CHELSA_BIOCLIM_DIR

# Where to save outputs
OUTPUT_DIR = BASELINE_COMPARE_DIR

# European bounding box (must match your pipeline's domain)
EUROPE_EXTENT = dict(lon_min=-22.0, lon_max=45.0, lat_min=27.0, lat_max=72.0)

# Number of top-precipitation cells to report for the BIO12 extreme analysis
TOP_N_EXTREMES = 200


TEST_BIO03_BUG_HYPOTHESIS = True


# Helpers


def log_print(msg, log_handle=None):
    """Print to stdout and to log file simultaneously."""
    print(msg)
    if log_handle is not None:
        log_handle.write(msg + "\n")
        log_handle.flush()


def find_pipeline_file(bio_n: int) -> Path | None:
    """Locate your pipeline's BIO file. Tries common naming patterns."""
    candidates = [
        PIPELINE_BASELINE_DIR / f"BIO{bio_n:02d}.nc",
        PIPELINE_BASELINE_DIR / f"BIO{bio_n:02d}_1km.nc",
        PIPELINE_BASELINE_DIR / f"bio{bio_n:02d}.nc",
        PIPELINE_BASELINE_DIR / f"BIO{bio_n}.nc",
        PIPELINE_BASELINE_DIR / f"bio{bio_n}.nc",
    ]
    return next((c for c in candidates if c.exists()), None)


def find_chelsa_file(bio_n: int) -> Path | None:
    """
    Locate CHELSA's distributed BIO file.
    Primary structure (confirmed):
      {CHELSA_BASELINE_DIR}/bio{NN}/1981-2010/CHELSA_bio{NN}_1981-2010_V.2.1.tif
    where NN is zero-padded (bio01, bio02, ..., bio19).
    """
    bio_dir_padded   = CHELSA_BASELINE_DIR / f"bio{bio_n:02d}" / "1981-2010"
    bio_dir_unpadded = CHELSA_BASELINE_DIR / f"bio{bio_n}"     / "1981-2010"
    candidates = [
        # Nested structure with zero-padding (the confirmed pattern)
        bio_dir_padded   / f"CHELSA_bio{bio_n:02d}_1981-2010_V.2.1.tif",
        # Nested structure without zero-padding (fallback)
        bio_dir_unpadded / f"CHELSA_bio{bio_n}_1981-2010_V.2.1.tif",
        # Flat fallbacks (in case some files live differently)
        CHELSA_BASELINE_DIR / f"CHELSA_bio{bio_n:02d}_1981-2010_V.2.1.tif",
        CHELSA_BASELINE_DIR / f"CHELSA_bio{bio_n}_1981-2010_V.2.1.tif",
        CHELSA_BASELINE_DIR / f"bio{bio_n}_1981-2010.tif",
    ]
    return next((c for c in candidates if c.exists()), None)


def load_pipeline_bio(path: Path, bio_n: int) -> xr.DataArray:
    """Load a pipeline BIO NetCDF file. Returns DataArray (y, x) in deg N/E."""
    ds = xr.open_dataset(path)
    # Find the data variable (one of these likely names)
    var_names = [f"BIO{bio_n:02d}", f"BIO{bio_n}", f"bio{bio_n}",
                 f"bio{bio_n:02d}", "__xarray_dataarray_variable__"]
    da = None
    for v in var_names:
        if v in ds.data_vars:
            da = ds[v]
            break
    if da is None:
        # If no recognized name, take the first variable
        da = ds[list(ds.data_vars)[0]]
    # Standardize coordinate names to y/x
    rename = {}
    for src, dst in [("lat", "y"), ("latitude", "y"),
                     ("lon", "x"), ("longitude", "x")]:
        if src in da.dims:
            rename[src] = dst
    if rename:
        da = da.rename(rename)
    # Sort to standard order (S to N, W to E)
    da = da.sortby("y").sortby("x")
    return da


def load_chelsa_bio(path: Path, bio_n: int) -> xr.DataArray:

    da = rxr.open_rasterio(path, masked=True, mask_and_scale=True).squeeze()
    # Standardize to y/x naming (rioxarray uses y/x by default but be safe)
    if "lat" in da.coords:
        da = da.rename({"lat": "y", "lon": "x"})
    # Clip to Europe (CHELSA distributable is global)
    da = da.sel(
        x=slice(EUROPE_EXTENT["lon_min"], EUROPE_EXTENT["lon_max"]),
        y=slice(EUROPE_EXTENT["lat_max"], EUROPE_EXTENT["lat_min"]),  # CHELSA y is N->S
    )
    # Sort to S->N to match pipeline convention
    da = da.sortby("y")
    return da


def align_grids(pipeline_da: xr.DataArray, chelsa_da: xr.DataArray):

    # Reindex CHELSA onto pipeline's exact coordinates
    chelsa_aligned = chelsa_da.interp(
        y=pipeline_da.y.values,
        x=pipeline_da.x.values,
        method="nearest",
        kwargs={"fill_value": np.nan},
    )
    return pipeline_da, chelsa_aligned


def compute_stats(pipe_vals: np.ndarray, chelsa_vals: np.ndarray,
                  weights: np.ndarray) -> dict:

    if pipe_vals.size == 0:
        return dict(n_cells=0, bias=np.nan, rmse=np.nan, r2=np.nan,
                    pipe_mean=np.nan, chelsa_mean=np.nan,
                    pipe_min=np.nan, pipe_max=np.nan,
                    chelsa_min=np.nan, chelsa_max=np.nan)
    diff = pipe_vals - chelsa_vals
    # Area-weighted bias
    bias = np.sum(weights * diff) / np.sum(weights)
    # Area-weighted RMSE
    rmse = np.sqrt(np.sum(weights * diff**2) / np.sum(weights))
    # Area-weighted Pearson r squared
    pipe_mean   = np.sum(weights * pipe_vals)   / np.sum(weights)
    chelsa_mean = np.sum(weights * chelsa_vals) / np.sum(weights)
    cov = np.sum(weights * (pipe_vals - pipe_mean) * (chelsa_vals - chelsa_mean)) / np.sum(weights)
    var_p = np.sum(weights * (pipe_vals - pipe_mean)**2) / np.sum(weights)
    var_c = np.sum(weights * (chelsa_vals - chelsa_mean)**2) / np.sum(weights)
    denom = np.sqrt(var_p * var_c)
    r2 = (cov / denom)**2 if denom > 0 else np.nan
    return dict(
        n_cells=int(pipe_vals.size),
        bias=float(bias),
        rmse=float(rmse),
        r2=float(r2),
        pipe_mean=float(pipe_mean),
        chelsa_mean=float(chelsa_mean),
        pipe_min=float(np.min(pipe_vals)),
        pipe_max=float(np.max(pipe_vals)),
        chelsa_min=float(np.min(chelsa_vals)),
        chelsa_max=float(np.max(chelsa_vals)),
    )


def cosine_weights(y_coords: np.ndarray, shape: tuple) -> np.ndarray:

    cos_lat = np.cos(np.deg2rad(y_coords))
    return np.broadcast_to(cos_lat[:, None], shape).copy()



# Pre-flight checks


def preflight(log_handle):
    log_print("=" * 78, log_handle)
    log_print("CHELSA V2.1 BASELINE COMPARISON — PRE-FLIGHT", log_handle)
    log_print("=" * 78, log_handle)
    log_print(f"Pipeline baseline dir : {PIPELINE_BASELINE_DIR}", log_handle)
    log_print(f"CHELSA baseline dir   : {CHELSA_BASELINE_DIR}", log_handle)
    log_print(f"Output dir            : {OUTPUT_DIR}", log_handle)
    log_print("", log_handle)

    if not PIPELINE_BASELINE_DIR.exists():
        log_print(f"  ERROR: pipeline baseline dir does not exist", log_handle)
        return False
    if not CHELSA_BASELINE_DIR.exists():
        log_print(f"  ERROR: CHELSA baseline dir does not exist", log_handle)
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_print("Locating files for BIO01-BIO19 ...", log_handle)
    missing_pipeline = []
    missing_chelsa = []
    for n in range(1, 20):
        p = find_pipeline_file(n)
        c = find_chelsa_file(n)
        pm = "FOUND" if p else "MISSING"
        cm = "FOUND" if c else "MISSING"
        log_print(f"  BIO{n:02d}: pipeline={pm:8s}  chelsa={cm}", log_handle)
        if not p:
            missing_pipeline.append(n)
        if not c:
            missing_chelsa.append(n)

    if missing_pipeline:
        log_print(f"\n  WARNING: pipeline BIO files missing for: {missing_pipeline}", log_handle)
        log_print(f"           Adjust PIPELINE_BASELINE_DIR or the naming patterns", log_handle)
        log_print(f"           in find_pipeline_file().", log_handle)
    if missing_chelsa:
        log_print(f"\n  WARNING: CHELSA BIO files missing for: {missing_chelsa}", log_handle)
        log_print(f"           Adjust CHELSA_BASELINE_DIR or download missing files.", log_handle)

    if len(missing_pipeline) == 19 or len(missing_chelsa) == 19:
        log_print("\n  All files missing in one or both directories. Aborting.", log_handle)
        return False
    if len(missing_pipeline) > 5 or len(missing_chelsa) > 5:
        log_print("\n  More than 5 files missing. Verify paths and re-run.", log_handle)
        return False

    log_print("", log_handle)
    return True



# Per-BIO comparison


def compare_one_bio(bio_n: int, log_handle) -> dict | None:
    """Compare one BIO variable. Returns stats dict or None on failure."""
    log_print(f"\n--- BIO{bio_n:02d} ---", log_handle)

    p_path = find_pipeline_file(bio_n)
    c_path = find_chelsa_file(bio_n)
    if p_path is None or c_path is None:
        log_print(f"  Skipping (file not found)", log_handle)
        return None

    log_print(f"  Pipeline file : {p_path.name}", log_handle)
    log_print(f"  CHELSA file   : {c_path.name}", log_handle)

    t_load = time.time()
    pipe_da = load_pipeline_bio(p_path, bio_n)
    chelsa_da = load_chelsa_bio(c_path, bio_n)
    log_print(f"  Loaded in {time.time()-t_load:.1f}s "
              f"(pipe shape={pipe_da.shape}, chelsa shape={chelsa_da.shape})",
              log_handle)

    t_align = time.time()
    pipe_da, chelsa_aligned = align_grids(pipe_da, chelsa_da)
    log_print(f"  Aligned in {time.time()-t_align:.1f}s", log_handle)

    pipe_vals = pipe_da.values
    chelsa_vals = chelsa_aligned.values

    # Build cosine-of-latitude weights
    weights_2d = cosine_weights(pipe_da.y.values, pipe_vals.shape)

    # Valid mask: both datasets finite
    valid = np.isfinite(pipe_vals) & np.isfinite(chelsa_vals)
    n_valid = int(valid.sum())
    log_print(f"  Valid cells (both finite): {n_valid:,}", log_handle)

    if n_valid == 0:
        log_print(f"  WARNING: no valid cells, skipping stats", log_handle)
        return None

    stats = compute_stats(pipe_vals[valid], chelsa_vals[valid], weights_2d[valid])
    log_print(
        f"  bias={stats['bias']:+.4f}  rmse={stats['rmse']:.4f}  r2={stats['r2']:.4f}",
        log_handle,
    )
    log_print(
        f"  Pipeline range:  [{stats['pipe_min']:.2f}, {stats['pipe_max']:.2f}]  "
        f"mean={stats['pipe_mean']:.2f}",
        log_handle,
    )
    log_print(
        f"  CHELSA range:    [{stats['chelsa_min']:.2f}, {stats['chelsa_max']:.2f}]  "
        f"mean={stats['chelsa_mean']:.2f}",
        log_handle,
    )

    stats["bio"] = f"BIO{bio_n:02d}"

    # Special handling for BIO03: also test the Karger bug hypothesis
    if bio_n == 3 and TEST_BIO03_BUG_HYPOTHESIS:
        log_print("", log_handle)
        log_print("  TESTING BIO03 BUG HYPOTHESIS (Karger BioClim.py line 255):",
                  log_handle)
        log_print("  Comparing CHELSA's 'BIO03' file against YOUR BIO07 ...",
                  log_handle)
        p07 = find_pipeline_file(7)
        if p07 is not None:
            your_bio07 = load_pipeline_bio(p07, 7)
            _, chelsa_aligned_b3 = align_grids(your_bio07, chelsa_da)
            b07_vals = your_bio07.values
            ch_b3_vals = chelsa_aligned_b3.values
            valid_b = np.isfinite(b07_vals) & np.isfinite(ch_b3_vals)
            if valid_b.sum() > 0:
                w = cosine_weights(your_bio07.y.values, b07_vals.shape)
                bug_stats = compute_stats(b07_vals[valid_b], ch_b3_vals[valid_b],
                                          w[valid_b])
                log_print(
                    f"    Your BIO07 vs CHELSA's BIO03: "
                    f"bias={bug_stats['bias']:+.4f}  r2={bug_stats['r2']:.4f}",
                    log_handle,
                )
                if bug_stats["r2"] > 0.99:
                    log_print(
                        f"    >>> r2 > 0.99 confirms hypothesis: "
                        f"CHELSA's 'BIO03' file is actually BIO07 <<<",
                        log_handle,
                    )
                else:
                    log_print(
                        f"    r2 < 0.99: hypothesis not confirmed at this strength.",
                        log_handle,
                    )
                stats["bio03_as_bio07_r2"] = bug_stats["r2"]
                stats["bio03_as_bio07_bias"] = bug_stats["bias"]

    # Special handling for BIO12: extreme-value analysis
    if bio_n == 12:
        log_print("", log_handle)
        log_print("  BIO12 EXTREME-VALUE ANALYSIS:", log_handle)
        log_print(f"  Cells > 4000 mm/yr  (pipeline): "
                  f"{int(((pipe_vals > 4000) & valid).sum()):,}", log_handle)
        log_print(f"  Cells > 4000 mm/yr  (CHELSA):   "
                  f"{int(((chelsa_vals > 4000) & valid).sum()):,}", log_handle)
        log_print(f"  Cells > 6000 mm/yr  (pipeline): "
                  f"{int(((pipe_vals > 6000) & valid).sum()):,}", log_handle)
        log_print(f"  Cells > 6000 mm/yr  (CHELSA):   "
                  f"{int(((chelsa_vals > 6000) & valid).sum()):,}", log_handle)
        log_print(f"  Cells > 8000 mm/yr  (pipeline): "
                  f"{int(((pipe_vals > 8000) & valid).sum()):,}", log_handle)
        log_print(f"  Cells > 8000 mm/yr  (CHELSA):   "
                  f"{int(((chelsa_vals > 8000) & valid).sum()):,}", log_handle)
        log_print(f"  Cells > 10000 mm/yr (pipeline): "
                  f"{int(((pipe_vals > 10000) & valid).sum()):,}", log_handle)
        log_print(f"  Cells > 10000 mm/yr (CHELSA):   "
                  f"{int(((chelsa_vals > 10000) & valid).sum()):,}", log_handle)

        # Save top-N cells from each dataset
        export_top_n_extremes(pipe_da, chelsa_aligned, valid, log_handle)

    return stats


def export_top_n_extremes(pipe_da, chelsa_da, valid, log_handle):
    """Save the top-N highest BIO12 cells from each dataset, with lat/lon."""
    pipe_vals = pipe_da.values
    chelsa_vals = chelsa_da.values

    y_coords = pipe_da.y.values
    x_coords = pipe_da.x.values
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Pipeline top-N
    flat_pipe = np.where(valid, pipe_vals, -np.inf).ravel()
    top_idx_pipe = np.argpartition(-flat_pipe, TOP_N_EXTREMES)[:TOP_N_EXTREMES]
    top_idx_pipe = top_idx_pipe[np.argsort(-flat_pipe[top_idx_pipe])]

    # CHELSA top-N
    flat_chelsa = np.where(valid, chelsa_vals, -np.inf).ravel()
    top_idx_chelsa = np.argpartition(-flat_chelsa, TOP_N_EXTREMES)[:TOP_N_EXTREMES]
    top_idx_chelsa = top_idx_chelsa[np.argsort(-flat_chelsa[top_idx_chelsa])]

    out_path = OUTPUT_DIR / "baseline_bio12_extremes.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "source", "lat", "lon",
            "pipeline_BIO12_mm_yr", "chelsa_BIO12_mm_yr", "diff"
        ])
        for rank, idx in enumerate(top_idx_pipe, 1):
            lat = float(yy.ravel()[idx])
            lon = float(xx.ravel()[idx])
            pv = float(pipe_vals.ravel()[idx])
            cv = float(chelsa_vals.ravel()[idx])
            w.writerow([rank, "pipeline_top",
                        f"{lat:.4f}", f"{lon:.4f}",
                        f"{pv:.1f}", f"{cv:.1f}", f"{pv-cv:+.1f}"])
        for rank, idx in enumerate(top_idx_chelsa, 1):
            lat = float(yy.ravel()[idx])
            lon = float(xx.ravel()[idx])
            pv = float(pipe_vals.ravel()[idx])
            cv = float(chelsa_vals.ravel()[idx])
            w.writerow([rank, "chelsa_top",
                        f"{lat:.4f}", f"{lon:.4f}",
                        f"{pv:.1f}", f"{cv:.1f}", f"{pv-cv:+.1f}"])
    log_print(f"  Top-{TOP_N_EXTREMES} extreme cells saved: {out_path.name}",
              log_handle)

    # Quick overlap stat
    pipe_top_set = set(top_idx_pipe.tolist())
    chelsa_top_set = set(top_idx_chelsa.tolist())
    overlap = len(pipe_top_set & chelsa_top_set)
    log_print(
        f"  Top-{TOP_N_EXTREMES} cell overlap: "
        f"{overlap}/{TOP_N_EXTREMES} ({100*overlap/TOP_N_EXTREMES:.1f}%) "
        f"— do both datasets identify the same hot spots?",
        log_handle,
    )



# Main


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "baseline_comparison_log.txt"
    log_handle = open(log_path, "w")

    t_start = time.time()
    if not preflight(log_handle):
        log_handle.close()
        sys.exit(1)

    log_print("\n" + "=" * 78, log_handle)
    log_print("PER-BIO BASELINE COMPARISON", log_handle)
    log_print("=" * 78, log_handle)

    all_stats = []
    for n in range(1, 20):
        try:
            s = compare_one_bio(n, log_handle)
            if s is not None:
                all_stats.append(s)
        except Exception as e:
            log_print(f"  ERROR processing BIO{n:02d}: {e}", log_handle)

    # Write summary CSV
    summary_path = OUTPUT_DIR / "baseline_comparison_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "BIO", "n_cells", "bias", "rmse", "r2",
            "pipe_mean", "chelsa_mean",
            "pipe_min", "pipe_max", "chelsa_min", "chelsa_max",
            "bio03_as_bio07_r2", "bio03_as_bio07_bias",
        ])
        for s in all_stats:
            w.writerow([
                s["bio"], s["n_cells"],
                f"{s['bias']:+.4f}", f"{s['rmse']:.4f}", f"{s['r2']:.4f}",
                f"{s['pipe_mean']:.4f}", f"{s['chelsa_mean']:.4f}",
                f"{s['pipe_min']:.2f}", f"{s['pipe_max']:.2f}",
                f"{s['chelsa_min']:.2f}", f"{s['chelsa_max']:.2f}",
                s.get("bio03_as_bio07_r2", ""),
                s.get("bio03_as_bio07_bias", ""),
            ])

    log_print("\n" + "=" * 78, log_handle)
    log_print("DONE", log_handle)
    log_print("=" * 78, log_handle)
    log_print(f"Summary CSV : {summary_path}", log_handle)
    log_print(f"Full log    : {log_path}", log_handle)
    log_print(f"Total time  : {time.time()-t_start:.1f}s", log_handle)
    log_handle.close()


if __name__ == "__main__":
    main()