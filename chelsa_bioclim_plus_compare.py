"""
chelsa_bioclim_plus_compare.py

Usage:
    python chelsa_bioclim_plus_compare.py \\
        --pipeline_dir   "/path/to/output_bioclim_1km/ssp585/2071-2100" \\
        --bioclimplus_root "/path/to/chelsa02/chelsa/global/bioclim" \\
        --output_dir     "/path/to/output"

@author: amirrezaborzoueinia
"""

import sys
import argparse
from pathlib import Path
import warnings

import numpy as np
import xarray as xr
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# CHELSA-BIOCLIM+ uses -273.15 (absolute zero) as the ocean/NaN sentinel
# for temperature variables. Precipitation files may use 0 or another value.
NAN_SENTINEL_TEMP = -273.15

# Variables that use the absolute-zero NaN sentinel
TEMP_VARS = {f"bio{i:02d}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)}


SCALE_FACTOR_FROM_CHELSA_TO_THESIS = {}

BIO_LIST = [f"bio{i:02d}" for i in range(1, 20)]


def find_pipeline_file(pipeline_dir, bio):
    """Find this thesis's BIO file. Tries common naming patterns."""
    bio_upper = bio.upper()
    for name in (f"{bio_upper}_1km.nc", f"{bio}_1km.nc",
                 f"{bio_upper}.nc", f"{bio}.nc"):
        p = pipeline_dir / name
        if p.exists():
            return p
    return None


def find_bioclimplus_file(bioclimplus_root, bio,
                          period="2071-2100", gcm="IPSL-CM6A-LR",
                          scenario="ssp585"):
    """Find the CHELSA-BIOCLIM+ TIFF for a given BIO variable."""
    subdir = bioclimplus_root / bio / period / gcm / scenario
    if not subdir.is_dir():
        return None
    # Pattern: CHELSA_ipsl-cm6a-lr_ssp585_bio01_2071-2100_V.2.1.tif
    pattern = f"CHELSA_{gcm.lower()}_{scenario}_{bio}_{period}_V.2.1.tif"
    p = subdir / pattern
    if p.exists():
        return p
    # Fallback: try any .tif in the directory
    tifs = list(subdir.glob("*.tif"))
    if len(tifs) == 1:
        return tifs[0]
    return None


def load_pipeline(file_path, bio):
    """Load this thesis's 1km BIO field. Returns xr.DataArray with
    coordinates 'y' (latitude descending) and 'x' (longitude ascending)."""
    ds = xr.open_dataset(file_path)
    bio_upper = bio.upper()
    if bio_upper in ds:
        da = ds[bio_upper]
    elif bio in ds:
        da = ds[bio]
    else:
        da = ds[list(ds.data_vars)[0]]
    return da


def load_bioclimplus_european(file_path, bio, european_bbox):
    """Load CHELSA-BIOCLIM+ TIFF, crop to European bounding box.
    Returns xr.DataArray with the bbox subset. Applies NaN sentinel mask."""
    ds = xr.open_dataset(file_path, engine="rasterio")
    da = ds["band_data"].squeeze(drop=True)

    # The TIFF has 'x' (lon) ascending, 'y' (lat) descending typically
    lat_name = "y"
    lon_name = "x"
    lat_descending = da[lat_name].values[0] > da[lat_name].values[-1]

    # Bounding box
    minlat, maxlat, minlon, maxlon = european_bbox

    if lat_descending:
        da_eu = da.sel(
            {lat_name: slice(maxlat, minlat), lon_name: slice(minlon, maxlon)}
        )
    else:
        da_eu = da.sel(
            {lat_name: slice(minlat, maxlat), lon_name: slice(minlon, maxlon)}
        )

    # Apply NaN sentinel for temperature variables
    if bio in TEMP_VARS:
        da_eu = da_eu.where(da_eu > NAN_SENTINEL_TEMP + 0.5)

    return da_eu


def align_to_pipeline(chelsa_plus_da, pipeline_da):
    """Reindex the CHELSA-BIOCLIM+ data onto the pipeline's grid.
    Both are nominally at 30 arcsec, so floating-point differences in
    coordinate values are the only issue; we use nearest-neighbour
    reindexing to handle this without resampling."""
    pipe_lat_name = "y" if "y" in pipeline_da.coords else "latitude"
    pipe_lon_name = "x" if "x" in pipeline_da.coords else "longitude"
    cb_lat_name = "y" if "y" in chelsa_plus_da.coords else "latitude"
    cb_lon_name = "x" if "x" in chelsa_plus_da.coords else "longitude"

    pipe_lats = pipeline_da[pipe_lat_name].values
    pipe_lons = pipeline_da[pipe_lon_name].values

    aligned = chelsa_plus_da.reindex(
        {cb_lat_name: pipe_lats, cb_lon_name: pipe_lons},
        method="nearest",
        tolerance=0.005,  # ~500 m tolerance, well below 30 arcsec
    )
    # Rename coordinates to match the pipeline's convention for clean compare
    if cb_lat_name != pipe_lat_name:
        aligned = aligned.rename({cb_lat_name: pipe_lat_name})
    if cb_lon_name != pipe_lon_name:
        aligned = aligned.rename({cb_lon_name: pipe_lon_name})
    return aligned


def weighted_stats(thesis_da, chelsa_plus_da, bio):
    """Cosine-of-latitude-weighted bias, RMSE, R^2 over land cells."""
    pipe_lat_name = "y" if "y" in thesis_da.coords else "latitude"

    # Apply the BIO16-19 scale factor convention to align units
    if bio in SCALE_FACTOR_FROM_CHELSA_TO_THESIS:
        scale = SCALE_FACTOR_FROM_CHELSA_TO_THESIS[bio]
        chelsa_plus_adjusted = chelsa_plus_da * scale
        scale_note = f"x{scale:.0f}"
    else:
        chelsa_plus_adjusted = chelsa_plus_da
        scale_note = "x1"

    t = thesis_da.values.astype(np.float64)
    c = chelsa_plus_adjusted.values.astype(np.float64)

    if t.shape != c.shape:
        return {
            "n_land": 0, "bias": np.nan, "rmse": np.nan, "r2": np.nan,
            "thesis_mean": np.nan, "chelsa_plus_mean": np.nan,
            "scale_note": scale_note, "err": f"shape mismatch t={t.shape} c={c.shape}",
        }

    mask = np.isfinite(t) & np.isfinite(c)
    n_land = int(mask.sum())
    if n_land < 100:
        return {
            "n_land": n_land, "bias": np.nan, "rmse": np.nan, "r2": np.nan,
            "thesis_mean": np.nan, "chelsa_plus_mean": np.nan,
            "scale_note": scale_note, "err": "too few cells",
        }

    lats = thesis_da[pipe_lat_name].values
    weights_1d = np.cos(np.deg2rad(lats)).astype(np.float64)
    weights_2d = np.broadcast_to(weights_1d[:, None], t.shape).astype(np.float64)

    w = weights_2d[mask]
    tm = t[mask]
    cm = c[mask]
    wsum = float(w.sum())

    tmean = float((w * tm).sum() / wsum)
    cmean = float((w * cm).sum() / wsum)
    bias = tmean - cmean
    diff = tm - cm
    rmse = float(np.sqrt((w * diff ** 2).sum() / wsum))

    ta = tm - tmean
    ca = cm - cmean
    cov = float((w * ta * ca).sum() / wsum)
    var_t = float((w * ta ** 2).sum() / wsum)
    var_c = float((w * ca ** 2).sum() / wsum)
    r2 = float((cov / np.sqrt(var_t * var_c)) ** 2) if var_t > 0 and var_c > 0 else np.nan

    return {
        "n_land": n_land,
        "bias": bias,
        "rmse": rmse,
        "r2": r2,
        "thesis_mean": tmean,
        "chelsa_plus_mean": cmean,
        "scale_note": scale_note,
        "err": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="CHELSA-BIOCLIM+ methodological consistency check"
    )
    parser.add_argument("--pipeline_dir", required=True,
                        help="Directory with thesis's BIO files for ssp585 2071-2100")
    parser.add_argument("--bioclimplus_root", required=True,
                        help="Root of CHELSA-BIOCLIM+ bioclim/ tree")
    parser.add_argument("--output_dir", default=".",
                        help="Where to write outputs")
    parser.add_argument("--period", default="2071-2100")
    parser.add_argument("--gcm", default="IPSL-CM6A-LR")
    parser.add_argument("--scenario", default="ssp585")
    args = parser.parse_args()

    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve()
    bioclimplus_root = Path(args.bioclimplus_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine the European bounding box from the pipeline's BIO01 grid
    sample_path = find_pipeline_file(pipeline_dir, "bio01")
    if sample_path is None:
        print(f"ERROR: no BIO01 file found in {pipeline_dir}")
        sys.exit(1)
    sample = load_pipeline(sample_path, "bio01")
    lat_name = "y" if "y" in sample.coords else "latitude"
    lon_name = "x" if "x" in sample.coords else "longitude"
    lats = sample[lat_name].values
    lons = sample[lon_name].values
    european_bbox = (
        float(np.min(lats)) - 0.01,  # min lat
        float(np.max(lats)) + 0.01,  # max lat
        float(np.min(lons)) - 0.01,  # min lon
        float(np.max(lons)) + 0.01,  # max lon
    )
    print(f"\nEuropean bounding box from pipeline's BIO01 grid:")
    print(f"  lat: {european_bbox[0]:.3f} to {european_bbox[1]:.3f}")
    print(f"  lon: {european_bbox[2]:.3f} to {european_bbox[3]:.3f}")
    print(f"  grid: {len(lats)} x {len(lons)}")

    print("\n" + "=" * 70)
    print("Comparing CHELSA-BIOCLIM+ vs this thesis's pipeline")
    print(f"  GCM:      {args.gcm}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Period:   {args.period}")
    print("=" * 70)

    rows = []
    for bio in BIO_LIST:
        print(f"  {bio.upper()}: ", end="", flush=True)

        pipe_path = find_pipeline_file(pipeline_dir, bio)
        if pipe_path is None:
            print("PIPELINE FILE MISSING")
            continue

        cb_path = find_bioclimplus_file(bioclimplus_root, bio,
                                         period=args.period, gcm=args.gcm,
                                         scenario=args.scenario)
        if cb_path is None:
            print("CHELSA-BIOCLIM+ FILE MISSING")
            continue

        try:
            thesis_da = load_pipeline(pipe_path, bio)
            chelsa_plus_da = load_bioclimplus_european(
                cb_path, bio, european_bbox
            )
            chelsa_plus_aligned = align_to_pipeline(chelsa_plus_da, thesis_da)
            stats = weighted_stats(thesis_da, chelsa_plus_aligned, bio)
            stats["variable"] = bio.upper()
            rows.append(stats)
            print(
                f"n_land={stats['n_land']:>9,}  "
                f"bias={stats['bias']:>9.3f}  "
                f"rmse={stats['rmse']:>9.3f}  "
                f"r2={stats['r2']:>6.3f}  "
                f"scale={stats['scale_note']}"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            rows.append({
                "variable": bio.upper(), "n_land": 0,
                "bias": np.nan, "rmse": np.nan, "r2": np.nan,
                "thesis_mean": np.nan, "chelsa_plus_mean": np.nan,
                "scale_note": "n/a", "err": str(e),
            })

    # Build report
    lines = []
    lines.append("=" * 88)
    lines.append("CHELSA-BIOCLIM+ METHODOLOGICAL CONSISTENCY CHECK")
    lines.append("=" * 88)
    lines.append("")
    lines.append(f"This thesis's pipeline vs Brun et al. (2022) CHELSA-BIOCLIM+ dataset")
    lines.append(f"  GCM:        {args.gcm}")
    lines.append(f"  Scenario:   {args.scenario}")
    lines.append(f"  Period:     {args.period}")
    lines.append("")
    lines.append("Both datasets use:")
    lines.append("  - Same CHELSA V2.1 baseline (Karger et al. 2021)")
    lines.append("  - Same delta-change downscaling method (Karger et al. 2023)")
    lines.append("  - Same GCM, scenario, period")
    lines.append("Differences therefore reflect implementation details only.")
    lines.append("")
    lines.append("BIO16-19 SCALING NOTE:")
    lines.append("  CHELSA-BIOCLIM+ file metadata labels these as 'Mean Monthly")
    lines.append("  Precipitation' with units 'kg m-2 month-1', but empirical")
    lines.append("  inspection of the stored values shows they are quarterly sums")
    lines.append("  (mm/quarter), matching this thesis's convention. No scaling")
    lines.append("  factor is therefore applied. The file metadata label appears")
    lines.append("  to be mislabeled in CHELSA-BIOCLIM+.")
    lines.append("")
    lines.append("Statistics: cos(latitude)-weighted over cells where both finite")
    lines.append("")
    lines.append("  Var      n_land       bias        rmse         r2      thesis_mean    cb_plus_mean   scale")
    lines.append("  " + "-" * 95)
    for row in rows:
        if "err" in row and row.get("err"):
            lines.append(f"  {row['variable']:<7}  ERROR: {row['err']}")
            continue
        lines.append(
            f"  {row['variable']:<7}{row['n_land']:>9,}  "
            f"{row['bias']:>10.4f}  {row['rmse']:>10.4f}  "
            f"{row['r2']:>10.4f}  {row['thesis_mean']:>13.4f}  "
            f"{row['chelsa_plus_mean']:>13.4f}    {row.get('scale_note', '')}"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("  bias            = (thesis) - (chelsa-bioclim+ adjusted), area-weighted")
    lines.append("  rmse            = root mean squared error, area-weighted")
    lines.append("  r2              = squared weighted Pearson correlation (spatial)")
    lines.append("  thesis_mean     = area-weighted mean of this thesis's output")
    lines.append("  cb_plus_mean    = area-weighted mean of CHELSA-BIOCLIM+ (after scale adj.)")
    lines.append("  scale           = scale factor applied to CHELSA-BIOCLIM+ for comparison")
    lines.append("")
    lines.append("References:")
    lines.append("  Brun, P., Zimmermann, N.E., Hari, C., Pellissier, L., & Karger, D.N. (2022).")
    lines.append("  Global climate-related predictors at kilometre resolution for the past and")
    lines.append("  future. Earth System Science Data, 14, 5573-5603.")
    lines.append("  https://doi.org/10.5194/essd-14-5573-2022")
    lines.append("")
    lines.append("  Dataset: https://doi.org/10.16904/envidat.332")
    lines.append("=" * 88)

    text = "\n".join(lines) + "\n"

    report_path = out_dir / "chelsa_bioclim_plus_comparison_report.txt"
    csv_path = out_dir / "chelsa_bioclim_plus_comparison.csv"

    with open(report_path, "w") as f:
        f.write(text)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nReport: {report_path}")
    print(f"CSV:    {csv_path}")
    print("\nPaste the report back so we can interpret the numbers.")


if __name__ == "__main__":
    main()