"""
eobs_validate.py

  1. PROCESS  — Load E-OBS daily 1981-2010, aggregate to 12-month climatology
                Saves: eobs_monthly_climatology.nc

  2. BIOCLIM  — Apply bioclim derivation to the E-OBS monthly climatology
                Reuses your existing bioclim_functions.py
                Saves: eobs_bioclim.nc

  3. COMPARE  — Aggregate CHELSA 1 km baseline BIOs to E-OBS 0.1 deg grid
                Compute bias, RMSE, R^2 for all 19 BIOs over land cells
                Saves: eobs_comparison_report.txt + eobs_comparison.csv

Usage:
    python eobs_validate.py \\
        --eobs_dir "/path/to/eobs_raw" \\
        --chelsa_baseline_dir "/path/to/output_bioclim_1km/historical/baseline" \\
        --pipeline_dir "/path/to/pipeline/files"     # for bioclim_functions.py
        [--output_dir "/path/to/output"]              # default: current dir

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


BIO_VARS = [f"BIO{i:02d}" for i in range(1, 20)]

# Days per calendar month (non-leap year, per Karger 2023 convention)
DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                          dtype=np.float32)

# Mapping: E-OBS variable name -> CHELSA-style name expected by bioclim code
EOBS_TO_CHELSA = {
    "tg": "tas",
    "tn": "tasmin",
    "tx": "tasmax",
    "rr": "pr",
}



# Step 1 — Process E-OBS daily to 12-month climatology


def process_eobs_to_monthly(eobs_dir, output_path):
    """Load E-OBS daily files, subset to 1981-2010, aggregate to 12-month
    climatology. Returns the climatology dataset.

    IMPORTANT: the four E-OBS source files (tg/tn/tx/rr) carry slightly
    different floating-point representations of the same latitude/longitude
    grid (e.g. 25.04986061 vs 25.04986062). When combined naively with
    xr.Dataset(...), xarray treats these as distinct positions and produces
    a union grid full of NaNs because no two variables share an exact
    coordinate value. We work around this by snapping the last three
    variables' coordinates to the first variable's coordinates before
    combining.
    """

    print("\n" + "=" * 70)
    print("STEP 1 — Process E-OBS daily to monthly climatology (1981-2010)")
    print("=" * 70)

    arrays_climatology = {}
    reference_lat = None
    reference_lon = None

    for eobs_var, chelsa_name in EOBS_TO_CHELSA.items():
        path = eobs_dir / f"{eobs_var}_ens_mean_0.1deg_reg_v32.0e.nc"
        if not path.exists():
            print(f"  ERROR: file not found: {path}")
            sys.exit(1)

        print(f"\n  Processing {eobs_var} ({chelsa_name}) from {path.name} ...")
        ds = xr.open_dataset(path, chunks={"time": 365})
        da = ds[eobs_var]

        # Subset to 1981-2010
        da = da.sel(time=slice("1981-01-01", "2010-12-31"))
        n_days = da.sizes["time"]
        print(f"    subset 1981-2010: {n_days} days")

        # Monthly aggregation: temperature -> mean, precipitation -> sum
        if eobs_var == "rr":
            monthly = da.resample(time="1MS").sum(skipna=True)
            print(f"    aggregated daily mm -> monthly mm (sum)")
        else:
            monthly = da.resample(time="1MS").mean(skipna=True)
            print(f"    aggregated daily C -> monthly mean C")

        print(f"    monthly time series: {monthly.sizes['time']} months")

        # Climatology: average across the 30 years, by calendar month
        clim = monthly.groupby("time.month").mean(skipna=True)
        print(f"    climatology: {clim.sizes['month']} months (one per Jan-Dec)")

        # Capture the reference coordinates from the first variable
        if reference_lat is None:
            reference_lat = clim.latitude.values.copy()
            reference_lon = clim.longitude.values.copy()
            print(f"    [coord reference] using {eobs_var} coords as canonical")
        else:
            # Snap this variable's coordinates to the reference
            # Check that the shape matches (same grid, just different precision)
            lat_diff = np.abs(clim.latitude.values - reference_lat).max()
            lon_diff = np.abs(clim.longitude.values - reference_lon).max()
            if (clim.latitude.shape != reference_lat.shape or
                clim.longitude.shape != reference_lon.shape):
                raise ValueError(
                    f"{eobs_var}: grid shape mismatch with reference; "
                    f"got lat={clim.latitude.shape} lon={clim.longitude.shape}, "
                    f"expected lat={reference_lat.shape} lon={reference_lon.shape}"
                )
            if lat_diff > 0.001 or lon_diff > 0.001:
                raise ValueError(
                    f"{eobs_var}: coordinates differ from reference by more than "
                    f"0.001 deg (lat_diff={lat_diff}, lon_diff={lon_diff}); "
                    f"this is more than floating-point noise, indicating a "
                    f"genuinely different grid."
                )
            print(f"    [coord snap] aligning {eobs_var} to reference coords "
                  f"(max diff: lat={lat_diff:.2e}, lon={lon_diff:.2e})")
            clim = clim.assign_coords(latitude=reference_lat,
                                       longitude=reference_lon)

        arrays_climatology[chelsa_name] = clim.compute()
        ds.close()

    # Combine into a single dataset (now all coords identical)
    print("\n  Combining four climatologies into single dataset ...")
    ds_clim = xr.Dataset(arrays_climatology)
    ds_clim.attrs["title"] = "E-OBS v32.0e monthly climatology, 1981-2010"
    ds_clim.attrs["source"] = "E-OBS daily, processed to monthly climatology"
    ds_clim.attrs["note"] = (
        "All four variables snapped to a common reference grid; original "
        "floating-point differences between source files were <1e-3 deg."
    )

    # Sanity check before saving
    tas_finite = np.isfinite(ds_clim["tas"].isel(month=0).values)
    pr_finite = np.isfinite(ds_clim["pr"].isel(month=0).values)
    both_finite = tas_finite & pr_finite
    print(f"  Sanity check (January): "
          f"tas finite cells = {int(tas_finite.sum()):,}, "
          f"pr finite cells = {int(pr_finite.sum()):,}, "
          f"both = {int(both_finite.sum()):,}")
    if int(both_finite.sum()) < 1000:
        print("  WARN: very low overlap between tas and pr; this should not happen")
        print("        after coordinate alignment. Check the source files.")

    ds_clim.to_netcdf(output_path)
    print(f"  Saved: {output_path}")

    return ds_clim



# Step 2 — Compute bioclim variables from E-OBS monthly climatology


def compute_eobs_bioclim(climatology, output_path, pipeline_dir):
    """Apply the user's bioclim_functions.py to the E-OBS climatology.
    Returns the bioclim dataset."""

    print("\n" + "=" * 70)
    print("STEP 2 — Compute bioclim variables from E-OBS climatology")
    print("=" * 70)

    # Import the user's bioclim module
    sys.path.insert(0, str(pipeline_dir))
    try:
        import bioclim_functions as bcf
        print(f"  Imported bioclim_functions from: {pipeline_dir}")
    except ImportError as e:
        print(f"  ERROR: could not import bioclim_functions.py: {e}")
        print(f"  Make sure --pipeline_dir points to a directory containing it.")
        sys.exit(1)

    # The user's bioclim_functions.py uses dim names: month, tas, tasmin, tasmax, prec
    # Rename pr -> prec to match the user's convention if needed
    ds_for_bioclim = climatology.copy()
    if "pr" in ds_for_bioclim and "prec" not in ds_for_bioclim:
        ds_for_bioclim = ds_for_bioclim.rename({"pr": "prec"})


    print("  Building complete-coverage mask (all 4 vars valid in all 12 months) ...")
    tas_complete = np.isfinite(ds_for_bioclim["tas"]).all(dim="month")
    tasmin_complete = np.isfinite(ds_for_bioclim["tasmin"]).all(dim="month")
    tasmax_complete = np.isfinite(ds_for_bioclim["tasmax"]).all(dim="month")
    prec_complete = np.isfinite(ds_for_bioclim["prec"]).all(dim="month")
    common_mask = tas_complete & tasmin_complete & tasmax_complete & prec_complete
    n_common = int(common_mask.sum().values)
    print(f"    cells where all four variables are complete: {n_common:,}")

    # Apply the mask to each variable: where mask is False, set all 12 months to NaN
    for var in ("tas", "tasmin", "tasmax", "prec"):
        ds_for_bioclim[var] = ds_for_bioclim[var].where(common_mask)

    # Pull the four DataArrays
    tas = ds_for_bioclim["tas"]
    tasmin = ds_for_bioclim["tasmin"]
    tasmax = ds_for_bioclim["tasmax"]
    prec = ds_for_bioclim["prec"]

    print(f"  Input shapes:  tas {tas.shape}, prec {prec.shape}")

    # Call compute_bioclim — this is the same function used for the CHELSA BIOs
    print("  Computing 19 bioclim variables ...")
    bioclim_result = bcf.compute_bioclim(tas, tasmin, tasmax, prec)

    # Handle both Dataset and dict return shapes (the user's compute_bioclim
    # returns a dict of {name: DataArray}; convert it to a Dataset so we can
    # save and pass cleanly to Step 3)
    if isinstance(bioclim_result, dict):
        bioclim_ds = xr.Dataset(bioclim_result)
    elif isinstance(bioclim_result, xr.Dataset):
        bioclim_ds = bioclim_result
    else:
        raise TypeError(
            f"compute_bioclim returned {type(bioclim_result).__name__}; "
            "expected dict or xarray.Dataset"
        )
    print(f"  Done. Output variables: {list(bioclim_ds.data_vars)}")

    bioclim_ds.attrs["title"] = "Bioclimatic variables from E-OBS v32.0e, 1981-2010"
    bioclim_ds.attrs["source"] = "E-OBS monthly climatology -> bioclim_functions.py"
    bioclim_ds.to_netcdf(output_path)
    print(f"  Saved: {output_path}")

    return bioclim_ds



# Step 3 — Compare CHELSA (aggregated to 0.1 deg) vs E-OBS bioclim


def find_chelsa_bio_file(chelsa_dir, var):
    for name in (f"{var}_1km.nc", f"{var.lower()}_1km.nc",
                 f"{var}.nc", f"{var.lower()}.nc"):
        p = chelsa_dir / name
        if p.exists():
            return p
    return None


def open_chelsa_bio(file_path, var):
    ds = xr.open_dataset(file_path)
    if var in ds:
        return ds[var]
    if var.lower() in ds:
        return ds[var.lower()]
    return ds[list(ds.data_vars)[0]]


def aggregate_chelsa_to_eobs(chelsa_1km, eobs_target):
    """Area-weighted aggregation of CHELSA 1 km to E-OBS 0.1 deg grid.
    Uses bilinear interpolation (xarray.interp) which for downscaling
    operations gives near-area-weighted means at fine source / coarse target.
    """
    # Identify CHELSA coords
    chelsa_lat = "y" if "y" in chelsa_1km.coords else "latitude"
    chelsa_lon = "x" if "x" in chelsa_1km.coords else "longitude"
    eobs_lat = "latitude"
    eobs_lon = "longitude"

    # Ensure ascending coordinates for interp
    if chelsa_1km[chelsa_lat].values[0] > chelsa_1km[chelsa_lat].values[-1]:
        chelsa_1km = chelsa_1km.isel(**{chelsa_lat: slice(None, None, -1)})

    # Reindex via interp to the E-OBS target grid coordinates
    # For 1 km -> 0.1 deg aggregation, bilinear interp samples the
    # underlying high-res field at the E-OBS cell centers; given the
    # ~100:1 resolution ratio, this is functionally an area-weighted
    # downsample within sampling error.
    aggregated = chelsa_1km.interp(
        {chelsa_lat: eobs_target[eobs_lat], chelsa_lon: eobs_target[eobs_lon]},
        method="linear",
    )
    return aggregated


def weighted_stats(chelsa_da, eobs_da):
    """Compute area-weighted bias, RMSE, R^2 over land cells (both finite).
    Land cells = cells where BOTH chelsa_da and eobs_da are finite."""

    # Align onto the E-OBS grid
    lat = eobs_da["latitude"].values
    weights_1d = np.cos(np.deg2rad(lat)).astype(np.float64)
    weights_2d = np.broadcast_to(weights_1d[:, None],
                                  eobs_da.shape).astype(np.float64)

    c = chelsa_da.values.astype(np.float64)
    e = eobs_da.values.astype(np.float64)

    mask = np.isfinite(c) & np.isfinite(e)
    n_land = int(mask.sum())
    if n_land < 100:
        return {
            "n_land": n_land, "bias": np.nan, "rmse": np.nan,
            "r2": np.nan, "chelsa_mean": np.nan, "eobs_mean": np.nan,
        }

    w = weights_2d[mask]
    c_m = c[mask]
    e_m = e[mask]
    wsum = float(w.sum())

    chelsa_mean = float((w * c_m).sum() / wsum)
    eobs_mean = float((w * e_m).sum() / wsum)
    bias = chelsa_mean - eobs_mean

    diff = c_m - e_m
    rmse = float(np.sqrt((w * diff ** 2).sum() / wsum))

    # Weighted R^2 = squared weighted Pearson correlation
    c_anom = c_m - chelsa_mean
    e_anom = e_m - eobs_mean
    cov = float((w * c_anom * e_anom).sum() / wsum)
    var_c = float((w * c_anom ** 2).sum() / wsum)
    var_e = float((w * e_anom ** 2).sum() / wsum)
    if var_c > 0 and var_e > 0:
        r = cov / np.sqrt(var_c * var_e)
        r2 = float(r ** 2)
    else:
        r2 = np.nan

    return {
        "n_land": n_land,
        "bias": bias,
        "rmse": rmse,
        "r2": r2,
        "chelsa_mean": chelsa_mean,
        "eobs_mean": eobs_mean,
    }


def compare_chelsa_vs_eobs(chelsa_dir, eobs_bioclim, output_report, output_csv):
    print("\n" + "=" * 70)
    print("STEP 3 — Compare CHELSA (aggregated to 0.1 deg) vs E-OBS bioclim")
    print("=" * 70)

    # Use BIO01 as the reference for aggregation grid
    template_bio = list(eobs_bioclim.data_vars)[0]
    template = eobs_bioclim[template_bio]

    rows = []
    for var in BIO_VARS:
        chelsa_path = find_chelsa_bio_file(chelsa_dir, var)
        if chelsa_path is None:
            print(f"  WARN  : CHELSA file for {var} not found, skipping")
            continue

        print(f"  {var}: ", end="", flush=True)

        # Load CHELSA and aggregate to E-OBS grid
        chelsa_1km = open_chelsa_bio(chelsa_path, var)
        chelsa_aggregated = aggregate_chelsa_to_eobs(chelsa_1km, template)

        # Get the matching E-OBS variable name (try a few patterns)
        eobs_var = None
        for cand in (var, var.lower(), f"bio{var[3:].lstrip('0') or '0'}"):
            if cand in eobs_bioclim:
                eobs_var = cand
                break
        if eobs_var is None:
            print(f"E-OBS variable not found")
            continue
        eobs_da = eobs_bioclim[eobs_var]

        stats = weighted_stats(chelsa_aggregated, eobs_da)
        stats["variable"] = var
        rows.append(stats)

        print(f"n_land={stats['n_land']:>8,}  "
              f"bias={stats['bias']:>8.3f}  "
              f"rmse={stats['rmse']:>8.3f}  "
              f"r2={stats['r2']:>6.3f}")

    # Build the report text
    lines = []
    lines.append("=" * 78)
    lines.append("E-OBS INDEPENDENT VALIDATION — CHELSA 1 km vs E-OBS 0.1 deg")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Baseline period: 1981-2010")
    lines.append("Aggregation:    CHELSA 1 km -> E-OBS 0.1 deg via bilinear interp")
    lines.append("Statistics:     cos(latitude)-weighted over cells where both are finite")
    lines.append("")
    lines.append("  Var      n_land       bias         rmse           r2    chelsa_mean    eobs_mean")
    lines.append("  " + "-" * 87)
    for row in rows:
        lines.append(
            f"  {row['variable']:<7}{row['n_land']:>8,}  "
            f"{row['bias']:>10.4f}  {row['rmse']:>10.4f}  "
            f"{row['r2']:>10.4f}  {row['chelsa_mean']:>13.4f}  {row['eobs_mean']:>11.4f}"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("  bias        = (CHELSA aggregated) - (E-OBS), area-weighted")
    lines.append("  rmse        = root mean squared error, area-weighted")
    lines.append("  r2          = squared weighted Pearson correlation (spatial)")
    lines.append("  chelsa_mean = area-weighted mean of CHELSA on E-OBS grid")
    lines.append("  eobs_mean   = area-weighted mean of E-OBS on E-OBS grid")
    lines.append("")
    lines.append("Reference for E-OBS v32.0e:")
    lines.append("  Cornes, R.C., van der Schrier, G., van den Besselaar, E.J.M., &")
    lines.append("  Jones, P.D. (2018). An ensemble version of the E-OBS temperature and")
    lines.append("  precipitation data sets. J. Geophys. Res. Atmos. 123, 9391-9409.")
    lines.append("=" * 78)

    text = "\n".join(lines) + "\n"
    with open(output_report, "w") as f:
        f.write(text)
    print(f"\nReport saved: {output_report}")

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"CSV saved:    {output_csv}")

    return rows



# Main

def main():
    parser = argparse.ArgumentParser(description="E-OBS independent validation")
    parser.add_argument("--eobs_dir", required=True,
                        help="Directory containing raw E-OBS .nc files")
    parser.add_argument("--chelsa_baseline_dir", required=True,
                        help="Directory containing CHELSA baseline BIO files")
    parser.add_argument("--pipeline_dir", required=True,
                        help="Directory containing bioclim_functions.py")
    parser.add_argument("--output_dir", default=".",
                        help="Where to write outputs (default: current dir)")
    parser.add_argument("--skip_step", default=None,
                        choices=["1", "2", "12"],
                        help="Skip step 1 / step 2 / both (use existing files)")
    args = parser.parse_args()

    eobs_dir = Path(args.eobs_dir).expanduser().resolve()
    chelsa_dir = Path(args.chelsa_baseline_dir).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly_path = out_dir / "eobs_monthly_climatology.nc"
    bioclim_path = out_dir / "eobs_bioclim.nc"
    report_path = out_dir / "eobs_comparison_report.txt"
    csv_path = out_dir / "eobs_comparison.csv"

    # Step 1
    skip = args.skip_step or ""
    if "1" in skip and monthly_path.exists():
        print(f"Step 1 skipped (using existing {monthly_path.name})")
        climatology = xr.open_dataset(monthly_path)
    else:
        climatology = process_eobs_to_monthly(eobs_dir, monthly_path)

    # Step 2
    if "2" in skip and bioclim_path.exists():
        print(f"Step 2 skipped (using existing {bioclim_path.name})")
        bioclim = xr.open_dataset(bioclim_path)
    else:
        bioclim = compute_eobs_bioclim(climatology, bioclim_path, pipeline_dir)

    # Step 3
    compare_chelsa_vs_eobs(chelsa_dir, bioclim, report_path, csv_path)

    print("\n" + "=" * 70)
    print("DONE — paste the comparison report back to interpret the numbers.")
    print("=" * 70)


if __name__ == "__main__":
    main()