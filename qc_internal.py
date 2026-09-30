"""
qc_internal.py

Usage:
    python qc_internal.py <bio_dir> [--label <label>]

@author: amirrezaborzoueinia
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd



# Configuration


BIO_VARS = [f"BIO{i:02d}" for i in range(1, 20)]

# Variables to inspect for upper-tail spatial location
# Format: (variable, mode, units_label)
TOP_LOC_TARGETS = [
    ("BIO5",  "max", "°C  (warmest month tasmax)"),
    ("BIO6",  "min", "°C  (coldest month tasmin)"),
    ("BIO12", "max", "mm/yr  (annual precipitation)"),
    ("BIO15", "max", "%      (precip seasonality, CV)"),
]



# Helpers


def find_bio_file(bio_dir, var):
    """Find the NetCDF file for a given BIO variable using common patterns."""
    candidates = [
        bio_dir / f"{var}_1km.nc",
        bio_dir / f"{var.lower()}_1km.nc",
        bio_dir / f"{var}.nc",
        bio_dir / f"{var.lower()}.nc",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def open_bio(file_path, var):
    """Open a BIO NetCDF and return the DataArray."""
    ds = xr.open_dataset(file_path)
    if var in ds:
        return ds[var]
    if var.lower() in ds:
        return ds[var.lower()]
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    raise ValueError(
        f"Cannot find {var} in {file_path.name}; "
        f"variables present: {list(ds.data_vars)}"
    )


def get_lat_lon_names(da):
    """Identify the latitude and longitude coordinate names."""
    lat_name = next((n for n in ("y", "lat", "latitude") if n in da.coords), None)
    lon_name = next((n for n in ("x", "lon", "longitude") if n in da.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot identify lat/lon coordinates from {list(da.coords)}")
    return lat_name, lon_name


def cosine_weights(da):
    """Return a 1-D DataArray of cos(lat) weights aligned with da."""
    lat_name, _ = get_lat_lon_names(da)
    lat = da[lat_name]
    weights = np.cos(np.deg2rad(lat))
    weights.name = "weights"
    return weights


def headline_stats(da, weights):
    """Compute min, max, percentiles, and area-weighted mean."""
    vals = da.values.ravel()
    valid_mask = ~np.isnan(vals)
    n_valid = int(valid_mask.sum())
    n_nan = int((~valid_mask).sum())

    if n_valid == 0:
        return {
            "n_valid": 0, "n_nan": n_nan,
            "min": np.nan, "max": np.nan,
            "p001": np.nan, "p1": np.nan, "p50": np.nan,
            "p99": np.nan, "p9999": np.nan,
            "area_weighted_mean": np.nan,
        }

    try:
        awm = float(da.weighted(weights).mean(skipna=True).values)
    except Exception:
        awm = float(np.nan)

    valid = vals[valid_mask]
    return {
        "n_valid": n_valid,
        "n_nan": n_nan,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p001": float(np.percentile(valid, 0.01)),
        "p1": float(np.percentile(valid, 1)),
        "p50": float(np.percentile(valid, 50)),
        "p99": float(np.percentile(valid, 99)),
        "p9999": float(np.percentile(valid, 99.99)),
        "area_weighted_mean": awm,
    }


def get_top_locations(da, n_top=10, mode="max"):
    """Return the top-N cells (max or min) as a list of (lat, lon, value)."""
    arr = da.values
    flat = arr.ravel()
    finite = np.isfinite(flat)
    if not finite.any():
        return []

    if mode == "max":
        ranked = np.argsort(np.where(finite, flat, -np.inf))[::-1]
    else:
        ranked = np.argsort(np.where(finite, flat, np.inf))

    top_flat_idx = ranked[:n_top]
    top_2d = np.unravel_index(top_flat_idx, arr.shape)

    lat_name, lon_name = get_lat_lon_names(da)
    lats = da[lat_name].values
    lons = da[lon_name].values

    out = []
    for k in range(len(top_flat_idx)):
        i, j = top_2d[0][k], top_2d[1][k]
        lat_val = float(lats[i]) if lats.ndim == 1 else float(lats[i, j])
        lon_val = float(lons[j]) if lons.ndim == 1 else float(lons[i, j])
        out.append((lat_val, lon_val, float(arr[i, j])))
    return out


def cross_variable_checks(bio_data):
    """Cell-wise cross-variable consistency."""
    results = {}
    tol = 1e-3  # Celsius tolerance for floating-point equality

    if all(v in bio_data for v in ("BIO05", "BIO06", "BIO07")):
        diff = bio_data["BIO07"] - (bio_data["BIO05"] - bio_data["BIO06"])
        abs_diff = np.abs(diff.values)
        results["BIO07 == BIO05 - BIO06"] = {
            "max_abs_diff": float(np.nanmax(abs_diff)),
            "cells_violating_tol": int((abs_diff > tol).sum()),
            "passes": int((abs_diff > tol).sum()) == 0,
        }

    if all(v in bio_data for v in ("BIO02", "BIO07")):
        diff = bio_data["BIO07"].values - bio_data["BIO02"].values
        violations = int((diff < -tol).sum())
        worst = float(np.nanmin(diff)) if np.isfinite(diff).any() else np.nan
        results["BIO02 <= BIO07"] = {
            "worst_value_of_BIO07_minus_BIO02": worst,
            "cells_violating": violations,
            "passes": violations == 0,
        }

    if all(v in bio_data for v in ("BIO13", "BIO14")):
        diff = bio_data["BIO13"].values - bio_data["BIO14"].values
        violations = int((diff < -tol).sum())
        worst = float(np.nanmin(diff)) if np.isfinite(diff).any() else np.nan
        results["BIO14 <= BIO13"] = {
            "worst_value_of_BIO13_minus_BIO14": worst,
            "cells_violating": violations,
            "passes": violations == 0,
        }

    if all(v in bio_data for v in ("BIO01", "BIO05", "BIO06")):
        upper = bio_data["BIO05"].values - bio_data["BIO01"].values
        lower = bio_data["BIO01"].values - bio_data["BIO06"].values
        upper_v = int((upper < -tol).sum())
        lower_v = int((lower < -tol).sum())
        results["BIO06 <= BIO01 <= BIO05"] = {
            "BIO01 > BIO05 violations": upper_v,
            "BIO01 < BIO06 violations": lower_v,
            "passes": upper_v == 0 and lower_v == 0,
        }

    return results


def nan_propagation_check(bio_data):
    """Verify all variables share the same land mask (NaN pattern of BIO01)."""
    if "BIO01" not in bio_data:
        return {"error": "BIO01 not present"}

    ref_mask = np.isnan(bio_data["BIO01"].values)
    results = {}
    for var, da in bio_data.items():
        m = np.isnan(da.values)
        extra = int((m & ~ref_mask).sum())
        missing = int((ref_mask & ~m).sum())
        results[var] = {
            "n_nan": int(m.sum()),
            "extra_nan_vs_BIO01": extra,
            "missing_nan_vs_BIO01": missing,
            "consistent": extra == 0 and missing == 0,
        }
    return results



# Report writer


def write_report(label, results, output_path):
    L = []
    L.append("=" * 78)
    L.append(f"INTERNAL QC REPORT — {label}")
    L.append("=" * 78)

    # 1. Cell counts
    L.append("\n1. LAND/OCEAN CELL COUNTS")
    L.append("-" * 78)
    if "BIO01" in results["stats"]:
        s = results["stats"]["BIO01"]
        total = s["n_valid"] + s["n_nan"]
        L.append(f"  Total land cells (BIO01, non-NaN):  {s['n_valid']:>15,}")
        L.append(f"  NaN cells (ocean + masked):         {s['n_nan']:>15,}")
        L.append(f"  Total grid cells:                   {total:>15,}")

    # 2. Headline statistics
    L.append("\n2. HEADLINE STATISTICS")
    L.append("-" * 78)
    header = f"  {'Var':<7}{'min':>12}{'p1':>12}{'p50':>12}{'p99':>12}{'p99.99':>12}{'max':>12}{'awm':>12}"
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for var in BIO_VARS:
        if var in results["stats"]:
            s = results["stats"][var]
            L.append(
                f"  {var:<7}{s['min']:>12.3f}{s['p1']:>12.3f}{s['p50']:>12.3f}"
                f"{s['p99']:>12.3f}{s['p9999']:>12.3f}{s['max']:>12.3f}"
                f"{s['area_weighted_mean']:>12.3f}"
            )
    L.append("\n  awm = area-weighted mean (cosine-of-latitude weighted)")
    L.append("  BIO3, BIO4, BIO15 are scaled by ×100 (WorldClim convention).")

    # 3. Upper-tail spatial locations
    L.append("\n3. UPPER-TAIL SPATIAL LOCATIONS (top 5 cells per variable)")
    L.append("-" * 78)
    for var, mode, units in TOP_LOC_TARGETS:
        if var not in results["top_locs"]:
            continue
        info = results["top_locs"][var]
        L.append(f"\n  {var}  [mode={info['mode']}, units = {units}]")
        for i, (lat, lon, val) in enumerate(info["locs"][:5], 1):
            lat_h = f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'}"
            lon_h = f"{abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"
            L.append(f"    #{i}  ({lat_h:>10}, {lon_h:>10})   value = {val:>10.2f}")

    # 4. Cross-variable consistency
    L.append("\n4. CROSS-VARIABLE CONSISTENCY")
    L.append("-" * 78)
    for check_name, info in results["cross"].items():
        status = "PASS" if info.get("passes") else "FAIL"
        L.append(f"  [{status}]  {check_name}")
        for k, v in info.items():
            if k == "passes":
                continue
            if isinstance(v, float):
                L.append(f"          {k}: {v:.6g}")
            else:
                L.append(f"          {k}: {v}")

    # 5. NaN propagation
    L.append("\n5. NAN PROPAGATION CHECK")
    L.append("-" * 78)
    nan_check = results["nan_check"]
    if "error" in nan_check:
        L.append(f"  ERROR: {nan_check['error']}")
    else:
        all_ok = all(v["consistent"] for v in nan_check.values())
        L.append(f"  Overall: {'ALL VARIABLES CONSISTENT WITH BIO01 LAND MASK' if all_ok else 'INCONSISTENCIES FOUND'}")
        if not all_ok:
            L.append("")
            L.append(f"  {'Var':<7}{'n_nan':>15}{'extra vs BIO01':>20}{'missing vs BIO01':>20}")
            for var in BIO_VARS:
                if var in nan_check:
                    info = nan_check[var]
                    if not info["consistent"]:
                        L.append(
                            f"  {var:<7}{info['n_nan']:>15,}"
                            f"{info['extra_nan_vs_BIO01']:>20,}"
                            f"{info['missing_nan_vs_BIO01']:>20,}"
                        )

    L.append("\n" + "=" * 78)
    L.append("END OF REPORT")
    L.append("=" * 78)

    text = "\n".join(L) + "\n"
    with open(output_path, "w") as f:
        f.write(text)
    return text



# Main


def main():
    parser = argparse.ArgumentParser(
        description="Internal QC for bioclimatic variables"
    )
    parser.add_argument("bio_dir",
                        help="Directory containing BIO01_1km.nc ... BIO19_1km.nc")
    parser.add_argument("--label", default="qc",
                        help="Label for output files (e.g. 'baseline')")
    args = parser.parse_args()

    bio_dir = Path(args.bio_dir).expanduser().resolve()
    if not bio_dir.is_dir():
        print(f"ERROR: {bio_dir} is not a directory")
        sys.exit(1)

    print(f"Loading BIO variables from: {bio_dir}")
    bio_data = {}
    for var in BIO_VARS:
        path = find_bio_file(bio_dir, var)
        if path is None:
            print(f"  WARN  : {var} not found")
            continue
        try:
            bio_data[var] = open_bio(path, var)
            print(f"  loaded: {var}  shape={bio_data[var].shape}  file={path.name}")
        except Exception as e:
            print(f"  ERROR loading {var}: {e}")

    if not bio_data:
        print("ERROR: no BIO files loaded; aborting.")
        sys.exit(1)

    ref_var = "BIO01" if "BIO01" in bio_data else next(iter(bio_data))
    weights = cosine_weights(bio_data[ref_var])

    # 1+2: Headline statistics
    print("\nComputing headline statistics ...")
    stats = {}
    for var, da in bio_data.items():
        stats[var] = headline_stats(da, weights)
        print(f"  {var}: min={stats[var]['min']:>9.2f}  "
              f"max={stats[var]['max']:>9.2f}  "
              f"awm={stats[var]['area_weighted_mean']:>9.2f}")

    # 3: Upper-tail locations
    print("\nFinding upper-tail spatial locations ...")
    top_locs = {}
    for var, mode, _ in TOP_LOC_TARGETS:
        if var in bio_data:
            top_locs[var] = {
                "mode": mode,
                "locs": get_top_locations(bio_data[var], n_top=10, mode=mode),
            }
            top1 = top_locs[var]["locs"][0]
            print(f"  {var:<6}({mode:<3}): #1 cell @ ({top1[0]:.3f}, {top1[1]:.3f}) = {top1[2]:.2f}")

    # 4: Cross-variable
    print("\nCross-variable consistency ...")
    cross = cross_variable_checks(bio_data)
    for k, v in cross.items():
        print(f"  {'PASS' if v['passes'] else 'FAIL'}: {k}")

    # 5: NaN propagation
    print("\nNaN propagation ...")
    nan_check = nan_propagation_check(bio_data)
    if "error" not in nan_check:
        all_ok = all(v["consistent"] for v in nan_check.values())
        print(f"  All variables consistent with BIO01 land mask: {all_ok}")

    results = {
        "stats": stats,
        "top_locs": top_locs,
        "cross": cross,
        "nan_check": nan_check,
    }

    out_dir = Path(".").resolve()
    report_path = out_dir / f"qc_report_{args.label}.txt"
    csv_path = out_dir / f"qc_stats_{args.label}.csv"

    write_report(args.label, results, report_path)
    print(f"\nReport written : {report_path}")

    rows = [{"variable": v, **s} for v, s in stats.items()]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV written    : {csv_path}")

    print("\n" + "=" * 70)
    print("DONE — paste the .txt report back so we can diagnose anything weird.")
    print("=" * 70)


if __name__ == "__main__":
    main()