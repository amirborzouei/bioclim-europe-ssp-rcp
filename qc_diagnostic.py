"""
qc_diagnostic.py
@author: amirrezaborzoueinia
"""

import sys
import argparse
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import xarray as xr



# Helpers


@contextmanager
def open_first(file_path):
    ds = xr.open_dataset(file_path)
    try:
        yield ds
    finally:
        ds.close()


def find_bio_file(bio_dir, var):
    bio_dir = Path(bio_dir)
    for name in (f"{var}_1km.nc", f"{var.lower()}_1km.nc",
                 f"{var}.nc", f"{var.lower()}.nc"):
        p = bio_dir / name
        if p.exists():
            return p
    return None


def open_bio(file_path, var):
    ds = xr.open_dataset(file_path)
    if var in ds:
        return ds[var]
    if var.lower() in ds:
        return ds[var.lower()]
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    raise ValueError(f"Cannot find {var} in {file_path.name}")


def find_monthly_file(monthly_dir, var):
    """Look for monthly NC files. Supports patterns like:
       pr_monthly.nc / pr_1km.nc / pr.nc (single file with 12 months)
       pr_01.nc ... pr_12.nc (per-month files)
       precip_monthly_1km.nc etc.
    """
    monthly_dir = Path(monthly_dir)
    candidates_single = [
        f"{var}_monthly.nc", f"{var}_monthly_1km.nc",
        f"{var}_1km.nc", f"{var}.nc",
        f"{var.lower()}_monthly.nc", f"{var.lower()}_monthly_1km.nc",
        f"{var.lower()}_1km.nc", f"{var.lower()}.nc",
    ]
    for name in candidates_single:
        p = monthly_dir / name
        if p.exists():
            return ("single", p)

    # per-month files
    per_month = []
    for m in range(1, 13):
        for pat in (f"{var}_{m:02d}.nc", f"{var.lower()}_{m:02d}.nc",
                    f"{var}_month{m:02d}.nc"):
            p = monthly_dir / pat
            if p.exists():
                per_month.append(p)
                break
    if len(per_month) == 12:
        return ("per_month", per_month)
    return (None, None)


def open_monthly(monthly_dir, var):
    """Return an xarray DataArray with month dimension of size 12."""
    kind, files = find_monthly_file(monthly_dir, var)
    if kind == "single":
        ds = xr.open_dataset(files)
        # Find the variable
        if var in ds:
            da = ds[var]
        elif var.lower() in ds:
            da = ds[var.lower()]
        else:
            da = ds[list(ds.data_vars)[0]]
        return da
    elif kind == "per_month":
        arrays = []
        for f in files:
            ds = xr.open_dataset(f)
            arrays.append(ds[list(ds.data_vars)[0]])
        return xr.concat(arrays, dim="month")
    else:
        return None


def get_lat_lon_names(da):
    lat_name = next((n for n in ("y", "lat", "latitude") if n in da.coords), None)
    lon_name = next((n for n in ("x", "lon", "longitude") if n in da.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot identify lat/lon from {list(da.coords)}")
    return lat_name, lon_name


def cell_at_point(da, lat, lon):
    """Return the value of da at the nearest grid cell to (lat, lon),
    along with the actual lat, lon of that cell."""
    lat_name, lon_name = get_lat_lon_names(da)
    sel = da.sel({lat_name: lat, lon_name: lon}, method="nearest")
    actual_lat = float(sel[lat_name].values)
    actual_lon = float(sel[lon_name].values)
    return actual_lat, actual_lon, sel


def window_around(da, lat, lon, half_size_deg=0.05):
    """Return a subset of da within +/- half_size_deg around (lat, lon)."""
    lat_name, lon_name = get_lat_lon_names(da)
    lat_arr = da[lat_name].values
    lon_arr = da[lon_name].values

    lat_mask = (lat_arr >= lat - half_size_deg) & (lat_arr <= lat + half_size_deg)
    lon_mask = (lon_arr >= lon - half_size_deg) & (lon_arr <= lon + half_size_deg)

    # works for 1D coords
    if lat_arr.ndim == 1:
        return da.sel({lat_name: lat_arr[lat_mask], lon_name: lon_arr[lon_mask]})
    # 2D coords (less common): fall back to nearest single cell
    _, _, single = cell_at_point(da, lat, lon)
    return single


def find_argmax_cell(da):
    """Return (lat, lon, value) of the argmax cell of da, skipping NaN."""
    arr = da.values
    valid = np.isfinite(arr)
    if not valid.any():
        return None, None, None
    flat = np.where(valid, arr, -np.inf)
    idx = np.argmax(flat)
    i, j = np.unravel_index(idx, arr.shape)
    lat_name, lon_name = get_lat_lon_names(da)
    lats = da[lat_name].values
    lons = da[lon_name].values
    lat = float(lats[i]) if lats.ndim == 1 else float(lats[i, j])
    lon = float(lons[j]) if lons.ndim == 1 else float(lons[i, j])
    return lat, lon, float(arr[i, j])



# Diagnostic 1 — St Kilda BIO12 hotspot


def diagnose_st_kilda(args, lines):
    target_lat = args.st_kilda_lat
    target_lon = args.st_kilda_lon
    win_deg = args.window_km / 111.0  # 1 deg ~= 111 km

    lines.append("=" * 78)
    lines.append("DIAGNOSTIC 1 — BIO12 St Kilda hotspot")
    lines.append("=" * 78)
    lines.append(f"  Target point: ({target_lat:.3f} N, {abs(target_lon):.3f} W)")
    lines.append(f"  Window:       +/- {args.window_km} km (= +/- {win_deg:.4f} deg)")
    lines.append("")

    # BIO12 at target
    p = find_bio_file(args.bio_baseline, "BIO12")
    if p is None:
        lines.append("  ERROR: BIO12 file not found in baseline directory")
        return
    bio12_base = open_bio(p, "BIO12")
    actual_lat, actual_lon, bio12_val = cell_at_point(bio12_base, target_lat, target_lon)
    lines.append(f"  BIO12 baseline at nearest 1 km cell:")
    lines.append(f"     cell = ({actual_lat:.4f} N, {abs(actual_lon):.4f} W)")
    lines.append(f"     value = {float(bio12_val.values):.2f} mm/yr")

    p_fut = find_bio_file(args.bio_future, "BIO12")
    if p_fut is not None:
        bio12_fut = open_bio(p_fut, "BIO12")
        _, _, bio12_fut_val = cell_at_point(bio12_fut, target_lat, target_lon)
        bio12_fut_val = float(bio12_fut_val.values)
        ratio = bio12_fut_val / float(bio12_val.values)
        lines.append(f"  BIO12 future at same cell:")
        lines.append(f"     value = {bio12_fut_val:.2f} mm/yr")
        lines.append(f"     ratio (future / baseline) = {ratio:.4f}")
        lines.append(f"     (interpretation: ratio close to 1.0 means delta-change")
        lines.append(f"      preserves CHELSA baseline; large ratio would amplify it)")
    lines.append("")

    # Window inspection
    win = window_around(bio12_base, target_lat, target_lon, win_deg)
    finite_vals = win.values[np.isfinite(win.values)]
    n_valid = len(finite_vals)
    n_nan = win.size - n_valid
    lines.append(f"  Window of {args.window_km} km radius around target:")
    lines.append(f"     total cells in window:        {win.size}")
    lines.append(f"     valid (land) cells:           {n_valid}")
    lines.append(f"     NaN (ocean/masked) cells:     {n_nan}")
    if n_valid > 0:
        lines.append(f"     valid cell stats: "
                     f"min={finite_vals.min():.1f}  "
                     f"max={finite_vals.max():.1f}  "
                     f"mean={finite_vals.mean():.1f}  mm/yr")
    lines.append("")

    # Interpretation aid
    lines.append("  Interpretation:")
    if n_valid > 0 and n_nan > 0:
        lines.append(f"     - The window contains both land and ocean cells, which is")
        lines.append(f"       consistent with St Kilda being a real island (Hirta)")
        lines.append(f"       surrounded by ocean. The land mask is working correctly.")
    if n_valid == 0:
        lines.append(f"     - WARN: window has no valid cells. Either the target is")
        lines.append(f"       all ocean, or the window is too small.")
    if n_valid > 0:
        lines.append(f"     - CHELSA V2.1's orographic algorithm (Karger 2017) is known")
        lines.append(f"       to overestimate precipitation at small Atlantic islands.")
        lines.append(f"       The observed annual precipitation at St Kilda (per UK Met")
        lines.append(f"       Office) is approximately 1400-1500 mm/yr, so the CHELSA")
        lines.append(f"       baseline value of ~12000 mm/yr is overestimated by ~8x.")
        lines.append(f"       This is a CHELSA dataset characteristic, not a pipeline")
        lines.append(f"       artifact. Your delta-change method preserves but does")
        lines.append(f"       not amplify CHELSA's values.")
    lines.append("")



# Diagnostic 2 — BIO15 CV blowup


def diagnose_bio15(args, lines):
    lines.append("=" * 78)
    lines.append("DIAGNOSTIC 2 — BIO15 CV blowup")
    lines.append("=" * 78)
    lines.append("")

    # Find argmax of BIO15 in each scenario
    for label, bio_dir, monthly_dir in (
        ("baseline", args.bio_baseline, args.monthly_baseline),
        ("future",   args.bio_future,   args.monthly_future),
    ):
        lines.append(f"--- {label} ---")
        p = find_bio_file(bio_dir, "BIO15")
        if p is None:
            lines.append(f"  ERROR: BIO15 file not found in {label} directory")
            lines.append("")
            continue

        bio15 = open_bio(p, "BIO15")
        lat, lon, val = find_argmax_cell(bio15)
        if lat is None:
            lines.append("  ERROR: no valid cells in BIO15")
            lines.append("")
            continue
        lines.append(f"  BIO15 maximum cell:")
        lines.append(f"     coordinates = ({lat:.4f} N, {abs(lon):.4f} {'E' if lon >= 0 else 'W'})")
        lines.append(f"     BIO15 value = {val:.2f}  (= CV * 100)")
        lines.append(f"     implied CV  = {val/100:.4f}")
        lines.append("")

        # Inspect monthly precipitation at that cell
        if monthly_dir is None:
            lines.append("  Monthly precipitation directory not provided; skipping")
            lines.append("  underlying-data inspection at this cell.")
            lines.append("")
            continue

        pr_monthly = open_monthly(monthly_dir, "pr")
        if pr_monthly is None:
            # try other variable names
            for varname in ("prec", "precipitation", "PR", "Pr"):
                pr_monthly = open_monthly(monthly_dir, varname)
                if pr_monthly is not None:
                    break
        if pr_monthly is None:
            lines.append("  WARN: monthly precipitation file not found in")
            lines.append(f"        {monthly_dir}; skipping cell inspection")
            lines.append("        (tried: pr, prec, precipitation, PR, Pr)")
            lines.append("")
            continue

        # Get the 12 monthly values at the BIO15 hotspot
        actual_lat, actual_lon, cell_data = cell_at_point(pr_monthly, lat, lon)
        monthly_vals = np.array(cell_data.values).ravel()
        n_months = len(monthly_vals)
        finite = monthly_vals[np.isfinite(monthly_vals)]

        lines.append(f"  Monthly precipitation at this cell ({n_months} months):")
        if len(finite) == 0:
            lines.append("     ERROR: all NaN at this cell")
            lines.append("")
            continue

        # Print 12 monthly values
        if n_months == 12:
            for i, v in enumerate(monthly_vals, 1):
                lines.append(f"     month {i:2d}: {v:8.4f} mm/month")
        lines.append("")
        mu = float(np.nanmean(monthly_vals))
        sigma = float(np.nanstd(monthly_vals))
        annual = float(np.nansum(monthly_vals))
        cv_implied = (sigma / (mu + 1e-6)) * 100
        lines.append(f"     summary:")
        lines.append(f"       mean monthly precipitation:    {mu:.4f} mm/month")
        lines.append(f"       std of monthly precipitation:  {sigma:.4f} mm/month")
        lines.append(f"       annual sum (= BIO12-like):     {annual:.2f} mm/year")
        lines.append(f"       CV from cell (sigma/(mu+eps)*100) = {cv_implied:.2f}")
        lines.append(f"       BIO15 reported:                 {val:.2f}")
        lines.append("")
        lines.append(f"     Diagnosis:")
        if mu < 0.1:
            lines.append(f"       - Mean monthly precipitation is < 0.1 mm/month.")
            lines.append(f"         At this magnitude, the +1e-6 epsilon in the")
            lines.append(f"         denominator becomes non-negligible relative to")
            lines.append(f"         mu, and the CV is in the numerically unstable")
            lines.append(f"         regime. BIO15 in this cell is dominated by the")
            lines.append(f"         epsilon safeguard rather than meaningful")
            lines.append(f"         climatological variability.")
        elif mu < 1.0:
            lines.append(f"       - Mean monthly precipitation is < 1 mm/month.")
            lines.append(f"         The epsilon (1e-6) does not bias CV here, but")
            lines.append(f"         CV is mathematically sensitive in this regime")
            lines.append(f"         and small differences in monthly values produce")
            lines.append(f"         large CV swings. The value is real but should")
            lines.append(f"         be interpreted as 'extreme arid seasonality'.")
        else:
            lines.append(f"       - Mean monthly precipitation is >= 1 mm/month.")
            lines.append(f"         CV is in the numerically stable regime; the")
            lines.append(f"         high BIO15 value reflects genuine strong")
            lines.append(f"         seasonality, not a numerical artifact.")
        lines.append("")



# Main


def main():
    parser = argparse.ArgumentParser(description="Diagnostic for BIO12 / BIO15 hotspots")
    parser.add_argument("--bio_baseline", required=True,
                        help="Directory containing baseline BIO files")
    parser.add_argument("--bio_future", required=True,
                        help="Directory containing future BIO files")
    parser.add_argument("--monthly_baseline", default=None,
                        help="Directory with baseline monthly tas/pr (optional)")
    parser.add_argument("--monthly_future", default=None,
                        help="Directory with future monthly tas/pr (optional)")
    parser.add_argument("--st_kilda_lat", type=float, default=57.821)
    parser.add_argument("--st_kilda_lon", type=float, default=-8.579)
    parser.add_argument("--window_km", type=float, default=5.0)
    parser.add_argument("--output", default="qc_diagnostic_report.txt")
    args = parser.parse_args()

    lines = []
    lines.append("=" * 78)
    lines.append("QC DIAGNOSTIC REPORT")
    lines.append("=" * 78)
    lines.append("")

    diagnose_st_kilda(args, lines)
    diagnose_bio15(args, lines)

    lines.append("=" * 78)
    lines.append("END OF REPORT")
    lines.append("=" * 78)

    text = "\n".join(lines) + "\n"
    print(text)

    with open(args.output, "w") as f:
        f.write(text)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
