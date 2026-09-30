"""
check_outputs.py
@author: amirrezaborzoueinia
"""

import xarray as xr
import numpy as np
from pathlib import Path
from config import OUTPUT_ROOT, OUTPUT_ROOT_1KM


EXPECTED_RANGES = {
    "BIO01": (-60,   40,  "degC"),
    "BIO02": (  0,   30,  "degC"),
    "BIO03": (  0,  100,  "%"),
    "BIO04": (  0, 1500,  "degC*100"),
    "BIO05": (-40,   60,  "degC"),   # max summer month, SE domain edge
    "BIO06": (-70,   30,  "degC"),
    "BIO07": (  0,   80,  "degC"),
    "BIO08": (-60,   50,  "degC"),   # quarterly mean, raised for SE domain
    "BIO09": (-60,   50,  "degC"),   # quarterly mean, raised for SE domain
    "BIO10": (-40,   50,  "degC"),   # warmest quarter mean
    "BIO11": (-60,   30,  "degC"),
    "BIO12": (  0, 5000,  "mm/yr"),
    "BIO13": (  0,  650,  "mm/mo"),  # raised for future Norwegian precip
    "BIO14": (  0,  300,  "mm/mo"),
    "BIO15": (  0,  250,  "%"),      # raised for arid Mediterranean CV
    "BIO16": (  0, 1700,  "mm/qtr"),
    "BIO17": (  0, 1500,  "mm/qtr"),
    "BIO18": (  0, 1700,  "mm/qtr"),
    "BIO19": (  0, 1700,  "mm/qtr"),
}


#  Geographic location labels for extreme value reporting             

def _geo_label(lat: float, lon: float) -> str:
    """Return a human-readable geographic label for a lat/lon point."""
    if lon < -14 and 62 < lat < 68:
        return "Iceland"
    if lon < -14:
        return "Atlantic Ocean — likely artifact"
    if -14 < lon < -5 and 57 < lat < 61:
        return "Hebrides / Western Scotland"
    if -8 < lon < -4 and 51 < lat < 56:
        return "Ireland / Wales coast"
    if 4 < lon < 9 and 59 < lat < 63:
        return "Western Norway (Sognefjord region)"
    if 5 < lon < 30 and 68 < lat < 72:
        return "Northern Norway / Svalbard"
    if -5 < lon < 35 and 35 < lat < 42:
        return "Mediterranean basin"
    if 25 < lon < 45 and 36 < lat < 42:
        return "Turkey / Eastern Mediterranean"
    if 20 < lon < 45 and 42 < lat < 50:
        return "Eastern Europe / Balkans"
    return f"Europe (lat={lat:.1f}, lon={lon:.1f})"



#  Core check function                                                 

def check_folder(folder: Path, suffix: str = "_1km") -> dict:
    """
    Check all BioClim NetCDF files in a folder.

    Parameters
    ----------
    folder : Path to folder containing BIO01*.nc ... BIO19*.nc
    suffix : file suffix after variable name ("_1km" or "")

    Returns
    -------
    dict with pass/fail counts and any issues found
    """
    nc_files = sorted(folder.glob(f"BIO*{suffix}.nc"))
    if not nc_files:
        # Try without suffix
        nc_files = sorted(folder.glob("BIO*.nc"))

    if not nc_files:
        print(f"  ✗ No BioClim files found in {folder}")
        return {"pass": 0, "fail": 1, "issues": ["No files found"]}

    print(f"\n  Checking {len(nc_files)} files in:")
    print(f"  {folder.relative_to(folder.parents[2])}")
    print(f"  {'Variable':<8} {'Shape':<16} {'Min':>8} {'Max':>10} "
          f"{'NaN%':>6}  {'Status'}")
    print(f"  {'─'*70}")

    results = {"pass": 0, "fail": 0, "issues": [], "stats": {}}

    for f in nc_files:
        varname = f.stem.replace(suffix, "").replace("_1km", "")
        if varname not in EXPECTED_RANGES:
            continue

        ds    = xr.open_dataset(f)
        da    = ds[varname]
        vals  = da.values
        vmin  = float(np.nanmin(vals))
        vmax  = float(np.nanmax(vals))
        nan_pct = float(np.isnan(vals).sum() / vals.size * 100)
        shape = da.shape

        lo, hi, unit = EXPECTED_RANGES[varname]
        issues = []

        if vmin < lo:
            issues.append(f"min {vmin:.1f} below physical minimum {lo}")
        if vmax > hi:
            # Find location of maximum
            idx_flat = np.nanargmax(vals)
            yi, xi   = np.unravel_index(idx_flat, vals.shape)
            lat      = float(da.y[yi]) if "y" in da.coords else float(da.lat[yi])
            lon      = float(da.x[xi]) if "x" in da.coords else float(da.lon[xi])
            loc      = _geo_label(lat, lon)
            issues.append(
                f"max {vmax:.1f} exceeds physical cap {hi} "
                f"at {loc} (lat={lat:.3f}, lon={lon:.3f})"
            )
        # NaN% ocean masking check only applies to 1km outputs.
        # Coarse GCM outputs (shape 35x27) are never land-masked by design —
        # the IPSL grid covers ocean cells and that is scientifically correct.
        # Only flag if: 1km output (many cells) AND precipitation variable
        # AND NaN% is near zero (land mask not applied).
        is_1km_output  = vals.size > 100000   # 1km Europe ~22M cells
        is_precip_var  = varname in ("BIO12", "BIO13", "BIO14")
        if is_1km_output and is_precip_var and nan_pct < 1:
            issues.append(f"NaN%={nan_pct:.0f}% — ocean masking may be missing")

        status = "✓ PASS" if not issues else f"✗ FAIL ({len(issues)} issue{'s' if len(issues)>1 else ''})"
        print(f"  {varname:<8} {str(shape):<16} {vmin:>8.1f} {vmax:>10.1f} "
              f"{nan_pct:>5.0f}%  {status}")

        for issue in issues:
            print(f"           ⚠ {issue}")
            results["issues"].append(f"{varname}: {issue}")

        results["stats"][varname] = {
            "min": vmin, "max": vmax, "nan_pct": nan_pct
        }

        if issues:
            results["fail"] += 1
        else:
            results["pass"] += 1

        ds.close()

    print(f"\n  Result: {results['pass']} passed, {results['fail']} failed")
    return results



#  Check all available outputs                                         

def check_all(root: Path, suffix: str = "_1km"):
    """Check all scenario/period folders under root."""
    print("=" * 60)
    print(f"  BioClim Output QC Check")
    print(f"  Root: {root}")
    print("=" * 60)

    total_pass = total_fail = 0
    all_issues = []

    for scen_dir in sorted(root.iterdir()):
        if not scen_dir.is_dir():
            continue
        for period_dir in sorted(scen_dir.iterdir()):
            if not period_dir.is_dir():
                continue
            r = check_folder(period_dir, suffix)
            total_pass += r["pass"]
            total_fail += r["fail"]
            all_issues += r["issues"]

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed")
    if all_issues:
        print(f"\n  Issues to review:")
        for issue in all_issues[:20]:
            print(f"    ✗ {issue}")
        if len(all_issues) > 20:
            print(f"    ... and {len(all_issues)-20} more")
    else:
        print(f"  All outputs within physical plausibility bounds.")
    print()



#  Quick single-folder check                                           

def quick_check_baseline():
    """Quick check of the historical baseline — run this first."""
    base_1km   = OUTPUT_ROOT_1KM / "historical" / "baseline"
    base_coarse = OUTPUT_ROOT    / "historical" / "baseline"

    if base_1km.exists():
        check_folder(base_1km, suffix="_1km")
    if base_coarse.exists():
        check_folder(base_coarse, suffix="")



#  Run when executed directly                                          

if __name__ == "__main__":
    print("\nChecking 1km outputs...")
    if OUTPUT_ROOT_1KM.exists():
        check_all(OUTPUT_ROOT_1KM, suffix="_1km")
    else:
        print(f"  1km output folder not found: {OUTPUT_ROOT_1KM}")

    print("\nChecking coarse-resolution outputs...")
    if OUTPUT_ROOT.exists():
        check_all(OUTPUT_ROOT, suffix="")
    else:
        print(f"  Coarse output folder not found: {OUTPUT_ROOT}")
