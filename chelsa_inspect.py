"""
chelsa_inspect.py

STEP 1 OF DOWNSCALING — Run this before chelsa_prep.py or anything else.

@author: amirrezaborzoueinia
"""

from pathlib import Path
import numpy as np

# ── CHANGE THIS to your CHELSA folder ───────────────────────────────

from config import CHELSA_EUROPE_DIR

CHELSA_SAMPLE_DIR = CHELSA_EUROPE_DIR
# ────────────────────────────────────────────────────────────────────

import rioxarray as rxr


def inspect_file(filepath: Path, label: str, known_ocean_lon: float = -15.0,
                  known_ocean_lat: float = 50.0, known_land_lon: float = 2.5,
                  known_land_lat: float = 47.0):
    """
    Open one GeoTIFF and print all diagnostically useful information.
    Checks one known ocean point (mid-Atlantic) and one land point (France).
    """
    print(f"\n{'─'*55}")
    print(f"  FILE: {filepath.name}")
    print(f"  LABEL: {label}")
    print(f"{'─'*55}")

    # Open WITHOUT mask_and_scale first to see raw packed values
    da_raw = rxr.open_rasterio(filepath).squeeze()
    raw_mean = float(np.nanmean(da_raw.values))
    print(f"  Raw mean (no scaling)  : {raw_mean:.4f}")
    print(f"  scale_factor attr      : {da_raw.attrs.get('scale_factor', 'NOT PRESENT')}")
    print(f"  add_offset attr        : {da_raw.attrs.get('add_offset',   'NOT PRESENT')}")
    print(f"  units attr             : {da_raw.attrs.get('units',        'NOT PRESENT')}")
    print(f"  nodata value           : {da_raw.rio.nodata}")

    # Open WITH mask_and_scale=True to see real-world values
    da = rxr.open_rasterio(filepath, masked=True, mask_and_scale=True).squeeze()
    real_mean = float(np.nanmean(da.values))
    real_min  = float(np.nanmin(da.values))
    real_max  = float(np.nanmax(da.values))
    print(f"\n  After mask_and_scale=True:")
    print(f"  Mean  : {real_mean:.4f}")
    print(f"  Min   : {real_min:.4f}")
    print(f"  Max   : {real_max:.4f}")
    print(f"  Shape : {da.shape}")
    print(f"  Dims  : {da.dims}")
    print(f"  x range: {float(da.x.min()):.3f} → {float(da.x.max()):.3f}")
    print(f"  y range: {float(da.y.min()):.3f} → {float(da.y.max()):.3f}")

    # Ocean mask check
    ocean_val = float(da.sel(x=known_ocean_lon, y=known_ocean_lat,
                              method="nearest").values)
    land_val  = float(da.sel(x=known_land_lon,  y=known_land_lat,
                              method="nearest").values)
    nan_pct   = np.isnan(da.values).sum() / da.values.size * 100

    print(f"\n  Ocean point ({known_ocean_lat}N, {known_ocean_lon}E): {ocean_val}")
    print(f"  Land  point ({known_land_lat}N,  {known_land_lon}E): {land_val:.4f}")
    print(f"  NaN percentage: {nan_pct:.1f}%")

    # Diagnose
    print(f"\n  DIAGNOSIS:")
    if "tasmax" in label.lower() or "tasmin" in label.lower() or "tas_" in label.lower():
        if real_mean > 100:
            print(f"  ✓ Temperature is in KELVIN (mean={real_mean:.1f}K)")
            print(f"    → Set CHELSA_TEMP_IN_KELVIN = True in config.py")
        elif -60 < real_mean < 60:
            print(f"  ✓ Temperature is in CELSIUS (mean={real_mean:.1f}°C)")
            print(f"    → Set CHELSA_TEMP_IN_KELVIN = False in config.py")
        else:
            print(f"  ✗ UNEXPECTED range. Raw={raw_mean:.2f} Scaled={real_mean:.4f}")
            print(f"    → File may need different opening approach")

    elif "pr" in label.lower():
        if 5 < real_mean < 500:
            print(f"  ✓ Precipitation in mm/month (mean={real_mean:.1f} mm)")
            print(f"    → No conversion needed for CHELSA pr")
        elif real_mean < 0.01:
            print(f"  ✗ Precipitation in kg/m²/s (mean={real_mean:.2e})")
            print(f"    → Needs conversion in downscaling.py")
        else:
            print(f"  ⚠ Unexpected precipitation range: {real_mean:.4f}")

    if np.isnan(ocean_val):
        print(f"  ✓ Ocean is NaN — land mask already applied by CHELSA")
        print(f"    → Your downscaled outputs will automatically mask ocean")
    else:
        print(f"  ✗ Ocean has value {ocean_val:.2f} — no land mask")
        print(f"    → You will need to apply a land mask separately")

    return {
        "mean": real_mean, "min": real_min, "max": real_max,
        "ocean_is_nan": np.isnan(ocean_val),
        "nan_pct": nan_pct,
    }


def main():
    print("=" * 55)
    print("  CHELSA FILE INSPECTION")
    print("  Run this before chelsa_prep.py or downscaling.py")
    print("=" * 55)

    # Find sample files
    results = {}

    for var in ["tas", "tasmax", "tasmin", "pr"]:
        # Try to find any file for this variable
        candidates = list(CHELSA_SAMPLE_DIR.rglob(f"*{var}*01*.tif"))
        if not candidates:
            candidates = list(CHELSA_SAMPLE_DIR.rglob(f"*{var}*.tif"))

        if not candidates:
            print(f"\n  ✗ No {var} files found in {CHELSA_SAMPLE_DIR}")
            print(f"    Check CHELSA_SAMPLE_DIR path at top of this file")
            continue

        f = candidates[0]
        results[var] = inspect_file(f, var)

    # Final summary
    print(f"\n{'='*55}")
    print(f"  SUMMARY — copy these settings to config.py")
    print(f"{'='*55}")

    if "tasmax" in results:
        kelvin = results["tasmax"]["mean"] > 100
        print(f"  CHELSA_TEMP_IN_KELVIN = {kelvin}")

    if "pr" in results:
        needs_conv = results["pr"]["mean"] < 0.01
        print(f"  CHELSA_PR_NEEDS_CONVERSION = {needs_conv}")
        if not needs_conv:
            print(f"  (CHELSA pr is already in mm/month — no conversion needed)")

    ocean_masked = all(r["ocean_is_nan"] for r in results.values() if "ocean" in str(r))
    if results:
        first = list(results.values())[0]
        if first["ocean_is_nan"]:
            print(f"  ✓ Ocean mask: CHELSA NaN ocean cells will propagate")
            print(f"    through delta math automatically — no extra masking needed")
        else:
            print(f"  ✗ Ocean mask: not present in CHELSA files")
            print(f"    Apply land mask after downscaling")

    print()


if __name__ == "__main__":
    main()
