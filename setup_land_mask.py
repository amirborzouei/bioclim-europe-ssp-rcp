"""
setup_land_mask.py
@author: amirrezaborzoueinia
"""

import numpy as np
import xarray as xr
from pathlib import Path

from config import CHELSA_EUROPE_DIR, OUTPUT_ROOT, NC_LAT_DIM, NC_LON_DIM
from land_mask_utils import (
    create_land_mask,
    coarsen_mask_to_gcm,
)


def main():
    print("=" * 55)
    print("  Land Mask Setup")
    print("=" * 55)

    # ---- Step 1: Find a reference CHELSA file ----
    # Any clipped Europe file will do — we just need the grid
    candidates = list(CHELSA_EUROPE_DIR.rglob("*.tif"))
    if not candidates:
        print(f"\n  ✗ No clipped CHELSA files found in {CHELSA_EUROPE_DIR}")
        print(f"    Run chelsa_prep.py first.")
        return

    ref_file   = candidates[0]
    mask_path  = CHELSA_EUROPE_DIR / "land_mask_1km.tif"

    print(f"\n  Reference file : {ref_file.name}")
    print(f"  Output mask    : {mask_path}")

    # ---- Step 2: Create 1km mask ----
    if mask_path.exists():
        print(f"\n  1km mask already exists ({mask_path.stat().st_size/1e6:.1f} MB)")
        print(f"  Delete {mask_path.name} and rerun if you need to recreate it.")
        from land_mask_utils import load_land_mask
        mask_1km = load_land_mask(mask_path)
    else:
        print("\n  Creating 1km land mask from GSHHG L1 full-resolution polygons...")
        mask_1km = create_land_mask(ref_file, mask_path)

    # ---- Step 3: Quick visual check ----
    land_pct = float(mask_1km.mean().values) * 100
    print(f"\n  Land coverage over clipped domain: {land_pct:.1f}%")
    print(f"  (Expected ~40-50% for Europe + surrounding ocean strip)")

    # ---- Step 4: Coarsen to GCM resolution ----
    # Load one GCM file to get the lat/lon grid
    gcm_nc_files = list(OUTPUT_ROOT.rglob("BIO01.nc"))
    if gcm_nc_files:
        ds_gcm   = xr.open_dataset(gcm_nc_files[0])
        gcm_lats = ds_gcm[NC_LAT_DIM].values
        gcm_lons = ds_gcm[NC_LON_DIM].values
        ds_gcm.close()

        print("\n  Coarsening mask to GCM resolution...")
        mask_gcm = coarsen_mask_to_gcm(mask_1km, gcm_lats, gcm_lons)

        # Save GCM mask as NetCDF
        gcm_mask_path = CHELSA_EUROPE_DIR / "gcm_land_mask.nc"
        mask_gcm.name = "land_mask"
        mask_gcm.attrs = {
            "description": "Land mask coarsened to IPSL-CM6A-LR grid",
            "source":      "GSHHG v2.3.7 L1 full-resolution, threshold=0.5",
        }
        mask_gcm.to_dataset(name="land_mask").to_netcdf(gcm_mask_path)
        gcm_land_pct = float(mask_gcm.mean().values) * 100
        print(f"  GCM land coverage: {gcm_land_pct:.1f}%")
        print(f"  Saved: {gcm_mask_path.name}")
    else:
        print("\n  ⚠ No BioClim outputs found — skipping GCM mask creation.")
        print(f"    Run main.py baseline first, then re-run this script.")

    # ---- Summary ----
    print(f"\n{'='*55}")
    print(f"  Done. Files saved to: {CHELSA_EUROPE_DIR}")
    print(f"\n  How to use in your code:")
    print(f"    from land_mask_utils import load_land_mask, apply_land_mask")
    print(f"    mask = load_land_mask(CHELSA_EUROPE_DIR / 'land_mask_1km.tif')")
    print(f"    da_land = apply_land_mask(da, mask)")
    print()


if __name__ == "__main__":
    main()

