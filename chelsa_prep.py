"""
chelsa_prep.py

  1. Set CHELSA_GLOBAL_DIR to where your downloaded .tif files are
  2. Verify CHELSA_EUROPE_DIR matches config.py CHELSA_EUROPE_DIR

@author: amirrezaborzoueinia
"""

import rioxarray as rxr
from pathlib import Path

from config import EUROPE_EXTENT, CHELSA_GLOBAL_DIR, CHELSA_EUROPE_DIR


def clip_all_chelsa():
    print("=" * 55)
    print("  CHELSA Europe Clipping")
    print(f"  Source : {CHELSA_GLOBAL_DIR}")
    print(f"  Output : {CHELSA_EUROPE_DIR}")
    print("=" * 55)

    CHELSA_EUROPE_DIR.mkdir(parents=True, exist_ok=True)

    e = EUROPE_EXTENT

    variables = ["tas", "tasmax", "tasmin", "pr"]
    total_written = 0
    total_skipped = 0
    total_missing = 0

    for var in variables:
        out_var_dir = CHELSA_EUROPE_DIR / var
        out_var_dir.mkdir(exist_ok=True)
        print(f"\n  Variable: {var}")

        for m in range(1, 13):
            # Try all common CHELSA V2.1 filename patterns
            candidates = [
                CHELSA_GLOBAL_DIR / f"CHELSA_{var}_{m:02d}_1981-2010_V.2.1.tif",
                CHELSA_GLOBAL_DIR / f"CHELSA_{var}_{m:02d}_1981-2010_V2.1.tif",
                CHELSA_GLOBAL_DIR / f"CHELSA_{var}_{m:02d}_1981-2010.tif",
                # Some downloads put files in subfolders:
                CHELSA_GLOBAL_DIR / var / f"CHELSA_{var}_{m:02d}_1981-2010_V.2.1.tif",
                CHELSA_GLOBAL_DIR / var / f"CHELSA_{var}_{m:02d}_1981-2010_V2.1.tif",
            ]

            src = next((c for c in candidates if c.exists()), None)

            if src is None:
                print(f"    ✗ Month {m:02d}: NOT FOUND")
                print(f"      Tried: {candidates[0].name}, {candidates[1].name}")
                total_missing += 1
                continue

            # Output filename matches the source filename for clarity
            dst = out_var_dir / src.name

            if dst.exists():
                size_mb = dst.stat().st_size / 1e6
                print(f"    ⚠ Month {m:02d}: SKIP (already exists, {size_mb:.1f} MB)")
                total_skipped += 1
                continue

            print(f"    Clipping month {m:02d} ...", end=" ", flush=True)

            da = rxr.open_rasterio(src, masked=True, mask_and_scale=True).squeeze()
            da_clip = da.rio.clip_box(
                minx=e["lon_min"],
                maxx=e["lon_max"],
                miny=e["lat_min"],
                maxy=e["lat_max"]
            )

            # Save as compressed float32 GeoTIFF
            # LZW compression: lossless, ~3-5x smaller than uncompressed
            da_clip.rio.to_raster(dst, compress="lzw", dtype="float32")

            size_mb = dst.stat().st_size / 1e6
            print(f"done ({size_mb:.1f} MB)")
            total_written += 1

    # Summary
    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Files written : {total_written}")
    print(f"  Files skipped : {total_skipped} (already existed)")
    print(f"  Files missing : {total_missing}")

    if total_missing > 0:
        print(f"\n  ✗ {total_missing} source files not found.")
        print(f"    Check CHELSA_GLOBAL_DIR in config.py and filename patterns.")
    else:
        total_mb = sum(
            f.stat().st_size for f in CHELSA_EUROPE_DIR.rglob("*.tif")
        ) / 1e6
        print(f"  Total size    : {total_mb:.0f} MB")
        print(f"\n  Next step: run chelsa_inspect.py on a clipped file")
        print(f"  to confirm units are correct, then run run_downscaling_full.py")


if __name__ == "__main__":
    clip_all_chelsa()
