"""
run_downscaling_full.py

Change scenario_key at the bottom to the scenario you want

@author: amirrezaborzoueinia
"""

import gc
import time
import xarray as xr
from pathlib import Path

from config import (
    PERIODS, SCENARIOS, SKIP_PERIODS,
    CHELSA_EUROPE_DIR, OUTPUT_ROOT_1KM,
    CHELSA_TEMP_IN_KELVIN,
)
from data_utils import get_climatology
from downscaling import downscale_scenario
from bioclim_functions import compute_bioclim


def save_bioclim_1km(bioclim_dict: dict,
                      scenario_key: str,
                      period_label: str):
    """Save all 19 1km BioClim variables for one scenario-period."""
    out_dir = OUTPUT_ROOT_1KM / scenario_key / period_label
    out_dir.mkdir(parents=True, exist_ok=True)

    for var_name, da in bioclim_dict.items():
        # Note: 1km outputs use _1km suffix to distinguish from coarse outputs
        out_path = out_dir / f"{var_name}_1km.nc"
        da.name  = var_name
        da.attrs.update({
            "scenario":   scenario_key,
            "period":     period_label,
            "source_gcm": "IPSL-CM6A-LR",
            "baseline":   "CHELSA_V2.1_1981-2010",
            "method":     "delta_downscaling",
            "resolution": "1km",
        })
        ds = da.to_dataset(name=var_name)
        ds.to_netcdf(
            out_path,
            encoding={var_name: {
                "zlib":      True,
                "complevel": 4,
                "dtype":     "float32",
            }}
        )
    return out_dir


def run_scenario(scenario_key: str):
    """
    Run the full 1km downscaling pipeline for one scenario.
    Processes all available periods sequentially.
    """
    t_total = time.time()

    print(f"\n{'='*58}")
    print(f"  1km Downscaling: {scenario_key.upper()}")
    print(f"  CHELSA: {CHELSA_EUROPE_DIR}")
    print(f"  Output: {OUTPUT_ROOT_1KM}")
    print(f"{'='*58}")

    # ---- Load GCM HISTORICAL climatology ONCE ----
    # This is reused for every future period as the baseline for delta.
    # Four variables: tas (mean), tasmin/tasmax (extremes), pr (precipitation).
    print("\n  Loading GCM historical climatology (1981-2010)...")
    gcm_hist = {}
    for var in ("tas", "tasmin", "tasmax", "pr"):
        gcm_hist[var] = get_climatology(
            var, "historical", "baseline", 1981, 2010
        )
        if gcm_hist[var] is None:
            print(f"  ✗ Failed to load historical {var}. Aborting.")
            return
    print("  Done.")

    # Determine which periods to process
    if scenario_key == "historical":
        periods_to_run = {"baseline": PERIODS["baseline"]}
    else:
        periods_to_run = {
            k: v for k, v in PERIODS.items()
            if k != "baseline"
        }

    # ---- Process each period ----
    for period_label, (y0, y1) in periods_to_run.items():

        # Skip check
        if scenario_key in SKIP_PERIODS:
            if period_label in SKIP_PERIODS[scenario_key]:
                print(f"\n  ⚠ SKIP {scenario_key}/{period_label} "
                      f"(SSP5-3.4OS missing 2015-2039)")
                continue

        # Check if already done
        out_dir = OUTPUT_ROOT_1KM / scenario_key / period_label
        existing = list(out_dir.glob("*_1km.nc")) if out_dir.exists() else []
        if len(existing) == 19:
            print(f"\n  ⚠ SKIP {period_label}: already complete (19 files found)")
            continue

        print(f"\n  Period: {period_label} ({y0}-{y1})")
        t_period = time.time()

        # Load future GCM climatology (4 variables)
        print("  Loading future climatology...")
        gcm_future = {}
        skip = False
        for var in ("tas", "tasmin", "tasmax", "pr"):
            if scenario_key == "historical":
                gcm_future[var] = gcm_hist[var]  # for historical baseline
            else:
                gcm_future[var] = get_climatology(
                    var, scenario_key, period_label, y0, y1
                )
            if gcm_future[var] is None:
                print(f"  ✗ Failed to load {var}. Skipping period.")
                skip = True
                break
        if skip:
            continue

        # Delta downscaling to 1km
        print("  Running delta downscaling:")
        try:
            ds_1km = downscale_scenario(
                gcm_hist, gcm_future,
                CHELSA_EUROPE_DIR,
                CHELSA_TEMP_IN_KELVIN
            )
        except Exception as exc:
            print(f"  ✗ Downscaling failed: {exc}")
            continue

        print(f"  Output shape: {ds_1km['tasmin'].shape}")
        print(f"  Output dtype: {ds_1km['tasmin'].dtype}")

        # Compute 19 BioClim variables at 1km (two logic paths: tas + extremes)
        print("  Computing BIO01-BIO19 at 1km ...", end=" ", flush=True)
        try:
            bioclim = compute_bioclim(
                ds_1km["tas"],
                ds_1km["tasmin"],
                ds_1km["tasmax"],
                ds_1km["pr"]
            )
        except Exception as exc:
            print(f"\n  ✗ BioClim computation failed: {exc}")
            continue
        print("done")

        # Save outputs
        print("  Saving 19 files ...", end=" ", flush=True)
        try:
            out_dir = save_bioclim_1km(bioclim, scenario_key, period_label)
        except Exception as exc:
            print(f"\n  ✗ Save failed: {exc}")
            continue
        print(f"done → {out_dir.relative_to(OUTPUT_ROOT_1KM.parent)}")

        elapsed = time.time() - t_period
        print(f"  Period complete in {elapsed/60:.1f} min")

        del ds_1km, bioclim, gcm_future
        gc.collect()

    total = time.time() - t_total
    print(f"\n  {'='*50}")
    print(f"  Scenario {scenario_key} complete in {total/60:.1f} min")
    print(f"  Output: {OUTPUT_ROOT_1KM / scenario_key}")



#  CHANGE THIS to run different scenarios                             
#  Uncomment one line per session                                     


if __name__ == "__main__":
    # run_scenario("historical")    # Session 1 
    # run_scenario("ssp585")        # Session 2 
    # run_scenario("ssp245")        # Session 3
    # run_scenario("ssp119")        # Session 4
    # run_scenario("ssp126")        # Session 5
    # run_scenario("ssp370")        # Session 6
    # run_scenario("ssp434")        # Session 7
    # run_scenario("ssp460")        # Session 8
     run_scenario("ssp534os")      # Session 9 (only 2 periods)
