"""
run_seasonal.py
Change scenario_key at the bottom.

@author: amirrezaborzoueinia
"""

import gc
import time
from pathlib import Path

from config import (
    PERIODS, SKIP_PERIODS,
    CHELSA_EUROPE_DIR, CHELSA_TEMP_IN_KELVIN, OUTPUT_ROOT_ECON,
)
from data_utils import get_climatology
from downscaling import downscale_scenario
from seasonal_function import compute_seasonal, save_seasonal


def run_scenario(scenario_key: str):
    t_total = time.time()
    print(f"\n{'='*58}\n  Seasonal covariates: {scenario_key.upper()}\n{'='*58}")

    # ---- GCM historical climatology (loaded once; needed by downscale_scenario) ----
    print("\n  Loading GCM historical climatology (1981-2010)...")
    gcm_hist = {}
    for var in ("tas", "tasmin", "tasmax", "pr"):
        gcm_hist[var] = get_climatology(var, "historical", "baseline", 1981, 2010)
        if gcm_hist[var] is None:
            print(f"  Failed to load historical {var}. Aborting.")
            return
    print("  Done.")

    if scenario_key == "historical":
        periods_to_run = {"baseline": PERIODS["baseline"]}
    else:
        periods_to_run = {k: v for k, v in PERIODS.items() if k != "baseline"}

    for period_label, (y0, y1) in periods_to_run.items():
        if scenario_key in SKIP_PERIODS and period_label in SKIP_PERIODS[scenario_key]:
            print(f"\n  SKIP {scenario_key}/{period_label} (missing years)")
            continue
        out_file = OUTPUT_ROOT_ECON / scenario_key / period_label / "ricardian_covariates_1km.nc"
        if out_file.exists():
            print(f"\n  SKIP {period_label}: already done ({out_file.name})")
            continue

        print(f"\n  Period: {period_label} ({y0}-{y1})")
        t_p = time.time()

        # future GCM climatology
        gcm_future = {}
        skip = False
        for var in ("tas", "tasmin", "tasmax", "pr"):
            if scenario_key == "historical":
                gcm_future[var] = gcm_hist[var]
            else:
                gcm_future[var] = get_climatology(var, scenario_key, period_label, y0, y1)
            if gcm_future[var] is None:
                print(f"  Failed to load {var}. Skipping period.")
                skip = True
                break
        if skip:
            continue

        # 1 km downscaling (reuses the pipeline; bilinear)
        print("  Downscaling to 1 km ...")
        try:
            ds_1km = downscale_scenario(gcm_hist, gcm_future,
                                        CHELSA_EUROPE_DIR, CHELSA_TEMP_IN_KELVIN)
        except Exception as exc:
            print(f"  Downscaling failed: {exc}")
            continue

        # seasonal Ricardian covariates from the monthly tas and pr
        print("  Computing seasonal covariates ...", end=" ", flush=True)
        cov = compute_seasonal(ds_1km["tas"], ds_1km["pr"])
        out_path = save_seasonal(cov, scenario_key, period_label, OUTPUT_ROOT_ECON)
        print(f"done -> {out_path.relative_to(OUTPUT_ROOT_ECON.parent)}")
        print(f"  Period complete in {(time.time()-t_p)/60:.1f} min")

        del ds_1km, cov, gcm_future
        gc.collect()

    print(f"\n  Scenario {scenario_key} complete in {(time.time()-t_total)/60:.1f} min")



#  Uncomment one line per session (or wrap in a loop)                

if __name__ == "__main__":
    # run_scenario("historical")      # baseline first
    # run_scenario("ssp585")
    # run_scenario("ssp245")
    # run_scenario("ssp119")
    # run_scenario("ssp126")
    # run_scenario("ssp370")
    # run_scenario("ssp434")
    # run_scenario("ssp460")
     run_scenario("ssp534os")
