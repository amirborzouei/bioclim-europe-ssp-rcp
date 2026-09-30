"""
verify_time_periods.py
@author: amirrezaborzoueinia
"""

import numpy as np
import xarray as xr
from config import SCENARIOS, PERIODS, NC_TIME_DIM

# Use one variable for the test — tasmax is most diagnostic
VARIABLE     = "tasmax"
TEST_SCENARIO = "ssp585"   # use highest-emission for clearest signal


def check_period(label, variable, scenario_key,
                  year_start, year_end):
    """
    Load one period, print diagnostics, return the climatology.
    """
    from data_utils import get_climatology

    print(f"\n  {'─'*50}")
    print(f"  {label}  ({year_start}–{year_end})")
    print(f"  {'─'*50}")


    # Step A: load the raw time series (before climatological averaging)
    # This is the most direct way to verify the splice

    from data_utils import load_variable, splice_historical_and_future
    from config import SKIP_PERIODS

    if label == "Baseline":
        # Historical only
        da_raw = load_variable(variable, "historical", year_start, year_end)
    else:
        hist_end = SCENARIOS["historical"]["years"][1]  # 2014
        if year_start <= hist_end:
            # This period needs the splice
            da_raw = splice_historical_and_future(
                variable, scenario_key, year_start, year_end, hist_end
            )
        else:
            # Pure future — just slice the SSP file
            da_raw = load_variable(variable, scenario_key, year_start, year_end)

    # Compute the actual years present in the data
    years = np.unique(da_raw[NC_TIME_DIM].dt.year.values)
    n_months = len(da_raw[NC_TIME_DIM])

    print(f"  Years in data  : {int(years.min())} → {int(years.max())}")
    print(f"  Months loaded  : {n_months}  "
          f"(expected {(year_end - year_start + 1) * 12})")

    # Check for gaps
    expected_years = set(range(year_start, year_end + 1))
    actual_years   = set(int(y) for y in years)
    missing        = expected_years - actual_years
    extra          = actual_years   - expected_years
    if missing:
        print(f"  ✗ MISSING YEARS: {sorted(missing)}")
    else:
        print(f"  ✓ No missing years")
    if extra:
        print(f"  ✗ EXTRA YEARS: {sorted(extra)}")

    # Check the splice boundary (2014 → 2015) if applicable
    if year_start <= 2014 < year_end:
        # Find months in 2014 and 2015
        months_2014 = da_raw.sel(
            {NC_TIME_DIM: da_raw[NC_TIME_DIM].dt.year == 2014}
        )
        months_2015 = da_raw.sel(
            {NC_TIME_DIM: da_raw[NC_TIME_DIM].dt.year == 2015}
        )
        print(f"  Months in 2014 : {len(months_2014[NC_TIME_DIM])}  "
              f"(from Historical file)")
        print(f"  Months in 2015 : {len(months_2015[NC_TIME_DIM])}  "
              f"(from SSP file)")

        # Check December 2014 / January 2015 values are physically continuous
        dec_2014 = float(months_2014.isel(
            {NC_TIME_DIM: -1}
        ).mean().values)
        jan_2015 = float(months_2015.isel(
            {NC_TIME_DIM: 0}
        ).mean().values)
        diff = abs(dec_2014 - jan_2015)
        print(f"  Dec 2014 mean  : {dec_2014:.2f} °C  (historical)")
        print(f"  Jan 2015 mean  : {jan_2015:.2f} °C  (SSP)")
        if diff < 15:
            print(f"  ✓ Splice boundary is physically continuous (diff={diff:.2f}°C)")
        else:
            print(f"  ✗ Large jump at splice boundary ({diff:.2f}°C) — check units")


    # Step B: compute the 12-month climatology

    clim = get_climatology(variable, scenario_key if label != "Baseline"
                           else "historical",
                           label, year_start, year_end)

    if clim is not None:
        ann_mean = float(clim.mean().values)
        print(f"  Climatology mean: {ann_mean:.2f} °C  "
              f"(annual average of all 12 months × all cells)")
        return ann_mean
    else:
        print(f"  ✗ Climatology returned None")
        return None



#  MAIN VERIFICATION RUN                                              


print("=" * 55)
print("  TIME PERIOD VERIFICATION")
print(f"  Variable: {VARIABLE}  |  Scenario: {TEST_SCENARIO}")
print("=" * 55)

means = {}

# Check 1 — Baseline
m = check_period("Baseline", VARIABLE, "historical", 1981, 2010)
if m: means["baseline"] = m

# Check 2 — 2011-2040 (the critical splice period)
m = check_period("2011-2040", VARIABLE, TEST_SCENARIO, 2011, 2040)
if m: means["2011-2040"] = m

# Check 3 — 2041-2070
m = check_period("2041-2070", VARIABLE, TEST_SCENARIO, 2041, 2070)
if m: means["2041-2070"] = m

# Check 4 — 2071-2100
m = check_period("2071-2100", VARIABLE, TEST_SCENARIO, 2071, 2100)
if m: means["2071-2100"] = m


#  WARMING PROGRESSION CHECK                                          


print(f"\n{'='*55}")
print(f"  WARMING PROGRESSION (should increase monotonically)")
print(f"{'='*55}")
labels_order = ["baseline", "2011-2040", "2041-2070", "2071-2100"]
prev = None
all_increasing = True
for lbl in labels_order:
    if lbl in means:
        val  = means[lbl]
        flag = ""
        if prev is not None and val <= prev:
            flag = "  ✗ NOT WARMING — check data"
            all_increasing = False
        else:
            flag = "  ✓"
        print(f"  {lbl:<12}: {val:6.2f} °C{flag}")
        prev = val

print()
if all_increasing:
    print("  ✓ ALL PERIODS PASS — warming progression is monotonic")
    print("  ✓ Splice and slicing logic is working correctly")
    print(f"\n  You are ready to run the full pipeline:")
    print(f"    from main import main")
    print(f"    main()   # regenerates all 456 coarse outputs")
else:
    print("  ✗ WARMING IS NOT MONOTONIC — investigate before running pipeline")
    print("    Check that 2011-2014 historical files are in the Historical folder")
    print("    Check SCENARIOS years in config.py")
