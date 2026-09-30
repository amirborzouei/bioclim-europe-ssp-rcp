"""
main.py
@author: amirrezaborzoueinia
"""


import argparse
import gc
import sys
import time
from pathlib import Path

from config import (
    PERIODS, FUTURE_SCENARIOS, SKIP_PERIODS,
    DASK_WORKERS, FORCE_COMPUTE_AND_RELEASE,
)
from data_utils import get_climatology, release
from bioclim_functions import compute_bioclim
from output_utils import save_bioclim_variable

ALL_BIOCLIM = [f"BIO{i:02d}" for i in range(1, 20)]



#  Console output helpers                                             


def _banner(msg):
    w = 60
    print(f"\n{'═'*w}")
    print(f"  {msg}")
    print(f"{'═'*w}")

def _step(msg):
    print(f"\n  ── {msg}")

def _ok(msg):
    print(f"     ✓  {msg}")

def _skip(msg):
    print(f"     ⚠  {msg}")

def _fail(msg):
    print(f"     ✗  {msg}", file=sys.stderr)



#  Process one scenario × period combination                          


def process_scenario_period(scenario_key, period_label,
                              year_start, year_end,
                              variables_to_compute,
                              dry_run=False):
    """
    Load climatologies, compute BioClim variables, save outputs.
    Returns (n_success, n_skip, n_fail).
    """
    # Skip-list check
    if scenario_key in SKIP_PERIODS:
        if period_label in SKIP_PERIODS[scenario_key]:
            _skip(f"SKIP {scenario_key}/{period_label} "
                  f"(SSP5-3.4OS missing 2015-2039)")
            return 0, 1, 0

    if dry_run:
        print(f"     [DRY RUN] {scenario_key}/{period_label}/{variables_to_compute}")
        return 0, 0, 0

    t0 = time.time()
    _step(f"{scenario_key.upper()}  |  {period_label}  ({year_start}–{year_end})")

    # Load four monthly climatologies
    # tas is used directly for mean-based BIO variables (BIO01, BIO08-11, BIO04)
    # tasmin/tasmax for range-based variables (BIO02, BIO05-07)
    print("     Loading tas    ...", end=" ", flush=True)
    tas_  = get_climatology("tas",    scenario_key, period_label, year_start, year_end)
    if tas_ is None:
        return 0, 1, 0
    print("done")

    print("     Loading tasmin ...", end=" ", flush=True)
    tmin = get_climatology("tasmin", scenario_key, period_label, year_start, year_end)
    if tmin is None:
        return 0, 1, 0
    print("done")

    print("     Loading tasmax ...", end=" ", flush=True)
    tmax = get_climatology("tasmax", scenario_key, period_label, year_start, year_end)
    if tmax is None:
        return 0, 1, 0
    print("done")

    print("     Loading pr     ...", end=" ", flush=True)
    prec = get_climatology("pr", scenario_key, period_label, year_start, year_end)
    if prec is None:
        return 0, 1, 0
    print("done")

    # Compute all 19 BioClim variables (two logic paths inside compute_bioclim)
    print("     Computing BIO01–BIO19 ...", end=" ", flush=True)
    try:
        all_bio = compute_bioclim(tas_, tmin, tmax, prec)
    except Exception as exc:
        _fail(f"Computation failed: {exc}")
        return 0, 0, 1
    print("done")

    # Free raw climatologies immediately
    if FORCE_COMPUTE_AND_RELEASE:
        release(tas_); release(tmin); release(tmax); release(prec)

    # Save only the requested variables
    n_ok = n_fail = 0
    for var_name in variables_to_compute:
        if var_name not in all_bio:
            _fail(f"{var_name} not found — check spelling (e.g. BIO01 not Bio01)")
            n_fail += 1
            continue
        try:
            path = save_bioclim_variable(
                all_bio[var_name], var_name,
                scenario_key, period_label
            )
            rel = path.relative_to(path.parents[3])
            _ok(f"Saved  {var_name}  →  {rel}")
            n_ok += 1
        except Exception as exc:
            _fail(f"Save failed for {var_name}: {exc}")
            n_fail += 1
        finally:
            if FORCE_COMPUTE_AND_RELEASE:
                release(all_bio[var_name])

    elapsed = time.time() - t0
    print(f"\n     Finished in {elapsed/60:.1f} min  "
          f"({n_ok} saved, {n_fail} failed)")
    return n_ok, 0, n_fail



#  Main orchestration                                                  


def main(scenario=None, period=None, variables_to_run=None, dry_run=False):
    """
    Run the BioClim pipeline.

    Parameters (all optional — defaults run everything)
    ----------
    scenario         : single scenario key e.g. "ssp585", or None for all
    period           : single period e.g. "baseline", or None for all
    variables_to_run : list e.g. ["BIO01","BIO12"], or None for all 19
    dry_run          : True shows plan without writing files
    """
    import dask
    if DASK_WORKERS == 1:
        dask.config.set(scheduler="synchronous")
    else:
        dask.config.set(num_workers=DASK_WORKERS)

    # Resolve scenarios
    if scenario is not None:
        scenarios_to_run = [scenario]
    else:
        scenarios_to_run = ["historical"] + FUTURE_SCENARIOS

    # Resolve periods
    if period is not None:
        if period not in PERIODS:
            print(f"Unknown period '{period}'. "
                  f"Choose from: {list(PERIODS.keys())}")
            sys.exit(1)
        periods_to_run = {period: PERIODS[period]}
    else:
        periods_to_run = PERIODS

    # Resolve variables
    vars_to_compute = variables_to_run if variables_to_run else ALL_BIOCLIM

    _banner("BioClim Pipeline  –  IPSL-CM6A-LR")
    print(f"  Scenarios : {scenarios_to_run}")
    print(f"  Periods   : {list(periods_to_run.keys())}")
    print(f"  Variables : {vars_to_compute}")
    if dry_run:
        print("  MODE      : DRY RUN — nothing will be written")

    totals    = {"success": 0, "skip": 0, "fail": 0}
    run_start = time.time()

    for scen in scenarios_to_run:
        _banner(f"Scenario: {scen.upper()}")
        for p_label, (y0, y1) in periods_to_run.items():
            # baseline only runs on historical
            if p_label == "baseline" and scen != "historical":
                continue
            # future periods don't run on historical
            if p_label != "baseline" and scen == "historical":
                continue
            n_ok, n_skip, n_fail = process_scenario_period(
                scen, p_label, y0, y1, vars_to_compute, dry_run
            )
            totals["success"] += n_ok
            totals["skip"]    += n_skip
            totals["fail"]    += n_fail
        gc.collect()

    # Summary
    total_elapsed = time.time() - run_start
    _banner("Pipeline Complete")
    print(f"  Total time  : {total_elapsed/60:.1f} minutes")
    print(f"  Saved       : {totals['success']} variable files")
    print(f"  Skipped     : {totals['skip']} scenario/period combos")
    print(f"  Failed      : {totals['fail']}")
    print(f"  Output root : {Path('output_bioclim').resolve()}")
    if totals["fail"] > 0:
        print("\n  Re-run failed combos with specific --scenario and --period flags.")



#  Bottom of file — change this block to control what runs in Spyder  


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Derive BIO01-BIO19 from IPSL-CM6A-LR CMIP6 data."
    )
    parser.add_argument("--scenario",  default=None)
    parser.add_argument("--period",    default=None)
    parser.add_argument("--variable",  default=None)
    parser.add_argument("--variables", nargs="+", default=None)
    parser.add_argument("--dryrun",    action="store_true")
    args = parser.parse_args()

    if args.variable:
        vrun = [args.variable.upper()]
    elif args.variables:
        vrun = [v.upper() for v in args.variables]
    else:
        vrun = None

    main(scenario=args.scenario, period=args.period,
         variables_to_run=vrun, dry_run=args.dryrun)

