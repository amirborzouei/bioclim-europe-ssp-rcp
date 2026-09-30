"""
extract_seasonal.py

USAGE
    python extract_seasonal.py
    python extract_seasonal.py --root "/path/to/files"
    python extract_seasonal.py --skip-grid      # NUTS3 only; runs in seconds

@author: amirrezaborzoueinia
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import xarray as xr
from config import PIPELINE_ROOT   # default root; override with $BIOCLIM_ROOT

SCENARIOS = ["ssp119", "ssp126", "ssp245", "ssp370",
             "ssp434", "ssp460", "ssp534os", "ssp585"]
LABEL = {"ssp119": "SSP1-1.9", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
         "ssp370": "SSP3-7.0", "ssp434": "SSP4-3.4", "ssp460": "SSP4-6.0",
         "ssp534os": "SSP5-3.4-OS", "ssp585": "SSP5-8.5"}
PERIODS = ["2011-2040", "2041-2070", "2071-2100"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARS = [f"T_{s}" for s in SEASONS] + [f"P_{s}" for s in SEASONS]

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(PIPELINE_ROOT))
ap.add_argument("--out", default=".")
ap.add_argument("--skip-grid", action="store_true",
                help="compute NUTS3 only (fast)")
args = ap.parse_args()

ECON = os.path.join(args.root, "output_economic_1km")
NUTS = os.path.join(args.root, "output_economic_nuts3")
CACHE = os.path.join(args.out, ".seasonal_cache.json")

cache = {}
if os.path.exists(CACHE):
    try:
        cache = json.load(open(CACHE))
        print(f"Resuming: {len(cache)} values already cached.")
    except Exception:
        cache = {}


def nc_path(scen, period):
    d = (os.path.join(ECON, "historical", "baseline") if scen == "baseline"
         else os.path.join(ECON, scen, period))
    p = os.path.join(d, "ricardian_covariates_1km.nc")
    return p if os.path.exists(p) else None


def csv_path(scen, period):
    d = (os.path.join(NUTS, "historical", "baseline") if scen == "baseline"
         else os.path.join(NUTS, scen, period))
    p = os.path.join(d, "ricardian_covariates_nuts3.csv")
    return p if os.path.exists(p) else None


def grid_means(scen, period):
    """All eight covariates from one file; cached per scenario-period."""
    key = f"grid|{scen}|{period}"
    if key in cache:
        return cache[key]
    p = nc_path(scen, period)
    if p is None:
        cache[key] = {v: float("nan") for v in VARS}
        json.dump(cache, open(CACHE, "w"))
        return cache[key]
    ds = xr.open_dataset(p)
    yname = "y" if "y" in ds.coords else "latitude"
    w = np.cos(np.deg2rad(ds[yname]))
    out = {}
    for v in VARS:
        out[v] = (float(ds[v].squeeze().weighted(w).mean(skipna=True).values)
                  if v in ds else float("nan"))
    ds.close()
    cache[key] = out
    json.dump(cache, open(CACHE, "w"))
    return out


def nuts_table(scen, period):
    p = csv_path(scen, period)
    if p is None:
        return None
    with open(p) as fh:
        return {r["NUTS_ID"]: {v: float(r[v]) for v in VARS if r.get(v)}
                for r in csv.DictReader(fh)}


def anom(fut, base, var):
    if fut != fut or base != base:
        return float("nan")
    return fut - base if var.startswith("T_") else (fut - base) / (abs(base) + 1e-6) * 100.0


print(f"Economic root: {ECON}")
print(f"NUTS3 root:    {NUTS}\n")


grid_anom, grid_base = {}, {}
if not args.skip_grid:
    print("--- 1 km grid ---")
    grid_base = grid_means("baseline", "baseline")
    print("  baseline: " + ", ".join(f"{v}={grid_base[v]:.2f}" for v in VARS[:4]))
    for scen in SCENARIOS:
        for period in PERIODS:
            m = grid_means(scen, period)
            for v in VARS:
                grid_anom[(v, scen, period)] = anom(m[v], grid_base[v], v)
            ok = "ok" if m[VARS[0]] == m[VARS[0]] else "missing"
            print(f"  {LABEL[scen]:<12} {period}  {ok}", flush=True)


print("\n--- NUTS3 regions ---")
nuts_anom = {}
nb = nuts_table("baseline", None)
if nb is None:
    print("  baseline NUTS3 CSV not found; skipping NUTS3 level.")
else:
    print(f"  baseline: {len(nb)} regions")
    for scen in SCENARIOS:
        for period in PERIODS:
            nf = nuts_table(scen, period)
            if nf is None:
                for v in VARS:
                    nuts_anom[(v, scen, period)] = float("nan")
                print(f"  {LABEL[scen]:<12} {period}  missing")
                continue
            common = set(nb) & set(nf)
            for v in VARS:
                vals = [anom(nf[r].get(v, float('nan')),
                             nb[r].get(v, float('nan')), v) for r in common]
                vals = [x for x in vals if x == x]
                nuts_anom[(v, scen, period)] = float(np.mean(vals)) if vals else float("nan")
            print(f"  {LABEL[scen]:<12} {period}  {len(common)} regions", flush=True)


def write_csv(path, table):
    cols = [f"{LABEL[s]} {p}" for s in SCENARIOS for p in PERIODS]
    with open(path, "w") as fh:
        fh.write("Variable," + ",".join(cols) + "\n")
        for v in VARS:
            row = []
            for s in SCENARIOS:
                for p in PERIODS:
                    x = table.get((v, s, p), float("nan"))
                    row.append("" if x != x else f"{x:.3f}")
            fh.write(v + "," + ",".join(row) + "\n")

if grid_anom:
    write_csv(os.path.join(args.out, "seasonal_anomalies_grid.csv"), grid_anom)
if nuts_anom:
    write_csv(os.path.join(args.out, "seasonal_anomalies_nuts3.csv"), nuts_anom)

with open(os.path.join(args.out, "seasonal_baseline.csv"), "w") as fh:
    fh.write("Variable,grid_baseline\n")
    for v in VARS:
        b = grid_base.get(v, float("nan"))
        fh.write(f"{v},{'' if b != b else f'{b:.4f}'}\n")



def block(table, title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for p in PERIODS:
        print(f"\n--- {p} ---")
        print(f"{'Var':<8}" + "".join(f"{LABEL[s]:>13}" for s in SCENARIOS))
        for v in VARS:
            cells = []
            for s in SCENARIOS:
                x = table.get((v, s, p), float("nan"))
                cells.append(f"{'--':>13}" if x != x else f"{x:>+13.3f}")
            print(f"{v:<8}" + "".join(cells))

if grid_anom:
    block(grid_anom, "SEASONAL ANOMALIES -- 1 km grid, area-weighted "
                     "(T in degC, P in per cent)")
if nuts_anom:
    block(nuts_anom, "SEASONAL ANOMALIES -- mean across NUTS3 regions "
                     "(T in degC, P in per cent)")

if grid_base:
    print("\n--- grid baseline values ---")
    for v in VARS:
        print(f"{v:<8}{grid_base[v]:>12.4f}")

print("\nWrote CSVs to " + os.path.abspath(args.out))
print("Paste the blocks above, or send the CSV files.")
