"""
extract_anomalies.py
@author: amirrezaborzoueinia
"""

import os
import sys
import json
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
BIOS = [f"BIO{n:02d}" for n in range(1, 20)]
TEMP = [f"BIO{n:02d}" for n in range(1, 12)]      # absolute difference
PREC = [f"BIO{n:02d}" for n in range(12, 20)]     # percentage change

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(PIPELINE_ROOT))
ap.add_argument("--coarsen", type=int, default=1)
ap.add_argument("--out", default=".")
args = ap.parse_args()

ROOT = args.root
BIO_ROOT = os.path.join(ROOT, "output_bioclim_1km")
BASELINE_DIR = os.path.join(BIO_ROOT, "historical", "baseline")
CACHE = os.path.join(args.out, ".anomaly_cache.json")

cache = {}
if os.path.exists(CACHE):
    try:
        cache = json.load(open(CACHE))
        print(f"Resuming: {len(cache)} values already cached.")
    except Exception:
        cache = {}


def find_file(d, var):
    for name in (f"{var}_1km.nc", f"{var.lower()}_1km.nc",
                 f"{var}.nc", f"{var.lower()}.nc"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def weighted_mean(path, var, coarsen):
    """Cosine-latitude area-weighted mean over all finite (land) cells."""
    ds = xr.open_dataset(path)
    da = ds[var] if var in ds else ds[list(ds.data_vars)[0]]
    da = da.squeeze()
    yname = "y" if "y" in da.coords else "latitude"
    xname = "x" if "x" in da.coords else "longitude"
    if coarsen > 1:
        da = da.coarsen({yname: coarsen, xname: coarsen}, boundary="trim").mean()
    w = np.cos(np.deg2rad(da[yname]))
    val = float(da.weighted(w).mean(skipna=True).values)
    ds.close()
    return val


def get_mean(scen, period, bio):
    key = f"{scen}|{period}|{bio}|c{args.coarsen}"
    if key in cache:
        return cache[key]
    d = BASELINE_DIR if scen == "baseline" else os.path.join(BIO_ROOT, scen, period)
    path = find_file(d, bio)
    val = float("nan") if path is None else weighted_mean(path, bio, args.coarsen)
    cache[key] = val
    json.dump(cache, open(CACHE, "w"))
    return val



print(f"Root: {BIO_ROOT}")
if args.coarsen > 1:
    print("*" * 70)
    print(f"WARNING: --coarsen {args.coarsen} is a smoke test only.")
    print("Coarsening averages across the land-ocean boundary and changes the")
    print("means materially. Do NOT use these numbers in the thesis.")
    print("Re-run with --coarsen 1 (the default) for reportable values.")
    print("*" * 70)
else:
    print("Coarsen: 1 (exact; matches the figures)")
print()

base = {}
for i, bio in enumerate(BIOS, 1):
    base[bio] = get_mean("baseline", "baseline", bio)
    print(f"  baseline {bio}  ({i}/19)  mean = {base[bio]:.4f}", flush=True)

means, anom = {}, {}
total = len(BIOS) * len(SCENARIOS) * len(PERIODS)
done = 0
for scen in SCENARIOS:
    for period in PERIODS:
        for bio in BIOS:
            done += 1
            v = get_mean(scen, period, bio)
            means[(bio, scen, period)] = v
            b = base[bio]
            if np.isnan(v) or np.isnan(b):
                anom[(bio, scen, period)] = float("nan")
            elif bio in TEMP:
                anom[(bio, scen, period)] = v - b
            else:
                anom[(bio, scen, period)] = (v - b) / (abs(b) + 1e-6) * 100.0
        print(f"  {LABEL[scen]:<12} {period}  ({done}/{total})", flush=True)


def write_csv(path, table, fmt="%.4f"):
    cols = [f"{LABEL[s]} {p}" for s in SCENARIOS for p in PERIODS]
    with open(path, "w") as fh:
        fh.write("BIO," + ",".join(cols) + "\n")
        for bio in BIOS:
            row = []
            for s in SCENARIOS:
                for p in PERIODS:
                    v = table[(bio, s, p)]
                    row.append("" if np.isnan(v) else fmt % v)
            fh.write(bio + "," + ",".join(row) + "\n")

f1 = os.path.join(args.out, "anomaly_means_all19.csv")
f2 = os.path.join(args.out, "anomaly_table_all19.csv")
write_csv(f1, means)
write_csv(f2, anom, fmt="%.3f")

with open(os.path.join(args.out, "baseline_means_all19.csv"), "w") as fh:
    fh.write("BIO,baseline_mean\n")
    for bio in BIOS:
        fh.write(f"{bio},{base[bio]:.4f}\n")



print("\n" + "=" * 78)
print("ANOMALIES  (BIO1-11 absolute; BIO12-19 per cent change)")
print("=" * 78)
for p in PERIODS:
    print(f"\n--- {p} ---")
    print(f"{'BIO':<7}" + "".join(f"{LABEL[s]:>13}" for s in SCENARIOS))
    for bio in BIOS:
        cells = []
        for s in SCENARIOS:
            v = anom[(bio, s, p)]
            cells.append(f"{'--':>13}" if np.isnan(v) else f"{v:>+13.3f}")
        print(f"{bio:<7}" + "".join(cells))

print("\n--- baseline means ---")
for bio in BIOS:
    print(f"{bio:<7}{base[bio]:>12.4f}")

print(f"\nWrote:\n  {f1}\n  {f2}\n  baseline_means_all19.csv")
print("Paste the three blocks above, or send the CSV files.")
