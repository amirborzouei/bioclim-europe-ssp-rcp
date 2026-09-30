"""
extract_spatial_structure.py

WHAT IT COMPUTES, per variable, scenario and period
    latitude profile   area-weighted mean anomaly in 5-degree latitude bands,
                       which is what "the change is stronger in the south" has
                       to mean if it is to be checked
    spread             area-weighted 5th, 50th and 95th percentiles of the
                       anomaly across land cells, plus the unweighted extremes
    concentration      share of land area beyond a threshold, so the section can
                       say how much of Europe passes a given level rather than
                       only what the average does

    and for BIO6, the freezing crossing as a SPATIAL statistic: the share of
    land area whose coldest-month minimum is below 0 degC at the baseline, and
    the share of THAT area which rises above 0 degC.

@author: amirrezaborzoueinia
"""

import os
import sys
import json
import argparse
import numpy as np
import xarray as xr
from config import PIPELINE_ROOT   # default root; override with $BIOCLIM_ROOT

TEMP = {f"BIO{n:02d}" for n in range(1, 12)}
LABEL = {"ssp119": "SSP1-1.9", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
         "ssp370": "SSP3-7.0", "ssp434": "SSP4-3.4", "ssp460": "SSP4-6.0",
         "ssp534os": "SSP5-3.4-OS", "ssp585": "SSP5-8.5"}

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(PIPELINE_ROOT))
ap.add_argument("--scenario", default="ssp585", help="comma-separated")
ap.add_argument("--period", default="2071-2100", help="comma-separated")
ap.add_argument("--vars", default="BIO01,BIO05,BIO06,BIO17", help="comma-separated")
ap.add_argument("--band", type=float, default=5.0, help="latitude band width, degrees")
ap.add_argument("--min-lat", type=float, default=None,
                help="restrict every statistic to cells at or above this "
                     "latitude. The domain is a rectangle and its land mask "
                     "includes North Africa and the Middle East, which are not "
                     "Europe; use this to report Europe-only figures.")
ap.add_argument("--max-lat", type=float, default=None)
ap.add_argument("--out", default=".")
args = ap.parse_args()

BIO_ROOT = os.path.join(args.root, "output_bioclim_1km")
BASE_DIR = os.path.join(BIO_ROOT, "historical", "baseline")
VARS = [v.strip().upper() for v in args.vars.split(",")]
SCENS = [s.strip() for s in args.scenario.split(",")]
PERIODS = [p.strip() for p in args.period.split(",")]
CACHE = os.path.join(args.out, ".spatial_cache.json")

# Thresholds for the concentration statistic, in each variable's own units.
THRESHOLD = {"BIO01": 6.0, "BIO05": 8.0, "BIO06": 6.0, "BIO17": -10.0}

# Bump this whenever the set of statistics computed per field changes. It is
# part of the cache key, so old entries are ignored rather than silently
# returned with the new fields missing. Version 1 did not compute the
# ratio-of-means aggregation and produced empty columns for it on a re-run.
CACHE_VERSION = 3

cache = {}
if os.path.exists(CACHE):
    try:
        raw = json.load(open(CACHE))
        cache = {k: v for k, v in raw.items() if k.endswith(f"|v{CACHE_VERSION}")}
        stale = len(raw) - len(cache)
        if cache:
            print(f"Resuming: {len(cache)} results already cached.")
        if stale:
            print(f"Ignoring {stale} cached results from an earlier version of "
                  f"this script; those fields will be recomputed.")
    except Exception:
        cache = {}


def find_file(d, var):
    for name in (f"{var}_1km.nc", f"{var.lower()}_1km.nc", f"{var}.nc", f"{var.lower()}.nc"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def load(d, var):
    p = find_file(d, var)
    if p is None:
        return None, None
    ds = xr.open_dataset(p)
    da = (ds[var] if var in ds else ds[list(ds.data_vars)[0]]).squeeze()
    yname = "y" if "y" in da.coords else "latitude"
    arr = da.values.astype(np.float32, copy=False)
    lat = da[yname].values.astype(np.float32)
    ds.close()
    return arr, lat


def wq(v, w, qs):
    o = np.argsort(v, kind="stable")
    v = v[o]; w = w[o].astype(np.float64)
    cw = np.cumsum(w)
    return np.interp(qs, (cw - 0.5 * w) / cw[-1], v)


def analyse(var, scen, period):
    key = (f"{var}|{scen}|{period}|b{args.band:g}"
           f"|lat{args.min_lat}-{args.max_lat}|v{CACHE_VERSION}")
    if key in cache:
        return cache[key]
    fut_dir = os.path.join(BIO_ROOT, scen, period)
    b, lat = load(BASE_DIR, var)
    f, _ = load(fut_dir, var)
    if b is None or f is None:
        cache[key] = None
        json.dump(cache, open(CACHE, "w"))
        return None

    # Optional latitude restriction. The domain is a rectangle, so its land
    # mask contains North Africa and the Middle East as well as Europe; below
    # roughly 35 degrees north it is almost entirely North Africa. Masking here
    # rather than at the reporting stage keeps every statistic consistent.
    if args.min_lat is not None or args.max_lat is not None:
        keep = np.ones(lat.shape, bool)
        if args.min_lat is not None:
            keep &= lat >= args.min_lat
        if args.max_lat is not None:
            keep &= lat <= args.max_lat
        b = np.where(keep[:, None], b, np.nan)
        f = np.where(keep[:, None], f, np.nan)

    anom = (f - b) if var in TEMP else (f - b) / (np.abs(b) + 1e-6) * 100.0
    unit = "degC" if var in TEMP else "%"
    w1d = np.cos(np.deg2rad(lat)).astype(np.float32)
    w2d = np.broadcast_to(w1d[:, None], anom.shape)
    ok = np.isfinite(anom)
    v = anom[ok]; w = w2d[ok]
    wsum = float(w.sum(dtype=np.float64))
    mean = float(np.dot(v.astype(np.float64), w.astype(np.float64)) / wsum)
    p5, p50, p95 = wq(v, w, [0.05, 0.50, 0.95])

    # TWO WAYS TO AVERAGE A PERCENTAGE CHANGE, AND THEY DO NOT AGREE.
    #
    # "mean" above is the area-weighted mean of the per-cell percentage change:
    # a mean of ratios. extract_anomalies.py instead takes the area-weighted
    # mean of the baseline and of the future field and forms one ratio from
    # them: a ratio of means. For temperature the two are identical, because
    # the anomaly is a difference and averaging commutes with subtraction. For
    # precipitation they differ, sometimes by more than a percentage point,
    # because a cell with a near-zero baseline can post a very large relative
    # change and drag the mean of ratios with it.

    def ratio_of_means(bb, ff, ww):
        m = np.isfinite(bb) & np.isfinite(ff)
        if not m.any():
            return None
        bw = float(np.dot(bb[m].astype(np.float64), ww[m].astype(np.float64)))
        fw = float(np.dot(ff[m].astype(np.float64), ww[m].astype(np.float64)))
        sw = float(ww[m].sum(dtype=np.float64))
        bmean, fmean = bw / sw, fw / sw
        return (fmean - bmean) / (abs(bmean) + 1e-6) * 100.0

    mean_rom = None if var in TEMP else ratio_of_means(b, f, w2d)

    thr = THRESHOLD.get(var)
    if thr is None:
        share = None
    elif var == "BIO17":
        share = float(w[v < thr].sum(dtype=np.float64) / wsum * 100)
    else:
        share = float(w[v > thr].sum(dtype=np.float64) / wsum * 100)

    edges = np.arange(np.floor(lat.min() / args.band) * args.band,
                      np.ceil(lat.max() / args.band) * args.band + args.band, args.band)
    prof, prof_rom, prof_area = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = (lat >= lo) & (lat < hi)
        if not rows.any():
            prof.append(None); prof_rom.append(None); prof_area.append(None); continue
        sub_ok = np.isfinite(anom[rows, :])
        prof_area.append(float(w2d[rows, :][sub_ok].sum(dtype=np.float64)) / wsum * 100)
        sub = anom[rows, :]; subw = w2d[rows, :]
        m = np.isfinite(sub)
        prof.append(float(np.dot(sub[m].astype(np.float64), subw[m].astype(np.float64))
                          / subw[m].sum(dtype=np.float64)) if m.any() else None)
        prof_rom.append(None if var in TEMP else
                        ratio_of_means(b[rows, :], f[rows, :], subw))

    rec = {"var": var, "scenario": LABEL.get(scen, scen), "period": period, "unit": unit,
           "mean": mean, "mean_ratio_of_means": mean_rom,
           "p5": float(p5), "p50": float(p50), "p95": float(p95),
           "min": float(v.min()), "max": float(v.max()),
           "threshold": thr, "share_beyond_threshold_pct": share,
           "bands": [float(x) for x in edges], "profile": prof,
           "profile_ratio_of_means": prof_rom, "profile_area_share_pct": prof_area}

    if var == "BIO06":
        both = np.isfinite(b) & np.isfinite(f)
        below = both & (b < 0)
        crossed = below & (f > 0)
        bw = float(w2d[below].sum(dtype=np.float64))
        rec["baseline_below_freezing_pct"] = bw / wsum * 100
        rec["crossed_above_freezing_pct"] = float(w2d[crossed].sum(dtype=np.float64)) / wsum * 100
        rec["of_those_below_pct_crossing"] = (
            float(w2d[crossed].sum(dtype=np.float64)) / bw * 100 if bw else None)

    cache[key] = rec
    json.dump(cache, open(CACHE, "w"))
    return rec


print(f"baseline: {BASE_DIR}")
print(f"variables {VARS}  scenarios {SCENS}  periods {PERIODS}\n")

results = []
for scen in SCENS:
    for period in PERIODS:
        for var in VARS:
            r = analyse(var, scen, period)
            if r is None:
                print(f"  {LABEL.get(scen,scen):<12} {period}  {var}: missing")
                continue
            results.append(r)
            print(f"  {LABEL.get(scen,scen):<12} {period}  {var:<6} "
                  f"mean {r['mean']:+8.3f} {r['unit']:<5} "
                  f"p5 {r['p5']:+8.3f}  p95 {r['p95']:+8.3f}", flush=True)
            if var == "BIO06":
                print(f"{'':>27}below freezing at baseline "
                      f"{r['baseline_below_freezing_pct']:.1f}% of land area; "
                      f"of those {r['of_those_below_pct_crossing']:.1f}% rise above")


if results:
    keys = ["var", "scenario", "period", "unit", "mean", "mean_ratio_of_means",
            "p5", "p50", "p95", "min", "max",
            "threshold", "share_beyond_threshold_pct",
            "baseline_below_freezing_pct", "crossed_above_freezing_pct",
            "of_those_below_pct_crossing"]

    def cell(r, k):
        v = r.get(k)
        if v is None:
            return ""
        return v if isinstance(v, str) else f"{v:.4f}"

    with open(os.path.join(args.out, "spatial_summary.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in results:
            fh.write(",".join(cell(r, k) for k in keys) + "\n")

    with open(os.path.join(args.out, "spatial_latitude_profile.csv"), "w") as fh:
        fh.write("scenario,period,variable,band_south,band_north,"
                 "mean_anomaly,mean_anomaly_ratio_of_means,land_area_share_pct\n")
        for r in results:
            e = r["bands"]
            rom = r.get("profile_ratio_of_means") or [None] * len(r["profile"])
            ar = r.get("profile_area_share_pct") or [None] * len(r["profile"])
            for i, x in enumerate(r["profile"]):
                if x is None:
                    continue
                q = "" if rom[i] is None else f"{rom[i]:.4f}"
                a = "" if ar[i] is None else f"{ar[i]:.4f}"
                fh.write(f"{r['scenario']},{r['period']},{r['var']},"
                         f"{e[i]:.0f},{e[i+1]:.0f},{x:.4f},{q},{a}\n")

    # printed latitude table, for the first scenario-period only
    first = [r for r in results
             if r["scenario"] == results[0]["scenario"] and r["period"] == results[0]["period"]]
    if len(first) > 1:
        e = first[0]["bands"]
        print("\n" + "=" * 78)
        print(f"LATITUDE PROFILE, {first[0]['scenario']} {first[0]['period']}, "
              f"{args.band:.0f}-degree bands")
        print("=" * 78)
        print(f"{'band':<12}" + "".join(f"{r['var']:>12}" for r in first)
              + f"{'land %':>10}")
        for i in range(len(e) - 1):
            cells = "".join(f"{'--':>12}" if r["profile"][i] is None
                            else f"{r['profile'][i]:>+12.3f}" for r in first)
            ar = first[0].get("profile_area_share_pct") or []
            share = "" if i >= len(ar) or ar[i] is None else f"{ar[i]:>10.1f}"
            print(f"{e[i]:>3.0f}-{e[i+1]:<8.0f}" + cells + share)
        print("\nland % is each band's share of the total land area in the run. "
              "The domain is a rectangle:\nbelow about 35 N its land is almost "
              "entirely North Africa, not Europe. Use --min-lat to exclude it.")

    cross = [r for r in results if r["var"] == "BIO06"]
    if len(cross) > 1:
        print("\n" + "=" * 78)
        print("FREEZING CROSSING ACROSS PATHWAYS")
        print("=" * 78)
        print(f"{'scenario':<14}{'period':<12}{'below at baseline':>20}{'of those, crossing':>22}")
        for r in cross:
            print(f"{r['scenario']:<14}{r['period']:<12}"
                  f"{r['baseline_below_freezing_pct']:>19.1f}%"
                  f"{r['of_those_below_pct_crossing']:>21.1f}%")

    print(f"\nWrote spatial_summary.csv and spatial_latitude_profile.csv to "
          f"{os.path.abspath(args.out)}")
    print("Send me both and I will write Section 4.6.")