"""
nuts3_aggregate.py

INPUT
  output_economic_1km/{scenario}/{period}/ricardian_covariates_1km.nc  (8 vars)
  Eurostat GISCO NUTS 2021, level 3 polygons (WGS84):
      NUTS_RG_01M_2021_4326.shp   (filter LEVL_CODE == 3)
OUTPUT
  output_economic_nuts3/{scenario}/{period}/ricardian_covariates_nuts3.csv
      columns: NUTS_ID, T_DJF..T_SON, P_DJF..P_SON  (area-weighted regional means)

@author: amirrezaborzoueinia
"""

import numpy as np
import pandas as pd
import xarray as xr

VARS = ["T_DJF", "T_MAM", "T_JJA", "T_SON", "P_DJF", "P_MAM", "P_JJA", "P_SON"]


def weighted_region_means(values: dict, region_ids: np.ndarray,
                          weights: np.ndarray) -> pd.DataFrame:

    reg = region_ids.ravel()
    w = weights.ravel()
    inside = reg >= 0
    out = {}
    for name, arr in values.items():
        v = arr.ravel()
        ok = inside & np.isfinite(v)
        d = pd.DataFrame({"reg": reg[ok], "wv": w[ok] * v[ok], "w": w[ok]})
        g = d.groupby("reg").sum()
        out[name] = g["wv"] / g["w"]
    return pd.DataFrame(out)


def rasterize_nuts3(nuts_gdf, like: xr.DataArray):

    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    y = like["y"].values
    x = like["x"].values
    H, W = len(y), len(x)
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    # pixel edges (assume regular grid); north-up transform
    dx = (x1 - x0) / (W - 1); dy = (y1 - y0) / (H - 1)
    transform = from_bounds(x0 - dx / 2, y0 - dy / 2, x1 + dx / 2, y1 + dy / 2, W, H)
    shapes = ((geom, i) for i, geom in enumerate(nuts_gdf.geometry))
    ridx = rasterize(shapes, out_shape=(H, W), transform=transform,
                     fill=-1, dtype="int32", all_touched=False)
    # rasterize returns rows top->bottom; align to y orientation
    if y[0] < y[-1]:            # ascending y -> flip to match north-up raster
        ridx = ridx[::-1, :]
    return ridx



def _nuts_id_column(gdf):

    for c in ["NUTS_ID", "nuts_id", "NUTS_id", "NUTS3_ID", "id", "FID"]:
        if c in gdf.columns:
            return c
    # fallback: among columns of short, two-letter-prefixed codes, pick the one
    # with the most distinct values (the region id, not the country code)
    best, best_n = None, -1
    for c in gdf.columns:
        s = gdf[c].astype(str)
        looks = (s.str[:2].str.isalpha() & s.str.len().between(4, 5)).mean()
        if looks > 0.8 and s.nunique() > best_n:
            best, best_n = c, s.nunique()
    if best is not None:
        return best
    raise KeyError("No NUTS id column found. Columns present: " + str(list(gdf.columns)))


def aggregate_scenario(nc_path, nuts_gdf, out_csv):
    ds = xr.open_dataset(nc_path)
    like = ds[VARS[0]]
    region_ids = rasterize_nuts3(nuts_gdf, like)
    lat = like["y"].values
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, like.shape[1]))
    vals = {v: ds[v].values for v in VARS}
    res = weighted_region_means(vals, region_ids, weights)
    idcol = _nuts_id_column(nuts_gdf)
    res.index = nuts_gdf[idcol].values[res.index]
    res.index.name = "NUTS_ID"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv)
    return res
