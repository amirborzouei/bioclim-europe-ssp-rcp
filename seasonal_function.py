"""
seasonal_function.py

INPUT (identical to the temperature/precipitation inputs of bioclim_functions):
  tas  : monthly mean temperature    [degC]      dims (month, y, x), month 1..12
  prec : monthly precipitation total [mm/month]  dims (month, y, x)

OUTPUT: dict {name: DataArray(y, x)} with eight covariates:
  T_DJF, T_MAM, T_JJA, T_SON  seasonal mean temperature    [degC]
  P_DJF, P_MAM, P_JJA, P_SON  seasonal precipitation total [mm/season]

Seasons (meteorological, Northern Hemisphere):
  DJF = Dec, Jan, Feb (winter)   MAM = Mar, Apr, May (spring)
  JJA = Jun, Jul, Aug (summer)   SON = Sep, Oct, Nov (autumn)

@author: amirrezaborzoueinia
"""

import numpy as np
import xarray as xr

SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
           "JJA": [6, 7, 8], "SON": [9, 10, 11]}

def compute_seasonal(tas: xr.DataArray, prec: xr.DataArray) -> dict:
    """Eight Ricardian seasonal covariates from monthly tas and prec."""
    out = {}
    for name, months in SEASONS.items():
        tsub = tas.sel(month=months)
        psub = prec.sel(month=months)
        t = tsub.mean(dim="month", skipna=True)
        p = psub.sum(dim="month", skipna=True).where(psub.notnull().any(dim="month"))
        t.attrs = {"long_name": f"Mean temperature of {name}", "units": "degC",
                   "season": name,
                   "definition": "mean of the three monthly mean temperatures",
                   "purpose": "Ricardian climate covariate (Mendelsohn et al. 1994)"}
        p.attrs = {"long_name": f"Precipitation of {name}", "units": "mm",
                   "season": name,
                   "definition": "sum of the three monthly precipitation totals",
                   "purpose": "Ricardian climate covariate (Mendelsohn et al. 1994)"}
        out[f"T_{name}"] = t
        out[f"P_{name}"] = p
    return out

def save_seasonal(cov: dict, scenario: str, period: str, out_root) -> "Path":
    """Write the eight variables as one compressed NetCDF per scenario-period."""
    from pathlib import Path
    out_dir = Path(out_root) / scenario / period
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(cov)
    ds.attrs.update({
        "title": "Ricardian seasonal climate variables, 1 km",
        "scenario": scenario, "period": period,
        "source_gcm": "IPSL-CM6A-LR", "run": "r1i1p1f1",
        "baseline": "CHELSA_V2.1_1981-2010",
        "method": "delta-change downscaling (bilinear) then seasonal aggregation",
        "reference": "Mendelsohn, Nordhaus & Shaw 1994, Am. Econ. Rev. 84(4):753-771",
    })
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    out_path = out_dir / "ricardian_variables_1km.nc"
    ds.to_netcdf(out_path, encoding=enc)
    return out_path
