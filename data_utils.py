"""
data_utils.py
@author: amirrezaborzoueinia
"""

import gc
import warnings
import numpy as np
import xarray as xr
from pathlib import Path

# New xarray API for CMIP6 cftime dates (avoids FutureWarning)
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

from config import (
    DATA_ROOT, SCENARIOS, EUROPE_EXTENT,
    NC_TIME_DIM, NC_LAT_DIM, NC_LON_DIM, NC_VAR_NAMES,
    TEMP_OFFSET_K, PR_KG_TO_MM, DASK_CHUNKS, SKIP_PERIODS,
)



#  Internal helpers                                                    


def _nc_files(variable: str, scenario_key: str) -> list[Path]:
    folder = DATA_ROOT / variable / SCENARIOS[scenario_key]["folder"]
    files  = sorted(folder.glob("*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No .nc files found in:\n  {folder}\n"
            f"Check DATA_ROOT in config.py and folder name spelling."
        )
    if len(files) > 1:
        print(f"    [INFO] {len(files)} files in "
              f"{variable}/{SCENARIOS[scenario_key]['folder']} "
              f"— concatenating along time")
    return files


def _open_chunked(files: list[Path]) -> xr.Dataset:
    """Open one or more NetCDF files with Dask lazy loading."""
    kwargs = dict(decode_times=TIME_CODER, chunks=DASK_CHUNKS)
    if len(files) == 1:
        return xr.open_dataset(files[0], **kwargs)
    return xr.open_mfdataset(sorted(files), combine="by_coords", **kwargs)


def _fix_coords(ds: xr.Dataset) -> xr.Dataset:
    """Normalise coordinate names and order."""
    rename = {}
    if "latitude"  in ds.dims and NC_LAT_DIM not in ds.dims:
        rename["latitude"]  = NC_LAT_DIM
    if "longitude" in ds.dims and NC_LON_DIM not in ds.dims:
        rename["longitude"] = NC_LON_DIM
    if rename:
        ds = ds.rename(rename)
    # 0-360 -> -180/180
    if ds[NC_LON_DIM].values.max() > 180:
        ds = ds.assign_coords(
            {NC_LON_DIM: (ds[NC_LON_DIM] + 180) % 360 - 180}
        ).sortby(NC_LON_DIM)
    # South-to-North
    if ds[NC_LAT_DIM].values[0] > ds[NC_LAT_DIM].values[-1]:
        ds = ds.sortby(NC_LAT_DIM)
    return ds


def _clip_europe(ds: xr.Dataset) -> xr.Dataset:
    e = EUROPE_EXTENT
    return ds.sel(**{
        NC_LON_DIM: slice(e["lon_min"], e["lon_max"]),
        NC_LAT_DIM: slice(e["lat_min"], e["lat_max"]),
    })


def _convert_units(da: xr.DataArray, variable: str) -> xr.DataArray:
    """Convert CMIP6 raw units to analysis-ready units."""
    if variable in ("tas", "tasmax", "tasmin"):
        if da.attrs.get("units", "K") == "K" or float(
            da.isel({NC_TIME_DIM: 0, NC_LAT_DIM: 0, NC_LON_DIM: 0}).values
        ) > 100:
            da = da - TEMP_OFFSET_K
            da.attrs["units"] = "degC"
    elif variable == "pr" and PR_KG_TO_MM:
        days = da[NC_TIME_DIM].dt.days_in_month
        da   = da * days * 86400.0
        da.attrs["units"] = "mm/month"
    return da



#  Public API                                                          


def load_variable(variable: str, scenario_key: str,
                  year_start: int, year_end: int) -> xr.DataArray:
    """
    Lazily load one climate variable for a scenario and year range.
    Returns DataArray with dims (time, lat, lon). Data is NOT in RAM.
    """
    files    = _nc_files(variable, scenario_key)
    var_name = NC_VAR_NAMES[variable]
    ds = _open_chunked(files)
    ds = _fix_coords(ds)
    ds = _clip_europe(ds)
    da = ds[var_name]
    da = _convert_units(da, variable)
    da = da.sel({NC_TIME_DIM: slice(str(year_start), str(year_end))})
    if da.sizes[NC_TIME_DIM] == 0:
        raise ValueError(
            f"No timesteps for {variable}/{scenario_key} "
            f"in {year_start}-{year_end}. "
            f"Check SCENARIOS year ranges in config.py."
        )
    return da


def splice_historical_and_future(variable: str,
                                  future_scenario: str,
                                  year_start: int,
                                  year_end: int,
                                  hist_end: int = 2014) -> xr.DataArray:
    """
    Concatenate historical + future data for a period that straddles
    the historical/future boundary (e.g. 2011-2040 needs
    Historical 2011-2014 + SSP 2015-2040).

    For the 2011-2040 period:
      - Years 2011-2014 come from the Historical folder
        (observed forcings — scientifically required by CMIP6 protocol)
      - Years 2015-2040 come from the SSP scenario folder

    """
    parts = []
    if year_start <= hist_end:
        da_h = load_variable(
            variable, "historical",
            year_start, min(hist_end, year_end)
        )
        parts.append(da_h)
    fut_start = SCENARIOS[future_scenario]["years"][0]
    fut_slice_start = max(fut_start, hist_end + 1, year_start)
    if fut_slice_start <= year_end:
        da_f = load_variable(
            variable, future_scenario,
            fut_slice_start, year_end
        )
        parts.append(da_f)
    if not parts:
        raise ValueError(
            f"No data assembled for {variable}/{future_scenario} "
            f"{year_start}-{year_end}"
        )
    return xr.concat(parts, dim=NC_TIME_DIM) if len(parts) > 1 else parts[0]


def compute_monthly_climatology(da: xr.DataArray,
                                 year_start: int,
                                 year_end: int) -> xr.DataArray:
    """
    Compute 12-month climatological means from a time series.
    Triggers Dask computation — reads from disk here.
    Returns DataArray with dim 'month' (1-12), shape (12, lat, lon).
    """
    years = np.unique(da[NC_TIME_DIM].dt.year.values)
    if len(years) < 5:
        warnings.warn(
            f"Only {len(years)} years for climatology "
            f"({year_start}-{year_end}). Results may be unreliable.",
            UserWarning, stacklevel=2
        )
    return da.groupby(f"{NC_TIME_DIM}.month").mean(dim=NC_TIME_DIM).compute()


def get_climatology(variable: str, scenario_key: str,
                    period_label: str,
                    year_start: int, year_end: int) -> xr.DataArray | None:
    """
    High-level entry point: load -> splice if needed -> climatology.
    Returns None if period is on the skip list or data is missing.
    """
    if scenario_key in SKIP_PERIODS:
        if period_label in SKIP_PERIODS[scenario_key]:
            print(f"    [SKIP] {scenario_key}/{period_label}: "
                  f"SSP5-3.4OS data starts 2041")
            return None
    hist_end = SCENARIOS["historical"]["years"][1]  # 2014
    try:
        if scenario_key == "historical":
            da = load_variable(variable, "historical", year_start, year_end)
        elif year_start <= hist_end:
            da = splice_historical_and_future(
                variable, scenario_key, year_start, year_end, hist_end
            )
        else:
            da = load_variable(variable, scenario_key, year_start, year_end)
        return compute_monthly_climatology(da, year_start, year_end)
    except (FileNotFoundError, ValueError) as exc:
        print(f"    [ERROR] {variable}/{scenario_key}/{period_label}: {exc}")
        return None


def release(obj):
    """Delete an object and run garbage collection to free RAM."""
    del obj
    gc.collect()
