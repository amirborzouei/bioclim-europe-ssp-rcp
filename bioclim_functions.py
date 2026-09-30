"""
bioclim_functions.py
@author: amirrezaborzoueinia
"""

import numpy as np
import xarray as xr



#  Rolling quarter helpers                                             


def _rolling_quarter_sums(monthly: xr.DataArray) -> xr.DataArray:
    """12 rolling 3-month sums. Quarter 0=Jan+Feb+Mar, ..., wrapping."""
    months = [monthly.sel(month=m) for m in range(1, 13)]
    quarters = [
        months[i] + months[(i + 1) % 12] + months[(i + 2) % 12]
        for i in range(12)
    ]
    return xr.concat(quarters, dim="quarter")


def _rolling_quarter_means(monthly: xr.DataArray) -> xr.DataArray:
    return _rolling_quarter_sums(monthly) / 3.0



#  NaN-safe quarter selection using numpy.take_along_axis             


def _select_quarter(q_data: xr.DataArray,
                     criterion_data: xr.DataArray,
                     mode: str) -> xr.DataArray:
    """
    Select one quarter per spatial cell based on argmax or argmin
    of criterion_data, using numpy.take_along_axis.

    NaN handling: cells where all quarters are NaN (ocean) are identified
    first; NaN is filled with a neutral value so argmax/argmin complete;
    result is set back to NaN for all-NaN cells.

    Parameters
    ----------
    q_data         : DataArray (quarter=12, y, x) - values to select from
    criterion_data : DataArray (quarter=12, y, x) - values to find max/min of
    mode           : "max" or "min"

    Returns
    -------
    DataArray (y, x) with NaN for masked/ocean cells
    """
    crit_np = criterion_data.values
    data_np = q_data.values

    valid = np.isfinite(crit_np).any(axis=0)

    if mode == "max":
        fill   = np.nanmin(crit_np) if np.any(np.isfinite(crit_np)) else 0.0
        filled = np.where(np.isfinite(crit_np), crit_np, fill)
        idx    = np.argmax(filled, axis=0)
    else:
        fill   = np.nanmax(crit_np) if np.any(np.isfinite(crit_np)) else 0.0
        filled = np.where(np.isfinite(crit_np), crit_np, fill)
        idx    = np.argmin(filled, axis=0)

    selected = np.take_along_axis(data_np, idx[np.newaxis], axis=0)[0]
    selected = np.where(valid, selected, np.nan)

    spatial_dims     = [d for d in q_data.dims if d != "quarter"]
    spatial_dims_set = set(spatial_dims)
    spatial_coords   = {
        k: v for k, v in q_data.coords.items()
        if k != "quarter" and
        set(getattr(v, "dims", ())).issubset(spatial_dims_set)
    }
    return xr.DataArray(selected, dims=spatial_dims, coords=spatial_coords)



#  Main function                                                       


def compute_bioclim(tas: xr.DataArray,
                    tmin: xr.DataArray,
                    tmax: xr.DataArray,
                    prec: xr.DataArray) -> dict[str, xr.DataArray]:
    """
    Compute all 19 BioClim variables from monthly climatologies.

    TWO LOGIC PATHS:
      Mean-based variables use 'tas' directly (the model's physically
      integrated daily mean) - NOT the midpoint approximation
      (tasmax + tasmin) / 2.

      Range/extremes-based variables use 'tasmax' and 'tasmin' as
      required by the WorldClim definitions.

    Parameters
    ----------
    tas  : (month, lat, lon)  [degC]   - daily-mean temperature
    tmin : (month, lat, lon)  [degC]   - daily-minimum temperature
    tmax : (month, lat, lon)  [degC]   - daily-maximum temperature
    prec : (month, lat, lon)  [mm/month]

    Returns
    -------
    dict {"BIO01": DataArray, ..., "BIO19": DataArray}
    NaN cells (ocean, masked) remain NaN in all outputs.
    """
    # Build rolling quarters on tas (for mean-based quarter indices)
    # and on prec (for precipitation-quarter indices).
    q_tas  = _rolling_quarter_means(tas)
    q_prec = _rolling_quarter_sums(prec)

    # ---- MEAN-BASED variables (use tas directly) ----------------------
    # BIO01: Annual mean of tas
    bio01 = tas.mean(dim="month")
    bio01.attrs = {"long_name": "Annual Mean Temperature",
                   "units": "degC",
                   "computed_from": "tas (model-integrated daily mean)"}

    # BIO04: Temperature seasonality = std-dev of monthly tas * 100
    bio04 = tas.std(dim="month", ddof=0, skipna=True) * 100.0
    bio04.attrs = {"long_name": "Temperature Seasonality",
                   "units": "degC*100",
                   "computed_from": "std-dev of tas across 12 months"}

    # ---- RANGE/EXTREMES variables (use tasmax/tasmin) ---------------
    # BIO02: Mean diurnal range = mean(tmax - tmin)
    bio02 = (tmax - tmin).mean(dim="month")
    bio02.attrs = {"long_name": "Mean Diurnal Range",
                   "units": "degC",
                   "computed_from": "mean(tasmax - tasmin)"}

    # BIO05: Max temperature of warmest month
    bio05 = tmax.max(dim="month")
    bio05.attrs = {"long_name": "Max Temperature of Warmest Month",
                   "units": "degC", "computed_from": "max of tasmax"}

    # BIO06: Min temperature of coldest month
    bio06 = tmin.min(dim="month")
    bio06.attrs = {"long_name": "Min Temperature of Coldest Month",
                   "units": "degC", "computed_from": "min of tasmin"}

    # BIO07: Annual temperature range = BIO05 - BIO06
    bio07 = bio05 - bio06
    bio07.attrs = {"long_name": "Temperature Annual Range",
                   "units": "degC", "computed_from": "BIO05 - BIO06"}

    # BIO03: Isothermality = BIO02 / BIO07 * 100 (uses extremes)
    bio03 = (bio02 / bio07) * 100.0
    bio03.attrs = {"long_name": "Isothermality",
                   "units": "%", "computed_from": "BIO02/BIO07 * 100"}

    # ---- PRECIPITATION variables -------------------------------------
    bio12 = prec.sum(dim="month", skipna=True).where(
        prec.notnull().any(dim="month")
    )
    bio12.attrs = {"long_name": "Annual Precipitation", "units": "mm"}

    bio13 = prec.max(dim="month", skipna=True)
    bio13.attrs = {"long_name": "Precipitation of Wettest Month",
                   "units": "mm"}

    bio14 = prec.min(dim="month", skipna=True)
    bio14.attrs = {"long_name": "Precipitation of Driest Month",
                   "units": "mm"}

    prec_mean = prec.mean(dim="month", skipna=True)
    prec_std  = prec.std(dim="month",  ddof=0, skipna=True)
    bio15 = (prec_std / (prec_mean + 1e-6)) * 100.0
    bio15.attrs = {"long_name": "Precipitation Seasonality", "units": "%"}

    # ---- QUARTER-BASED variables (numpy take_along_axis) -------------
    # Quarter selection criteria use either q_tas (mean) or q_prec.

    # Wettest quarter
    bio16 = _select_quarter(q_prec, q_prec, "max")
    bio16.attrs = {"long_name": "Precipitation of Wettest Quarter",
                   "units": "mm"}
    # BIO08: mean temperature of wettest quarter -- uses tas
    bio08 = _select_quarter(q_tas, q_prec, "max")
    bio08.attrs = {"long_name": "Mean Temperature of Wettest Quarter",
                   "units": "degC",
                   "computed_from": "tas selected by wettest q_prec"}

    # Driest quarter
    bio17 = _select_quarter(q_prec, q_prec, "min")
    bio17.attrs = {"long_name": "Precipitation of Driest Quarter",
                   "units": "mm"}
    # BIO09: mean temperature of driest quarter -- uses tas
    bio09 = _select_quarter(q_tas, q_prec, "min")
    bio09.attrs = {"long_name": "Mean Temperature of Driest Quarter",
                   "units": "degC",
                   "computed_from": "tas selected by driest q_prec"}

    # Warmest quarter -- selection criterion is q_tas
    # BIO10: mean temperature of warmest quarter -- uses tas
    bio10 = _select_quarter(q_tas, q_tas, "max")
    bio10.attrs = {"long_name": "Mean Temperature of Warmest Quarter",
                   "units": "degC", "computed_from": "max of q_tas"}
    bio18 = _select_quarter(q_prec, q_tas, "max")
    bio18.attrs = {"long_name": "Precipitation of Warmest Quarter",
                   "units": "mm",
                   "computed_from": "prec selected by warmest q_tas"}

    # Coldest quarter -- selection criterion is q_tas
    # BIO11: mean temperature of coldest quarter -- uses tas
    bio11 = _select_quarter(q_tas, q_tas, "min")
    bio11.attrs = {"long_name": "Mean Temperature of Coldest Quarter",
                   "units": "degC", "computed_from": "min of q_tas"}
    bio19 = _select_quarter(q_prec, q_tas, "min")
    bio19.attrs = {"long_name": "Precipitation of Coldest Quarter",
                   "units": "mm",
                   "computed_from": "prec selected by coldest q_tas"}



    # Collect and clean
    result = {
        "BIO01": bio01, "BIO02": bio02, "BIO03": bio03,
        "BIO04": bio04, "BIO05": bio05, "BIO06": bio06,
        "BIO07": bio07, "BIO08": bio08, "BIO09": bio09,
        "BIO10": bio10, "BIO11": bio11, "BIO12": bio12,
        "BIO13": bio13, "BIO14": bio14, "BIO15": bio15,
        "BIO16": bio16, "BIO17": bio17, "BIO18": bio18,
        "BIO19": bio19,
    }

    # Drop lingering coordinates
    for key in result:
        for coord in ["quarter", "month"]:
            if coord in result[key].coords:
                result[key] = result[key].drop_vars(coord)

    return result