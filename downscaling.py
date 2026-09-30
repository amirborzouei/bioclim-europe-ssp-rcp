"""
downscaling.py
@author: amirrezaborzoueinia
"""

import numpy as np
import xarray as xr
import rioxarray as rxr
from pathlib import Path



#  Load CHELSA baseline                                               


def load_chelsa_baseline(chelsa_dir: Path,
                          variable: str,
                          temp_in_kelvin: bool = True) -> xr.DataArray:

    monthly = []
    var_dir = chelsa_dir / variable

    for m in range(1, 13):
        # Try common CHELSA V2.1 filename patterns
        candidates = [
            var_dir / f"CHELSA_{variable}_{m:02d}_1981-2010_V.2.1.tif",
            var_dir / f"CHELSA_{variable}_{m:02d}_1981-2010_V2.1.tif",
            var_dir / f"CHELSA_{variable}_{m:02d}_1981-2010.tif",
        ]
        f = next((c for c in candidates if c.exists()), None)
        if f is None:
            raise FileNotFoundError(
                f"Cannot find CHELSA {variable} month {m:02d} in {var_dir}\n"
                f"Tried:\n" +
                "\n".join(f"  {c.name}" for c in candidates) +
                f"\nCheck filenames match one of these patterns."
            )

        # mask_and_scale=True applies scale_factor and add_offset automatically
        # masked=True sets NoData (ocean) pixels to NaN
        da = rxr.open_rasterio(f, masked=True, mask_and_scale=True).squeeze()

        # Unit check and conversion
        mean_val = float(np.nanmean(da.values))
        if variable in ("tas", "tasmax", "tasmin"):
            if temp_in_kelvin and mean_val > 100:
                da = da - 273.15  # K -> degC
            elif not temp_in_kelvin and mean_val > 100:
                raise ValueError(
                    f"CHELSA_TEMP_IN_KELVIN=False but mean value is {mean_val:.1f}. "
                    f"Looks like Kelvin. Set CHELSA_TEMP_IN_KELVIN=True in config.py."
                )
        # Precipitation: CHELSA V2.1 pr is in kg/m2/month = mm/month, no conversion needed

        monthly.append(da)

    stacked = xr.concat(monthly, dim="month")
    stacked["month"] = range(1, 13)
    return stacked



#  Delta computation                                                   


def compute_delta_temperature(gcm_hist: xr.DataArray,
                               gcm_future: xr.DataArray) -> xr.DataArray:
    """
    Additive delta for temperature.
    delta = GCM_future_monthly_mean - GCM_historical_monthly_mean
    Both inputs have dim 'month' (1-12), units degC.
    """
    delta = gcm_future - gcm_hist
    delta.attrs["downscaling_method"] = "additive_delta_temperature"
    return delta


def compute_delta_precipitation(gcm_hist: xr.DataArray,
                                 gcm_future: xr.DataArray) -> xr.DataArray:
    """
    Multiplicative ratio for precipitation, following the reference
    implementation of Karger et al. 2023

        ratio_m = (P_fut_m + epsilon) / (P_hist_m + epsilon)

    with epsilon = 0.01 mm/day, added to BOTH numerator and denominator.

    The function receives precipitation in mm/month (the unit produced
    by data_utils._convert_units). It internally converts to mm/day for
    each calendar month using the standard non-leap-year month lengths
    before applying epsilon, so that the value of epsilon matches the
    Karger et al. (2023) specification exactly. The ratio itself is
    dimensionless and is applied multiplicatively to the CHELSA 1 km
    baseline (in mm/month) without further unit conversion.

    An upper clip of 100 is retained as a safety margin against pathological
    cells where the historical climatology is below epsilon and the future
    climatology is many times larger. No lower clip is needed: the epsilon
    on the denominator guarantees finite ratios in every cell.
    """
    # Standard non-leap-year days per calendar month
    # (1981-2010 has 8 leap years out of 30, so February's true average
    # is 28.25 days; using 28 introduces ~0.9% error on Feb only, which
    # is well below the methodological precision of the delta change method)
    days_per_month = np.array(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        dtype=np.float64,
    )
    days_da = xr.DataArray(
        days_per_month,
        dims=["month"],
        coords={"month": np.arange(1, 13)},
    )

    # Convert mm/month -> mm/day per calendar month
    hist_mm_per_day = gcm_hist / days_da
    fut_mm_per_day  = gcm_future / days_da

    # Apply Karger et al. (2023) Eq. 3: epsilon on BOTH terms
    epsilon = 0.01  # mm/day
    ratio = (fut_mm_per_day + epsilon) / (hist_mm_per_day + epsilon)

    # Upper safety clip only; lower bound handled by epsilon on denominator
    # ratio = ratio.clip(max=100.0)

    ratio.attrs["downscaling_method"] = "multiplicative_ratio_precipitation"
    ratio.attrs["epsilon_mm_per_day"] = float(epsilon)
    ratio.attrs["reference"] = (
        "Karger et al. (2023) chelsa-cmip6 1.0, Eq. 3, "
        "doi:10.1111/ecog.06535"
    )
    return ratio



#  Interpolation to 1km grid                                          


def interpolate_delta_to_chelsa(delta: xr.DataArray,
                                  chelsa_baseline: xr.DataArray) -> xr.DataArray:
   
    # Extract CHELSA coordinates
    chelsa_lats_raw = chelsa_baseline.y.values
    chelsa_lons_raw = chelsa_baseline.x.values

    # Sort y to ascending (South→North) — required for xarray interp
    chelsa_lats_asc = np.sort(chelsa_lats_raw)
    chelsa_lons_asc = np.sort(chelsa_lons_raw)

    # Rename GCM lat/lon → y/x to match CHELSA coordinate names
    delta_renamed = delta.rename({"lat": "y", "lon": "x"})

    # Ensure GCM coords are also ascending
    delta_renamed = delta_renamed.sortby("y").sortby("x")


    delta_1km = delta_renamed.interp(
        y=chelsa_lats_asc,
        x=chelsa_lons_asc,
        method="linear",
        kwargs={"fill_value": None}
    )

    # Reindex back to original CHELSA y order (North→South)
    # so result aligns with CHELSA data for the arithmetic step
    delta_1km = delta_1km.reindex(y=chelsa_lats_raw, x=chelsa_lons_raw)

    return delta_1km



#  Apply deltas                                                        


def apply_delta_temperature(chelsa_hist: xr.DataArray,
                              delta_1km: xr.DataArray) -> xr.DataArray:

    import numpy as np
    result_vals = chelsa_hist.values + delta_1km.values
    result = xr.DataArray(
        result_vals,
        dims=chelsa_hist.dims,
        coords=chelsa_hist.coords
    )
    result.attrs["processing"] = (
        "CHELSA_1981-2010_degC + IPSL-CM6A-LR_additive_delta_degC"
    )
    return result


def apply_delta_precipitation(chelsa_hist: xr.DataArray,
                                delta_1km: xr.DataArray) -> xr.DataArray:
    """
    Multiply CHELSA 1km baseline by interpolated precipitation ratio.

    Uses numpy arithmetic for the same reason as apply_delta_temperature:
    xarray inner-join coordinate alignment collapses the y dimension.
    clip(min=0) ensures no negative precipitation values.
    """
    import numpy as np
    result_vals = np.clip(chelsa_hist.values * delta_1km.values, 0.0, None)
    result = xr.DataArray(
        result_vals,
        dims=chelsa_hist.dims,
        coords=chelsa_hist.coords
    )
    result.attrs["processing"] = (
        "CHELSA_1981-2010_mm_x_IPSL-CM6A-LR_multiplicative_ratio"
    )
    return result



#  Full downscaling for one scenario-period combination               


def downscale_scenario(gcm_hist_clims: dict,
                        gcm_future_clims: dict,
                        chelsa_dir: Path,
                        temp_in_kelvin: bool = True) -> dict:

    # Load land mask if available (recommended — CHELSA V2.1 has no ocean NaN)
    land_mask = None
    mask_path = chelsa_dir / "land_mask_1km.tif"
    if mask_path.exists():
        from land_mask_utils import load_land_mask
        land_mask = load_land_mask(mask_path)
        print(f"    Land mask loaded: {mask_path.name}")
    else:
        print(f"    ⚠ No land mask found at {mask_path}")
        print(f"      Run setup_land_mask.py to create it.")
        print(f"      Continuing without mask — ocean cells will have values.")

    result = {}
    # NOTE: tas is processed alongside tasmin/tasmax/pr.
    # Same additive-delta logic applies for tas as for tasmin/tasmax,
    # since all three are temperature fields.
    for var in ("tas", "tasmin", "tasmax", "pr"):
        print(f"    Downscaling {var} ...", end=" ", flush=True)

        chelsa = load_chelsa_baseline(chelsa_dir, var, temp_in_kelvin)
        gcm_h  = gcm_hist_clims[var]
        gcm_f  = gcm_future_clims[var]

        if var == "pr":
            delta     = compute_delta_precipitation(gcm_h, gcm_f)
            delta_1km = interpolate_delta_to_chelsa(delta, chelsa)
            ds        = apply_delta_precipitation(chelsa, delta_1km)
        else:
            # Same additive-delta path for tas, tasmin, tasmax
            delta     = compute_delta_temperature(gcm_h, gcm_f)
            delta_1km = interpolate_delta_to_chelsa(delta, chelsa)
            ds        = apply_delta_temperature(chelsa, delta_1km)

        # Apply land mask: ocean cells become NaN
        # This is necessary because CHELSA V2.1 has values in ocean cells
        if land_mask is not None:
            from land_mask_utils import apply_land_mask
            ds = apply_land_mask(ds, land_mask)



        result[var] = ds
        print("done")
    return result