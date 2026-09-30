"""
output_utils.py
@author: amirrezaborzoueinia
"""

import xarray as xr
from pathlib import Path
from config import OUTPUT_ROOT, OUTPUT_FORMAT, COMPRESS, NC_LAT_DIM, NC_LON_DIM


def _output_path(scenario_key: str,
                  period_label: str,
                  var_name: str,
                  root: Path | None = None) -> Path:
    """Build and create the output directory, return file path."""
    base = root if root is not None else OUTPUT_ROOT
    out_dir = base / scenario_key / period_label
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".nc" if OUTPUT_FORMAT == "netcdf" else ".tif"
    return out_dir / f"{var_name}{ext}"


def save_bioclim_variable(da: xr.DataArray,
                           var_name: str,
                           scenario_key: str,
                           period_label: str,
                           root: Path | None = None) -> Path:
   
    out_path = _output_path(scenario_key, period_label, var_name, root)

    # Attach metadata
    attrs = da.attrs.copy()
    attrs.update({
        "scenario":   scenario_key,
        "period":     period_label,
        "source_gcm": "IPSL-CM6A-LR",
        "experiment": "CMIP6",
        "created_by": "bioclim_thesis_pipeline",
    })
    da.attrs = attrs

    if OUTPUT_FORMAT == "netcdf":

        encoding = {}
        if COMPRESS:
            encoding[var_name] = {
                "zlib":      True,
                "complevel": 4,
                "dtype":     "float32",
            }
        ds_out = da.to_dataset(name=var_name)
        ds_out.to_netcdf(
            out_path,
            encoding=encoding if encoding else None
        )

    elif OUTPUT_FORMAT == "geotiff":
        try:
            import rioxarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "rioxarray is required for GeoTIFF output.\n"
                "Install: conda install -c conda-forge rioxarray"
            )
        da_geo = da.rio.set_spatial_dims(
            x_dim=NC_LON_DIM, y_dim=NC_LAT_DIM
        ).rio.write_crs("EPSG:4326")
        da_geo.rio.to_raster(
            out_path,
            compress="lzw" if COMPRESS else None
        )

    return out_path


def save_all_bioclim(bioclim_dict: dict,
                      scenario_key: str,
                      period_label: str,
                      root: Path | None = None) -> list[Path]:

    written = []
    for var_name, da in bioclim_dict.items():
        path = save_bioclim_variable(
            da, var_name, scenario_key, period_label, root
        )
        written.append(path)
    return written
