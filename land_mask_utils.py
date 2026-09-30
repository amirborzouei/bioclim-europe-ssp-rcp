"""
land_mask_utils.py
@author: amirrezaborzoueinia
"""

import numpy as np
import xarray as xr
import rioxarray as rxr
from pathlib import Path



#  Create mask from GSHHG polygons                                     


def create_land_mask(chelsa_reference_file: Path,
                      output_path: Path,
                      gshhg_shp: Path = None) -> xr.DataArray:

    
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
    except ImportError as e:
        raise ImportError(
            f"Missing package: {e}\n"
            f"Install with: conda install -c conda-forge geopandas -y\n"
            f"(rasterio is installed as part of rioxarray)"
        )

    # Resolve GSHHG path from config if not provided
    if gshhg_shp is None:
        try:
            import config
            gshhg_shp = Path(config.GSHHG_SHP)
        except (ImportError, AttributeError):
            raise ValueError(
                "GSHHG shapefile path not provided and config.GSHHG_SHP "
                "is not set. Download GSHHG version 2.3.7 from "
                "and set config.GSHHG_SHP to the path of GSHHS_f_L1.shp"
            )
    gshhg_shp = Path(gshhg_shp)
    if not gshhg_shp.exists():
        raise FileNotFoundError(
            f"GSHHG shapefile not found at: {gshhg_shp}\n"
            f"Expected file: GSHHS_shp/f/GSHHS_f_L1.shp"
        )

    print("  Loading reference CHELSA grid ...", end=" ", flush=True)
    ref = rxr.open_rasterio(
        chelsa_reference_file,
        masked=True, mask_and_scale=True
    ).squeeze()
    height = ref.shape[0]
    width  = ref.shape[1]
    left   = float(ref.x.min()) - abs(float(ref.rio.resolution()[0])) / 2
    right  = float(ref.x.max()) + abs(float(ref.rio.resolution()[0])) / 2
    bottom = float(ref.y.min()) - abs(float(ref.rio.resolution()[1])) / 2
    top    = float(ref.y.max()) + abs(float(ref.rio.resolution()[1])) / 2
    transform = from_bounds(left, bottom, right, top, width, height)
    print("done")

    print(f"  Loading GSHHG L1 full-resolution polygons from {gshhg_shp.name} ...",
          end=" ", flush=True)
    gdf = gpd.read_file(gshhg_shp)
    gdf = gdf.to_crs("EPSG:4326")

    
    gdf = gdf.cx[left:right, bottom:top]
    print(f"done ({len(gdf)} polygons in domain)")

    print("  Rasterizing land polygons to 1km grid ...", end=" ", flush=True)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    land_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,          # ocean = 0
        dtype=np.uint8,
        all_touched=False
    )
    # all_touched=False: only pixels whose centre falls inside a polygon
    # are marked as land. This gives the sharpest coastline.
    print("done")

    # Build DataArray matching the reference grid
    mask_da = xr.DataArray(
        land_raster.astype(bool),
        dims=["y", "x"],
        coords={"y": ref.y.values, "x": ref.x.values}
    )
    mask_da.attrs = {
        "description": "Land mask derived from GSHHG L1 full-resolution polygons",
        "True_means":  "land cell",
        "False_means": "ocean or lake cell",
    }
    mask_da.rio.write_crs("EPSG:4326", inplace=True)

    # Save as GeoTIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_da.astype(np.uint8).rio.to_raster(
        output_path, compress="lzw", dtype="uint8"
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Land mask saved: {output_path.name}  ({size_mb:.1f} MB)")
    print(f"  Land coverage: "
          f"{land_raster.sum():,} cells / {land_raster.size:,} total "
          f"({land_raster.mean()*100:.1f}%)")

    return mask_da



#  Load existing mask                                                  


def load_land_mask(mask_path: Path) -> xr.DataArray:

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Land mask not found: {mask_path}\n"
            f"Run create_land_mask() first (see setup_land_mask.py)."
        )
    da = rxr.open_rasterio(mask_path, masked=False).squeeze()
    return da.astype(bool)



#  Apply mask to any DataArray                                         


def apply_land_mask(da: xr.DataArray,
                     mask: xr.DataArray) -> xr.DataArray:
   
    # Detect spatial resolution of da vs mask
    lat_coord = "lat" if "lat" in da.coords else "y"

    da_spatial_size  = da.sizes.get(lat_coord, 0)
    mask_spatial_size = mask.sizes.get("y", mask.sizes.get("lat", 0))

    if da_spatial_size == mask_spatial_size:

        
        return da.where(mask.values.astype(bool))

    else:

        lon_coord = "lon" if "lon" in da.coords else "x"

        mask_r = mask
        if lat_coord == "lat" and "y" in mask.coords:
            mask_r = mask.rename({"y": "lat", "x": "lon"})

        mask_coarse = mask_r.interp(
            {lat_coord: da[lat_coord].values,
             lon_coord: da[lon_coord].values},
            method="nearest"
        ).astype(bool)

        # Use .values here too for the same reason
        return da.where(mask_coarse.values.astype(bool))


#  Coarsen mask to GCM resolution (for GCM-native statistics)         


def coarsen_mask_to_gcm(mask_1km: xr.DataArray,
                         gcm_lats: np.ndarray,
                         gcm_lons: np.ndarray,
                         threshold: float = 0.5) -> xr.DataArray:

    # Rename to lat/lon for GCM alignment
    if "y" in mask_1km.dims:
        mask_r = mask_1km.rename({"y": "lat", "x": "lon"})
    else:
        mask_r = mask_1km

    mask_float = mask_r.astype(float)
    land_frac  = mask_float.interp(
        lat=gcm_lats,
        lon=gcm_lons,
        method="linear",
        kwargs={"fill_value": 0.0}
    )
    return land_frac >= threshold
