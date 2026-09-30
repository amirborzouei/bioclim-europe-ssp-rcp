"""
config.py
@author: amirrezaborzoueinia
"""

from pathlib import Path
import os


#  SECTION 1 — PATHS                                                  

PIPELINE_ROOT = Path(os.environ.get(
    "BIOCLIM_ROOT", Path(__file__).resolve().parent))

#  EXTERNAL INPUTS — not redistributed with the code.

# Folder containing tas/, tasmax/, tasmin/, pr/ from GCM.
DATA_ROOT = Path(os.environ.get(
    "BIOCLIM_IPSL_DIR", "/path/to/GCM"))

# Global CHELSA V2.1 monthly climatology GeoTIFFs, before clipping.
CHELSA_GLOBAL_DIR = Path(os.environ.get(
    "BIOCLIM_CHELSA_GLOBAL", "/path/to/chelsa/climatologies"))

# Global CHELSA V2.1 bioclim GeoTIFFs, used by chelsa_baseline_compare.py.
# Structure: {CHELSA_BIOCLIM_DIR}/bio{NN}/1981-2010/CHELSA_bio{NN}_1981-2010_V.2.1.tif
CHELSA_BIOCLIM_DIR = Path(os.environ.get(
    "BIOCLIM_CHELSA_BIOCLIM", "/path/to/chelsa/global/bioclim"))

# Coastline for the land mask here (GSHHG).
#   https://www.soest.hawaii.edu/pwessel/gshhg/index.html
# (f = full resolution, L1 = land vs ocean polygons).
GSHHG_SHP = Path(os.environ.get(
    "BIOCLIM_GSHHG_DIR", "/path/to/gshhg-shp-2.3.7"))


#  DERIVED FROM PIPELINE_ROOT — no need to edit

NUTS_SHP               = PIPELINE_ROOT / "NUTS_2024_shp.shp"

OUTPUT_ROOT            = PIPELINE_ROOT / "output_bioclim"
OUTPUT_ROOT_1KM        = PIPELINE_ROOT / "output_bioclim_1km"
OUTPUT_ROOT_ECON       = PIPELINE_ROOT / "output_economic_1km"
OUTPUT_ROOT_ECON_NUTS3 = PIPELINE_ROOT / "output_economic_nuts3"

CHELSA_EUROPE_DIR      = PIPELINE_ROOT / "chelsa_europe"
FIGURES_DIR            = PIPELINE_ROOT / "figures"
RESULTS_FIGURES_DIR    = PIPELINE_ROOT / "results_figures"
VALIDATION_DIR         = PIPELINE_ROOT / "validation_outputs"
BASELINE_COMPARE_DIR   = VALIDATION_DIR / "baseline_compare"

BASELINE_DIR_1KM       = OUTPUT_ROOT_1KM / "historical" / "baseline"


#  SECTION 2 — SCENARIOS AND PERIODS                                  


VARIABLES = ["tas", "tasmax", "tasmin", "pr"]

# "folder" must match EXACTLY what is on disk (case, dashes, spaces).
# "years" covers the full range available in that folder's .nc files.
SCENARIOS = {
    "historical": {"folder": "Historical",  "years": (1981, 2014)},
    "ssp119":     {"folder": "SSP1-1.9",    "years": (2015, 2100)},
    "ssp126":     {"folder": "SSP1-2.6",    "years": (2015, 2100)},
    "ssp434":     {"folder": "SSP4-3.4",    "years": (2015, 2100)},
    "ssp534os":   {"folder": "SSP5-3.4OS",  "years": (2041, 2100)},
    "ssp245":     {"folder": "SSP2-4.5",    "years": (2015, 2100)},
    "ssp460":     {"folder": "SSP4-6.0",    "years": (2015, 2100)},
    "ssp370":     {"folder": "SSP3-7.0",    "years": (2015, 2100)},
    "ssp585":     {"folder": "SSP5-8.5",    "years": (2015, 2100)},
}

# 30-year analysis periods
PERIODS = {
    "baseline":  (1981, 2010),
    "2011-2040": (2011, 2040),
    "2041-2070": (2041, 2070),
    "2071-2100": (2071, 2100),
}

FUTURE_SCENARIOS = [s for s in SCENARIOS if s != "historical"]

# SSP5-3.4OS starts 2041 — 2011-2040 cannot be assembled
SKIP_PERIODS = {
    "ssp534os": ["2011-2040"],
}


#  SECTION 3 — NetCDF COORDINATE NAMES                                
#  Edit these if pipeline_check.py reports a coordinate mismatch.    


NC_TIME_DIM = "time"
NC_LAT_DIM  = "lat"
NC_LON_DIM  = "lon"

NC_VAR_NAMES = {
    "tas":    "tas",
    "tasmax": "tasmax",
    "tasmin": "tasmin",
    "pr":     "pr",
}


#  SECTION 4 — UNIT CONVERSIONS                                       

TEMP_OFFSET_K = 273.15  # subtract from Kelvin to get Celsius
PR_KG_TO_MM   = True    # IPSL pr: kg/m2/s -> mm/month

# Set this after running chelsa_inspect.py
# True  = CHELSA temperature files are in Kelvin (raw mean ~280)
# False = CHELSA temperature files are in Celsius (raw mean ~7)
CHELSA_TEMP_IN_KELVIN = True


#  SECTION 5 — SPATIAL DOMAIN                                         


EUROPE_EXTENT = {
    "lon_min": -25.0,
    "lon_max":  45.0,
    "lat_min":  27.0,
    "lat_max":  72.0,
}


#  SECTION 6 — MEMORY CONTROL (M3 MacBook Air 16GB)                  

DASK_CHUNKS = {
    "time": 6,    # process 6 months at a time
    "lat":  -1,   # all latitudes at once (fine for IPSL coarse grid)
    "lon":  -1,
}

DASK_WORKERS              = 1     # single-threaded: safest on MacBook Air
FORCE_COMPUTE_AND_RELEASE = True  # free RAM after each variable


#  SECTION 7 — OUTPUT FORMAT                                          


OUTPUT_FORMAT = "netcdf"  # "netcdf" (.nc) or "geotiff" (.tif)
COMPRESS      = True      # zlib compression


#  SECTION 8 — FIGURE STYLES                                          

SCENARIO_STYLES = {
    "ssp119":   {"label": "SSP1-1.9",   "color": "#1a9850", "ls": "-"},
    "ssp126":   {"label": "SSP1-2.6",   "color": "#66bd63", "ls": "-"},
    "ssp245":   {"label": "SSP2-4.5",   "color": "#fee090", "ls": "-"},
    "ssp434":   {"label": "SSP4-3.4",   "color": "#fdae61", "ls": "-"},
    "ssp460":   {"label": "SSP4-6.0",   "color": "#f46d43", "ls": "-"},
    "ssp370":   {"label": "SSP3-7.0",   "color": "#d73027", "ls": "-"},
    "ssp585":   {"label": "SSP5-8.5",   "color": "#a50026", "ls": "-"},
    "ssp534os": {"label": "SSP5-3.4OS", "color": "#9970ab", "ls": "--"},
}
