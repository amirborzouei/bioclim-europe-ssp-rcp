"""
run_nuts3.py
@author: amirrezaborzoueinia
"""
from pathlib import Path
import geopandas as gpd
from nuts3_aggregate import aggregate_scenario

# ---- 1) paths come from config.py; nothing to edit here ----------------------
from config import NUTS_SHP, OUTPUT_ROOT_ECON, OUTPUT_ROOT_ECON_NUTS3

# ---- 2) load NUTS level 3 in WGS84 degrees (to match the 1 km grid) ----------
nuts = gpd.read_file(NUTS_SHP)
nuts = nuts[nuts["LEVL_CODE"] == 3].to_crs(4326).reset_index(drop=True)
print(f"Loaded {len(nuts)} NUTS3 regions")

# ---- 3) loop over every 1 km output and write a NUTS3 CSV ---------------------
IN_ROOT  = OUTPUT_ROOT_ECON
OUT_ROOT = OUTPUT_ROOT_ECON_NUTS3
files = sorted(IN_ROOT.glob("*/*/ricardian_covariates_1km.nc"))
if not files:
    print("No 1 km files found. Run run_seasonal.py first.")
for nc in files:
    scenario, period = nc.parts[-3], nc.parts[-2]
    out_csv = OUT_ROOT / scenario / period / "ricardian_covariates_nuts3.csv"
    res = aggregate_scenario(nc, nuts, out_csv)
    print(f"  {scenario}/{period}: {len(res)} regions -> {out_csv}")
print("done")