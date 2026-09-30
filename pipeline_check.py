"""
pipeline_check.py
@author: amirrezaborzoueinia
"""


import sys
import time
from pathlib import Path

# Colour codes for macOS terminal
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(label, msg=""):
    print(f"  {GREEN}✓  PASS{RESET}  {BOLD}{label}{RESET}  {msg}")

def fail(label, msg):
    print(f"  {RED}✗  FAIL{RESET}  {BOLD}{label}{RESET}")
    print(f"         {RED}→ {msg}{RESET}")

def warn(label, msg):
    print(f"  {YELLOW}⚠  WARN{RESET}  {BOLD}{label}{RESET}")
    print(f"         {YELLOW}→ {msg}{RESET}")

def section(title):
    print(f"\n{BOLD}{'─'*55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*55}{RESET}")

failures      = []
warnings_list = []



#  CHECK 1 — Python packages                                          

section("1  Python package imports")

required = {
    "xarray":     "conda install -c conda-forge xarray",
    "numpy":      "conda install -c conda-forge numpy",
    "netCDF4":    "conda install -c conda-forge netCDF4",
    "dask":       "conda install -c conda-forge dask",
    "scipy":      "conda install -c conda-forge scipy",
    "pandas":     "conda install -c conda-forge pandas",
    "cftime":     "pip install cftime",
    "matplotlib": "conda install -c conda-forge matplotlib",
    "cartopy":    "conda install -c conda-forge cartopy",
    "rioxarray":  "conda install -c conda-forge rioxarray",
}

for pkg, cmd in required.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        ok(pkg, f"version {ver}")
    except ImportError:
        fail(pkg, f"Not installed. Fix: {cmd}")
        failures.append(f"Missing package: {pkg}")



#  CHECK 2 — config.py                                                

section("2  config.py")

try:
    import config
    ok("config.py", "imports without errors")
except Exception as e:
    fail("config.py", f"Import error: {e}")
    failures.append("config.py broken")
    print("\n  Cannot continue. Fix config.py first.")
    sys.exit(1)

if str(config.DATA_ROOT).endswith("IPSL-CM6A-LR (France)") and \
   not config.DATA_ROOT.exists():
    fail("DATA_ROOT",
         f"Path does not exist: {config.DATA_ROOT}\n"
         f"         Check the path is typed correctly.")
    failures.append("DATA_ROOT path not found")
elif not config.DATA_ROOT.exists():
    fail("DATA_ROOT",
         f"Does not exist: {config.DATA_ROOT}")
    failures.append("DATA_ROOT path not found")
else:
    ok("DATA_ROOT", str(config.DATA_ROOT))

# GSHHG shapefile used by setup_land_mask.py
if hasattr(config, "GSHHG_SHP"):
    if Path(config.GSHHG_SHP).exists():
        ok("GSHHG_SHP", str(config.GSHHG_SHP))
    else:
        warn("GSHHG_SHP",
             f"Path does not exist: {config.GSHHG_SHP}\n"
             f"         Required by setup_land_mask.py to build the land mask.\n"
             f"         Download GSHHG v2.3.7 from\n"
             f"         https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/\n"
             f"         and extract the archive. The expected file is\n"
             f"         GSHHS_shp/f/GSHHS_f_L1.shp inside the extracted folder.")
else:
    warn("GSHHG_SHP",
         "config.GSHHG_SHP is not defined. The land-mask step will fail.\n"
         "         Add GSHHG_SHP = Path('...') to config.py.")



#  CHECK 3 — Folder structure                                         

section("3  Folder structure on disk")

import xarray as xr
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

missing_count = 0
test_file  = None
test_var   = None

for var in config.VARIABLES:
    var_dir = config.DATA_ROOT / var
    if not var_dir.exists():
        fail(f"{var}/", f"Folder not found: {var_dir}")
        failures.append(f"Missing variable folder: {var}")
        continue
    ok(f"{var}/", "folder exists")

    for scen_key, scen_info in config.SCENARIOS.items():
        scen_dir = var_dir / scen_info["folder"]
        if not scen_dir.exists():
            fail(f"  {var}/{scen_info['folder']}/",
                 f"Not found. Check spelling exactly: '{scen_info['folder']}'")
            missing_count += 1
            continue
        nc_files = sorted(scen_dir.glob("*.nc"))
        if not nc_files:
            fail(f"  {var}/{scen_info['folder']}/",
                 "Folder exists but has no .nc files")
            failures.append(f"Empty: {scen_dir}")
        else:
            mb = sum(f.stat().st_size for f in nc_files) / 1e6
            ok(f"  {var}/{scen_info['folder']}/",
               f"{len(nc_files)} file(s) — {mb:.0f} MB")
            if test_file is None:
                test_file = nc_files[0]
                test_var  = var

if missing_count:
    warn("Missing folders",
         f"{missing_count} folders not found. Check SCENARIOS in config.py.")



#  CHECK 4 — Open one NetCDF file and inspect contents               

section("4  NetCDF file inspection")

if test_file is None:
    fail("NetCDF open", "No files found. Fix folder structure first.")
    failures.append("No files to inspect")
else:
    try:
        ds   = xr.open_dataset(test_file, decode_times=TIME_CODER)
        dims = list(ds.dims)
        ok("open", f"{test_file.name}")
        ok("dimensions", str(dims))

        # Check lat/lon coordinates
        for expected in [config.NC_LAT_DIM, config.NC_LON_DIM]:
            if expected in ds.dims:
                vals = ds[expected].values
                ok(f"  coord '{expected}'",
                   f"range {float(vals.min()):.2f} → {float(vals.max()):.2f}")
                if expected == config.NC_LON_DIM and float(vals.max()) > 180:
                    warn(f"  coord '{expected}'",
                         "0-360 longitude detected — pipeline auto-converts. No action needed.")
                    warnings_list.append("Longitude 0-360 (auto-handled)")
            else:
                similar = [d for d in dims if expected[:3].lower() in d.lower()]
                hint = (f"Found '{similar[0]}' — set NC_{expected.upper()}_DIM"
                        f" = '{similar[0]}' in config.py") if similar else ""
                fail(f"  coord '{expected}'",
                     f"Not found. {hint}")
                failures.append(f"Wrong coord name: {expected}")

        # Check time coordinate using cftime-safe .dt.year
        if config.NC_TIME_DIM in ds.dims:
            years = ds[config.NC_TIME_DIM].dt.year.values
            ok(f"  coord '{config.NC_TIME_DIM}'",
               f"{int(years.min())}–{int(years.max())} "
               f"({len(years)} monthly timesteps)")
        else:
            fail(f"  coord '{config.NC_TIME_DIM}'",
                 "Time dimension not found. Check NC_TIME_DIM in config.py.")
            failures.append("Missing time coordinate")

        # Check variable name
        var_name = config.NC_VAR_NAMES[test_var]
        if var_name in ds.data_vars:
            da   = ds[var_name]
            unit = da.attrs.get("units", "not specified")
            ok(f"  variable '{var_name}'",
               f"shape {dict(da.sizes)}  units: {unit}")
        else:
            actual = [v for v in ds.data_vars
                      if v not in ("lat_bnds","lon_bnds","time_bnds")]
            fail(f"  variable '{var_name}'",
                 f"Not found. Actual variables: {actual}. "
                 f"Update NC_VAR_NAMES in config.py.")
            failures.append(f"Wrong variable name: {var_name}")

        ds.close()

    except Exception as e:
        fail("NetCDF open", str(e))
        failures.append("Cannot open NetCDF file")



#  CHECK 5 — Unit sanity                                              

section("5  Unit sanity check")

try:
    import numpy as np

    # Temperature check
    for tvar in ["tasmin", "tasmax", "tas"]:
        scen_dir = config.DATA_ROOT / tvar / config.SCENARIOS["historical"]["folder"]
        files = sorted(scen_dir.glob("*.nc"))
        if not files:
            continue
        ds    = xr.open_dataset(files[0], decode_times=TIME_CODER)
        da    = ds[config.NC_VAR_NAMES[tvar]]
        units = da.attrs.get("units", "unknown")
        # Strip non-spatial coords before mean to avoid cftime float() errors
        da_sl = da.isel({config.NC_TIME_DIM: 0})
        drop  = [c for c in da_sl.coords
                 if c not in (config.NC_LAT_DIM, config.NC_LON_DIM)]
        sample = float(da_sl.drop_vars(drop, errors="ignore").mean().values)
        if units == "K" or sample > 100:
            ok(f"{tvar} units",
               f"Kelvin (mean={sample:.1f}K) — auto-converts to °C")
        elif -80 < sample < 60:
            ok(f"{tvar} units",
               f"Already °C (mean={sample:.1f}°C)")
        else:
            warn(f"{tvar} units",
                 f"Unexpected (mean={sample:.4f}, units={units}) — check manually")
            warnings_list.append(f"Unexpected unit range for {tvar}")
        ds.close()
        break

    # Precipitation check
    scen_dir = config.DATA_ROOT / "pr" / config.SCENARIOS["historical"]["folder"]
    files = sorted(scen_dir.glob("*.nc"))
    if files:
        ds    = xr.open_dataset(files[0], decode_times=TIME_CODER)
        da    = ds[config.NC_VAR_NAMES["pr"]]
        units = da.attrs.get("units", "unknown")
        da_sl = da.isel({config.NC_TIME_DIM: 0})
        drop  = [c for c in da_sl.coords
                 if c not in (config.NC_LAT_DIM, config.NC_LON_DIM)]
        sample = float(da_sl.drop_vars(drop, errors="ignore").mean().values)
        if units in ("kg m-2 s-1","kg/m2/s","kg m**-2 s**-1") or sample < 0.1:
            ok("pr units",
               f"kg/m²/s (mean={sample:.2e}) — auto-converts to mm/month")
        else:
            warn("pr units",
                 f"units='{units}', mean={sample:.4f} — may already be mm. "
                 "If so, set PR_KG_TO_MM=False in config.py")
            warnings_list.append("Verify pr units manually")
        ds.close()

except Exception as e:
    warn("Unit check", f"Could not complete: {e}")



#  CHECK 6 — Mini compute test                                        

section("6  Mini compute test (1 month slice)")

try:
    t0 = time.time()
    scen_dir = (config.DATA_ROOT / "tasmin" /
                config.SCENARIOS["historical"]["folder"])
    files = sorted(scen_dir.glob("*.nc"))
    if files:
        ds  = xr.open_dataset(files[0], decode_times=TIME_CODER,
                               chunks=config.DASK_CHUNKS)
        da  = ds[config.NC_VAR_NAMES["tasmin"]]
        da  = da.sel(**{
            config.NC_LON_DIM: slice(
                config.EUROPE_EXTENT["lon_min"],
                config.EUROPE_EXTENT["lon_max"]),
            config.NC_LAT_DIM: slice(
                config.EUROPE_EXTENT["lat_min"],
                config.EUROPE_EXTENT["lat_max"]),
        })
        result = da.isel({config.NC_TIME_DIM: 0}).compute()
        elapsed = time.time() - t0
        ok("compute one slice",
           f"shape={dict(result.sizes)}  "
           f"time={elapsed:.2f}s  "
           f"mean={float(result.mean().values):.2f}")
        ds.close()
    else:
        warn("Mini compute", "No tasmin/historical files to test with")
except Exception as e:
    fail("Mini compute", str(e))
    failures.append("Compute test failed")



#  CHECK 7 — Output directory writeable                               

section("7  Output directory")

try:
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    test_f = config.OUTPUT_ROOT / ".write_test"
    test_f.write_text("ok")
    test_f.unlink()
    ok("OUTPUT_ROOT", str(config.OUTPUT_ROOT.resolve()))
except Exception as e:
    fail("OUTPUT_ROOT", f"Cannot write: {e}")
    failures.append("Output directory not writable")



#  CHECK 8 — Disk space estimate                                      

section("8  Disk space")

import shutil
free_gb = shutil.disk_usage(config.OUTPUT_ROOT.parent).free / 1e9
est_gb  = 19 * 25 * 5 / 1000   # ~2.4 GB for coarse-resolution outputs
if free_gb > est_gb * 2:
    ok("disk space", f"{free_gb:.1f} GB free (need ~{est_gb:.1f} GB)")
elif free_gb > est_gb:
    warn("disk space",
         f"{free_gb:.1f} GB free — tight. Need ~{est_gb:.1f} GB.")
    warnings_list.append("Low disk space")
else:
    fail("disk space",
         f"Only {free_gb:.1f} GB free. Need ~{est_gb:.1f} GB.")
    failures.append("Insufficient disk space")


#  SUMMARY                                                            

print(f"\n{'═'*55}")
print(f"{BOLD}  SUMMARY{RESET}")
print(f"{'═'*55}")

if not failures and not warnings_list:
    print(f"  {GREEN}{BOLD}ALL CHECKS PASSED.{RESET}")
    print(f"\n  Next step — run your first test in Spyder console:")
    print(f"    from main import main")
    print(f"    main(scenario='historical', period='baseline', "
          f"variables_to_run=['BIO01'])\n")
elif not failures:
    print(f"  {YELLOW}{BOLD}{len(warnings_list)} warning(s) — "
          f"review but pipeline will likely work.{RESET}")
    for w in warnings_list:
        print(f"    ⚠  {w}")
    print(f"\n  When ready:")
    print(f"    from main import main")
    print(f"    main(scenario='historical', period='baseline', "
          f"variables_to_run=['BIO01'])\n")
else:
    print(f"  {RED}{BOLD}{len(failures)} FAILURE(S) — fix before running:{RESET}")
    for f_ in failures:
        print(f"    ✗  {f_}")
    if warnings_list:
        print(f"\n  {YELLOW}Warnings:{RESET}")
        for w in warnings_list:
            print(f"    ⚠  {w}")
    print()
