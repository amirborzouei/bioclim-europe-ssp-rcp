# High-resolution bioclimatic and seasonal climate variables for Europe under eight SSP-RCP pathways

Pipeline code for the dataset described in Borzoueinia, A. R. (2026), *The Creation and
Analysis of Bioclimatic Variables across Europe under SSP-RCP Scenarios*, M.Sc. thesis,
Brandenburg University of Technology Cottbus-Senftenberg.

The pipeline downscales IPSL-CM6A-LR monthly fields onto the CHELSA V2.1 baseline
climatology by delta change, then derives 19 WorldClim bioclimatic variables and 8
seasonal covariates for a 1981-2010 baseline and eight SSP-RCP pathways across three
future periods, at 30 arc-seconds (about 1 km).

**This repository contains the code only.** The derived dataset is deposited separately:
see *Related deposits* below.

Figure-generating scripts are deliberately not included. Everything plotted in the thesis
can be reproduced from the published dataset with any mapping library; shipping the
plotting code would add dependencies without adding reproducibility.

---

## Requirements

Python 3.11 with xarray, NumPy, pandas, GeoPandas, Shapely, rioxarray, rasterio, SciPy,
Dask, Matplotlib and Cartopy. See `environment.yml`.

## External inputs

Three inputs are not redistributed here. Download each and point `config.py` at it, or
export the matching environment variable.

| Input | Used for | Licence | Variable |
|---|---|---|---|
| IPSL-CM6A-LR monthly `tas`, `tasmax`, `tasmin`, `pr` (CMIP6, r1i1p1f1), via ESGF | change factors | CC BY-NC-SA 4.0 | `DATA_ROOT` / `$BIOCLIM_IPSL_DIR` |
| CHELSA V2.1 monthly climatology, 1981-2010 | baseline climatology | CC0 1.0 | `CHELSA_GLOBAL_DIR` / `$BIOCLIM_CHELSA_GLOBAL` |
| CHELSA V2.1 bioclim GeoTIFFs | validation only | CC0 1.0 | `CHELSA_BIOCLIM_DIR` / `$BIOCLIM_CHELSA_BIOCLIM` |
| GSHHG full-resolution shoreline, v2.3.7 | land mask | LGPL v3 | `GSHHG_SHP` / `$BIOCLIM_GSHHG_DIR` |

Everything else is derived from `PIPELINE_ROOT`, which defaults to the folder `config.py`
sits in. A fresh clone therefore runs without editing anything except those four paths.

## Running the pipeline

All 26 modules sit in the repository root, so the scripts find one another with no path
configuration. Run the six entry points in this order:

```
python setup_land_mask.py        # land mask from GSHHG
python chelsa_prep.py            # clip CHELSA to the domain
python main.py                   # BIO01-BIO19 at the driving model's resolution
python run_downscaling_full.py   # 1 km bioclimatic fields
python run_seasonal.py           # 1 km seasonal covariates
python run_nuts3.py              # aggregate covariates to NUTS3
```

**Core library.** `config.py` (all paths and constants), `bioclim_functions.py` (BIO1-BIO19),
`downscaling.py` (the delta-change calculation), `seasonal_function.py` (the eight seasonal
covariates), `land_mask_utils.py` (GSHHG land mask), `nuts3_aggregate.py` (regional
aggregation), `data_utils.py` and `output_utils.py` (loading and NetCDF writing).

**Quality control and validation**, run afterwards and independent of production:
`pipeline_check.py`, `qc_internal.py`, `qc_diagnostic.py`, `check_outputs.py`,
`verify_time_periods.py`, `chelsa_inspect.py`, `chelsa_baseline_compare.py`,
`chelsa_bioclim_plus_compare.py`, `eobs_validate.py`.

**Analysis**, producing the tables reported in the thesis: `extract_anomalies.py`,
`extract_seasonal.py`, `extract_spatial_structure.py`.

## Output layout

Every path takes the form `<root>/<scenario>/<period>/<file>`:

- `output_bioclim_1km/` &rarr; `BIOxx_1km.nc`, one file per variable
- `output_economic_1km/` &rarr; `ricardian_covariates_1km.nc`, eight covariates per file
- `output_economic_nuts3/` &rarr; `ricardian_covariates_nuts3.csv`, 1,345 NUTS3 regions

Scenario keys are `historical`, `ssp119`, `ssp126`, `ssp245`, `ssp370`, `ssp434`,
`ssp460`, `ssp534os`, `ssp585`. `ssp534os` has two future periods rather than three,
because the pathway branches from SSP5-8.5 only in 2040. NetCDF files are NetCDF-4 with
zlib compression at level 4, stored as 32-bit float.

---

## Licensing

**Code in this repository: MIT.** See `LICENSE`.

**The derived dataset: CC BY-NC-SA 4.0.** This is not a free choice. Every change factor
in the pipeline is computed from IPSL-CM6A-LR output, which is released under
CC BY-NC-SA 4.0. Both the non-commercial and the share-alike conditions carry over to
adapted material, so the derived fields inherit them. Users who need a permissively
licensed equivalent should re-run this pipeline against a driving model released under
CC BY 4.0; nothing in the code depends on IPSL specifically.

CHELSA V2.1 is released under CC0 1.0 and places no condition on derived products. GSHHG
is LGPL v3 and is used only as an input when constructing the land mask; its polygons are
not redistributed here.

## Required acknowledgement

The CMIP6 Terms of Use require the following, and it must be reproduced by anyone
publishing work based on this dataset:

> We acknowledge the World Climate Research Programme, which, through its Working Group
> on Coupled Modelling, coordinated and promoted CMIP6. We thank the climate modeling
> groups for producing and making available their model output, the Earth System Grid
> Federation (ESGF) for archiving the data and providing access, and the multiple funding
> agencies who support CMIP6 and ESGF.

| `source_id` | `institution_id` | Institution |
|---|---|---|
| IPSL-CM6A-LR | IPSL | Institut Pierre-Simon Laplace, Paris, France |

Variant label `r1i1p1f1`.

## Citing the inputs

Karger, D. N., Conrad, O., Böhner, J., Kawohl, T., Kreft, H., Soria-Auza, R. W.,
Zimmermann, N. E., Linder, H. P., Kessler, M. (2017). Climatologies at high resolution
for the earth's land surface areas. *Scientific Data* 4, 170122.
https://doi.org/10.1038/sdata.2017.122

Karger, D. N. et al. (2021). Climatologies at high resolution for the earth's land
surface areas. EnviDat. https://doi.org/10.16904/envidat.228

Wessel, P., & Smith, W. H. F. (1996). A global, self-consistent, hierarchical,
high-resolution shoreline database. *Journal of Geophysical Research* 101(B4).
https://doi.org/10.1029/96JB00104

## Related deposits

- Code: [CODE DOI]
- Dataset: [DATA DOI]

## Known limitations

The change factors derive from a single model and a single ensemble member, so no
uncertainty range attaches to any field and forced change cannot be separated from
internal variability. IPSL-CM6A-LR sits at the upper end of the CMIP6 range of
equilibrium climate sensitivity, so absolute magnitudes should be read as upper-side
estimates; the ordering of pathways and the spatial and seasonal patterns are far less
sensitive to that choice. Delta change shifts a distribution without reshaping it, so
nothing here supports statistics of extreme events or any question that turns on a change
in variability. Three variables, BIO08, BIO09 and BIO15, agree less closely with the
reference product than the rest for reasons written into their definitions; see the
thesis, Section 5.2.
