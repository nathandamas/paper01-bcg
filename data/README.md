# Data dictionary

All spatial files use SIRGAS 2000 / UTM zone 22S (EPSG:31982) unless their embedded GeoParquet metadata states otherwise. Cell identifiers are IBGE Statistical Grid identifiers.

## `analysis-ready/`

### `osm_area_timeseries_curitiba.parquet`

Long-format monthly OSM mapped-building-area history.

| Column | Meaning | Unit |
|---|---|---|
| `cell_id` | 1 × 1 km IBGE grid-cell identifier | — |
| `timestamp` | monthly snapshot, UTC | date |
| `osm_area_m2` | mapped area of OSM building polygons in the complete cell | m² |

Expected structure: 502 cells × 218 months, January 2008–February 2026. Monthly decreases are retained in this file; the cumulative-maximum correction is applied by the fitting script.

### `logistic_fits_area.parquet`

Archived four-parameter logistic fits for the OSM area series. The fitted model is

`y(t) = a + (b-a) / (1 + exp((m-t)/tau))`.

Legacy column names `c`/`c_area` and `d`/`d_area` correspond to `m` (inflection month) and `tau` (time constant), respectively. Area-family parameters `a_area` and `b_area` are expressed in m²; `c_area` and `d_area` are months. Standard-error, fit-quality, saturation, monotonicity-correction and validation fields accompany the estimates.

### `cell_analysis_input.geoparquet`

Frozen 502-cell input used for downstream regression and spatial analysis. It combines cell geometries, area-curve parameters and standard errors, final OSM area, VIDA reference count/area, area completeness, IBGE population variables and cell-level contextual fields. This is the canonical starting point for exact reproduction of the manuscript's statistical results.

Primary variables:

| Column | Manuscript quantity | Unit |
|---|---|---|
| `ref_completeness_area` | `C`, final cumulative-max OSM area / VIDA area | ratio |
| `b_area` | fitted mapped-area level `b-hat` | m² |
| `c_area` | inflection month `m-hat` | months after January 2008 |
| `d_area` | time constant `tau-hat` | months |
| `y_current_area` | final cumulative-max OSM building area | m² |
| `ref_area` | VIDA building-footprint area | km² |
| `cell_effective_density` | effective population density | inhabitants/km² of domiciliated support |

### `vida_reference_by_cell.parquet`

Cell-level count and area summaries derived from the pre-OSM-layer version of VIDA Combined. No source building geometry is included.

### `curitiba_boundary.parquet`

Curitiba municipal boundary used as cartographic context. Analytical grid cells were not clipped to this boundary.

## `reference-results/`

These files record the archived outputs against which a new run is compared.

| File | Purpose |
|---|---|
| `regression_sample.geoparquet` | final 207 cells, OLS fitted values/residuals and LISA labels |
| `mgwr_local_standardized.csv` | standardised MGWR local coefficients |
| `mgwr_inference_standardized.csv` | local t statistics and corrected-filtered t statistics |
| `diagnostics_global_spatial.json` | sample gate, OLS, Moran/LM, LISA, lag/error and GWR values |
| `diagnostics_nested_sem.json` | nested models, Rook sensitivity and SEM joint Wald test |
| `diagnostics_mgwr.json` | like-for-like standardised GWR/MGWR comparison |
| `diagnostics_mgwr_inference.json` | MGWR bandwidth intervals, ENP and corrected local inference |

## Integrity

Run `python scripts/00_validate.py --strict` to verify file hashes, record counts, schemas and expected numerical values. Hashes are stored in `data/checksums.sha256`.

