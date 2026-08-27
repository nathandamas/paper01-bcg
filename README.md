# OSM building-area growth and completeness: reproduction package

Anonymous review repository for the manuscript assessing whether parameters of four-parameter logistic curves fitted to OpenStreetMap (OSM) building-area histories are associated with independently measured building-area completeness in Curitiba, Brazil.

The repository contains only the data products and code needed to reproduce the reported analytical sample, Tables 1–4, and Figures 1–7. It intentionally omits the manuscript, author metadata, exploratory notebooks, development logs, superseded model families, and unrelated intermediate files.

## What is reproducible

Two routes are provided:

- **Quick reproduction** regenerates Tables 1–4 and Figures 4–7 from the archived analysis-ready inputs and reference model outputs. It also copies the exact publication assets for Figures 1–3. Runtime is normally under a few minutes.
- **Full model reproduction** rebuilds the 207-cell analytical sample, OLS and nested models, spatial diagnostics, spatial-lag and spatial-error models, GWR, and standardised MGWR. MGWR bandwidth selection is computationally intensive and may take one to two hours depending on hardware.
- **Curve-fit audit** independently refits the four-parameter logistic curves from the 502 monthly OSM building-area series. Because nonlinear optimisers can show small platform-dependent differences, the manuscript's downstream analysis starts from the frozen cell-level input in `data/analysis-ready/cell_analysis_input.geoparquet`.

The raw building polygons are not redistributed. The archived data are cell-level aggregates sufficient for the analyses reported in the manuscript.

## Repository layout

```text
.
├── data/
│   ├── analysis-ready/       # 502-cell time series, fitted parameters and covariates
│   ├── reference-results/    # archived outputs used for exact comparison
│   └── checksums.sha256
├── docs/                     # provenance, output map, reproduction and anonymity notes
├── reference/figures/        # seven 300 dpi manuscript figures
├── results/                  # regenerated outputs (ignored except placeholders)
├── scripts/                  # numbered analysis and figure scripts
├── src/paper1/               # reusable fitting, sample and spatial-analysis code
├── tests/                    # numerical and structural checks
├── config.yml                # analysis choices and expected headline values
├── environment.yml           # recommended Conda environment
└── requirements-lock.txt     # pinned Python packages
```

## Installation

The reference environment uses Python 3.11.

```bash
conda env create -f environment.yml
conda activate osm-building-completeness
python -m pip install -e .
```

An equivalent pip-only installation is:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

## Reproduce

Validate the archive first:

```bash
python scripts/00_validate.py --strict
```

Quick route:

```bash
python scripts/reproduce.py --mode quick
```

Full model route:

```bash
python scripts/reproduce.py --mode full
```

Optional independent curve refit:

```bash
python scripts/01_refit_logistic.py
```

The Makefile provides the same entry points:

```bash
make validate
make quick
make full
make test
```

Generated tables are written to `results/tables/`, figures to `results/figures/`, and model artefacts to `results/models/`. See `docs/OUTPUT_MAP.md` for the correspondence with the manuscript.

## Analysis specification

- Spatial support: 502 complete 1 × 1 km IBGE Statistical Grid cells intersecting Curitiba.
- OSM history: 218 monthly snapshots, January 2008–February 2026.
- Curve: `y(t) = a + (b-a) / (1 + exp((m-t)/tau))` after cumulative-maximum correction.
- Regression variables: inflection year `2008 + m/12`, annual growth rate `12/tau`, fitted mapped-area level `b`, and effective population density.
- Final quality-filtered sample: 207 cells.
- Spatial weights: Queen contiguity for the primary analysis; Rook contiguity for sensitivity.
- Random seed: 42; permutation tests use 999 permutations.
- Local models: adaptive Gaussian kernel; response and predictors are z-standardised for the GWR/MGWR comparison.

All thresholds are machine-readable in `config.yml`.

## Expected headline checks

Small differences in permutation p-values or local-model optimisation can occur across operating systems and binary-library builds. The pinned environment should reproduce the archived values within the tolerances defined in `config.yml`.

| Quantity | Archived value |
|---|---:|
| Final sample | 207 cells |
| OLS R² / adjusted R² | 0.829690 / 0.826317 |
| OLS RMSE | 0.087 |
| Joint Wald p-value for temporal shape | 0.414501 |
| Queen residual Moran's I / permutation p | 0.169711 / 0.004 |
| Spatial-error λ / AIC | 0.293763 / −424.47 |
| Standardised GWR bandwidth / AICc | 51 / 203.27 |
| Standardised MGWR R² / AICc | 0.932966 / 133.62 |
| MGWR bandwidths: intercept, inflection, rate, level, population | 11, 179, 198, 6, 7 |

## Data and software licences

Code is released under the MIT License in `LICENSE`. The redistributed data products have separate source obligations described in `LICENSE-DATA.md`. In particular, the OSM-derived and VIDA Combined database products are provided under ODbL 1.0 terms and require attribution and share-alike treatment where applicable. IBGE-derived data require source attribution.

## Anonymous-review safeguard

This package contains no author names, affiliations, e-mail addresses, ORCIDs, local absolute paths, manuscript file, or source-repository history. For double-anonymous review it must be published from a neutral account or anonymous archival service using a fresh Git history; do not fork the identifiable development repository. See `docs/ANONYMIZATION.md` before publication.

