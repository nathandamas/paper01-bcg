# Manuscript-to-repository output map

| Manuscript item | Regeneration source | Script | Generated file |
|---|---|---|---|
| Table 1, data components | documented provenance constants | `05_make_tables.py` | `results/tables/Table_1_data_sources.*` |
| Table 2, screening cascade | 502-cell analysis input | `05_make_tables.py` | `results/tables/Table_2_screening.*` |
| Table 3A, full OLS coefficients | 207-cell analysis sample | `03_nested_models.py`, `05_make_tables.py` | `results/tables/Table_3_ols_nested.*` |
| Table 3B, nested models | 207-cell analysis sample | `03_nested_models.py`, `05_make_tables.py` | `results/tables/Table_3_ols_nested.*` |
| Table 4A, spatial-model progression | archived or refitted spatial diagnostics | `02_global_spatial.py`, `04_mgwr.py`, `05_make_tables.py` | `results/tables/Table_4_spatial_mgwr.*` |
| Table 4B, MGWR inference | standardised Gaussian MGWR | `04_mgwr.py`, `05_make_tables.py` | `results/tables/Table_4_spatial_mgwr.*` |
| Figure 1, OSM footprints and logistic curve | exact explanatory asset | copied by `reproduce.py` | `results/figures/Figure_1.png` |
| Figure 2, study area | exact cartographic asset | copied by `reproduce.py` | `results/figures/Figure_2.png` |
| Figure 3, workflow | exact explanatory asset | copied by `reproduce.py` | `results/figures/Figure_3.png` |
| Figure 4, spatial distributions | 502-cell input | `06_make_figures.py` | `results/figures/Figure_4.png` |
| Figure 5, OLS effects/nested models/area relation | 207-cell sample and diagnostics | `06_make_figures.py` | `results/figures/Figure_5.png` |
| Figure 6, residuals and LISA | 207-cell sample | `06_make_figures.py` | `results/figures/Figure_6.png` |
| Figure 7, MGWR local coefficients | local MGWR output joined to the sample | `06_make_figures.py` | `results/figures/Figure_7.png` |

CSV and Markdown versions of every table are generated. Figures are saved as PNG at 300 dpi and constrained to the journal's maximum pixel envelope (2138 × 2551 px).

