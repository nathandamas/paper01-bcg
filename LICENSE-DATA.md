# Data licences and attribution

The MIT licence in `LICENSE` applies to software only. It does not replace the terms attached to the source data represented by the redistributed aggregate products.

## OpenStreetMap-derived data

Files containing monthly mapped-building areas or quantities derived from them include:

- `data/analysis-ready/osm_area_timeseries_curitiba.parquet`
- `data/analysis-ready/logistic_fits_area.parquet`
- OSM-related columns in `data/analysis-ready/cell_analysis_input.geoparquet`
- corresponding columns in the archived regression and diagnostic outputs

These data are derived from OpenStreetMap and are made available under the [Open Data Commons Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Required attribution: **© OpenStreetMap contributors**. Users who publicly use, adapt, or redistribute the database or a substantial extraction must comply with the ODbL, including its attribution and share-alike requirements where applicable.

## VIDA Combined building data

The VIDA Combined cell aggregates in `vida_reference_by_cell.parquet`, and the reference-area quantities derived from them, originate from the pre-OSM-layer version of VIDA Combined. VIDA describes the combined database as distributed under the [Open Data Commons Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Attribute **VIDA Combined**, including its underlying Google Open Buildings and Microsoft GlobalML Building Footprints sources, as documented by VIDA.

Only cell-level aggregate footprint counts and areas are redistributed here; source building geometries are not included.

## IBGE-derived data

Grid geometry, census-population summaries, effective-population-density variables, and the municipal boundary originate from the Brazilian Institute of Geography and Statistics (IBGE). Attribute **Instituto Brasileiro de Geografia e Estatística (IBGE), 2022 Census / Statistical Grid and territorial boundary data**. Consult the [IBGE open-data portal](https://www.ibge.gov.br/acesso-informacao/dados-abertos.html) and the metadata accompanying the source products for applicable terms.

## Composite files

Several analysis-ready files combine fields derived from more than one source. Reuse must satisfy all applicable source terms. Nothing in this repository grants rights beyond those supplied by the original data providers.

## Basemap

The exact Figure 2 publication asset includes a CARTO/OpenStreetMap basemap credit rendered in the image. Basemap tiles are not redistributed as a database in this repository.

