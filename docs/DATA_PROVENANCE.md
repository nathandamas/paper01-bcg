Data provenance
Spatial support
The analytical framework consists of the 502 complete 1 × 1 km cells of the 2022 IBGE Statistical Grid that intersect the Curitiba municipal boundary. Cells were retained whole, including portions extending outside the municipality. Metric overlay and area calculations use SIRGAS 2000 / UTM zone 22S (EPSG:31982).

OpenStreetMap history
Monthly mapped-building areas were obtained through the ohsome API for January 2008–February 2026. The archived long table contains 218 timestamps for each of 502 cells.

Query semantics:

Endpoint: /elements/area/groupBy/boundary
Boundary support: cell bounding boxes supplied to ohsome
Geometry: polygons and multipolygon relations
Filter: building=* and exclusion of building=no
Output used here: total mapped building area by boundary and month
The API response is not assumed to be monotone because deletions, retagging and geometry replacement can reduce mapped area. The curve-fitting script preserves the raw final value for audit and applies numpy.maximum.accumulate before fitting.

The archived file is the exact cell-level aggregate used in the analysis. Re-querying a live history service is deliberately outside the deterministic workflow because upstream database repairs or software changes could alter a later response. The manuscript endpoint and filter are documented above so the extraction can be independently repeated if desired.

VIDA Combined reference
The reference denominator was computed from the pre-OSM-layer version of VIDA Combined, which harmonised Google Open Buildings and Microsoft GlobalML Building Footprints. Using the pre-OSM layer prevents the denominator from directly incorporating the OSM data being evaluated.

Source footprints were intersected with the same complete grid cells; count and planimetric area were aggregated by cell. Only those aggregates are archived. VIDA is treated as an operational reference, not error-free ground truth.

IBGE data
The package includes derived fields from:

the 2022 IBGE Statistical Grid and Demographic Census;
census-sector population and domiciliated-area information used to derive effective population density; and
the Curitiba municipal boundary used for map context.
Favelas and Urban Communities were used only as visual context in the exact Figure 2 asset; this layer was not a model covariate and is not required for the statistical pipeline.

Processing chain
Arrange the OSM area series by cell and month.
Preserve the final raw observation and apply cumulative-maximum correction.
Fit the four-parameter logistic curve in scaled area units.
Convert area parameters and standard errors back to m².
Join fitted parameters, final OSM area, VIDA aggregates, grid geometry and IBGE variables by cell identifier.
Compute area completeness and apply the seven-part curve-quality gate plus the VIDA minimum-area rule.
Run the global and spatial models on the resulting 207 cells.
data/checksums.sha256 identifies the exact archived files.
