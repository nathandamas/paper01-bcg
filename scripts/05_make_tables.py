#!/usr/bin/env python3
"""Generate manuscript Tables 1–4 as CSV and Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm

from paper1 import spatial
from paper1.paths import (
    ANALYSIS_READY,
    MODEL_RESULTS,
    REFERENCE_RESULTS,
    TABLE_RESULTS,
    ensure_results,
)
from paper1.sample import build_sample, compute_quality_flags


def write_frame(frame: pd.DataFrame, stem: str, heading: str | None = None) -> None:
    frame.to_csv(TABLE_RESULTS / f"{stem}.csv", index=False)
    text = ""
    if heading:
        text += f"# {heading}\n\n"
    text += frame.to_markdown(index=False) + "\n"
    (TABLE_RESULTS / f"{stem}.md").write_text(text, encoding="utf-8")


def table_1() -> None:
    frame = pd.DataFrame(
        [
            {
                "Data component": "OSM building history",
                "Original support": "Building polygons and multipolygon relations",
                "Period/version": "Monthly, Jan 2008–Feb 2026",
                "Cell-level processing": "ohsome mapped area; cumulative-maximum correction",
                "Analytical role": "Four-parameter logistic trajectories and final OSM area",
            },
            {
                "Data component": "VIDA Combined (pre-OSM layer)",
                "Original support": "Google/Microsoft building footprints",
                "Period/version": "Archived pre-OSM-layer release",
                "Cell-level processing": "Footprint count and intersected planimetric area",
                "Analytical role": "Independent operational reference denominator",
            },
            {
                "Data component": "2022 IBGE Statistical Grid and Census",
                "Original support": "1 × 1 km grid and census sectors",
                "Period/version": "2022 Census",
                "Cell-level processing": "Complete cells; population and domiciliated support",
                "Analytical role": "Spatial support and effective population density",
            },
            {
                "Data component": "IBGE municipal boundary",
                "Original support": "Municipal polygon",
                "Period/version": "2022 territorial division",
                "Cell-level processing": "Selection/context only; cells not clipped",
                "Analytical role": "Study-area definition and cartography",
            },
        ]
    )
    write_frame(frame, "Table_1_data_sources", "Table 1. Data components")


def table_2(cells: gpd.GeoDataFrame) -> None:
    series = pd.read_parquet(ANALYSIS_READY / "osm_area_timeseries_curitiba.parquet")
    ordered = series.sort_values(["cell_id", "timestamp"])
    summaries = []
    for cell_id, group in ordered.groupby("cell_id", sort=False):
        values = group["osm_area_m2"].to_numpy(dtype=float)
        corrected = np.maximum.accumulate(values)
        summaries.append(
            {
                "cell_id": cell_id,
                "raw_final": values[-1],
                "corrected_range": corrected.max() - corrected.min(),
            }
        )
    screening = pd.DataFrame(summaries)
    flags = compute_quality_flags(cells.reset_index(drop=True))
    rows = [
        ("Initial 1 × 1 km cells", len(cells)),
        ("Raw final mapped area > 1,000 m²", int((screening["raw_final"] > 1000).sum())),
        ("Cumulative-max-corrected range ≥ 5,000 m²", int((screening["corrected_range"] >= 5000).sum())),
        ("Convergent four-parameter logistic fit", int(flags["flag_convergence"].sum())),
        ("RSE(b) and RSE(m) ≤ 0.50", int(flags["flag_rse_b_infl"].sum())),
        ("RSE(tau) ≤ 1.00", int(flags["flag_rse_rate"].sum())),
        ("R² ≥ 0.90", int(flags["flag_r2"].sum())),
        ("Current area ≥ 0.85 × fitted level", int(flags["flag_saturation"].sum())),
        ("6 ≤ inflection month ≤ 210", int(flags["flag_c_window"].sum())),
        ("Non-degenerate fit (R² < 0.99999)", int(flags["flag_not_degenerate"].sum())),
        ("All criteria and VIDA area ≥ 5,000 m²", len(build_sample(cells)[0])),
    ]
    frame = pd.DataFrame(rows, columns=["Screening criterion", "Cells passing"])
    write_frame(frame, "Table_2_screening", "Table 2. Curve-screening cascade")


def nested_metrics(y: np.ndarray, X: pd.DataFrame) -> dict[str, dict[str, object]]:
    specifications = {
        "Mapped-area level only": ["b_asymptote"],
        "Level and population": ["b_asymptote", "pop_density"],
        "Full model": ["inflection", "rate", "b_asymptote", "pop_density"],
        "Without mapped-area level": ["inflection", "rate", "pop_density"],
    }
    output = {}
    for name, columns in specifications.items():
        model = sm.OLS(y, sm.add_constant(X[columns], has_constant="add")).fit()
        output[name] = {
            "predictors": columns,
            "R2": float(model.rsquared),
            "Adjusted R2": float(model.rsquared_adj),
            "RMSE": float(np.sqrt(np.mean(model.resid**2))),
            "AIC": float(model.aic),
            "BIC": float(model.bic),
        }
    return output


def table_3(cells: gpd.GeoDataFrame) -> None:
    y, X, _ = build_sample(cells)
    robust = spatial.run_ols(y, X)
    labels = {
        "const": "Intercept",
        "rate": "Annual growth rate",
        "inflection": "Inflection year",
        "b_asymptote": "Mapped-area level (m²)",
        "pop_density": "Effective population density (inhabitants/km²)",
    }
    order = ["const", "rate", "inflection", "b_asymptote", "pop_density"]
    panel_a = robust.coefficients.loc[order].reset_index(names="term")
    panel_a.insert(0, "Panel", "A. Full-model coefficient estimates")
    panel_a["Predictor"] = panel_a["term"].map(labels)
    panel_a = panel_a[
        ["Panel", "Predictor", "coef", "std_err", "t", "p", "beta_star"]
    ].rename(
        columns={
            "coef": "Coefficient",
            "std_err": "HC3 SE",
            "t": "t",
            "p": "p",
            "beta_star": "Standardised coefficient",
        }
    )

    nested = nested_metrics(y, X)
    predictor_symbols = {
        "Mapped-area level only": "b",
        "Level and population": "b + population density",
        "Full model": "inflection + rate + b + population density",
        "Without mapped-area level": "inflection + rate + population density",
    }
    panel_b = pd.DataFrame(
        [
            {
                "Panel": "B. Nested-model comparison",
                "Model": name,
                "Predictors": predictor_symbols[name],
                **values,
                "AIC / BIC": f"{values['AIC']:.2f} / {values['BIC']:.2f}",
            }
            for name, values in nested.items()
        ]
    ).drop(columns=["predictors", "AIC", "BIC"])

    combined = pd.concat([panel_a, panel_b], ignore_index=True, sort=False)
    write_frame(combined, "Table_3_ols_nested", "Table 3. Global OLS and nested models (n = 207)")


def load_spatial_values(source: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if source == "reference":
        global_payload = json.loads(
            (REFERENCE_RESULTS / "diagnostics_global_spatial.json").read_text(encoding="utf-8")
        )["results"]["ref_completeness_area"]
        mgwr_payload = json.loads(
            (REFERENCE_RESULTS / "diagnostics_mgwr.json").read_text(encoding="utf-8")
        )
        inference_payload = json.loads(
            (REFERENCE_RESULTS / "diagnostics_mgwr_inference.json").read_text(encoding="utf-8")
        )
        progression = [
            {"Model": "OLS", "Spatial parameter or bandwidth": "—", "R² measure": f"{global_payload['ols']['adj_r2']:.3f} adjusted", "AIC/AICc": f"{global_payload['ols']['aic']:.2f} AIC", "Main result": "Spatially autocorrelated residuals"},
            {"Model": "Spatial lag, ML", "Spatial parameter or bandwidth": f"rho = {global_payload['spatial_lag']['rho']:.3f}", "R² measure": f"{global_payload['spatial_lag']['R2']:.3f} pseudo-R²", "AIC/AICc": f"{global_payload['spatial_lag']['AIC']:.2f} AIC", "Main result": "Limited improvement over OLS"},
            {"Model": "Spatial error, ML", "Spatial parameter or bandwidth": f"lambda = {global_payload['spatial_error']['lambda']:.3f}", "R² measure": f"{global_payload['spatial_error']['R2']:.3f} pseudo-R²", "AIC/AICc": f"{global_payload['spatial_error']['AIC']:.2f} AIC", "Main result": "Temporal-shape joint test remains non-significant"},
            {"Model": "GWR, original scale", "Spatial parameter or bandwidth": f"{global_payload['gwr']['bw']:.0f} neighbours", "R² measure": f"{global_payload['gwr']['r2']:.3f}", "AIC/AICc": f"{global_payload['gwr']['aicc']:.2f} AICc", "Main result": "Single adaptive Gaussian bandwidth"},
            {"Model": "GWR (z scale)", "Spatial parameter or bandwidth": f"{mgwr_payload['gwr_std']['bw']:.0f} neighbours", "R² measure": f"{mgwr_payload['gwr_std']['r2']:.3f}", "AIC/AICc": f"{mgwr_payload['gwr_std']['aicc']:.2f} AICc", "Main result": "Like-for-like comparison model"},
            {"Model": "MGWR (z scale)", "Spatial parameter or bandwidth": "Predictor-specific", "R² measure": f"{mgwr_payload['mgwr_std']['r2']:.3f}", "AIC/AICc": f"{mgwr_payload['mgwr_std']['aicc']:.2f} AICc", "Main result": "Preferred to GWR on the same standardised scale"},
        ]
        intervals = inference_payload["bws_intervals_gaussian"]
        inference_by_name = {
            row["covariate"]: row for row in inference_payload["inference_gaussian_std"]
        }
        bws = mgwr_payload["mgwr_std"]["bandwidths"]
        terms = ["const", "inflection", "rate", "b_asymptote", "pop_density"]
        inference = [
            {
                "Term": term,
                "Bandwidth": int(bws[term]),
                "AICc-weighted 95% interval": f"{int(intervals[term][0])}–{int(intervals[term][1])}",
                "ENP_j": inference_by_name[term]["ENP_j"],
                "Median |t|": inference_by_name[term]["median_abs_t"],
                "Significant cells (%)": inference_by_name[term]["pct_sig_corrected"],
            }
            for term in terms
        ]
    else:
        global_payload = json.loads((MODEL_RESULTS / "global_spatial.json").read_text(encoding="utf-8"))
        mgwr_payload = json.loads((MODEL_RESULTS / "mgwr.json").read_text(encoding="utf-8"))
        progression = [
            {"Model": "OLS", "Spatial parameter or bandwidth": "—", "R² measure": f"{global_payload['ols']['adjusted_r2']:.3f} adjusted", "AIC/AICc": f"{global_payload['ols']['aic']:.2f} AIC", "Main result": "Spatially autocorrelated residuals"},
            {"Model": "Spatial lag, ML", "Spatial parameter or bandwidth": f"rho = {global_payload['spatial_lag']['rho']:.3f}", "R² measure": f"{global_payload['spatial_lag']['pseudo_r2']:.3f} pseudo-R²", "AIC/AICc": f"{global_payload['spatial_lag']['aic']:.2f} AIC", "Main result": "Limited improvement over OLS"},
            {"Model": "Spatial error, ML", "Spatial parameter or bandwidth": f"lambda = {global_payload['spatial_error']['lambda']:.3f}", "R² measure": f"{global_payload['spatial_error']['pseudo_r2']:.3f} pseudo-R²", "AIC/AICc": f"{global_payload['spatial_error']['aic']:.2f} AIC", "Main result": "Temporal-shape joint test remains non-significant"},
            {"Model": "GWR, original scale", "Spatial parameter or bandwidth": f"{global_payload['gwr_original']['bandwidth']:.0f} neighbours", "R² measure": f"{global_payload['gwr_original']['r2']:.3f}", "AIC/AICc": f"{global_payload['gwr_original']['aicc']:.2f} AICc", "Main result": "Single adaptive Gaussian bandwidth"},
            {"Model": "GWR (z scale)", "Spatial parameter or bandwidth": f"{mgwr_payload['gwr_standardized']['bandwidth']:.0f} neighbours", "R² measure": f"{mgwr_payload['gwr_standardized']['r2']:.3f}", "AIC/AICc": f"{mgwr_payload['gwr_standardized']['aicc']:.2f} AICc", "Main result": "Like-for-like comparison model"},
            {"Model": "MGWR (z scale)", "Spatial parameter or bandwidth": "Predictor-specific", "R² measure": f"{mgwr_payload['mgwr_standardized']['r2']:.3f}", "AIC/AICc": f"{mgwr_payload['mgwr_standardized']['aicc']:.2f} AICc", "Main result": "Preferred to GWR on the same standardised scale"},
        ]
        inference = [
            {
                "Term": row["term"],
                "Bandwidth": int(row["bandwidth"]),
                "AICc-weighted 95% interval": f"{int(row['bandwidth_interval'][0])}–{int(row['bandwidth_interval'][1])}",
                "ENP_j": row["enp_j"],
                "Median |t|": row["median_abs_t"],
                "Significant cells (%)": row["significant_cells_percent"],
            }
            for row in mgwr_payload["mgwr_standardized"]["inference"]
        ]
    return progression, inference


def table_4(source: str) -> None:
    progression, inference = load_spatial_values(source)
    labels = {
        "const": "Intercept",
        "inflection": "Inflection year",
        "rate": "Annual growth rate",
        "b_asymptote": "Mapped-area level",
        "pop_density": "Effective population density",
    }
    panel_a = pd.DataFrame(progression)
    panel_a.insert(0, "Panel", "A. Spatial-model progression")
    panel_b = pd.DataFrame(inference)
    panel_b["Term"] = panel_b["Term"].map(labels)
    panel_b.insert(0, "Panel", "B. Standardised MGWR bandwidths and local inference")
    combined = pd.concat([panel_a, panel_b], ignore_index=True, sort=False)
    write_frame(combined, "Table_4_spatial_mgwr", "Table 4. Spatial models and MGWR inference (n = 207)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["reference", "results"], default="reference")
    args = parser.parse_args()
    ensure_results()
    cells = gpd.read_parquet(ANALYSIS_READY / "cell_analysis_input.geoparquet")
    table_1()
    table_2(cells)
    table_3(cells)
    table_4(args.source)
    print("Generated Tables 1–4 in results/tables")


if __name__ == "__main__":
    main()
