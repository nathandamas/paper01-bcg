#!/usr/bin/env python3
"""Reproduce OLS coefficients, nested models and incremental-R2 inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import brentq

from paper1 import spatial
from paper1.paths import ANALYSIS_READY, MODEL_RESULTS, ensure_results, write_json
from paper1.sample import build_sample


SEED = 42


def fit_metrics(y: np.ndarray, X: pd.DataFrame) -> tuple[object, dict[str, float]]:
    model = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    return model, {
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adjusted_r2": float(model.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(model.resid**2))),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }


def fast_r2(y: np.ndarray, X: np.ndarray) -> float:
    design = np.column_stack((np.ones(len(y)), X))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return float(1.0 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2))


def pairs_bootstrap(y: np.ndarray, X: pd.DataFrame, replicates: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    full = X[["inflection", "rate", "b_asymptote", "pop_density"]].to_numpy()
    base = X[["b_asymptote", "pop_density"]].to_numpy()
    increments = np.empty(replicates)
    for index in range(replicates):
        draw = rng.integers(0, len(y), len(y))
        increments[index] = fast_r2(y[draw], full[draw]) - fast_r2(y[draw], base[draw])
    lower, upper = np.percentile(increments, [2.5, 97.5])
    return {
        "replicates": replicates,
        "mean_incremental_r2": float(increments.mean()),
        "percentile_2_5": float(lower),
        "percentile_97_5": float(upper),
    }


def minimum_detectable_increment(n: int, full_r2: float) -> dict[str, float]:
    df1, df2 = 2, n - 5
    critical = stats.f.isf(0.05, df1, df2)

    def power(f_squared: float) -> float:
        return float(stats.ncf.sf(critical, df1, df2, f_squared * n))

    f_squared = float(brentq(lambda value: power(value) - 0.80, 1e-8, 1.0))
    return {
        "power": 0.80,
        "alpha": 0.05,
        "f_squared": f_squared,
        "incremental_r2": f_squared * (1.0 - full_r2),
    }


def main() -> None:
    ensure_results()
    cells = gpd.read_parquet(ANALYSIS_READY / "cell_analysis_input.geoparquet")
    y, X, sample = build_sample(cells)
    specifications = {
        "mapped_area_level_only": ["b_asymptote"],
        "level_and_population": ["b_asymptote", "pop_density"],
        "full_model": ["inflection", "rate", "b_asymptote", "pop_density"],
        "without_mapped_area_level": ["inflection", "rate", "pop_density"],
    }
    nested: dict[str, dict[str, object]] = {}
    models = {}
    for name, columns in specifications.items():
        model, metrics = fit_metrics(y, X[columns])
        models[name] = model
        nested[name] = {"predictors": columns, **metrics}

    robust = spatial.run_ols(y, X)
    statistic, p_value = spatial.wald_test_joint(robust, ["inflection", "rate"])
    correlation = stats.pearsonr(sample["b_area"], sample["y_current_area"])
    incremental_r2 = (
        nested["full_model"]["r2"] - nested["level_and_population"]["r2"]
    )
    diagnostics = {
        "seed": SEED,
        "n": int(len(y)),
        "coefficients_hc3": robust.coefficients.reset_index(names="term").to_dict("records"),
        "vif": robust.vif.to_dict("records"),
        "shape_wald": {"statistic": statistic, "df": 2, "p": p_value},
        "nested_models": nested,
        "incremental_unadjusted_r2": incremental_r2,
        "incremental_adjusted_r2": (
            nested["full_model"]["adjusted_r2"]
            - nested["level_and_population"]["adjusted_r2"]
        ),
        "mapped_level_vs_current_area": {
            "r": float(correlation.statistic),
            "p": float(correlation.pvalue),
        },
        "pairs_bootstrap": pairs_bootstrap(y, X),
        "minimum_detectable_increment": minimum_detectable_increment(
            len(y), nested["full_model"]["r2"]
        ),
    }
    write_json(MODEL_RESULTS / "nested_models.json", diagnostics)
    robust.coefficients.reset_index(names="term").to_csv(
        MODEL_RESULTS / "ols_coefficients_hc3.csv", index=False
    )
    print("Nested-model reproduction complete")
    for name, values in nested.items():
        print(f"{name:28s} adjusted R2={values['adjusted_r2']:.6f}")
    print(f"b vs current OSM area r={correlation.statistic:.7f}")


if __name__ == "__main__":
    main()

