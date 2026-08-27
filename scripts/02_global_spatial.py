#!/usr/bin/env python3
"""Reproduce global OLS, residual dependence, spatial models and GWR."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran
from scipy import stats

from paper1 import spatial
from paper1.paths import ANALYSIS_READY, MODEL_RESULTS, ensure_results, write_json
from paper1.sample import build_sample, flag_pass_counts


SEED = 42


def dependence_dict(result: spatial.SpatialDependenceResult) -> dict[str, object]:
    return {
        "moran_i": result.morans_i,
        "moran_p": result.morans_p,
        "moran_z": result.morans_z,
        "moran_expected": result.morans_EI,
        "lm_lag_stat": result.lm_lag_stat,
        "lm_lag_p": result.lm_lag_p,
        "lm_error_stat": result.lm_error_stat,
        "lm_error_p": result.lm_error_p,
        "robust_lm_lag_stat": result.rlm_lag_stat,
        "robust_lm_lag_p": result.rlm_lag_p,
        "robust_lm_error_stat": result.rlm_error_stat,
        "robust_lm_error_p": result.rlm_error_p,
        "decision": result.decision,
    }


def sem_joint_wald(model: object, columns: list[str]) -> dict[str, float]:
    names = ["CONSTANT", *columns]
    indices = [names.index("inflection"), names.index("rate")]
    beta = np.asarray(model.betas).reshape(-1)[indices]
    covariance = np.asarray(model.vm)[np.ix_(indices, indices)]
    statistic = float(beta @ np.linalg.inv(covariance) @ beta)
    return {"statistic": statistic, "df": 2, "p": float(stats.chi2.sf(statistic, 2))}


def main() -> None:
    ensure_results()
    np.random.seed(SEED)
    cells = gpd.read_parquet(ANALYSIS_READY / "cell_analysis_input.geoparquet")
    y, X, sample = build_sample(cells)
    if len(y) != 207:
        raise RuntimeError(f"Expected 207 cells, obtained {len(y)}")

    queen = spatial.run_full_pipeline(
        y=y,
        X_df=X,
        gdf=sample,
        weights_method="queen",
        run_local_models=False,
        permutations=999,
        lisa_seed=SEED,
    )
    np.random.seed(SEED)
    rook = spatial.run_full_pipeline(
        y=y,
        X_df=X,
        gdf=sample,
        weights_method="rook",
        run_local_models=False,
        permutations=999,
        lisa_seed=SEED,
    )

    coordinates = np.column_stack((sample.geometry.centroid.x, sample.geometry.centroid.y))
    gwr = spatial.run_gwr(
        y,
        X.to_numpy(),
        coordinates,
        feature_names=list(X.columns),
        kernel="gaussian",
    )
    wald_stat, wald_p = spatial.wald_test_joint(queen.ols, ["inflection", "rate"])

    output_sample = sample.copy()
    output_sample["resid_ols"] = queen.ols.residuals
    output_sample["fitted_ols"] = queen.ols.fitted
    output_sample["lisa_cluster"] = queen.lisa.cluster_labels
    output_sample["lisa_local_I"] = queen.lisa.local_Is
    output_sample["lisa_p"] = queen.lisa.p_values
    output_sample.to_parquet(MODEL_RESULTS / "regression_sample_refitted.geoparquet")

    pd.DataFrame(gwr.params, columns=gwr.feature_names).assign(
        id=sample["id"].to_numpy()
    ).to_csv(MODEL_RESULTS / "gwr_local_original.csv", index=False)

    sem_model = queen.spatial_error.model
    filtered = getattr(sem_model, "e_filtered", None)
    filtered_moran = None
    if filtered is not None:
        weights = spatial.build_weights(sample, method="queen", transform="B")
        keep = np.array([index not in weights.islands for index in range(len(sample))])
        non_islands = sample.loc[keep].reset_index(drop=True)
        non_island_weights = spatial.build_weights(
            non_islands, method="queen", transform="B"
        )
        np.random.seed(SEED)
        filtered_test = Moran(
            np.asarray(filtered).reshape(-1)[keep],
            non_island_weights,
            permutations=999,
        )
        filtered_moran = {
            "n": int(keep.sum()),
            "i": float(filtered_test.I),
            "p": float(filtered_test.p_sim),
        }

    diagnostics = {
        "seed": SEED,
        "n": int(len(y)),
        "flag_pass_counts": flag_pass_counts(cells),
        "ols": {
            "r2": queen.ols.r2,
            "adjusted_r2": queen.ols.adj_r2,
            "aic": queen.ols.aic,
            "bic": queen.ols.bic,
            "rmse": queen.ols.rmse,
            "coefficients": queen.ols.coefficients.reset_index(names="term").to_dict("records"),
            "vif": queen.ols.vif.to_dict("records"),
            "shape_wald": {"statistic": wald_stat, "df": 2, "p": wald_p},
        },
        "queen": dependence_dict(queen.spatial_dependence)
        | {"lisa_counts": queen.lisa.counts},
        "rook": dependence_dict(rook.spatial_dependence),
        "spatial_lag": {
            "rho": queen.spatial_lag.rho,
            "pseudo_r2": queen.spatial_lag.pseudo_r2,
            "aic": queen.spatial_lag.aic,
        },
        "spatial_error": {
            "lambda": queen.spatial_error.lam,
            "pseudo_r2": queen.spatial_error.pseudo_r2,
            "aic": queen.spatial_error.aic,
            "shape_wald": sem_joint_wald(sem_model, list(X.columns)),
            "filtered_residual_moran": filtered_moran,
        },
        "gwr_original": {"bandwidth": gwr.bw, "r2": gwr.r2, "aicc": gwr.aicc},
    }
    write_json(MODEL_RESULTS / "global_spatial.json", diagnostics)
    print("Global/spatial reproduction complete")
    print(f"OLS adjusted R2: {queen.ols.adj_r2:.6f}")
    print(f"Queen Moran I: {queen.spatial_dependence.morans_i:.6f}")
    print(f"SEM lambda/AIC: {queen.spatial_error.lam:.6f} / {queen.spatial_error.aic:.2f}")


if __name__ == "__main__":
    main()
