#!/usr/bin/env python3
"""Reproduce like-for-like standardised GWR and Gaussian MGWR."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

from paper1 import spatial
from paper1.paths import ANALYSIS_READY, MODEL_RESULTS, ensure_results, write_json
from paper1.sample import build_sample


SEED = 42


def main() -> None:
    ensure_results()
    np.random.seed(SEED)
    cells = gpd.read_parquet(ANALYSIS_READY / "cell_analysis_input.geoparquet")
    y, X, sample = build_sample(cells)
    y_standardized = (y - y.mean()) / y.std()
    X_standardized = (X - X.mean()) / X.std()
    coordinates = np.column_stack((sample.geometry.centroid.x, sample.geometry.centroid.y))
    names = ["const", *X_standardized.columns]

    gwr = spatial.run_gwr(
        y_standardized,
        X_standardized.to_numpy(),
        coordinates,
        feature_names=list(X_standardized.columns),
        kernel="gaussian",
    )

    selector = Sel_BW(
        coordinates,
        y_standardized.reshape(-1, 1),
        X_standardized.to_numpy(),
        multi=True,
        kernel="gaussian",
        fixed=False,
        spherical=False,
    )
    bandwidths = selector.search(criterion="AICc", multi_bw_min=[2] * len(names))
    model = MGWR(
        coordinates,
        y_standardized.reshape(-1, 1),
        X_standardized.to_numpy(),
        selector=selector,
        kernel="gaussian",
        fixed=False,
        spherical=False,
    )
    result = model.fit()

    local = pd.DataFrame(result.params, columns=names)
    local["id"] = sample["id"].to_numpy()
    local.to_csv(MODEL_RESULTS / "mgwr_local_standardized.csv", index=False)

    t_values = np.asarray(result.tvalues)
    filtered = np.asarray(result.filter_tvals())
    inference_frame = pd.DataFrame({"id": sample["id"].to_numpy()})
    for index, name in enumerate(names):
        inference_frame[f"t_{name}"] = t_values[:, index]
        inference_frame[f"tfilt_{name}"] = filtered[:, index]
    inference_frame.to_csv(MODEL_RESULTS / "mgwr_inference_standardized.csv", index=False)

    intervals = result.get_bws_intervals(selector, level=0.95)
    enp = np.asarray(result.ENP_j, dtype=float)
    inference = []
    for index, name in enumerate(names):
        inference.append(
            {
                "term": name,
                "bandwidth": float(bandwidths[index]),
                "bandwidth_interval": [
                    float(intervals[index][0]),
                    float(intervals[index][1]),
                ],
                "enp_j": float(enp[index]),
                "median_abs_t": float(np.median(np.abs(t_values[:, index]))),
                "significant_cells_percent": float((filtered[:, index] != 0).mean() * 100),
            }
        )

    diagnostics = {
        "seed": SEED,
        "n": int(len(y)),
        "standardization": {
            "response_ddof": 0,
            "predictor_ddof": 1,
        },
        "gwr_standardized": {
            "bandwidth": gwr.bw,
            "r2": gwr.r2,
            "aicc": gwr.aicc,
        },
        "mgwr_standardized": {
            "r2": float(result.R2),
            "aicc": float(result.aicc),
            "bandwidths": {
                name: float(value) for name, value in zip(names, bandwidths)
            },
            "inference": inference,
        },
    }
    write_json(MODEL_RESULTS / "mgwr.json", diagnostics)
    print("Standardised GWR/MGWR reproduction complete")
    print(f"GWR: bw={gwr.bw:.0f}, R2={gwr.r2:.6f}, AICc={gwr.aicc:.2f}")
    print(
        "MGWR: "
        f"R2={result.R2:.6f}, AICc={result.aicc:.2f}, "
        f"bandwidths={[int(value) for value in bandwidths]}"
    )


if __name__ == "__main__":
    main()

