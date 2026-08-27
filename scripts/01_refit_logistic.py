#!/usr/bin/env python3
"""Refit all area trajectories from the archived monthly OSM table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from paper1.fitting import fit_all_cells
from paper1.paths import ANALYSIS_READY, MODEL_RESULTS, ensure_results, write_json


def main() -> None:
    ensure_results()
    source = ANALYSIS_READY / "osm_area_timeseries_curitiba.parquet"
    print(f"Reading {source.relative_to(ROOT)}")
    time_series = pd.read_parquet(source)
    fitted = fit_all_cells(time_series)
    parquet_path = MODEL_RESULTS / "logistic_fits_area_refitted.parquet"
    csv_path = MODEL_RESULTS / "logistic_fits_area_refitted.csv"
    fitted.to_parquet(parquet_path, index=False)
    fitted.to_csv(csv_path, index=False)
    summary = {
        "cells": int(len(fitted)),
        "successful": int(fitted["success"].sum()),
        "failed": int((~fitted["success"]).sum()),
        "methods": fitted.loc[fitted["success"], "method"].value_counts().to_dict(),
        "output": str(parquet_path.relative_to(ROOT)),
    }
    write_json(MODEL_RESULTS / "logistic_refit_summary.json", summary)
    print(f"Successful fits: {summary['successful']}/{summary['cells']}")
    print(f"Wrote {parquet_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

