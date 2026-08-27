"""Repository paths and small serialisation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ANALYSIS_READY = DATA / "analysis-ready"
REFERENCE_RESULTS = DATA / "reference-results"
REFERENCE_FIGURES = ROOT / "reference" / "figures"
RESULTS = ROOT / "results"
MODEL_RESULTS = RESULTS / "models"
TABLE_RESULTS = RESULTS / "tables"
FIGURE_RESULTS = RESULTS / "figures"
CONFIG = ROOT / "config.yml"


def ensure_results() -> None:
    """Create generated-output directories without touching archived files."""
    for path in (MODEL_RESULTS, TABLE_RESULTS, FIGURE_RESULTS):
        path.mkdir(parents=True, exist_ok=True)


def jsonable(value: Any) -> Any:
    """Convert common NumPy objects for JSON output."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=jsonable) + "\n",
        encoding="utf-8",
    )

