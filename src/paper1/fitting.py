"""Four-parameter logistic fitting for monthly OSM mapped-building area."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit


@dataclass(frozen=True)
class FittingSpecification:
    scale_m2: float = 1000.0
    raw_final_min_m2: float = 1000.0
    corrected_range_min_m2: float = 5000.0
    maxfev: int = 70000


SPEC = FittingSpecification()
METHODS: tuple[tuple[str, bool, float], ...] = (
    ("trf", True, 0.30),
    ("dogbox", True, 0.30),
    ("lm", False, 0.20),
)


def logistic(t: np.ndarray, a: float, b: float, m: float, tau: float) -> np.ndarray:
    """Numerically stable four-parameter logistic curve."""
    tau_safe = tau if abs(tau) > 1e-10 else 1e-10
    exponent = np.clip((m - np.asarray(t, dtype=float)) / tau_safe, -500, 500)
    return a + (b - a) / (1.0 + np.exp(exponent))


def initial_guess(y: np.ndarray, t: np.ndarray) -> tuple[float, float, float, float]:
    a0 = float(np.nanmin(y))
    b0 = float(np.nanmax(y))
    differences = np.diff(y)
    if differences.size and np.nanmax(differences) > 0:
        index = int(np.nanargmax(differences))
        m0 = float((t[index] + t[index + 1]) / 2.0)
    else:
        m0 = float(np.nanmean(t))
    tau0 = max(5.0, float(np.nanmax(t) - np.nanmin(t)) / 10.0)
    return a0, b0, m0, tau0


def parameter_bounds(y: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    n = len(y)
    y_max = float(np.nanmax(y))
    return (
        (-0.5 * y_max, 0.5 * y_max, -0.5 * n, 1.0),
        (0.9 * y_max, 100.0 * y_max, 2.0 * n, 2.0 * n),
    )


def _metrics(y: np.ndarray, fitted: np.ndarray, parameters: int = 4) -> dict[str, float]:
    residual = y - fitted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    rmse = float(np.sqrt(np.mean(residual**2)))
    data_range = float(np.max(y) - np.min(y))
    n = len(y)
    safe_sse = max(sse, np.finfo(float).tiny)
    return {
        "rmse": rmse,
        "nrmse": rmse / data_range if data_range > 0 else np.inf,
        "r_squared": 1.0 - sse / sst if sst > 0 else -np.inf,
        "aic": n * np.log(safe_sse / n) + 2 * parameters,
        "bic": n * np.log(safe_sse / n) + parameters * np.log(n),
    }


def fit_series(
    values_m2: Iterable[float],
    cell_id: str,
    spec: FittingSpecification = SPEC,
) -> dict[str, object]:
    """Fit one monthly series using the archived multi-optimiser cascade."""
    raw = np.asarray(list(values_m2), dtype=float)
    base = {
        "id": str(cell_id),
        "success": False,
        "raw_final_area_m2": float(raw[-1]) if raw.size else np.nan,
    }
    if raw.size < 5 or not np.isfinite(raw).all() or np.any(raw < 0):
        return base | {"error": "invalid or insufficient series"}
    if raw[-1] <= spec.raw_final_min_m2:
        return base | {"error": "raw final area does not exceed 1,000 m2"}

    corrected = np.maximum.accumulate(raw)
    corrected_range = float(corrected.max() - corrected.min())
    if corrected_range < spec.corrected_range_min_m2:
        return base | {"error": "corrected range is below 5,000 m2"}

    adjustment = float(np.abs(corrected - raw).sum() / max(raw.sum(), 1.0) * 100.0)
    decreases = int((np.diff(raw) < 0).sum())
    y = corrected / spec.scale_m2
    t = np.arange(len(y), dtype=float)
    p0 = initial_guess(y, t)
    bounds = parameter_bounds(y)
    candidates: list[dict[str, object]] = []

    for method, bounded, minimum_r2 in METHODS:
        kwargs: dict[str, object] = {"p0": p0, "method": method, "maxfev": spec.maxfev}
        if bounded:
            kwargs["bounds"] = bounds
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                estimates, covariance = curve_fit(logistic, t, y, **kwargs)
            a, b, m, tau = map(float, estimates)
            if b <= a or tau <= 0:
                continue
            fitted = logistic(t, *estimates)
            metrics = _metrics(y, fitted)
            if metrics["r_squared"] < minimum_r2:
                continue
            with np.errstate(invalid="ignore"):
                errors = np.sqrt(np.diag(covariance)).astype(float)
            candidates.append(
                {
                    "method": method,
                    "a": a,
                    "b": b,
                    "m": m,
                    "tau": tau,
                    "a_se": float(errors[0]),
                    "b_se": float(errors[1]),
                    "m_se": float(errors[2]),
                    "tau_se": float(errors[3]),
                    **metrics,
                }
            )
        except (RuntimeError, ValueError, FloatingPointError, OptimizeWarning):
            continue

    if not candidates:
        return base | {
            "error": "all optimisation methods failed",
            "y_current_area": float(corrected[-1]),
            "n_decreases_raw": decreases,
            "monotonicity_correction_pct": adjustment,
        }

    best = min(candidates, key=lambda item: (item["nrmse"], -item["r_squared"], item["aic"]))
    scale = spec.scale_m2
    return base | {
        "success": True,
        "error": "",
        "a_area": best["a"] * scale,
        "b_area": best["b"] * scale,
        "c_area": best["m"],
        "d_area": best["tau"],
        "a_area_std_error": best["a_se"] * scale,
        "b_area_std_error": best["b_se"] * scale,
        "c_area_std_error": best["m_se"],
        "d_area_std_error": best["tau_se"],
        "b_area_ci_lower": (best["b"] - 1.96 * best["b_se"]) * scale,
        "b_area_ci_upper": (best["b"] + 1.96 * best["b_se"]) * scale,
        "r_squared_area": best["r_squared"],
        "rmse_area_scaled": best["rmse"],
        "nrmse_area": best["nrmse"],
        "aic_area": best["aic"],
        "bic_area": best["bic"],
        "method": best["method"],
        "y_current_area": float(corrected[-1]),
        "corrected_range_area_m2": corrected_range,
        "n_decreases_raw": decreases,
        "monotonicity_correction_pct": adjustment,
    }


def fit_all_cells(long_table: pd.DataFrame, spec: FittingSpecification = SPEC) -> pd.DataFrame:
    """Fit every cell in a long table with columns cell_id, timestamp and osm_area_m2."""
    required = {"cell_id", "timestamp", "osm_area_m2"}
    missing = required - set(long_table.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    ordered = long_table.sort_values(["cell_id", "timestamp"])
    rows = [
        fit_series(group["osm_area_m2"].to_numpy(), str(cell_id), spec)
        for cell_id, group in ordered.groupby("cell_id", sort=True)
    ]
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

