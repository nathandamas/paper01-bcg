"""Construction of the manuscript's 207-cell regression sample."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleSpecification:
    start_year: float = 2008.0
    rse_b_max: float = 0.50
    rse_m_max: float = 0.50
    rse_tau_max: float = 1.00
    r2_min: float = 0.90
    saturation_fraction: float = 0.85
    inflection_month_min: float = 6.0
    inflection_month_max: float = 210.0
    degenerate_r2_exclusive: float = 0.99999
    reference_area_min_km2: float = 0.005


SPEC = SampleSpecification()


def add_derived_views(df: pd.DataFrame, spec: SampleSpecification = SPEC) -> pd.DataFrame:
    """Add manuscript-scale inflection year and annual growth rate."""
    out = df.copy().reset_index(drop=True)
    out = out.drop(
        columns=[c for c in ("inflection_year", "rate_per_year") if c in out],
        errors="ignore",
    )
    out["inflection_year"] = spec.start_year + out["c_area"] / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rate_per_year"] = np.where(
            out["d_area"].abs() > 1e-9,
            12.0 / out["d_area"],
            np.nan,
        )
    return out


def _safe_rse(df: pd.DataFrame, se_col: str, estimate_col: str) -> np.ndarray:
    estimate = df[estimate_col].abs().to_numpy(dtype=float)
    standard_error = df[se_col].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rse = np.where(estimate > 1e-9, standard_error / estimate, np.inf)
    return np.where(np.isfinite(rse), rse, np.inf)


def compute_quality_flags(
    df: pd.DataFrame,
    spec: SampleSpecification = SPEC,
) -> pd.DataFrame:
    """Evaluate the seven curve-quality criteria reported in Table 2."""
    flags = pd.DataFrame(index=df.index)
    flags["flag_convergence"] = df["b_area"].notna()
    flags["flag_rse_b_infl"] = (
        (_safe_rse(df, "b_area_std_error", "b_area") <= spec.rse_b_max)
        & (_safe_rse(df, "c_area_std_error", "c_area") <= spec.rse_m_max)
    )
    flags["flag_rse_rate"] = (
        _safe_rse(df, "d_area_std_error", "d_area") <= spec.rse_tau_max
    )
    flags["flag_r2"] = df["r_squared_area"] >= spec.r2_min
    with np.errstate(invalid="ignore"):
        flags["flag_saturation"] = (
            df["y_current_area"] >= spec.saturation_fraction * df["b_area"]
        )
    flags["flag_c_window"] = df["c_area"].between(
        spec.inflection_month_min,
        spec.inflection_month_max,
        inclusive="both",
    )
    flags["flag_not_degenerate"] = (
        df["r_squared_area"] < spec.degenerate_r2_exclusive
    )
    criteria = [
        "flag_convergence",
        "flag_rse_b_infl",
        "flag_rse_rate",
        "flag_r2",
        "flag_saturation",
        "flag_c_window",
        "flag_not_degenerate",
    ]
    flags["flag_all"] = flags[criteria].all(axis=1)
    return flags


def prepare_cells(df: pd.DataFrame, spec: SampleSpecification = SPEC) -> pd.DataFrame:
    """Return all 502 cells with derived variables and quality flags."""
    out = add_derived_views(df, spec)
    flags = compute_quality_flags(out, spec).reset_index(drop=True)
    return pd.concat([out.reset_index(drop=True), flags], axis=1)


def build_sample(
    df: pd.DataFrame,
    spec: SampleSpecification = SPEC,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return response, design matrix and Geo/DataFrame for the final sample."""
    cells = prepare_cells(df, spec)
    reference_ok = cells["ref_area"] >= spec.reference_area_min_km2
    response_ok = cells["ref_completeness_area"].notna() & np.isfinite(
        cells["ref_completeness_area"]
    )
    sample = cells[cells["flag_all"] & reference_ok & response_ok].copy()
    sample = sample.reset_index(drop=True)

    y = sample["ref_completeness_area"].to_numpy(dtype=float)
    X = pd.DataFrame(
        {
            "inflection": sample["inflection_year"].to_numpy(dtype=float),
            "rate": sample["rate_per_year"].to_numpy(dtype=float),
            "b_asymptote": sample["b_area"].to_numpy(dtype=float),
            "pop_density": sample["cell_effective_density"].to_numpy(dtype=float),
        }
    )
    valid = np.isfinite(y) & np.isfinite(X.to_numpy()).all(axis=1)
    return (
        y[valid],
        X.loc[valid].reset_index(drop=True),
        sample.loc[valid].reset_index(drop=True),
    )


def flag_pass_counts(df: pd.DataFrame, spec: SampleSpecification = SPEC) -> dict[str, int]:
    flags = compute_quality_flags(add_derived_views(df, spec), spec)
    return {column: int(flags[column].sum()) for column in flags.columns}

