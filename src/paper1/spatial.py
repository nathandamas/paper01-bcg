"""
spatial_pipeline.py — Spatial Regression Pipeline
==================================================
Encapsulates the spatial analysis decision tree into reusable
functions.  Each transition in the progression is driven by diagnostics
from the prior step — nothing is skipped, nothing is assumed.

Pipeline:
  OLS (HC3) -> Moran's I -> LM tests -> Spatial Lag/Error -> GWR -> MGWR

Sources:
  - documented decision framework used in the associated manuscript
  - Anselin (1988), Fotheringham et al. (2017), Oshan et al. (2019)
  - production implementation archived with this reproduction package
  - Anselin 1988                             (LM decision rule)

Environment:
  See ``environment.yml`` and ``requirements-lock.txt`` at repository root.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

import esda
import libpysal
import spreg
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, Queen, Rook
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW

logger = logging.getLogger(__name__)


def _clone_weights(W: libpysal.weights.W, transform: str) -> libpysal.weights.W:
    """Create a new W with the same neighbors but a different transform."""
    W_new = libpysal.weights.W(W.neighbors, silence_warnings=True)
    W_new.transform = transform
    return W_new


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGNIFICANCE_LEVEL = 0.05
GWR_MAX_N = 5000
MGWR_MAX_N = 3000
MAX_FEATURES = 15

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OLSResult:
    """OLS regression results with HC3 robust standard errors."""
    model: Any
    n: int
    r2: float
    adj_r2: float
    aic: float
    bic: float
    rmse: float
    coefficients: pd.DataFrame
    vif: pd.DataFrame
    residuals: np.ndarray
    fitted: np.ndarray
    beta_star: pd.Series

    def summary_dict(self) -> dict:
        return {
            "model": "OLS",
            "n": self.n,
            "R2": self.r2,
            "adj_R2": self.adj_r2,
            "AIC": self.aic,
            "AICc": None,
            "key_param": None,
        }


@dataclass
class SpatialDependenceResult:
    """Moran's I on residuals + Lagrange Multiplier tests."""
    morans_i: float
    morans_p: float
    morans_z: float
    morans_EI: float
    lm_lag_stat: float
    lm_lag_p: float
    lm_error_stat: float
    lm_error_p: float
    rlm_lag_stat: float
    rlm_lag_p: float
    rlm_error_stat: float
    rlm_error_p: float
    has_spatial_dependence: bool
    decision: str  # "ols_sufficient" | "spatial_lag" | "spatial_error" | "spatial_durbin"
    decision_log: List[str] = field(default_factory=list)


@dataclass
class LISAResult:
    """Local Moran's I (LISA) cluster analysis."""
    local_Is: np.ndarray
    p_values: np.ndarray
    quadrants: np.ndarray
    significant_mask: np.ndarray
    cluster_labels: np.ndarray
    counts: Dict[str, int]


@dataclass
class SpatialLagResult:
    """Spatial Lag model (ML estimation)."""
    model: Any
    pseudo_r2: float
    rho: float
    aic: float

    def summary_dict(self) -> dict:
        return {
            "model": "Spatial-Lag",
            "R2": self.pseudo_r2,
            "adj_R2": None,
            "AIC": self.aic,
            "AICc": None,
            "key_param": f"rho={self.rho:.4f}",
        }


@dataclass
class SpatialErrorResult:
    """Spatial Error model (ML estimation)."""
    model: Any
    pseudo_r2: float
    lam: float
    aic: float

    def summary_dict(self) -> dict:
        return {
            "model": "Spatial-Error",
            "R2": self.pseudo_r2,
            "adj_R2": None,
            "AIC": self.aic,
            "AICc": None,
            "key_param": f"lambda={self.lam:.4f}",
        }


@dataclass
class GWRResult:
    """Geographically Weighted Regression results."""
    model: Any
    bw: float
    r2: float
    aicc: float
    local_r2: np.ndarray
    params: np.ndarray
    tvalues: np.ndarray
    feature_names: List[str]

    def summary_dict(self) -> dict:
        return {
            "model": "GWR",
            "R2": self.r2,
            "adj_R2": None,
            "AIC": None,
            "AICc": self.aicc,
            "key_param": f"bw={self.bw:.0f}",
        }

    def local_significance_summary(self) -> pd.DataFrame:
        """Percentage of cells where |t| > 1.96 per covariate."""
        rows = []
        for j, name in enumerate(self.feature_names):
            t_abs = np.abs(self.tvalues[:, j])
            rows.append({
                "covariate": name,
                "pct_significant": (t_abs > 1.96).mean() * 100,
                "median_abs_t": float(np.median(t_abs)),
            })
        return pd.DataFrame(rows)


@dataclass
class MGWRResult:
    """Multiscale GWR results with per-covariate bandwidths."""
    model: Any
    bandwidths: np.ndarray
    r2: float
    aicc: float
    params: np.ndarray
    feature_names: List[str]
    enp: Optional[float] = None

    def summary_dict(self) -> dict:
        return {
            "model": "MGWR",
            "R2": self.r2,
            "adj_R2": None,
            "AIC": None,
            "AICc": self.aicc,
            "key_param": f"ENP={self.enp:.1f}" if self.enp else None,
        }

    def bandwidth_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "covariate": self.feature_names,
            "bandwidth": [int(b) for b in self.bandwidths],
        })

    def bandwidth_ratio(self, var_a: str, var_b: str) -> float:
        """Ratio of bandwidth between two covariates (larger / smaller)."""
        idx_a = self.feature_names.index(var_a)
        idx_b = self.feature_names.index(var_b)
        bw_a, bw_b = self.bandwidths[idx_a], self.bandwidths[idx_b]
        return max(bw_a, bw_b) / max(min(bw_a, bw_b), 1)


@dataclass
class PipelineResult:
    """Full pipeline output: all fitted models + decision log."""
    ols: OLSResult
    spatial_dependence: SpatialDependenceResult
    lisa: Optional[LISAResult] = None
    spatial_lag: Optional[SpatialLagResult] = None
    spatial_error: Optional[SpatialErrorResult] = None
    gwr: Optional[GWRResult] = None
    mgwr: Optional[MGWRResult] = None
    decision_log: List[str] = field(default_factory=list)

    def comparison_table(self) -> pd.DataFrame:
        """Model comparison table (AIC/AICc, R², key parameter)."""
        rows = [self.ols.summary_dict()]
        if self.spatial_lag:
            rows.append(self.spatial_lag.summary_dict())
        if self.spatial_error:
            rows.append(self.spatial_error.summary_dict())
        if self.gwr:
            rows.append(self.gwr.summary_dict())
        if self.mgwr:
            rows.append(self.mgwr.summary_dict())
        return pd.DataFrame(rows)

    def best_model_by_aicc(self) -> str:
        """Return name of model with lowest AICc (GWR/MGWR) or AIC (others)."""
        candidates = {}
        candidates["OLS"] = self.ols.aic
        if self.spatial_lag:
            candidates["Spatial-Lag"] = self.spatial_lag.aic
        if self.spatial_error:
            candidates["Spatial-Error"] = self.spatial_error.aic
        if self.gwr:
            candidates["GWR"] = self.gwr.aicc
        if self.mgwr:
            candidates["MGWR"] = self.mgwr.aicc
        return min(candidates, key=candidates.get)


# ---------------------------------------------------------------------------
# 1. OLS baseline
# ---------------------------------------------------------------------------

def run_ols(
    y: np.ndarray,
    X_df: pd.DataFrame,
    cov_type: str = "HC3",
) -> OLSResult:
    """
    OLS regression with robust standard errors.

    Parameters
    ----------
    y : array-like, shape (n,)
        Dependent variable.
    X_df : DataFrame, shape (n, k)
        Predictor matrix (without constant — added internally).
    cov_type : str
        Covariance estimator ("HC3" by default per White 1980).

    Returns
    -------
    OLSResult with model, diagnostics, coefficients, VIF, and
    standardised beta coefficients (beta*).
    """
    feature_names = list(X_df.columns)
    X = sm.add_constant(X_df)
    model = sm.OLS(y, X).fit(cov_type=cov_type)

    # VIF is computed with an intercept in the auxiliary regressions.  Report
    # predictor rows only; this is the definition used in the manuscript.
    vif_data = pd.DataFrame({
        "variable": feature_names,
        "VIF": [
            variance_inflation_factor(X.values, i)
            for i in range(1, X.shape[1])
        ],
    })

    # Standardised beta: beta* = beta_j * (std_x_j / std_y)
    std_y = np.std(y, ddof=1)
    beta_star_vals = {}
    for name in feature_names:
        coef = model.params[name]
        std_x = np.std(X_df[name].values, ddof=1)
        beta_star_vals[name] = coef * std_x / std_y if std_y > 0 else 0.0
    beta_star = pd.Series(beta_star_vals)

    # Coefficient table
    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p": model.pvalues,
    })
    coef_df.loc[feature_names, "beta_star"] = beta_star
    coef_df.loc["const", "beta_star"] = np.nan

    rmse = np.sqrt(np.mean(model.resid ** 2))

    logger.info(
        "OLS: n=%d  adj_R2=%.4f  AIC=%.2f  max_VIF=%.2f",
        model.nobs, model.rsquared_adj, model.aic, vif_data["VIF"].max(),
    )

    return OLSResult(
        model=model,
        n=int(model.nobs),
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        aic=float(model.aic),
        bic=float(model.bic),
        rmse=float(rmse),
        coefficients=coef_df,
        vif=vif_data,
        residuals=model.resid.values if hasattr(model.resid, "values") else model.resid,
        fitted=model.fittedvalues.values if hasattr(model.fittedvalues, "values") else model.fittedvalues,
        beta_star=beta_star,
    )


def wald_test_joint(
    ols_result: OLSResult,
    variables: List[str],
) -> Tuple[float, float]:
    """
    Wald test: H0 = all listed variables have coefficient == 0.

    Returns (F_statistic, p_value).
    """
    model = ols_result.model
    param_names = list(model.params.index)
    r_matrix = np.zeros((len(variables), len(param_names)))
    for i, var in enumerate(variables):
        r_matrix[i, param_names.index(var)] = 1
    result = model.wald_test(r_matrix, scalar=True)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# 2. Spatial weights
# ---------------------------------------------------------------------------

def build_weights(
    gdf: gpd.GeoDataFrame,
    method: str = "queen",
    transform: str = "R",
    **kwargs,
) -> libpysal.weights.W:
    """
    Build spatial weights matrix.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must be in a projected CRS for distance-based weights.
    method : str
        "queen", "rook", or "knn".
    transform : str
        "R" (row-standardized) or "B" (binary).
    **kwargs
        Extra args forwarded to the weights constructor (e.g. k=5 for KNN).
    """
    builders = {
        "queen": lambda: Queen.from_dataframe(gdf, use_index=False),
        "rook": lambda: Rook.from_dataframe(gdf, use_index=False),
        "knn": lambda: KNN.from_dataframe(gdf, k=kwargs.get("k", 5)),
    }
    if method not in builders:
        raise ValueError(f"Unknown method '{method}'. Use queen, rook, or knn.")

    W = builders[method]()
    W.transform = transform

    n_islands = len(W.islands)
    if n_islands > 0:
        logger.warning(
            "%d island(s) detected (0 neighbors). These will be excluded "
            "from Moran's I and LISA.", n_islands,
        )

    logger.info(
        "Weights (%s, %s): n=%d  mean_neighbors=%.1f  islands=%d",
        method, transform, W.n, W.mean_neighbors, n_islands,
    )
    return W


# ---------------------------------------------------------------------------
# 3. Spatial dependence diagnostics
# ---------------------------------------------------------------------------

def check_spatial_dependence(
    y: np.ndarray,
    X: np.ndarray,
    residuals: np.ndarray,
    W: libpysal.weights.W,
    feature_names: Optional[List[str]] = None,
    permutations: int = 999,
    alpha: float = SIGNIFICANCE_LEVEL,
) -> SpatialDependenceResult:
    """
    Global Moran's I on residuals + LM/RLM tests (Anselin decision rule).

    The Anselin (1988) decision rule:
      1. Both LM-Lag and LM-Error significant?
         - Compare Robust versions: whichever remains significant wins.
      2. Only one LM significant? Use that model.
      3. Neither significant? OLS is sufficient.
      4. Both Robust versions significant? Spatial Durbin.

    Parameters
    ----------
    y : (n,) dependent variable values.
    X : (n, k) predictor matrix (without constant).
    residuals : (n,) OLS residuals.
    W : spatial weights (will be row-standardized internally for LM tests).
    feature_names : optional variable names for spreg.
    permutations : number of permutations for Moran's I.
    alpha : significance threshold.
    """
    log = []

    # Moran's I (binary weights)
    W_binary = _clone_weights(W, "B")
    mi = Moran(residuals, W_binary, permutations=permutations)
    log.append(
        f"Moran's I = {mi.I:.4f}, p = {mi.p_sim:.4f}, z = {mi.z_sim:.4f}"
    )

    has_sa = mi.p_sim < alpha

    # LM tests via spreg (needs row-standardized weights)
    W_row = _clone_weights(W, "R")
    names = feature_names or [f"x{i}" for i in range(X.shape[1])]
    ols_spreg = spreg.OLS(
        y.reshape(-1, 1), X, w=W_row,
        spat_diag=True, name_y="y", name_x=names,
    )

    lm_lag = getattr(ols_spreg, "lm_lag", (np.nan, 1.0))
    lm_err = getattr(ols_spreg, "lm_error", (np.nan, 1.0))
    rlm_lag = getattr(ols_spreg, "rlm_lag", (np.nan, 1.0))
    rlm_err = getattr(ols_spreg, "rlm_error", (np.nan, 1.0))

    log.append(
        f"LM-Lag p={lm_lag[1]:.4f}  LM-Error p={lm_err[1]:.4f}  "
        f"RLM-Lag p={rlm_lag[1]:.4f}  RLM-Error p={rlm_err[1]:.4f}"
    )

    # Anselin decision rule
    if not has_sa:
        decision = "ols_sufficient"
        log.append("Moran's I not significant -> OLS sufficient")
    else:
        lag_sig = lm_lag[1] < alpha
        err_sig = lm_err[1] < alpha

        if lag_sig and err_sig:
            rlag_sig = rlm_lag[1] < alpha
            rerr_sig = rlm_err[1] < alpha
            if rlag_sig and rerr_sig:
                decision = "spatial_durbin"
                log.append("Both RLM significant -> Spatial Durbin")
            elif rlag_sig:
                decision = "spatial_lag"
                log.append("Only RLM-Lag significant -> Spatial Lag")
            elif rerr_sig:
                decision = "spatial_error"
                log.append("Only RLM-Error significant -> Spatial Error")
            else:
                decision = "spatial_error"
                log.append("Neither RLM significant; defaulting to Spatial Error (conservative)")
        elif lag_sig:
            decision = "spatial_lag"
            log.append("Only LM-Lag significant -> Spatial Lag")
        elif err_sig:
            decision = "spatial_error"
            log.append("Only LM-Error significant -> Spatial Error")
        else:
            decision = "ols_sufficient"
            log.append("Neither LM test significant -> OLS sufficient despite Moran's I")

    logger.info("Spatial dependence decision: %s", decision)

    return SpatialDependenceResult(
        morans_i=float(mi.I),
        morans_p=float(mi.p_sim),
        morans_z=float(mi.z_sim),
        morans_EI=float(mi.EI),
        lm_lag_stat=float(lm_lag[0]),
        lm_lag_p=float(lm_lag[1]),
        lm_error_stat=float(lm_err[0]),
        lm_error_p=float(lm_err[1]),
        rlm_lag_stat=float(rlm_lag[0]),
        rlm_lag_p=float(rlm_lag[1]),
        rlm_error_stat=float(rlm_err[0]),
        rlm_error_p=float(rlm_err[1]),
        has_spatial_dependence=has_sa,
        decision=decision,
        decision_log=log,
    )


# ---------------------------------------------------------------------------
# 4. LISA
# ---------------------------------------------------------------------------

def run_lisa(
    values: np.ndarray,
    W: libpysal.weights.W,
    permutations: int = 999,
    seed: int = 42,
    alpha: float = SIGNIFICANCE_LEVEL,
) -> LISAResult:
    """
    Local Moran's I (LISA) — identifies HH, LL, LH, HL clusters.

    Parameters
    ----------
    values : (n,) array — typically OLS residuals.
    W : spatial weights (binary recommended).
    """
    W_binary = _clone_weights(W, "B")
    lisa = Moran_Local(values, W_binary, permutations=permutations, seed=seed)

    sig_mask = lisa.p_sim < alpha
    quad_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    labels = np.where(
        sig_mask,
        np.vectorize(quad_map.get)(lisa.q),
        "ns",
    )

    counts = {}
    for label in ["HH", "LL", "LH", "HL", "ns"]:
        counts[label] = int((labels == label).sum())

    logger.info(
        "LISA: HH=%d  LL=%d  LH=%d  HL=%d  ns=%d",
        counts["HH"], counts["LL"], counts["LH"], counts["HL"], counts["ns"],
    )

    return LISAResult(
        local_Is=lisa.Is,
        p_values=lisa.p_sim,
        quadrants=lisa.q,
        significant_mask=sig_mask,
        cluster_labels=labels,
        counts=counts,
    )


# ---------------------------------------------------------------------------
# 5. Spatial Lag model
# ---------------------------------------------------------------------------

def run_spatial_lag(
    y: np.ndarray,
    X: np.ndarray,
    W: libpysal.weights.W,
    feature_names: Optional[List[str]] = None,
) -> SpatialLagResult:
    """Spatial Lag model via Maximum Likelihood (spreg.ML_Lag)."""
    W_row = _clone_weights(W, "R")
    names = feature_names or [f"x{i}" for i in range(X.shape[1])]

    model = spreg.ML_Lag(
        y.reshape(-1, 1), X, w=W_row,
        name_y="y", name_x=names,
    )
    ss_tot = np.sum((y - y.mean()) ** 2)
    pseudo_r2 = 1.0 - model.utu / ss_tot

    logger.info(
        "Spatial-Lag: pseudo_R2=%.4f  rho=%.4f  AIC=%.2f",
        pseudo_r2, model.rho, model.aic,
    )

    return SpatialLagResult(
        model=model,
        pseudo_r2=float(pseudo_r2),
        rho=float(model.rho),
        aic=float(model.aic),
    )


# ---------------------------------------------------------------------------
# 6. Spatial Error model
# ---------------------------------------------------------------------------

def run_spatial_error(
    y: np.ndarray,
    X: np.ndarray,
    W: libpysal.weights.W,
    feature_names: Optional[List[str]] = None,
) -> SpatialErrorResult:
    """Spatial Error model via Maximum Likelihood (spreg.ML_Error)."""
    W_row = _clone_weights(W, "R")
    names = feature_names or [f"x{i}" for i in range(X.shape[1])]

    model = spreg.ML_Error(
        y.reshape(-1, 1), X, w=W_row,
        name_y="y", name_x=names,
    )
    ss_tot = np.sum((y - y.mean()) ** 2)
    pseudo_r2 = 1.0 - model.utu / ss_tot

    logger.info(
        "Spatial-Error: pseudo_R2=%.4f  lambda=%.4f  AIC=%.2f",
        pseudo_r2, model.lam, model.aic,
    )

    return SpatialErrorResult(
        model=model,
        pseudo_r2=float(pseudo_r2),
        lam=float(model.lam),
        aic=float(model.aic),
    )


# ---------------------------------------------------------------------------
# 7. GWR
# ---------------------------------------------------------------------------

def run_gwr(
    y: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    feature_names: Optional[List[str]] = None,
    kernel: str = "gaussian",
    criterion: str = "AICc",
) -> GWRResult:
    """
    Geographically Weighted Regression with adaptive bandwidth.

    Parameters
    ----------
    y : (n,) dependent variable.
    X : (n, k) predictors (without constant — added internally by mgwr).
    coords : (n, 2) projected coordinates (MUST be in metres, not degrees).
    kernel : "gaussian" or "bisquare".
    criterion : bandwidth selection criterion ("AICc" or "AIC").

    Raises
    ------
    ValueError if n > GWR_MAX_N.
    """
    n = len(y)
    if n > GWR_MAX_N:
        raise ValueError(
            f"n={n} exceeds GWR_MAX_N={GWR_MAX_N}. "
            f"Subsample spatially before calling run_gwr."
        )

    names = ["const"] + (feature_names or [f"x{i}" for i in range(X.shape[1])])

    sel = Sel_BW(
        coords, y.reshape(-1, 1), X,
        kernel=kernel, fixed=False, spherical=False,
    )
    bw = sel.search(criterion=criterion)

    gwr_model = GWR(
        coords, y.reshape(-1, 1), X, bw=bw,
        kernel=kernel, fixed=False, spherical=False,
    )
    results = gwr_model.fit()

    logger.info(
        "GWR: bw=%.0f  R2=%.4f  AICc=%.2f", bw, results.R2, results.aicc,
    )

    return GWRResult(
        model=results,
        bw=float(bw),
        r2=float(results.R2),
        aicc=float(results.aicc),
        local_r2=results.localR2.flatten(),
        params=results.params,
        tvalues=results.tvalues,
        feature_names=names,
    )


# ---------------------------------------------------------------------------
# 8. MGWR
# ---------------------------------------------------------------------------

def run_mgwr(
    y: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    feature_names: Optional[List[str]] = None,
    kernel: str = "gaussian",
    criterion: str = "AICc",
) -> MGWRResult:
    """
    Multiscale GWR — per-covariate adaptive bandwidths.

    Parameters
    ----------
    y : (n,) dependent variable.
    X : (n, k) predictors (without constant).
    coords : (n, 2) projected coordinates in metres.

    Note: mgwr 2.2.1 raises NotImplementedError for tvalues and localR2
    on MGWR models.  Only params and global R2 are available.

    Raises
    ------
    ValueError if n > MGWR_MAX_N.
    """
    n = len(y)
    if n > MGWR_MAX_N:
        raise ValueError(
            f"n={n} exceeds MGWR_MAX_N={MGWR_MAX_N}. "
            f"Subsample spatially before calling run_mgwr."
        )

    names = ["const"] + (feature_names or [f"x{i}" for i in range(X.shape[1])])
    n_vars = X.shape[1] + 1  # +1 for intercept

    sel = Sel_BW(
        coords, y.reshape(-1, 1), X, multi=True,
        kernel=kernel, fixed=False, spherical=False,
    )
    bws = sel.search(criterion=criterion, multi_bw_min=[2] * n_vars)

    mgwr_model = MGWR(
        coords, y.reshape(-1, 1), X, selector=sel,
        kernel=kernel, fixed=False, spherical=False,
    )
    results = mgwr_model.fit()

    # ENP back-calculation from AICc used in the archived analysis:
    #   AICc = n*log(RSS/n) + n*log(2*pi) + n*(n+k)/(n-k-2)
    #   where k = ENP = trace(S)
    #   Solve: lhs = AICc - n*log(RSS/n) - n*log(2*pi)
    #          lhs = n*(n+k)/(n-k-2)
    #          k = (lhs*(n-2) - n^2) / (n + lhs)
    try:
        rss = results.RSS if hasattr(results, "RSS") else np.sum(results.resid_response ** 2)
        lhs = results.aicc - n * np.log(rss / n) - n * np.log(2 * np.pi)
        enp = (lhs * (n - 2) - n ** 2) / (n + lhs) if (n + lhs) != 0 else None
        if enp is not None and enp < 0:
            enp = None
    except Exception:
        enp = None

    bw_strs = ", ".join(f"{nm}={b:.0f}" for nm, b in zip(names, bws))
    logger.info("MGWR: R2=%.4f  AICc=%.2f  bws=[%s]", results.R2, results.aicc, bw_strs)

    return MGWRResult(
        model=results,
        bandwidths=np.array(bws, dtype=float),
        r2=float(results.R2),
        aicc=float(results.aicc),
        params=results.params,
        feature_names=names,
        enp=float(enp) if enp is not None else None,
    )


# ---------------------------------------------------------------------------
# 9. Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    results: Dict[str, Any],
    residuals_dict: Optional[Dict[str, np.ndarray]] = None,
    W: Optional[libpysal.weights.W] = None,
) -> pd.DataFrame:
    """
    Tabular comparison of all fitted models.

    Parameters
    ----------
    results : dict mapping model name -> result dataclass.
    residuals_dict : optional dict mapping model name -> residuals array
                     (for computing residual Moran's I).
    W : spatial weights for residual Moran's I.
    """
    rows = []
    for name, res in results.items():
        row = res.summary_dict() if hasattr(res, "summary_dict") else {"model": name}
        if residuals_dict and name in residuals_dict and W is not None:
            W_b = _clone_weights(W, "B")
            mi = Moran(residuals_dict[name], W_b, permutations=99)
            row["resid_Moran_I"] = float(mi.I)
            row["resid_Moran_p"] = float(mi.p_sim)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. Full pipeline orchestrator
# ---------------------------------------------------------------------------

def run_full_pipeline(
    y: np.ndarray,
    X_df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    weights_method: str = "queen",
    run_local_models: bool = True,
    kernel: str = "gaussian",
    alpha: float = SIGNIFICANCE_LEVEL,
    permutations: int = 999,
    lisa_seed: int = 42,
) -> PipelineResult:
    """
    Full spatial regression pipeline.

    Decision tree:
      OLS (HC3)
       -> Moran's I on residuals
          -> p >= alpha: STOP, OLS sufficient
          -> p < alpha: LM tests (Anselin rule)
             -> Spatial Lag and/or Spatial Error
       -> if run_local_models and n <= thresholds:
          -> GWR (bandwidth by AICc)
          -> MGWR (per-covariate bandwidth by AICc)

    Parameters
    ----------
    y : (n,) dependent variable.
    X_df : DataFrame (n, k) predictors.
    gdf : GeoDataFrame with geometry column (projected CRS recommended).
    weights_method : "queen", "rook", or "knn".
    run_local_models : if True and sample size allows, fit GWR and MGWR.
    kernel : GWR/MGWR kernel ("gaussian" or "bisquare").
    alpha : significance level for all tests.
    permutations : for Moran's I.
    lisa_seed : random seed for LISA permutations.

    Returns
    -------
    PipelineResult with all fitted models and a decision log.
    """
    decision_log = []
    feature_names = list(X_df.columns)
    X_arr = X_df.values
    n = len(y)

    # --- CRS check ---
    if gdf.crs and gdf.crs.is_geographic:
        warnings.warn(
            f"GeoDataFrame CRS is geographic ({gdf.crs.to_epsg()}). "
            "Distance-based operations (GWR, MGWR) require projected CRS. "
            "Consider reprojecting to a local UTM zone.",
            UserWarning,
        )
    decision_log.append(f"CRS: {gdf.crs}")

    # --- Stage 1: OLS ---
    decision_log.append("--- Stage 1: OLS baseline (HC3) ---")
    ols = run_ols(y, X_df)
    decision_log.append(f"OLS adj_R2={ols.adj_r2:.4f}  AIC={ols.aic:.2f}")

    if ols.vif["VIF"].max() > 10:
        decision_log.append(
            f"WARNING: max VIF={ols.vif['VIF'].max():.1f} > 10 — multicollinearity"
        )

    # --- Stage 2: Spatial weights ---
    decision_log.append(f"--- Stage 2: Spatial weights ({weights_method}) ---")
    W = build_weights(gdf, method=weights_method, transform="R")

    # --- Stage 3: Spatial dependence ---
    decision_log.append("--- Stage 3: Moran's I + LM tests ---")
    sp_dep = check_spatial_dependence(
        y, X_arr, ols.residuals, W,
        feature_names=feature_names,
        permutations=permutations, alpha=alpha,
    )
    decision_log.extend(sp_dep.decision_log)

    # --- Stage 4: LISA ---
    decision_log.append("--- Stage 4: LISA clusters ---")
    lisa = run_lisa(ols.residuals, W, permutations=permutations, seed=lisa_seed, alpha=alpha)
    decision_log.append(
        f"LISA: HH={lisa.counts['HH']} LL={lisa.counts['LL']} "
        f"LH={lisa.counts['LH']} HL={lisa.counts['HL']}"
    )

    # --- Stage 5: Spatial Lag / Error ---
    sp_lag = None
    sp_err = None

    if sp_dep.decision in ("spatial_lag", "spatial_durbin"):
        decision_log.append("--- Stage 5a: Spatial Lag ---")
        try:
            sp_lag = run_spatial_lag(y, X_arr, W, feature_names=feature_names)
            decision_log.append(
                f"Spatial-Lag: pseudo_R2={sp_lag.pseudo_r2:.4f}  "
                f"rho={sp_lag.rho:.4f}  AIC={sp_lag.aic:.2f}"
            )
        except Exception as e:
            decision_log.append(f"Spatial-Lag FAILED: {e}")

    if sp_dep.decision in ("spatial_error", "spatial_durbin"):
        decision_log.append("--- Stage 5b: Spatial Error ---")
        try:
            sp_err = run_spatial_error(y, X_arr, W, feature_names=feature_names)
            decision_log.append(
                f"Spatial-Error: pseudo_R2={sp_err.pseudo_r2:.4f}  "
                f"lambda={sp_err.lam:.4f}  AIC={sp_err.aic:.2f}"
            )
        except Exception as e:
            decision_log.append(f"Spatial-Error FAILED: {e}")

    # Always fit both if either is triggered (L9: no model skipped)
    if sp_lag and not sp_err:
        decision_log.append("Fitting Spatial-Error too (L9: no model skipped)")
        try:
            sp_err = run_spatial_error(y, X_arr, W, feature_names=feature_names)
        except Exception as e:
            decision_log.append(f"Spatial-Error (supplementary) FAILED: {e}")
    elif sp_err and not sp_lag:
        decision_log.append("Fitting Spatial-Lag too (L9: no model skipped)")
        try:
            sp_lag = run_spatial_lag(y, X_arr, W, feature_names=feature_names)
        except Exception as e:
            decision_log.append(f"Spatial-Lag (supplementary) FAILED: {e}")

    # --- Stage 6: GWR ---
    gwr_result = None
    mgwr_result = None

    if run_local_models:
        coords = np.column_stack([
            gdf.geometry.centroid.x,
            gdf.geometry.centroid.y,
        ])

        if n <= GWR_MAX_N:
            decision_log.append("--- Stage 6: GWR ---")
            try:
                gwr_result = run_gwr(
                    y, X_arr, coords,
                    feature_names=feature_names, kernel=kernel,
                )
                decision_log.append(
                    f"GWR: bw={gwr_result.bw:.0f}  R2={gwr_result.r2:.4f}  "
                    f"AICc={gwr_result.aicc:.2f}"
                )
            except Exception as e:
                decision_log.append(f"GWR FAILED: {e}")
        else:
            decision_log.append(
                f"GWR SKIPPED: n={n} > GWR_MAX_N={GWR_MAX_N}"
            )

        # --- Stage 7: MGWR ---
        if n <= MGWR_MAX_N:
            decision_log.append("--- Stage 7: MGWR ---")
            try:
                mgwr_result = run_mgwr(
                    y, X_arr, coords,
                    feature_names=feature_names, kernel=kernel,
                )
                bw_tbl = mgwr_result.bandwidth_table()
                bw_str = ", ".join(
                    f"{r['covariate']}={r['bandwidth']}"
                    for _, r in bw_tbl.iterrows()
                )
                decision_log.append(
                    f"MGWR: R2={mgwr_result.r2:.4f}  AICc={mgwr_result.aicc:.2f}  "
                    f"bws=[{bw_str}]"
                )
                if mgwr_result.enp:
                    decision_log.append(
                        f"MGWR ENP={mgwr_result.enp:.1f} "
                        f"({mgwr_result.enp/n*100:.1f}% of n)"
                    )
            except Exception as e:
                decision_log.append(f"MGWR FAILED: {e}")
        else:
            decision_log.append(
                f"MGWR SKIPPED: n={n} > MGWR_MAX_N={MGWR_MAX_N}"
            )
    else:
        decision_log.append("Local models SKIPPED: run_local_models=False")

    # --- Summary ---
    result = PipelineResult(
        ols=ols,
        spatial_dependence=sp_dep,
        lisa=lisa,
        spatial_lag=sp_lag,
        spatial_error=sp_err,
        gwr=gwr_result,
        mgwr=mgwr_result,
        decision_log=decision_log,
    )

    best = result.best_model_by_aicc()
    decision_log.append(f"--- Best model by AIC/AICc: {best} ---")
    logger.info("Pipeline complete. Best model: %s", best)

    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def benjamini_hochberg(
    p_values: Dict[str, float],
    q: float = 0.10,
) -> pd.DataFrame:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : dict mapping test name -> raw p-value.
    q : FDR threshold (default 0.10).

    Returns
    -------
    DataFrame with columns: test, raw_p, rank, bh_critical, bh_p, reject.
    """
    df = pd.DataFrame([
        {"test": k, "raw_p": v} for k, v in p_values.items()
    ]).sort_values("raw_p").reset_index(drop=True)
    m = len(df)
    df["rank"] = range(1, m + 1)
    df["bh_critical"] = df["rank"] / m * q

    # Adjusted p-values (step-up)
    adj_p = np.empty(m)
    adj_p[m - 1] = df["raw_p"].iloc[m - 1]
    for i in range(m - 2, -1, -1):
        adj_p[i] = min(adj_p[i + 1], df["raw_p"].iloc[i] * m / (i + 1))
    df["bh_p"] = np.minimum(adj_p, 1.0)
    df["reject"] = df["bh_p"] < q

    return df.sort_values("rank").reset_index(drop=True)


def print_decision_log(result: PipelineResult) -> None:
    """Pretty-print the pipeline decision log."""
    for line in result.decision_log:
        if line.startswith("---"):
            print(f"\n{line}")
        elif "FAILED" in line or "WARNING" in line:
            print(f"  !! {line}")
        else:
            print(f"  {line}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Import paper1.spatial from the numbered scripts in scripts/.")
