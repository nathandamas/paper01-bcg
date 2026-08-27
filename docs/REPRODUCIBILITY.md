# Reproducibility protocol

## Route A — exact/quick reproduction

Use this route to regenerate manuscript-facing tables and figures from the archived analytical outputs.

```bash
python scripts/00_validate.py --strict
python scripts/reproduce.py --mode quick
```

This route:

- verifies the archive;
- re-derives the quality-gate counts from the 502-cell input;
- refits OLS and nested OLS models with HC3 standard errors;
- recreates Tables 1–4;
- recreates Figures 4–7 from archived cell-level/model outputs; and
- places exact Figures 1–3 in the output directory.

Figures 1–3 are explanatory/context graphics rather than model outputs. Their final 300 dpi assets are versioned so the publication layout can be checked exactly.

## Route B — full downstream model reproduction

```bash
python scripts/reproduce.py --mode full
```

In addition to the quick route, this refits:

- full and nested OLS models;
- Queen and Rook residual Moran tests;
- Local Moran clusters;
- LM and robust-LM diagnostics;
- maximum-likelihood spatial-lag and spatial-error models;
- original-scale and z-standardised GWR; and
- z-standardised Gaussian MGWR, its AICc-weighted bandwidth intervals and corrected local inference.

The full route writes new model artefacts to `results/models/`. The MGWR stage is the slowest and can take one to two hours.

## Route C — curve-fit audit

```bash
python scripts/01_refit_logistic.py
```

This independently fits the area trajectories from the archived 502-cell × 218-month table. It is an audit of the nonlinear stage, not the default starting point for the downstream models. Small numerical differences can arise from SciPy optimiser and BLAS implementations; for exact manuscript reproduction, use the frozen `cell_analysis_input.geoparquet`.

## Randomness and tolerances

- Random seed: 42.
- Moran and LISA permutations: 999.
- Non-permutation global-model checks use a tolerance of `1e-6` unless noted.
- GWR/MGWR results use the tolerance in `config.yml` because bandwidth-search ties and floating-point linear algebra can differ slightly across platforms.

## Computing environment

The archived model run used Python 3.11 and the package versions pinned in `requirements-lock.txt`. The supplied GitHub Actions workflow installs the same environment and exercises the quick route and tests. The workflow intentionally does not run MGWR on every commit because of its runtime.

## Expected output policy

Version-controlled files under `data/reference-results/` and `reference/figures/` are immutable comparison targets. New runs write only beneath `results/`. `scripts/99_clean.py` removes regenerated outputs without touching any archived input or reference file.

