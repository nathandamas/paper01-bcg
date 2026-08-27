import numpy as np

from paper1.fitting import fit_series, logistic


def test_logistic_midpoint_is_average_of_asymptotes():
    a, b, m, tau = 2.0, 10.0, 50.0, 8.0
    value = logistic(np.array([m]), a, b, m, tau)[0]
    assert value == (a + b) / 2.0


def test_synthetic_series_recovers_valid_curve():
    t = np.arange(218, dtype=float)
    values = logistic(t, 0.0, 36_000.0, 112.0, 9.0)
    result = fit_series(values, "synthetic")
    assert result["success"] is True
    assert abs(result["b_area"] - 36_000.0) < 100.0
    assert abs(result["c_area"] - 112.0) < 0.2
    assert abs(result["d_area"] - 9.0) < 0.2
    assert result["r_squared_area"] > 0.999


def test_cumulative_max_correction_is_reported():
    t = np.arange(218, dtype=float)
    values = logistic(t, 0.0, 30_000.0, 100.0, 12.0)
    values[170] -= 2_000.0
    result = fit_series(values, "deletion")
    assert result["success"] is True
    assert result["n_decreases_raw"] >= 1
    assert result["monotonicity_correction_pct"] > 0

