from pathlib import Path

import geopandas as gpd

from paper1.sample import build_sample, flag_pass_counts


ROOT = Path(__file__).resolve().parents[1]


def test_quality_gate_reproduces_reported_sample():
    cells = gpd.read_parquet(ROOT / "data" / "analysis-ready" / "cell_analysis_input.geoparquet")
    y, X, sample = build_sample(cells)
    assert len(cells) == 502
    assert len(y) == len(X) == len(sample) == 207
    assert flag_pass_counts(cells) == {
        "flag_convergence": 319,
        "flag_rse_b_infl": 277,
        "flag_rse_rate": 244,
        "flag_r2": 300,
        "flag_saturation": 280,
        "flag_c_window": 292,
        "flag_not_degenerate": 300,
        "flag_all": 207,
    }


def test_design_matrix_uses_manuscript_scales():
    cells = gpd.read_parquet(ROOT / "data" / "analysis-ready" / "cell_analysis_input.geoparquet")
    _, X, sample = build_sample(cells)
    assert list(X.columns) == ["inflection", "rate", "b_asymptote", "pop_density"]
    assert ((X["inflection"] - (2008.0 + sample["c_area"] / 12.0)).abs() < 1e-12).all()
    assert ((X["rate"] - 12.0 / sample["d_area"]).abs() < 1e-12).all()

