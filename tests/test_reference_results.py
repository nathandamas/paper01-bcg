import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_archived_global_values_match_config():
    config = yaml.safe_load((ROOT / "config.yml").read_text(encoding="utf-8"))["expected"]
    payload = json.loads(
        (ROOT / "data" / "reference-results" / "diagnostics_global_spatial.json").read_text(encoding="utf-8")
    )["results"]["ref_completeness_area"]
    assert payload["ols"]["n"] == config["n"]
    assert payload["ols"]["r2"] == pytest.approx(config["ols"]["r2"], abs=1e-12)
    assert payload["ols"]["adj_r2"] == pytest.approx(config["ols"]["adjusted_r2"], abs=1e-12)
    assert payload["spatial"]["moran_I"] == pytest.approx(config["queen"]["moran_i"], abs=1e-12)
    assert payload["spatial"]["lisa_counts"] == config["queen"]["lisa_counts"]


def test_archived_mgwr_values_match_config():
    config = yaml.safe_load((ROOT / "config.yml").read_text(encoding="utf-8"))["expected"]
    payload = json.loads(
        (ROOT / "data" / "reference-results" / "diagnostics_mgwr.json").read_text(encoding="utf-8")
    )
    assert payload["gwr_std"]["bw"] == config["gwr_standardized"]["bandwidth"]
    assert payload["mgwr_std"]["r2"] == pytest.approx(config["mgwr_standardized"]["r2"], abs=1e-12)
    assert payload["mgwr_std"]["aicc"] == pytest.approx(config["mgwr_standardized"]["aicc"], abs=1e-12)
    assert payload["mgwr_std"]["bandwidths"] == config["mgwr_standardized"]["bandwidths"]

