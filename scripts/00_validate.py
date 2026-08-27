#!/usr/bin/env python3
"""Validate archive integrity, structure, results and anonymity."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from PIL import Image

from paper1.paths import ANALYSIS_READY, REFERENCE_FIGURES, REFERENCE_RESULTS


REQUIRED_FILES = [
    ANALYSIS_READY / "osm_area_timeseries_curitiba.parquet",
    ANALYSIS_READY / "logistic_fits_area.parquet",
    ANALYSIS_READY / "cell_analysis_input.geoparquet",
    ANALYSIS_READY / "vida_reference_by_cell.parquet",
    ANALYSIS_READY / "curitiba_boundary.parquet",
    REFERENCE_RESULTS / "regression_sample.geoparquet",
    REFERENCE_RESULTS / "mgwr_local_standardized.csv",
    REFERENCE_RESULTS / "mgwr_inference_standardized.csv",
    REFERENCE_RESULTS / "diagnostics_global_spatial.json",
    REFERENCE_RESULTS / "diagnostics_nested_sem.json",
    REFERENCE_RESULTS / "diagnostics_mgwr.json",
    REFERENCE_RESULTS / "diagnostics_mgwr_inference.json",
    *[REFERENCE_FIGURES / f"Figure_{index}.png" for index in range(1, 8)],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hashes(errors: list[str]) -> None:
    manifest = ROOT / "data" / "checksums.sha256"
    if not manifest.exists():
        errors.append("missing data/checksums.sha256")
        return
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.lstrip("*")
        if not path.exists():
            errors.append(f"checksum target missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"checksum mismatch: {relative}")


def check_images(errors: list[str]) -> None:
    for index in range(1, 8):
        path = REFERENCE_FIGURES / f"Figure_{index}.png"
        if not path.exists():
            continue
        with Image.open(path) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (0, 0))
        if width > 2138 or height > 2551:
            errors.append(f"{path.name}: {width}x{height} exceeds 2138x2551")
        if not dpi or min(dpi) < 299:
            errors.append(f"{path.name}: DPI metadata is {dpi}, expected approximately 300")


def check_reference_numbers(errors: list[str]) -> None:
    config = yaml.safe_load((ROOT / "config.yml").read_text(encoding="utf-8"))
    expected = config["expected"]
    global_diag = json.loads(
        (REFERENCE_RESULTS / "diagnostics_global_spatial.json").read_text(encoding="utf-8")
    )["results"]["ref_completeness_area"]
    mgwr_diag = json.loads(
        (REFERENCE_RESULTS / "diagnostics_mgwr.json").read_text(encoding="utf-8")
    )
    checks = {
        "n": (global_diag["ols"]["n"], expected["n"]),
        "OLS R2": (global_diag["ols"]["r2"], expected["ols"]["r2"]),
        "OLS adjusted R2": (
            global_diag["ols"]["adj_r2"],
            expected["ols"]["adjusted_r2"],
        ),
        "Queen Moran I": (
            global_diag["spatial"]["moran_I"],
            expected["queen"]["moran_i"],
        ),
        "MGWR R2": (mgwr_diag["mgwr_std"]["r2"], expected["mgwr_standardized"]["r2"]),
        "MGWR AICc": (
            mgwr_diag["mgwr_std"]["aicc"],
            expected["mgwr_standardized"]["aicc"],
        ),
    }
    for label, (actual, target) in checks.items():
        if abs(float(actual) - float(target)) > 1e-9:
            errors.append(f"{label}: archived {actual!r} != configured {target!r}")


def check_strict_data(errors: list[str]) -> None:
    required_modules = [
        "numpy", "pandas", "scipy", "matplotlib", "pyarrow", "geopandas",
        "statsmodels", "libpysal", "esda", "spreg", "mgwr",
    ]
    for name in required_modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"required package unavailable: {name} ({exc})")
    if any(message.startswith("required package") for message in errors):
        return

    import geopandas as gpd
    import pandas as pd
    from paper1.sample import build_sample, flag_pass_counts

    time_series = pd.read_parquet(ANALYSIS_READY / "osm_area_timeseries_curitiba.parquet")
    cells = gpd.read_parquet(ANALYSIS_READY / "cell_analysis_input.geoparquet")
    sample = gpd.read_parquet(REFERENCE_RESULTS / "regression_sample.geoparquet")
    local = pd.read_csv(REFERENCE_RESULTS / "mgwr_local_standardized.csv")

    if len(cells) != 502 or cells["id"].nunique() != 502:
        errors.append(f"cell input has {len(cells)} rows/{cells['id'].nunique()} unique ids")
    if time_series["cell_id"].nunique() != 502:
        errors.append("time-series archive does not contain 502 cells")
    monthly_counts = time_series.groupby("cell_id")["timestamp"].nunique()
    if not (monthly_counts == 218).all():
        errors.append("not every time series contains 218 unique months")
    y, X, rebuilt = build_sample(cells)
    if len(y) != 207 or len(sample) != 207 or len(local) != 207:
        errors.append(
            f"expected 207 rows: rebuilt={len(y)}, archived={len(sample)}, MGWR={len(local)}"
        )
    expected_flags = {
        "flag_convergence": 319,
        "flag_rse_b_infl": 277,
        "flag_rse_rate": 244,
        "flag_r2": 300,
        "flag_saturation": 280,
        "flag_c_window": 292,
        "flag_not_degenerate": 300,
        "flag_all": 207,
    }
    actual_flags = flag_pass_counts(cells)
    if actual_flags != expected_flags:
        errors.append(f"quality-gate counts differ: {actual_flags}")
    if list(X.columns) != ["inflection", "rate", "b_asymptote", "pop_density"]:
        errors.append(f"unexpected design columns: {list(X.columns)}")
    if rebuilt.crs is None or rebuilt.crs.to_epsg() != 31982:
        errors.append(f"unexpected sample CRS: {rebuilt.crs}")


def check_anonymity(errors: list[str]) -> None:
    forbidden = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"nathan" + r"damas",
            r"nathan\s+" + r"damas",
            r"paper01-" + r"saturation-tautology",
            r"github\.com/na" + r"thandamas",
            r"[A-Z]:\\" + r"Users\\",
            r"/Us" + r"ers/[^/]+/",
        )
    ]
    text_extensions = {
        ".md", ".py", ".yml", ".yaml", ".toml", ".txt", ".cff",
        ".gitignore", ".gitattributes", "",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in text_extensions and path.name not in {"LICENSE", "Makefile"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in forbidden:
            if pattern.search(text):
                errors.append(f"anonymity pattern {pattern.pattern!r} found in {path.relative_to(ROOT)}")
        for address in re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
            if not address.endswith("@invalid.example"):
                errors.append(f"e-mail address found in {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="read all Parquet files and import the full stack")
    parser.add_argument("--anonymity", action="store_true", help="scan tracked text for identifying strings")
    args = parser.parse_args()

    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")
    check_hashes(errors)
    check_images(errors)
    check_reference_numbers(errors)
    if args.strict:
        check_strict_data(errors)
    if args.anonymity:
        check_anonymity(errors)

    if errors:
        print("VALIDATION FAILED")
        for message in errors:
            print(f"- {message}")
        return 1
    print("VALIDATION PASSED")
    print(f"- {len(REQUIRED_FILES)} required artefacts present and checksummed")
    print("- seven reference figures are within 2138 × 2551 px and carry 300 dpi metadata")
    if args.strict:
        print("- Parquet schemas, row counts, CRS and 207-cell quality gate verified")
    if args.anonymity:
        print("- no configured identifying pattern found in repository text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
