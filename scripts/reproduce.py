#!/usr/bin/env python3
"""One-command orchestration for quick or full reproduction."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from paper1.paths import FIGURE_RESULTS, REFERENCE_FIGURES, ensure_results


def run(script: str, *arguments: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [sys.executable, str(ROOT / "scripts" / script), *arguments]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def copy_explanatory_figures() -> None:
    for number in (1, 2, 3):
        shutil.copy2(
            REFERENCE_FIGURES / f"Figure_{number}.png",
            FIGURE_RESULTS / f"Figure_{number}.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    ensure_results()

    if not args.skip_validation:
        run("00_validate.py", "--strict")
    copy_explanatory_figures()

    if args.mode == "quick":
        run("03_nested_models.py")
        run("05_make_tables.py", "--source", "reference")
        run("06_make_figures.py", "--source", "reference")
    else:
        run("02_global_spatial.py")
        run("03_nested_models.py")
        run("04_mgwr.py")
        run("05_make_tables.py", "--source", "results")
        run("06_make_figures.py", "--source", "results")

    print("\nReproduction complete")
    print(f"Tables:  {ROOT / 'results' / 'tables'}")
    print(f"Figures: {ROOT / 'results' / 'figures'}")


if __name__ == "__main__":
    main()

