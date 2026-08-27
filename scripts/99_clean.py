#!/usr/bin/env python3
"""Remove regenerated outputs while preserving tracked placeholders."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for directory in (ROOT / "results" / "models", ROOT / "results" / "tables", ROOT / "results" / "figures"):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.name != ".gitkeep" and path.is_file():
            path.unlink()
            print(f"Removed {path.relative_to(ROOT)}")

