#!/usr/bin/env python3
"""Regenerate data-driven manuscript Figures 4–7 at 300 dpi."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from paper1 import spatial
from paper1.paths import (
    ANALYSIS_READY,
    FIGURE_RESULTS,
    MODEL_RESULTS,
    REFERENCE_RESULTS,
    ensure_results,
)
from paper1.sample import build_sample


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save_300(fig: plt.Figure, number: int) -> None:
    path = FIGURE_RESULTS / f"Figure_{number}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path.relative_to(ROOT)}")


def map_frame(ax: plt.Axes, all_cells: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> None:
    all_cells.boundary.plot(ax=ax, color="#d9d9d9", linewidth=0.22, zorder=1)
    boundary.boundary.plot(ax=ax, color="#222222", linewidth=0.75, zorder=4)
    ax.set_axis_off()


def furniture(ax: plt.Axes, kilometres: int = 5) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.06 * (xmax - xmin)
    y0 = ymin + 0.035 * (ymax - ymin)
    ax.plot([x0, x0 + kilometres * 1000], [y0, y0], color="black", lw=1.3, zorder=8)
    ax.text(x0 + kilometres * 500, y0 + 0.012 * (ymax - ymin), f"{kilometres} km", ha="center", fontsize=6.5)
    xn = xmax - 0.07 * (xmax - xmin)
    yn = ymax - 0.12 * (ymax - ymin)
    ax.annotate("N", xy=(xn, yn + 0.07 * (ymax - ymin)), xytext=(xn, yn), ha="center", fontsize=7, arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#333333"})


def load_inputs(source: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    cells = gpd.read_parquet(ANALYSIS_READY / "cell_analysis_input.geoparquet")
    boundary = gpd.read_parquet(ANALYSIS_READY / "curitiba_boundary.parquet")
    if source == "results":
        sample_path = MODEL_RESULTS / "regression_sample_refitted.geoparquet"
        local_path = MODEL_RESULTS / "mgwr_local_standardized.csv"
    else:
        sample_path = REFERENCE_RESULTS / "regression_sample.geoparquet"
        local_path = REFERENCE_RESULTS / "mgwr_local_standardized.csv"
    sample = gpd.read_parquet(sample_path)
    local = pd.read_csv(local_path)
    return cells, sample, boundary, local


def figure_4(cells: gpd.GeoDataFrame, sample: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> None:
    if "rate_per_year" not in sample:
        sample["rate_per_year"] = 12.0 / sample["d_area"]
    if "inflection_year" not in sample:
        sample["inflection_year"] = 2008.0 + sample["c_area"] / 12.0
    sample["b_thousand_m2"] = sample["b_area"] / 1000.0
    panels = [
        ("rate_per_year", "(a) Annual growth rate", "viridis", "year$^{-1}$"),
        ("inflection_year", "(b) Inflection year", "viridis", "year"),
        ("b_thousand_m2", "(c) Mapped-area level", "YlOrRd", "$10^3$ m$^2$"),
        ("ref_completeness_area", "(d) Area completeness", "YlOrRd", "C"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.62), constrained_layout=True)
    for index, (ax, (column, title, cmap, label)) in enumerate(zip(axes, panels)):
        cells.plot(ax=ax, color="#f7f7f7", edgecolor="#dddddd", linewidth=0.18)
        sample.plot(ax=ax, column=column, cmap=cmap, edgecolor="white", linewidth=0.10)
        map_frame(ax, cells, boundary)
        ax.set_title(title, pad=3)
        values = sample[column].to_numpy(dtype=float)
        mapper = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(float(np.nanmin(values)), float(np.nanmax(values))),
            cmap=cmap,
        )
        colorbar = fig.colorbar(mapper, ax=ax, orientation="horizontal", fraction=0.045, pad=0.015)
        colorbar.ax.tick_params(labelsize=5.8, length=2)
        colorbar.set_label(label, fontsize=6.3)
        if index == 0:
            furniture(ax)
    save_300(fig, 4)


def figure_5(cells: gpd.GeoDataFrame) -> None:
    y, X, sample = build_sample(cells)
    robust = spatial.run_ols(y, X)
    order = ["b_asymptote", "pop_density", "inflection", "rate"]
    labels = ["Mapped-area level", "Population density", "Inflection year", "Annual growth rate"]
    standard_y = np.std(y, ddof=1)
    effects = []
    errors = []
    for name in order:
        ratio = np.std(X[name], ddof=1) / standard_y
        effects.append(float(robust.model.params[name] * ratio))
        errors.append(float(1.96 * robust.model.bse[name] * ratio))

    nested_specs = [
        ("b only", ["b_asymptote"]),
        ("b + population", ["b_asymptote", "pop_density"]),
        ("Full model", ["inflection", "rate", "b_asymptote", "pop_density"]),
        ("Without b", ["inflection", "rate", "pop_density"]),
    ]
    adjusted = []
    for _, columns in nested_specs:
        import statsmodels.api as sm
        model = sm.OLS(y, sm.add_constant(X[columns], has_constant="add")).fit()
        adjusted.append(float(model.rsquared_adj))

    fig = plt.figure(figsize=(7.0, 5.25))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.34, wspace=0.30)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    positions = np.arange(len(order))[::-1]
    ax_a.errorbar(effects, positions, xerr=errors, fmt="+", ms=8, color="#0b6789", ecolor="#76a9be", capsize=2, lw=1.2)
    ax_a.axvline(0, color="#999999", ls="--", lw=0.8)
    ax_a.set_yticks(positions, labels)
    ax_a.set_xlabel("Standardised coefficient, $\\beta^*$")
    ax_a.set_title("(a) Standardised OLS coefficients")
    ax_a.grid(axis="x", color="#eeeeee", lw=0.7)
    ax_a.spines[["top", "right", "left"]].set_visible(False)
    ax_a.tick_params(axis="y", length=0)

    positions_b = np.arange(len(nested_specs))[::-1]
    ax_b.barh(positions_b, adjusted, color="#146b8b", height=0.36)
    ax_b.set_yticks(positions_b, [name for name, _ in nested_specs])
    ax_b.set_xlim(0, 0.88)
    for position, value in zip(positions_b, adjusted):
        ax_b.text(value + 0.012, position, f"{value:.3f}", va="center", fontsize=7)
    ax_b.set_title("(b) Nested-model performance (adjusted $R^2$)")
    ax_b.spines[["top", "right", "bottom"]].set_visible(False)
    ax_b.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_b.tick_params(axis="y", length=0)

    current = sample["y_current_area"].to_numpy(dtype=float) / 1000.0
    mapped_level = sample["b_area"].to_numpy(dtype=float) / 1000.0
    ordering = np.argsort(current)
    maximum = max(float(np.max(current)), float(np.max(mapped_level)))
    axis_max = max(100.0, math.ceil(maximum / 50.0) * 50.0)
    ax_c.plot([0, axis_max], [0, axis_max], color="#8f8f8f", lw=1.0, ls="--", label="1:1 reference")
    ax_c.plot(current[ordering], mapped_level[ordering], color="#f26b2b", lw=1.1, marker="o", ms=2.0, label="cells sorted by current area")
    ax_c.set_xlim(0, axis_max)
    ax_c.set_ylim(0, axis_max)
    ax_c.set_xlabel("Current cumulative OSM area ($10^3$ m$^2$)")
    ax_c.set_ylabel("Mapped-area level, $\\hat b$ ($10^3$ m$^2$)")
    ax_c.set_title("(c) Fitted level and observed OSM area")
    correlation = np.corrcoef(current, mapped_level)[0, 1]
    ax_c.text(0.02, 0.94, f"r = {correlation:.4f}; n = {len(sample)}", transform=ax_c.transAxes, va="top", fontsize=7, style="italic")
    ax_c.grid(color="#dddddd", ls="--", lw=0.6)
    ax_c.spines[["top", "right"]].set_visible(False)
    save_300(fig, 5)


def figure_6(cells: gpd.GeoDataFrame, sample: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.28), constrained_layout=True)
    ax_a, ax_b = axes
    cells.plot(ax=ax_a, color="white", edgecolor="#dddddd", linewidth=0.25)
    limit = float(np.nanmax(np.abs(sample["resid_ols"])))
    sample.plot(ax=ax_a, column="resid_ols", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), edgecolor="white", linewidth=0.12)
    map_frame(ax_a, cells, boundary)
    ax_a.set_title("(a) OLS residuals", loc="left", fontsize=11)
    mapper = mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), cmap="RdBu_r")
    colorbar = fig.colorbar(mapper, ax=ax_a, orientation="horizontal", fraction=0.035, pad=0.01)
    colorbar.set_label("Observed minus fitted completeness", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    colors = {"HH": "#e31a1c", "LL": "#1f78b4", "LH": "#9bd8e8", "HL": "#fda343", "ns": "#efefef"}
    cells.plot(ax=ax_b, color="white", edgecolor="#dddddd", linewidth=0.25)
    for category in ["ns", "HH", "LL", "LH", "HL"]:
        subset = sample[sample["lisa_cluster"] == category]
        subset.plot(ax=ax_b, color=colors[category], edgecolor="white", linewidth=0.12)
    map_frame(ax_b, cells, boundary)
    ax_b.set_title("(b) LISA clusters", loc="left", fontsize=11)
    counts = sample["lisa_cluster"].value_counts().to_dict()
    handles = [
        mpatches.Patch(color=colors[name], label=f"{name} (n = {counts.get(name, 0)})")
        for name in ["HH", "LL", "LH", "HL", "ns"]
    ]
    ax_b.legend(handles=handles, loc="lower right", frameon=False, fontsize=7)
    furniture(ax_b)
    save_300(fig, 6)


def figure_7(cells: gpd.GeoDataFrame, sample: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, local: pd.DataFrame) -> None:
    mapped = sample.merge(local, on="id", how="inner", validate="one_to_one")
    panels = [
        ("rate", "(a) Annual growth rate coefficient", 198),
        ("inflection", "(b) Inflection-year coefficient", 179),
        ("b_asymptote", "(c) Mapped-area-level coefficient", 6),
        ("pop_density", "(d) Population-density coefficient", 7),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.8), constrained_layout=True)
    for index, (ax, (column, title, bandwidth)) in enumerate(zip(axes.ravel(), panels)):
        cells.plot(ax=ax, color="white", edgecolor="#e2e2e2", linewidth=0.20)
        values = mapped[column].to_numpy(dtype=float)
        limit = float(np.nanmax(np.abs(values)))
        norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
        mapped.plot(ax=ax, column=column, cmap="RdBu_r", norm=norm, edgecolor="white", linewidth=0.10)
        map_frame(ax, cells, boundary)
        ax.set_title(f"{title}, $\\beta(s)$", loc="left", fontsize=8.5)
        ax.text(0.02, 0.02, f"bw = {bandwidth}", transform=ax.transAxes, fontsize=7, weight="bold")
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
        colorbar = fig.colorbar(mapper, ax=ax, orientation="vertical", fraction=0.035, pad=0.012)
        colorbar.ax.tick_params(labelsize=5.5, length=2)
        if index == 1:
            furniture(ax)
    save_300(fig, 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["reference", "results"], default="reference")
    args = parser.parse_args()
    ensure_results()
    cells, sample, boundary, local = load_inputs(args.source)
    figure_4(cells, sample, boundary)
    figure_5(cells)
    figure_6(cells, sample, boundary)
    figure_7(cells, sample, boundary, local)


if __name__ == "__main__":
    main()

