"""Fig. 1 - correlation between candidate factors and the baseline's identity preservation (IDF1).

Four panels (a)-(d): swarm density, camera motion, edge density and local contrast against the per-sequence IDF1 of the baseline
(30 held-out sequences, one point each), with the least-squares line, its 95 % confidence band and Pearson r.
Data: results/root_cause (factor tables) and results/per_sequence/botsort.csv (IDF1 shown with one decimal, as in the paper tables).
Print size 4.803 x 1.658 in.

Usage:
    python -m src.figures.fig1_root_cause --out figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import stats

from src import common
from src.analysis import root_cause
from src.figures import style as S

PANELS = [("mean_density", "Swarm density\n(drones/frame)", "(a)"), ("mean_motion", "Camera motion\n(px)", "(b)"),
          ("mean_edge_density", "Edge density", "(c)"), ("mean_local_contrast", "Local contrast", "(d)")]
EXPECTED_R = {"mean_density": 0.18, "mean_motion": 0.37, "mean_edge_density": -0.60, "mean_local_contrast": -0.56}


def confidence_band(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    """Function returning the (lower, upper) 95 % confidence band of the regression line."""
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    dof = len(x) - 2
    t_val = stats.t.ppf(1 - alpha / 2, dof)
    se = np.sqrt(np.sum(resid ** 2) / dof)
    mean_x, sxx, n = np.mean(x), np.sum((x - np.mean(x)) ** 2), len(x)

    def band(x_new):
        x_new = np.asarray(x_new)
        y_new = slope * x_new + intercept
        half = t_val * se * np.sqrt(1 / n + (x_new - mean_x) ** 2 / sxx)
        return y_new - half, y_new + half
    return band


def signed(r: float) -> str:
    """Pearson r with two decimals and a true minus sign."""
    return f"{r:.2f}".replace("-", "−")


def build(factors: Path, per_sequence: Path):
    F = root_cause.load_factors(factors)
    Y = root_cause.baseline_outcomes(pd.read_csv(per_sequence / "botsort.csv"))
    df = pd.concat([F, Y[["IDF1_r1"]].rename(columns={"IDF1_r1": "IDF1"})], axis=1).dropna()
    fig = S.new_figure(1)
    axes = fig.subplots(1, 4, sharey=True, gridspec_kw=dict(left=0.083, right=0.995, bottom=0.265, top=0.975, wspace=0.075))
    shown = {}
    for ax, (col, xlabel, tag) in zip(axes, PANELS):
        x, y = df[col].to_numpy(float), df["IDF1"].to_numpy(float)
        ax.scatter(x, y, s=7, color="black", marker="o", alpha=0.65, linewidths=0, zorder=3)
        x_line = np.linspace(x.min(), x.max(), 60)
        ax.plot(x_line, np.poly1d(np.polyfit(x, y, 1))(x_line), color="black", linestyle="--", linewidth=0.8, zorder=2)
        lower, upper = confidence_band(x, y)(x_line)
        ax.fill_between(x_line, lower, upper, color="black", alpha=0.14, linewidth=0, zorder=1)
        r = float(stats.pearsonr(x, y)[0])
        shown[col] = r
        ax.text(0.97, 0.965, f"r = {signed(r)}", transform=ax.transAxes, fontsize=S.VALUE, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))
        ax.text(0.03, 0.965, tag, transform=ax.transAxes, fontsize=S.LABEL, fontweight="bold", va="top")
        ax.set_xlabel(xlabel, fontsize=S.LABEL, linespacing=0.95)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="both", labelsize=S.TICK)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylabel("IDF1 (%)", fontsize=S.LABEL)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    return fig, shown


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factors", type=Path, default=common.RESULTS / "root_cause")
    ap.add_argument("--per-sequence", type=Path, default=common.RESULTS / "per_sequence")
    ap.add_argument("--out", type=Path, default=common.REPO_ROOT / "figures")
    ap.add_argument("--png", action="store_true", help="also write the 1600 px web PNG")
    a = ap.parse_args()
    fig, shown = build(a.factors, a.per_sequence)
    bad = {k: (signed(v), signed(EXPECTED_R[k])) for k, v in shown.items() if signed(v) != signed(EXPECTED_R[k])}
    if bad:
        print(f"WARNING: r values differ from the paper: {bad}", file=sys.stderr)
    S.save_figure(fig, a.out / "fig1.pdf", a.out / "fig1.png" if a.png else None)
    print(f"written {a.out / 'fig1.pdf'} | r = " + ", ".join(f"{signed(v)}" for v in shown.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
