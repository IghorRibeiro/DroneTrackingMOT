"""Fig. 2 - MOTA, IDF1 (left axis) and identity switches (right axis) versus the GM-cost weight gamma.

Data: per-sequence counts of ``gm_nsa_g015 ... gm_nsa_g085`` (results/per_sequence), summed over the 30 held-out sequences and
NOT rounded before the criteria below are applied. Print size 3.362 x 1.700 in.

Parametrised criteria
  --stable-delta-pp  "stable region" = every gamma whose MOTA is within this many percentage points of the best MOTA (default 0.3)
  --selected-gamma   the gamma marked as selected: a value (default 0.55) or ``auto`` = best MOTA, then IDF1, then IDs
  --expect-region    abort (exit 3) before drawing if the computed region differs from LO HI (paper: 0.25 0.65)

With unrounded numbers the best MOTA is at gamma = 0.45 (0.02 pp above gamma = 0.55); gamma = 0.55 has the best IDF1 and the fewest
identity switches of the sweep, which is why it is the operating point. A warning is printed when ``--selected-gamma`` differs
from the ``auto`` choice.

Usage:
    python -m src.figures.fig2_gamma_sweep --out figures --expect-region 0.25 0.65
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src import common
from src.evaluation.mot_eval import idf1_from_counts, mota_from_counts, totals
from src.figures import style as S

GAMMAS = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
STABLE_DELTA_PP = 0.3
SELECTED_GAMMA = 0.55


def load_sweep(per_sequence: Path) -> pd.DataFrame:
    rows = []
    for g in GAMMAS:
        t = totals(pd.read_csv(per_sequence / f"gm_nsa_g{int(round(g * 100)):03d}.csv"))
        rows.append({"gamma": g, "MOTA": 100 * mota_from_counts(t), "IDF1": 100 * idf1_from_counts(t), "IDs": int(t["ids"])})
    return pd.DataFrame(rows)


def stable_region(sweep: pd.DataFrame, delta_pp: float) -> tuple[float, float, np.ndarray]:
    mota = sweep["MOTA"].to_numpy()
    mask = mota >= mota.max() - delta_pp
    g = sweep["gamma"].to_numpy()[mask]
    return float(g.min()), float(g.max()), mask


def pick_selected(sweep: pd.DataFrame, selected) -> int:
    if str(selected).lower() == "auto":
        return int(np.lexsort((sweep["IDs"].to_numpy(), -sweep["IDF1"].to_numpy(), -sweep["MOTA"].to_numpy()))[0])
    return int(np.argmin(np.abs(sweep["gamma"].to_numpy() - float(selected))))


def build(sweep: pd.DataFrame, delta_pp: float, selected):
    fig = S.new_figure(2)
    ax = fig.add_axes([0.125, 0.205, 0.755, 0.665])
    ids_ax = ax.twinx()
    gamma, mota, idf1, ids = (sweep["gamma"].to_numpy(), sweep["MOTA"].to_numpy(), sweep["IDF1"].to_numpy(), sweep["IDs"].to_numpy())
    oi = pick_selected(sweep, selected)
    opt = float(gamma[oi])
    lo, hi, _ = stable_region(sweep, delta_pp)

    y0, y1 = 69.0, 80.0
    ax.set_ylim(y0, y1)
    ids_ax.set_ylim(1300, 2600)
    ax.set_xlim(0.11, 0.89)
    ax.axvspan(lo, hi, color=S.SOFT_BLUE, zorder=0, linewidth=0)
    ax.text((lo + hi) / 2, y0 + 0.35, "stable region", ha="center", va="bottom", fontsize=S.TEXT_PT, color="#1B5E7B", fontweight="bold")
    ax.plot(gamma, mota, linestyle="-", marker="o", color=S.NAVY, linewidth=0.9, markersize=3.4, markerfacecolor="white", markeredgewidth=0.8, zorder=4)
    ax.plot(gamma, idf1, linestyle="--", marker="s", color=S.BLUE, linewidth=0.9, markersize=3.2, markerfacecolor="white", markeredgewidth=0.8, zorder=4)
    ids_ax.plot(gamma, ids, linestyle=(0, (4, 1.2, 1, 1.2)), marker="^", color=S.ORANGE, linewidth=0.9, markersize=3.6, markerfacecolor="white",
                markeredgewidth=0.8, zorder=3)
    ax.axvline(opt, color=S.RED, linestyle=(0, (3, 2)), linewidth=0.9, zorder=2)
    ax.scatter([opt], [mota[oi]], s=14, facecolor=S.RED, edgecolor="white", linewidth=0.6, zorder=6)
    ax.text(opt + 0.015, y1 - 1.55, rf"$\gamma$ = {opt:.2f} selected", color=S.RED, fontsize=S.TEXT_PT, fontweight="bold", ha="left", va="top",
            bbox={"facecolor": "white", "edgecolor": S.RED, "linewidth": 0.6, "boxstyle": "round,pad=0.18"}, zorder=7)
    ax.set_xticks(gamma)
    ax.set_xticklabels([f"{g:.2f}" for g in gamma], fontsize=S.TICK)
    ax.set_yticks([70, 72, 74, 76, 78])
    ax.set_xlabel(r"GM-cost weight, $\gamma$", fontsize=S.LABEL)
    ax.set_ylabel("MOTA / IDF1 (%)", fontsize=S.LABEL)
    ids_ax.set_ylabel("ID switches", fontsize=S.LABEL)
    ids_ax.set_yticks([1400, 1800, 2200, 2600])
    ids_ax.spines["right"].set_visible(True)
    ids_ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=S.TICK)
    ids_ax.tick_params(labelsize=S.TICK)
    handles = [Line2D([0], [0], color=S.NAVY, linestyle="-", marker="o", markerfacecolor="white", markersize=3.4, markeredgewidth=0.8, label="MOTA"),
               Line2D([0], [0], color=S.BLUE, linestyle="--", marker="s", markerfacecolor="white", markersize=3.2, markeredgewidth=0.8, label="IDF1"),
               Line2D([0], [0], color=S.ORANGE, linestyle=(0, (4, 1.2, 1, 1.2)), marker="^", markerfacecolor="white", markersize=3.6,
                      markeredgewidth=0.8, label="ID switches")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=S.TEXT_PT, handlelength=2.2, columnspacing=1.2,
               handletextpad=0.5, bbox_to_anchor=(0.5, 1.0), borderpad=0.1)
    return fig, (lo, hi, opt, oi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-sequence", type=Path, default=common.RESULTS / "per_sequence")
    ap.add_argument("--out", type=Path, default=common.REPO_ROOT / "figures")
    ap.add_argument("--stable-delta-pp", type=float, default=STABLE_DELTA_PP)
    ap.add_argument("--selected-gamma", default=str(SELECTED_GAMMA))
    ap.add_argument("--expect-region", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--png", action="store_true", help="also write the 1600 px web PNG")
    a = ap.parse_args()

    sweep = load_sweep(a.per_sequence)
    lo, hi, mask = stable_region(sweep, a.stable_delta_pp)
    oi, auto = pick_selected(sweep, a.selected_gamma), pick_selected(sweep, "auto")
    best = sweep["MOTA"].max()
    print(sweep.assign(gap_to_best_pp=best - sweep["MOTA"], in_stable_region=mask).round(4).to_string(index=False))
    print(f"stable region (MOTA >= best - {a.stable_delta_pp} pp) = [{lo:.2f}, {hi:.2f}] | marked gamma = {sweep.gamma[oi]:.2f}")
    if auto != oi:
        print(f"WARNING: the MOTA > IDF1 > IDs rule on unrounded values would pick gamma = {sweep.gamma[auto]:.2f}; the figure marks {sweep.gamma[oi]:.2f}.")
    if a.expect_region and not (abs(lo - a.expect_region[0]) < 1e-9 and abs(hi - a.expect_region[1]) < 1e-9):
        print(f"ERROR: computed region [{lo:.2f}, {hi:.2f}] differs from the expected {a.expect_region}. Nothing was drawn.", file=sys.stderr)
        return 3
    fig, _ = build(sweep, a.stable_delta_pp, a.selected_gamma)
    a.out.mkdir(parents=True, exist_ok=True)
    data_dir = common.OUT_ROOT / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    sweep.assign(gap_to_best_pp=best - sweep["MOTA"], in_stable_region=mask).to_csv(data_dir / "fig2_gamma_sweep_data.csv", index=False)
    S.save_figure(fig, a.out / "fig2.pdf", a.out / "fig2.png" if a.png else None)
    print(f"written {a.out / 'fig2.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
