"""Fig. 3 - evolution of MOTA, IDF1 (bars, top) and identity switches (line, bottom) from the baseline to HIT-Tracker.

Data: per-sequence counts in results/per_sequence for botsort, gm_nsa_g055, hem, hem_sahi512 and hit, summed over the 30 held-out
sequences. Print size 3.362 x 2.156 in.

Usage:
    python -m src.figures.fig3_pipeline_evolution --out figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src import common
from src.evaluation.mot_eval import idf1_from_counts, mota_from_counts, totals
from src.figures import style as S

STAGES = [("Baseline", "botsort"), ("GM Cost\n+ NSA", "gm_nsa_g055"), ("HEM", "hem"), ("HEM +\nSAHI512", "hem_sahi512"), ("HIT-\nTracker", "hit")]
EXPECTED = {"botsort": ("76.0", "71.2", 2355), "gm_nsa_g055": ("76.9", "74.8", 1611), "hem": ("76.7", "71.2", 2374),
            "hem_sahi512": ("79.6", "80.3", 712), "hit": ("79.9", "82.7", 494)}


def load_pipeline(per_sequence: Path) -> pd.DataFrame:
    rows = []
    for label, exp_id in STAGES:
        t = totals(pd.read_csv(per_sequence / f"{exp_id}.csv"))
        rows.append({"id": exp_id, "label": label, "MOTA": 100 * mota_from_counts(t), "IDF1": 100 * idf1_from_counts(t), "IDs": int(t["ids"])})
    return pd.DataFrame(rows)


def build(pipeline: pd.DataFrame):
    fig = S.new_figure(3)
    top = fig.add_axes([0.15, 0.487, 0.80, 0.493])
    bot = fig.add_axes([0.15, 0.185, 0.80, 0.245], sharex=top)
    x = np.arange(len(pipeline))
    w = 0.38
    top.bar(x - w / 2, pipeline["MOTA"], w, color=S.NAVY, edgecolor=S.NAVY, linewidth=0.6, zorder=3)
    top.bar(x + w / 2, pipeline["IDF1"], w, color="white", edgecolor=S.BLUE, hatch="//////", linewidth=0.7, zorder=3)
    top.bar(x + w / 2, pipeline["IDF1"], w, color="none", edgecolor=S.BLUE, linewidth=0.7, zorder=4)
    top.axvspan(3.55, 4.45, color=S.SOFT_GREEN, zorder=0, linewidth=0)
    top.text(4.0, 86.7, "FINAL", ha="center", va="top", fontsize=S.TEXT_PT, color=S.GREEN, fontweight="bold")
    for i, (m, f) in enumerate(zip(pipeline["MOTA"], pipeline["IDF1"])):
        close = abs(m - f) < 1.5                      # adjacent bars of similar height: stagger the labels
        top.text(i - w / 2, m + (0.25 if not close or m >= f else 0.25), f"{m:.1f}", ha="center", va="bottom", fontsize=S.VALUE, color=S.NAVY, fontweight="bold")
        top.text(i + w / 2, f + (0.25 + (1.0 if close and f > m else 0.0)), f"{f:.1f}", ha="center", va="bottom", fontsize=S.VALUE, color=S.BLUE, fontweight="bold")
    top.set_ylim(70.0, 87.0)
    top.set_yticks([70, 75, 80, 85])
    top.set_ylabel("MOTA / IDF1 (%)", fontsize=S.LABEL)
    top.tick_params(axis="x", labelbottom=False, length=0)
    top.tick_params(axis="y", labelsize=S.TICK)
    top.legend(handles=[Patch(facecolor=S.NAVY, edgecolor=S.NAVY, label="MOTA"), Patch(facecolor="white", edgecolor=S.BLUE, hatch="//////", label="IDF1")],
               loc="upper left", ncol=2, frameon=False, fontsize=S.TEXT_PT, handlelength=1.3, handleheight=0.8, columnspacing=0.9, borderpad=0.1,
               bbox_to_anchor=(0.0, 1.02))

    bot.plot(x, pipeline["IDs"], linestyle=(0, (4, 1.2, 1, 1.2)), marker="^", color=S.ORANGE, linewidth=0.9, markersize=3.8, markerfacecolor="white",
             markeredgewidth=0.8, zorder=3)
    bot.axvspan(3.55, 4.45, color=S.SOFT_GREEN, zorder=0, linewidth=0)
    for i, v in enumerate(pipeline["IDs"]):
        dy = 190
        va = "bottom"
        bot.text(i, v + dy, f"{int(v):,}", ha="center", va=va, fontsize=S.VALUE, color=S.ORANGE, fontweight="bold")
    reduction = 100.0 * (1.0 - pipeline["IDs"].iloc[-1] / pipeline["IDs"].iloc[0])
    bot.text(3.97, 1450, f"−{reduction:.1f}% vs. baseline", fontsize=S.TEXT_PT, color=S.GREEN, fontweight="bold", ha="center", va="bottom",
             bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": S.GREEN, "linewidth": 0.6}, zorder=5)
    bot.set_ylim(0, 3200)
    bot.set_yticks([0, 1000, 2000, 3000])
    bot.set_ylabel("ID switches", fontsize=S.LABEL)
    bot.set_xticks(x)
    bot.set_xticklabels(pipeline["label"], fontsize=S.TICK, linespacing=0.95)
    bot.set_xlim(-0.6, 4.6)
    bot.tick_params(axis="y", labelsize=S.TICK)
    return fig, reduction


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-sequence", type=Path, default=common.RESULTS / "per_sequence")
    ap.add_argument("--out", type=Path, default=common.REPO_ROOT / "figures")
    ap.add_argument("--png", action="store_true", help="also write the 1600 px web PNG")
    a = ap.parse_args()
    pipeline = load_pipeline(a.per_sequence)
    print(pipeline.round(2).to_string(index=False))
    for r in pipeline.itertuples():
        exp = EXPECTED[r.id]
        if (f"{r.MOTA:.1f}", f"{r.IDF1:.1f}", r.IDs) != exp:
            print(f"WARNING: {r.id} shows {(f'{r.MOTA:.1f}', f'{r.IDF1:.1f}', r.IDs)} but the paper states {exp}", file=sys.stderr)
    fig, reduction = build(pipeline)
    S.save_figure(fig, a.out / "fig3.pdf", a.out / "fig3.png" if a.png else None)
    print(f"written {a.out / 'fig3.pdf'} | ID reduction {reduction:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
