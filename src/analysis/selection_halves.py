"""How stable are the selected hyper-parameters? Half/half selection simulations (gamma and SAHI tile size).

For each of 1,000 random 15/15 splits of the 30 held-out sequences (seed 2026, both directions A->B and B->A):
choose the configuration with the selection criterion on one half and score it on the other half. Reported: how often
the paper's choice (gamma = 0.55, tile = 512) is selected, the selection-vs-evaluation gap and the regret with respect to
the best configuration on the evaluation half. Two criteria: 'paper' (MOTA, then IDF1, then IDs, on values rounded to one
decimal) and 'strict' (same order, unrounded).

Usage:
    python -m src.analysis.selection_halves --out outputs/statistics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import common
from src.analysis import stats_utils as su
from src.evaluation.mot_eval import COUNT_COLS

GAMMAS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
TILES = [320, 384, 448, 512, 576]


def load(per_sequence: Path, exp_id: str) -> pd.DataFrame:
    return pd.read_csv(per_sequence / f"{exp_id}.csv").set_index("seq").sort_index()


def stack(data: dict) -> tuple[list, np.ndarray, list[str]]:
    keys = list(data)
    seqs = sorted(set.intersection(*[set(v.index) for v in data.values()]))
    return keys, np.stack([data[k].loc[seqs, COUNT_COLS].to_numpy(float) for k in keys]), seqs


def scores(arr: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    gt, fp, fn, ids, tp, ifp, ifn = arr[:, idx, :].sum(axis=1).T
    return pd.DataFrame({"MOTA": 1 - (fn + fp + ids) / gt, "IDF1": 2 * tp / (2 * tp + ifp + ifn), "ids": ids})


def run_family(name: str, data: dict, paper_choice, nsplit: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys, arr, seqs = stack(data)
    n, half = len(seqs), len(seqs) // 2
    rng = np.random.default_rng(seed)
    kpaper = keys.index(paper_choice)
    full = scores(arr, np.arange(n)).set_axis(keys)
    full_pick = {c: su.select_config(full, c) for c in ("paper", "strict")}
    rows = []
    for k in range(nsplit):
        perm = rng.permutation(n)
        for direction, (sel, ev) in {"A->B": (perm[:half], perm[half:2 * half]), "B->A": (perm[half:2 * half], perm[:half])}.items():
            s_sel, s_ev = scores(arr, sel).set_axis(keys), scores(arr, ev).set_axis(keys)
            for crit in ("paper", "strict"):
                ch, best_ev = su.select_config(s_sel, crit), su.select_config(s_ev, crit)
                rows.append(dict(
                    family=name, split=k, direction=direction, criterion=crit, chosen=ch, oracle_eval_choice=best_ev,
                    sel_MOTA=s_sel.loc[ch, "MOTA"] * 100, eval_MOTA=s_ev.loc[ch, "MOTA"] * 100,
                    sel_IDF1=s_sel.loc[ch, "IDF1"] * 100, eval_IDF1=s_ev.loc[ch, "IDF1"] * 100,
                    sel_ids=s_sel.loc[ch, "ids"], eval_ids=s_ev.loc[ch, "ids"],
                    oracle_eval_MOTA=s_ev.loc[best_ev, "MOTA"] * 100, oracle_eval_IDF1=s_ev.loc[best_ev, "IDF1"] * 100,
                    paper_cfg_eval_MOTA=s_ev.iloc[kpaper]["MOTA"] * 100, paper_cfg_eval_IDF1=s_ev.iloc[kpaper]["IDF1"] * 100,
                    paper_cfg_sel_MOTA=s_sel.iloc[kpaper]["MOTA"] * 100, paper_cfg_sel_IDF1=s_sel.iloc[kpaper]["IDF1"] * 100))
    R = pd.DataFrame(rows)
    R["gap_MOTA_sel_minus_eval"] = R["sel_MOTA"] - R["eval_MOTA"]
    R["gap_IDF1_sel_minus_eval"] = R["sel_IDF1"] - R["eval_IDF1"]
    R["regret_MOTA"] = R["oracle_eval_MOTA"] - R["eval_MOTA"]
    R["regret_IDF1"] = R["oracle_eval_IDF1"] - R["eval_IDF1"]
    R["chosen_is_paper"] = R["chosen"] == paper_choice

    summ = []
    groups = list(R.groupby(["criterion", "direction"])) + [((c, "pooled"), R[R.criterion == c]) for c in ("paper", "strict")]
    for (crit, direction), g in groups:
        counts = g["chosen"].value_counts(normalize=True).sort_index()
        summ.append(dict(family=name, criterion=crit, direction=direction, n_selections=len(g), n_seq_total=n, n_seq_per_half=half,
                         freq_choose_paper_cfg=float(g["chosen_is_paper"].mean()), whole_sample_pick=full_pick[crit],
                         mean_gap_MOTA=g["gap_MOTA_sel_minus_eval"].mean(), mean_gap_IDF1=g["gap_IDF1_sel_minus_eval"].mean(),
                         gap_MOTA_p2_5=g["gap_MOTA_sel_minus_eval"].quantile(.025), gap_MOTA_p97_5=g["gap_MOTA_sel_minus_eval"].quantile(.975),
                         gap_IDF1_p2_5=g["gap_IDF1_sel_minus_eval"].quantile(.025), gap_IDF1_p97_5=g["gap_IDF1_sel_minus_eval"].quantile(.975),
                         mean_eval_MOTA_chosen=g["eval_MOTA"].mean(), mean_eval_MOTA_paper_cfg=g["paper_cfg_eval_MOTA"].mean(),
                         mean_eval_IDF1_chosen=g["eval_IDF1"].mean(), mean_eval_IDF1_paper_cfg=g["paper_cfg_eval_IDF1"].mean(),
                         mean_regret_MOTA=g["regret_MOTA"].mean(), mean_regret_IDF1=g["regret_IDF1"].mean(),
                         choice_distribution="; ".join(f"{k}={v:.3f}" for k, v in counts.items())))
    return R, pd.DataFrame(summ)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-sequence", type=Path, default=common.RESULTS / "per_sequence")
    ap.add_argument("--out", type=Path, default=common.OUT_ROOT / "statistics")
    ap.add_argument("--nsplit", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    gamma = {g: load(a.per_sequence, f"gm_nsa_g{int(round(g * 100)):03d}") for g in GAMMAS}
    tiles = {t: load(a.per_sequence, f"hem_sahi{t}") for t in TILES}
    full = scores(*stack(gamma)[1:2], np.arange(30)).set_axis(list(gamma))
    full["MOTA"], full["IDF1"] = full["MOTA"] * 100, full["IDF1"] * 100
    full.index.name = "gamma"
    full.to_csv(a.out / "c3_gamma_fullsample_clean.csv")

    r_g, s_g = run_family("gamma (clean sweep)", gamma, 0.55, a.nsplit, common.SEED)
    r_g.to_csv(a.out / "c3_gamma_splits.csv", index=False)
    r_t, s_t = run_family("tile (HEM + SAHI + BoT-SORT)", tiles, 512, a.nsplit, common.SEED)
    r_t.to_csv(a.out / "c3_tile_splits.csv", index=False)
    pd.concat([s_g, s_t]).to_csv(a.out / "c3_summary.csv", index=False)
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
