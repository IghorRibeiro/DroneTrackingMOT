"""Check every statistic and count quoted in the text of the paper against the values recomputed from results/.

Sources: results/per_sequence (counts), results/statistics (bootstrap, Wilcoxon, half-splits, kinematics, root cause).
Values are compared at the precision printed in the paper. Exit status 0 if all match.

Usage:
    python -m src.analysis.check_paper_statistics [--statistics DIR] [--per-sequence DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src import common
from src.evaluation.mot_eval import totals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--statistics", type=Path, default=common.RESULTS / "statistics")
    ap.add_argument("--per-sequence", type=Path, default=common.RESULTS / "per_sequence")
    a = ap.parse_args()
    S, P = a.statistics, a.per_sequence

    def src(name: str) -> Path:        # recomputed file if present, otherwise the shipped one
        return S / name if (S / name).exists() else common.RESULTS / "statistics" / name

    cnt = {i: totals(pd.read_csv(P / f"{i}.csv")) for i in ("botsort", "hem", "hem_sahi512", "hit", "gm_nsa_g055", "static", "lqb", "adaptive")}
    corr = pd.read_csv(src("c2_correlations_baseline_clean.csv")).set_index(["factor", "outcome"])
    wil = pd.read_csv(src("c1_wilcoxon.csv")).set_index(["comparison", "metric"])
    boot = pd.read_csv(src("c1_bootstrap.csv")).set_index(["comparison", "metric"])
    terc = pd.read_csv(src("c2_terciles_edge_density.csv"))
    link = pd.read_csv(src("c2_link_delta_vs_edge.csv")).set_index("outcome")
    half = pd.read_csv(src("c3_summary.csv"))
    kin = json.loads((src("kin_result.json")).read_text())
    pooled = half[(half.direction == "pooled") & (half.criterion == "paper")].set_index("family")

    def w(cmp_, metric, col):
        return float(wil.loc[(cmp_, metric), col])
    rows = []

    def chk(name, got, paper, digits):
        ok = round(float(got), digits) == round(float(paper), digits)
        rows.append((name, f"{float(got):.{digits + 1}f}", f"{paper}", "OK" if ok else "DIFFERS"))

    # Section 3.3 - root cause
    chk("density x IDF1: p", corr.loc[("mean_density", "IDF1"), "pearson_p"], 0.34, 2)
    chk("camera motion x IDF1: p", corr.loc[("mean_motion", "IDF1"), "pearson_p"], 0.045, 3)
    chk("edge density x IDF1: Holm p (8 factors)", corr.loc[("mean_edge_density", "IDF1"), "pearson_p_holm_8factors"], 0.004, 3)
    chk("local contrast x IDF1: Spearman rho", corr.loc[("mean_local_contrast", "IDF1"), "spearman_rho"], -0.21, 2)
    # Section 3.4 - detector
    chk("baseline FN", cnt["botsort"]["fn"], 65554, 0)
    chk("HEM FN", cnt["hem"]["fn"], 61597, 0)
    chk("HEM vs baseline MOTA: Wilcoxon p", w("hem_vs_baseline", "dMOTA (p.p.)", "p"), 0.25, 2)
    # Section 3.5 - GM + NSA vs baseline
    chk("GM+NSA vs baseline IDF1: Wilcoxon p", w("gmnsa_vs_baseline", "dIDF1 (p.p.)", "p"), 0.004, 3)
    chk("GM+NSA vs baseline IDs: Wilcoxon p", w("gmnsa_vs_baseline", "dIDs", "p"), 0.026, 3)
    chk("GM+NSA vs baseline MOTA: Wilcoxon p", w("gmnsa_vs_baseline", "dMOTA (p.p.)", "p"), 0.49, 2)
    chk("GM+NSA vs baseline: IDs reduction (%)", boot.loc[("gmnsa_vs_baseline", "Relative IDs reduction (%)"), "estimate"], 31.6, 1)
    chk("GM+NSA IDs at gamma=0.55", cnt["gm_nsa_g055"]["ids"], 1611, 0)
    # Section 3.6 - interventions
    chk("static filter FN (reference)", cnt["gm_nsa_g055"]["fn"], 58248, 0)
    chk("static filter FN", cnt["static"]["fn"], 115739, 0)
    chk("lost-query bank IDs", cnt["lqb"]["ids"], 3464, 0)
    chk("adaptive threshold IDs reduction (%)", 100 * (1 - cnt["adaptive"]["ids"] / cnt["gm_nsa_g055"]["ids"]), 4.7, 1)
    chk("kinematic check transitions", kin["n_transitions"], 464345, 0)
    chk("kinematic check Mann-Whitney p", kin["p_idswitch_center_shift"], 0.207, 3)
    # Section 4.2
    for metric, n_imp, tie in (("dMOTA (p.p.)", 27, None), ("dIDF1 (p.p.)", 25, None), ("dIDs", 29, 1)):
        chk(f"HIT vs baseline {metric}: sequences improved", w("hit_vs_baseline", metric, "improved"), n_imp, 0)
        chk(f"HIT vs baseline {metric}: p < 0.001", float(w("hit_vs_baseline", metric, "p") < 0.001), 1.0, 0)
    chk("HIT vs baseline IDs: ties", w("hit_vs_baseline", "dIDs", "tied"), 1, 0)
    chk("HEM+SAHI512 vs baseline: MOTA gain (p.p.)", boot.loc[("hemsahi512_vs_baseline", "dMOTA (p.p.)"), "estimate"], 3.6, 1)
    chk("baseline FN (again)", cnt["botsort"]["fn"], 65554, 0)
    chk("HEM+SAHI512 FN", cnt["hem_sahi512"]["fn"], 54341, 0)
    chk("baseline FP", cnt["botsort"]["fp"], 47148, 0)
    chk("HEM+SAHI512 FP", cnt["hem_sahi512"]["fp"], 42789, 0)
    chk("HIT vs HEM+SAHI512 IDF1 gain (p.p.)", boot.loc[("hit_vs_hemsahi512", "dIDF1 (p.p.)"), "estimate"], 2.5, 1)
    chk("HIT vs HEM+SAHI512 IDF1: p", w("hit_vs_hemsahi512", "dIDF1 (p.p.)", "p"), 0.040, 3)
    chk("HIT vs HEM+SAHI512 IDs reduction (%)", boot.loc[("hit_vs_hemsahi512", "Relative IDs reduction (%)"), "estimate"], 30.6, 1)
    chk("HIT vs HEM+SAHI512 IDs: p", w("hit_vs_hemsahi512", "dIDs", "p"), 0.011, 3)
    chk("HIT vs HEM+SAHI512 MOTA gain (p.p.)", boot.loc[("hit_vs_hemsahi512", "dMOTA (p.p.)"), "estimate"], 0.4, 1)
    chk("HIT vs HEM+SAHI512 MOTA: p", w("hit_vs_hemsahi512", "dMOTA (p.p.)", "p"), 0.045, 3)
    for i, (d_idf1, d_mota) in enumerate(((10.8, 2.4), (9.3, 3.1), (16.7, 7.8)), 1):
        chk(f"tercile {i} dIDF1", terc.d_idf1_pp.iloc[i - 1], d_idf1, 1)
        chk(f"tercile {i} dMOTA", terc.d_mota_pp.iloc[i - 1], d_mota, 1)
    chk("tercile IDs reduction min (%)", terc.ids_reduction_pct.min(), 75.7, 1)
    chk("tercile IDs reduction max (%)", terc.ids_reduction_pct.max(), 80.6, 1)
    chk("dIDF1 vs edge density: r", link.loc["dIDF1 (p.p.) HIT - Baseline", "pearson_r"], 0.47, 2)
    chk("dIDF1 vs edge density: p", link.loc["dIDF1 (p.p.) HIT - Baseline", "pearson_p"], 0.008, 3)
    # Section 5 - split-half optimism
    for fam, (m, f) in {"gamma (clean sweep)": (0.1, 0.6), "tile (HEM + SAHI + BoT-SORT)": (0.3, 0.3)}.items():
        chk(f"half-split optimism MOTA, {fam}", pooled.loc[fam, "mean_gap_MOTA"], m, 1)
        chk(f"half-split optimism IDF1, {fam}", pooled.loc[fam, "mean_gap_IDF1"], f, 1)

    out = pd.DataFrame(rows, columns=["statistic", "recomputed", "paper", "status"])
    print(out.to_string(index=False))
    bad = int((out.status != "OK").sum())
    print(f"\n{len(out) - bad}/{len(out)} statistics quoted in the paper reproduced")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
