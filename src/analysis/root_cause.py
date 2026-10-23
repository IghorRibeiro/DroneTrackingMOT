"""Root-cause analysis (Section 3.2, Fig. 2): which per-sequence factors explain the baseline's tracking performance?

Inputs (all in results/root_cause/ unless overridden):
  * the per-sequence factor tables produced by ``sequence_factors.py``
    (density, camera motion, image-quality metrics, target-background contrast);
  * the per-sequence counts of the baseline (``results/per_sequence/botsort.csv``) and, for the cause -> intervention
    link, of HIT-Tracker (``results/per_sequence/hit.csv``).

Outputs: Pearson / Spearman correlations with Fisher 95 % CI and Holm correction across the eight factors, the check
against the values quoted in the paper, factor collinearity, the link "edge density vs HIT-Tracker gain" and the
edge-density terciles.

Usage:
    python -m src.analysis.root_cause --out outputs/root_cause
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import common
from src.analysis import stats_utils as su

FACTORS = [  # (column, label)
    ("mean_density", "Swarm density (targets/frame)"),
    ("mean_motion", "Camera motion (px/frame)"),
    ("mean_edge_density", "Edge density (Canny)"),
    ("mean_local_contrast", "Local contrast (global, block-wise)"),
    ("mean_target_bg_contrast", "Target-background contrast (per GT box)"),
    ("mean_entropy", "Entropy"),
    ("mean_laplacian_var", "Laplacian variance"),
    ("mean_hot_pixel_pct", "Hot pixels (%)"),
]

# Pearson r quoted in the paper (Section 3.3), checked automatically against the recomputed values
PAPER_R = [
    ("mean_density", "IDF1", 0.180), ("mean_density", "MOTA", 0.195),
    ("mean_motion", "IDF1", 0.368), ("mean_motion", "MOTA", 0.224),
    ("mean_edge_density", "IDF1", -0.600), ("mean_edge_density", "MOTA", -0.348),
    ("mean_local_contrast", "IDF1", -0.564), ("mean_local_contrast", "MOTA", -0.268),
    ("mean_target_bg_contrast", "MOTA", 0.707), ("mean_target_bg_contrast", "IDF1", 0.600),
    ("mean_target_bg_contrast", "FN_rate", -0.712),
]
COLLINEARITY_PAPER = 0.906  # edge density x local contrast


def load_factors(factor_dir: Path) -> pd.DataFrame:
    q = pd.read_csv(factor_dir / "frame_quality_metrics.csv")
    m = pd.read_csv(factor_dir / "camera_motion_estimates.csv")[["sequence", "mean_motion"]]
    d = pd.read_csv(factor_dir / "density_per_sequence.csv")[["sequence", "mean_density"]]
    t = pd.read_csv(factor_dir / "target_bg_contrast_per_sequence.csv")
    return q.merge(m, on="sequence").merge(d, on="sequence").merge(t, on="sequence").set_index("sequence").sort_index()


def baseline_outcomes(counts: pd.DataFrame) -> pd.DataFrame:
    c = counts.set_index("seq").sort_index()
    out = pd.DataFrame({"MOTA": su.mota_from_counts(c) * 100, "IDF1": su.idf1_from_counts(c) * 100,
                        "FN_rate": c["fn"] / c["gt"], "FP_rate": c["fp"] / c["gt"], "IDs": c["ids"], "GT": c["gt"]})
    out["MOTA_r1"], out["IDF1_r1"] = out["MOTA"].round(1), out["IDF1"].round(1)   # one-decimal inputs, as in the paper tables
    out["IDs_per_1k"] = out["IDs"] / out["GT"] * 1000
    return out


def correlations(F: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, kind in [("MOTA", "primary"), ("IDF1", "primary"), ("FN_rate", "extra"), ("IDs_per_1k", "extra")]:
        block = []
        for col, lab in FACTORS:
            r = su.corr_row(F[col].loc[Y.index], Y[outcome])
            r_round = su.corr_row(F[col].loc[Y.index], Y[f"{outcome}_r1"])["pearson_r"] if outcome in ("MOTA", "IDF1") else np.nan
            block.append(dict(factor=col, label=lab, outcome=outcome, kind=kind, **r, pearson_r_rounded_inputs=r_round))
        for b, adj in zip(block, su.holm([b["pearson_p"] for b in block])):
            b["pearson_p_holm_8factors"] = adj
            rows.append(b)
    return pd.DataFrame(rows)


def paper_check(corr: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, outcome, value in PAPER_R:
        c = corr[(corr.factor == factor) & (corr.outcome == outcome)].iloc[0]
        tol = 0.0011  # half unit of the last printed digit (three decimals)
        exact, rounded = c["pearson_r"], c["pearson_r_rounded_inputs"]
        ok = abs(exact - value) <= tol or (np.isfinite(rounded) and abs(rounded - value) <= tol)
        rows.append(dict(factor=factor, outcome=outcome, paper_r=value, r_exact_inputs=exact, r_rounded_inputs=rounded, status="OK" if ok else "DIFFERS"))
    r_el = float(np.corrcoef(F["mean_edge_density"], F["mean_local_contrast"])[0, 1])
    rows.append(dict(factor="edge_density x local_contrast", outcome="collinearity", paper_r=COLLINEARITY_PAPER, r_exact_inputs=r_el,
                     r_rounded_inputs=np.nan, status="OK" if abs(r_el - COLLINEARITY_PAPER) <= 0.0011 else "DIFFERS"))
    return pd.DataFrame(rows)


def link_and_terciles(F: pd.DataFrame, base: pd.DataFrame, hit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    b, h = base.set_index("seq").sort_index(), hit.set_index("seq").sort_index()
    P = pd.DataFrame({"gt": b["gt"], "b_fp": b["fp"], "b_fn": b["fn"], "b_ids": b["ids"], "b_idtp": b["idtp"], "b_idfp": b["idfp"], "b_idfn": b["idfn"],
                      "h_fp": h["fp"], "h_fn": h["fn"], "h_ids": h["ids"], "h_idtp": h["idtp"], "h_idfp": h["idfp"], "h_idfn": h["idfn"]})
    P["b_idf1"] = 2 * P["b_idtp"] / (2 * P["b_idtp"] + P["b_idfp"] + P["b_idfn"])
    P["h_idf1"] = 2 * P["h_idtp"] / (2 * P["h_idtp"] + P["h_idfp"] + P["h_idfn"])
    P["b_mota"] = 1 - (P["b_fn"] + P["b_fp"] + P["b_ids"]) / P["gt"]
    P["h_mota"] = 1 - (P["h_fn"] + P["h_fp"] + P["h_ids"]) / P["gt"]
    P["d_idf1_pp"] = (P["h_idf1"] - P["b_idf1"]) * 100
    P["d_mota_pp"] = (P["h_mota"] - P["b_mota"]) * 100
    P["d_ids"] = P["h_ids"] - P["b_ids"]
    P["d_ids_per_1k"] = P["d_ids"] / P["gt"] * 1000
    edge = F["mean_edge_density"].loc[P.index]
    link = pd.DataFrame([dict(outcome=lab, factor="mean_edge_density", **su.corr_row(edge, P[y]))
                         for y, lab in [("d_idf1_pp", "dIDF1 (p.p.) HIT - Baseline"), ("d_ids", "dIDs HIT - Baseline"),
                                        ("d_ids_per_1k", "dIDs per 1000 GT"), ("d_mota_pp", "dMOTA (p.p.) HIT - Baseline")]])
    order = edge.sort_values(kind="mergesort").index
    terc_rows = []
    for name, ix in zip(["T1 (low edge density)", "T2", "T3 (high edge density)"], np.array_split(np.arange(len(order)), 3)):
        seqs = order[ix]
        s, e = P.loc[seqs], edge.loc[seqs]
        gt = s["gt"].sum()
        row = dict(tercile=name, n_seq=len(seqs), edge_min=e.min(), edge_max=e.max(), gt=int(gt),
                   base_mota_pct=(1 - (s.b_fn.sum() + s.b_fp.sum() + s.b_ids.sum()) / gt) * 100,
                   hit_mota_pct=(1 - (s.h_fn.sum() + s.h_fp.sum() + s.h_ids.sum()) / gt) * 100,
                   base_ids=int(s.b_ids.sum()), hit_ids=int(s.h_ids.sum()),
                   ids_reduction_pct=(1 - s.h_ids.sum() / s.b_ids.sum()) * 100 if s.b_ids.sum() else np.nan, seqs=" ".join(seqs),
                   base_idf1_pct=2 * s.b_idtp.sum() / (2 * s.b_idtp.sum() + s.b_idfp.sum() + s.b_idfn.sum()) * 100,
                   hit_idf1_pct=2 * s.h_idtp.sum() / (2 * s.h_idtp.sum() + s.h_idfp.sum() + s.h_idfn.sum()) * 100)
        row["d_idf1_pp"] = row["hit_idf1_pct"] - row["base_idf1_pct"]
        row["d_mota_pp"] = row["hit_mota_pct"] - row["base_mota_pct"]
        terc_rows.append(row)
    return link, pd.DataFrame(terc_rows), P.assign(edge_density=edge)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factors", type=Path, default=common.RESULTS / "root_cause")
    ap.add_argument("--per-sequence", type=Path, default=common.RESULTS / "per_sequence")
    ap.add_argument("--out", type=Path, default=common.OUT_ROOT / "root_cause")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    F = load_factors(a.factors)
    base = pd.read_csv(a.per_sequence / "botsort.csv")
    hit = pd.read_csv(a.per_sequence / "hit.csv")
    Y = baseline_outcomes(base)
    if list(F.index) != list(Y.index):
        raise RuntimeError("factor table and baseline cover different sequences")
    corr = correlations(F, Y)
    corr.to_csv(a.out / "c2_correlations_baseline_clean.csv", index=False)
    chk = paper_check(corr, F)
    chk.to_csv(a.out / "c2_vs_paper_text.csv", index=False)
    pd.concat([F, Y], axis=1).to_csv(a.out / "c2_per_seq_dataset.csv")
    F.corr(method="pearson").to_csv(a.out / "c2_factor_collinearity_pearson.csv")
    link, terc, per_seq = link_and_terciles(F, base, hit)
    link.to_csv(a.out / "c2_link_delta_vs_edge.csv", index=False)
    terc.to_csv(a.out / "c2_terciles_edge_density.csv", index=False)
    per_seq.to_csv(a.out / "c2_per_seq_delta.csv")
    print(chk.to_string(index=False))
    n_bad = int((chk.status != "OK").sum())
    print(f"\n{len(chk) - n_bad}/{len(chk)} values quoted in the paper reproduced")
    return 0 if n_bad == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
