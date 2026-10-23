"""Paired statistics between systems (Section 4): bootstrap confidence intervals and Wilcoxon signed-rank tests.

Resampling unit = the 30 held-out SEQUENCES (paired: the same resampled sequences are used for both systems).
10,000 bootstrap replicates, seed 2026. MOTA and IDF1 are recomputed from the summed counts in every replicate.

Outputs (in --out): c1_bootstrap.csv, c1_wilcoxon.csv, c1_absolute_ci.csv, c1_per_seq_hit_vs_baseline.csv

Usage:
    python -m src.analysis.paired_tests --out outputs/statistics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import common
from src.analysis import stats_utils as su
from src.evaluation.mot_eval import COUNT_COLS, idf1_from_counts, mota_from_counts, totals

COMPARISONS = [  # (id, base, other, description)
    ("hit_vs_baseline", "botsort", "hit", "HIT-Tracker - baseline"),
    ("hem_vs_baseline", "botsort", "hem", "HEM - baseline"),
    ("gmnsa_vs_baseline", "botsort", "gm_nsa_g055", "GM cost + NSA-Kalman (gamma=0.55) - baseline"),
    ("hemsahi512_vs_baseline", "botsort", "hem_sahi512", "HEM + SAHI512 - baseline"),
    ("hit_vs_hemsahi512", "hem_sahi512", "hit", "HIT-Tracker - HEM + SAHI512"),
]


def load(per_sequence: Path, exp_id: str) -> pd.DataFrame:
    return pd.read_csv(per_sequence / f"{exp_id}.csv").set_index("seq").sort_index()


def run(per_sequence: Path, out: Path, nboot: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pts = {p: load(per_sequence, p) for p in {x for c in COMPARISONS for x in c[1:3]}}
    boot_rows, wil_rows, abs_rows, per_seq = [], [], [], None
    for cid, b, o, desc in COMPARISONS:
        A, B = pts[b], pts[o]
        seqs = A.index.intersection(B.index)
        bs = su.paired_bootstrap(A, B, nboot, common.SEED)
        for m, name in [("d_mota_pp", "dMOTA (p.p.)"), ("d_idf1_pp", "dIDF1 (p.p.)"), ("ids_reduction_pct", "Relative IDs reduction (%)")]:
            r = bs[m]
            boot_rows.append(dict(comparison=cid, description=desc, metric=name, estimate=r["point"], boot_mean=r["boot_mean"],
                                  ci95_lo=r["ci95_lo"], ci95_hi=r["ci95_hi"], n_seq=bs["_n"], n_boot=nboot, seed=common.SEED))
        a_, b_ = A.loc[seqs], B.loc[seqs]
        d_mota = (mota_from_counts(b_) - mota_from_counts(a_)).to_numpy() * 100
        d_idf1 = (idf1_from_counts(b_) - idf1_from_counts(a_)).to_numpy() * 100
        d_ids = (b_["ids"] - a_["ids"]).to_numpy()
        for name, d, lower in [("dMOTA (p.p.)", d_mota, False), ("dIDF1 (p.p.)", d_idf1, False), ("dIDs", d_ids, True)]:
            wil_rows.append(dict(comparison=cid, metric=name, **su.wilcoxon_row(d, lower_is_better=lower)))
        if cid == "hit_vs_baseline":
            per_seq = pd.DataFrame({"seq": seqs, **{f"base_{k}": a_[k].to_numpy() for k in COUNT_COLS},
                                    **{f"hit_{k}": b_[k].to_numpy() for k in COUNT_COLS},
                                    "d_mota_pp": d_mota, "d_idf1_pp": d_idf1, "d_ids": d_ids})
        idx = np.random.default_rng(common.SEED).integers(0, len(seqs), size=(nboot, len(seqs)))
        for tag, df in [(b, a_), (o, b_)]:
            if any(r["system"] == tag for r in abs_rows):
                continue
            agg = {k: df[k].to_numpy(float)[idx].sum(axis=1) for k in COUNT_COLS}
            for m, v, pt in [("MOTA (%)", mota_from_counts(agg) * 100, mota_from_counts(totals(df)) * 100),
                             ("IDF1 (%)", idf1_from_counts(agg) * 100, idf1_from_counts(totals(df)) * 100), ("IDs", agg["ids"], float(df["ids"].sum()))]:
                lo, hi = np.percentile(v, [2.5, 97.5])
                abs_rows.append(dict(system=tag, metric=m, estimate=pt, ci95_lo=lo, ci95_hi=hi, n_seq=len(seqs), n_boot=nboot, seed=common.SEED))
    pd.DataFrame(boot_rows).to_csv(out / "c1_bootstrap.csv", index=False)
    pd.DataFrame(wil_rows).to_csv(out / "c1_wilcoxon.csv", index=False)
    pd.DataFrame(abs_rows).to_csv(out / "c1_absolute_ci.csv", index=False)
    per_seq.to_csv(out / "c1_per_seq_hit_vs_baseline.csv", index=False)
    print(f"{len(boot_rows)} bootstrap rows and {len(wil_rows)} Wilcoxon rows written to {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-sequence", type=Path, default=common.RESULTS / "per_sequence")
    ap.add_argument("--out", type=Path, default=common.OUT_ROOT / "statistics")
    ap.add_argument("--nboot", type=int, default=10000)
    a = ap.parse_args()
    run(a.per_sequence, a.out, a.nboot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
