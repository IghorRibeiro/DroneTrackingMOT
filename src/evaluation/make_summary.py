"""Write results/summary.csv: one row per configuration of the paper (totals over the 30 held-out sequences).

Columns: id, table, label, n_seq, gt, fp, fn, ids, mota_pct, idf1_pct (MOTA / IDF1 recomputed from the summed counts, unrounded).
Rows follow results/paper_numbers.csv (an id that appears in several tables appears once per table).

Usage:
    python -m src.evaluation.make_summary [--per-sequence DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import common
from src.evaluation import mot_eval


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-sequence", type=Path, default=common.RESULTS / "per_sequence")
    ap.add_argument("--out", type=Path, default=common.RESULTS / "summary.csv")
    a = ap.parse_args()
    paper = pd.read_csv(common.RESULTS / "paper_numbers.csv")
    rows = []
    for r in paper.itertuples(index=False):
        s = mot_eval.summarize(pd.read_csv(a.per_sequence / f"{r.id}.csv"))
        rows.append(dict(id=r.id, table=r.table, label=r.label, n_seq=s["n_seq"], gt=s["gt"], fp=s["fp"], fn=s["fn"], ids=s["ids"],
                         mota_pct=round(s["mota_pct"], 4), idf1_pct=round(s["idf1_pct"], 4)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{len(rows)} rows written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
