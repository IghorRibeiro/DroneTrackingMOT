"""Sequence-level hard-example mining: rank the TRAINING sequences by tracking difficulty and select the top 20.

Steps:
  1. track every training sequence with the base detector + BoT-SORT (configs/trackers/botsort.yaml, conf 0.25);
  2. evaluate against the ground truth (py-motmetrics, IoU >= 0.5);
  3. rank by the composite score

         score = 2 * (100 - MOTA) + (100 - IDF1) + 0.02 * FN + 0.005 * FP + 0.20 * IDs

     and write the 20 highest-scoring sequences to ``configs/splits/hem_hard_sequences_20.txt`` (or ``--out``).

Only training sequences are eligible, so no held-out sequence can leak into the mining.

NOTE: the list used for the released checkpoint is shipped in configs/splits/hem_hard_sequences_20.txt. It was produced
with a first version of step 1 that reused one detector object for all training sequences; re-running this script uses
the clean protocol (new model per sequence) and may rank a few sequences differently. To rebuild the released HEM
detector use the shipped list.

Usage:
    python -m src.detector.select_hard_examples --out outputs/hem_hard_sequences_20.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import common
from src.evaluation import mot_eval
from src.tracking import run_yolo_tracker

TOP_K = 20
TRAIN_TRAJ_ID = "botsort_train"


def score_table(per_seq: pd.DataFrame) -> pd.DataFrame:
    d = per_seq.set_index("seq").copy()
    d["MOTA(%)"] = 100 * (1 - (d["fn"] + d["fp"] + d["ids"]) / d["gt"])
    d["IDF1(%)"] = 100 * 2 * d["idtp"] / (2 * d["idtp"] + d["idfp"] + d["idfn"])
    d["score"] = 2 * (100 - d["MOTA(%)"]) + (100 - d["IDF1(%)"]) + 0.02 * d["fn"] + 0.005 * d["fp"] + 0.20 * d["ids"]
    return d.sort_values("score", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=common.OUT_ROOT / "train_predictions")
    ap.add_argument("--out", type=Path, default=common.OUT_ROOT / "hem_hard_sequences_20.txt")
    ap.add_argument("--skip-tracking", action="store_true", help="reuse the trajectories already in --traj-dir")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    train = common.train_sequences()
    if not a.skip_tracking:
        spec_id = "botsort"                                   # base detector + default BoT-SORT
        run_yolo_tracker.run(spec_id, train, a.traj_dir, None, False, a.device or common.device())
    per_seq = mot_eval.evaluate_dir(a.traj_dir, train)
    table = score_table(per_seq)
    hard = table.head(TOP_K)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(hard.index) + "\n")
    table.to_csv(a.out.with_suffix(".scores.csv"))
    print(hard[["MOTA(%)", "IDF1(%)", "fp", "fn", "ids", "score"]].round(2).to_string())
    print(f"\n{len(hard)} sequences written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
