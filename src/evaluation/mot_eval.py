"""MOT evaluation with py-motmetrics (CLEAR-MOT / IDF1, IoU >= 0.5).

Protocol used for every number in the paper:
  * full sequences (all frames; 22,839 frames and 478,548 ground-truth boxes on the 30 held-out sequences);
  * greedy-free IoU association with ``iou_matrix(max_iou=0.5)``, frame by frame;
  * no filtering by class or visibility;
  * overall MOTA / IDF1 are recomputed from the **summed** per-sequence counts, never averaged over sequences:

      MOTA = 1 - (FN + FP + IDs) / GT        IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)

Usage:
    python -m src.evaluation.mot_eval --traj-dir trajectories/hit --out results/per_sequence/hit.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "asfarray"):  # py-motmetrics 1.4.0 still calls np.asfarray, removed in NumPy 2
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

from src import common

IOU_MAX = 0.5
COUNT_COLS = ["gt", "fp", "fn", "ids", "idtp", "idfp", "idfn"]
_MM_METRICS = ["num_objects", "num_false_positives", "num_misses", "num_switches", "idtp", "idfp", "idfn", "mota", "idf1", "motp"]
_MOT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "a", "b", "c"]


def load_mot(path: Path, max_frames: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=_MOT_COLS)
    return df[df["frame"] <= max_frames] if max_frames is not None else df


def eval_sequence(seq: str, pred_path: Path, max_frames: int | None = None, label_root: Path | None = None) -> dict:
    """Evaluate one sequence against ``original_label_file/<seq>/gt/gt.txt``."""
    import motmetrics as mm
    gt = load_mot(Path(label_root or common.LABEL_ROOT) / seq / "gt" / "gt.txt", max_frames)
    pr = load_mot(pred_path, max_frames)
    gt_g = {f: (g["id"].to_numpy(), g[["x", "y", "w", "h"]].to_numpy()) for f, g in gt.groupby("frame")}
    pr_g = {f: (g["id"].to_numpy(), g[["x", "y", "w", "h"]].to_numpy()) for f, g in pr.groupby("frame")}
    empty = (np.empty(0), np.empty((0, 4)))
    acc = mm.MOTAccumulator(auto_id=True)
    for f in sorted(set(gt_g) | set(pr_g)):
        gi, gb = gt_g.get(f, empty)
        pi, pb = pr_g.get(f, empty)
        acc.update(gi, pi, mm.distances.iou_matrix(gb, pb, max_iou=IOU_MAX))
    s = mm.metrics.create().compute(acc, metrics=_MM_METRICS, name=seq).iloc[0]
    return {"seq": seq, "gt": int(s["num_objects"]), "fp": int(s["num_false_positives"]), "fn": int(s["num_misses"]),
            "ids": int(s["num_switches"]), "idtp": int(s["idtp"]), "idfp": int(s["idfp"]), "idfn": int(s["idfn"]),
            "mota": float(s["mota"]), "idf1": float(s["idf1"]), "motp": float(s["motp"])}


def mota_from_counts(c) -> float:
    return 1.0 - (c["fn"] + c["fp"] + c["ids"]) / c["gt"]


def idf1_from_counts(c) -> float:
    return 2.0 * c["idtp"] / (2.0 * c["idtp"] + c["idfp"] + c["idfn"])


def totals(df: pd.DataFrame) -> dict:
    return {k: float(df[k].sum()) for k in COUNT_COLS}


def summarize(df: pd.DataFrame) -> dict:
    t = totals(df)
    return {"n_seq": len(df), "gt": int(t["gt"]), "fp": int(t["fp"]), "fn": int(t["fn"]), "ids": int(t["ids"]),
            "mota_pct": 100.0 * mota_from_counts(t), "idf1_pct": 100.0 * idf1_from_counts(t)}


def evaluate_dir(traj_dir: Path, sequences: list[str] | None = None, max_frames: int | None = None) -> pd.DataFrame:
    """Evaluate ``<traj_dir>/<seq>.txt`` for every sequence (default: the 30 held-out sequences)."""
    sequences = sequences or common.val_sequences()
    rows = []
    for s in sequences:
        f = Path(traj_dir) / f"{s}.txt"
        if not f.exists():
            raise FileNotFoundError(f"missing trajectory: {f}")
        rows.append(eval_sequence(s, f, max_frames))
    return pd.DataFrame(rows)


def load_counts(csv: Path) -> pd.DataFrame:
    return pd.read_csv(csv).set_index("seq").sort_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, required=True, help="directory with <sequence>.txt MOT files")
    ap.add_argument("--out", type=Path, default=None, help="optional per-sequence CSV")
    ap.add_argument("--sequences", nargs="*", default=None)
    a = ap.parse_args()
    df = evaluate_dir(a.traj_dir, a.sequences)
    s = summarize(df)
    print(f"{a.traj_dir.name}: MOTA={s['mota_pct']:.2f} IDF1={s['idf1_pct']:.2f} IDs={s['ids']} FP={s['fp']} FN={s['fn']} GT={s['gt']} ({s['n_seq']} sequences)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
