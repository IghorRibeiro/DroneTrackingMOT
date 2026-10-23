"""Run any experiment of configs/experiments.yaml with the right runner.

    python -m src.tracking.run_experiment --id hit
    python -m src.tracking.run_experiment --id static          # post-processing of trajectories/gm_nsa_g055
    python -m src.tracking.run_experiment --id botsort --sequences UAV009 --max-frames 100

Trajectories are written to trajectories/<id>/<sequence>.txt (existing files are kept unless --force).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import common
from src.tracking import registry, run_sahi_tracker, run_yolo_tracker, static_filter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    spec = registry.get(a.id)
    seqs = a.sequences or common.val_sequences()
    out_dir = a.out or common.TRAJ_ROOT / a.id
    device = a.device or common.device()

    if spec["runner"] == "sahi":
        run_sahi_tracker.run(a.id, seqs, out_dir, a.max_frames, a.force, device)
    elif spec["runner"] == "yolo":
        log = run_yolo_tracker.run(a.id, seqs, out_dir, a.max_frames, a.force, device)
        if log is not None:                                    # adaptive threshold: prove the patch was active
            common.OUT_ROOT.mkdir(parents=True, exist_ok=True)
            log.to_csv(common.OUT_ROOT / "adaptive_threshold_per_frame.csv", index=False)
            eff = log["effective_track_high_thresh"]
            n_low, n_high = int((eff < 0.25 - 1e-9).sum()), int((eff > 0.25 + 1e-9).sum())
            print(f"[{a.id}] frames at threshold 0.20/0.25/0.30 = {n_low}/{len(eff) - n_low - n_high}/{n_high}")
            if n_low + n_high == 0:
                print("[adaptive] FAILURE: the effective threshold never differed from the default; the patch was inactive", file=sys.stderr)
                return 7
    elif spec["runner"] == "post":
        src_dir = common.TRAJ_ROOT / spec["source"]
        for seq in seqs:
            dst = out_dir / f"{seq}.txt"
            if dst.exists() and not a.force:
                continue
            df = pd.read_csv(src_dir / f"{seq}.txt", header=None, names=static_filter.MOT_COLS)
            kept, _ = static_filter.filter_static_tracks(df, window=spec["window"], std_threshold=spec["std_threshold"])
            common.write_atomic(dst, "\n".join(static_filter.df_to_mot_lines(kept)))
            print(f"[{a.id}] {seq}: {len(df)} -> {len(kept)} rows", flush=True)
    else:
        raise ValueError(f"unknown runner '{spec['runner']}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
