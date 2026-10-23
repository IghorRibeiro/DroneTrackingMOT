"""Detector + tracker through Ultralytics ``YOLO.track`` (Tables 2 and 4, Fig. 3, Table 5, HEM, GM+NSA).

Clean-run protocol (the one behind every number of the paper):
  * a NEW ``YOLO`` object is created for every sequence, so tracker state, track identities and global motion
    compensation start from scratch. ``model.track(..., persist=True)`` keeps trackers alive across calls, so reusing
    one model across sequences leaks identities between videos;
  * frames are decoded with OpenCV and fed one by one to ``model.track(frame, persist=True)``;
  * detector confidence 0.25, tracker defaults from configs/trackers/*.yaml, device CUDA > MPS > CPU.

Usage:
    python -m src.tracking.run_yolo_tracker --id botsort
    python -m src.tracking.run_yolo_tracker --id gm_nsa_g055 --sequences UAV009 --max-frames 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src import common
from src.tracking import registry

CONF = 0.25


def install_patches(spec: dict, log_rows: list | None, state: dict):
    """Install the association / state-estimation patches of an experiment; returns ``(undo, lqb_module)``."""
    undo, lqb = [], None
    if spec.get("gm") is not None:
        from src.tracking import gm_cost, nsa_kalman
        gm = registry.gm_parameters(spec["tracker"], spec["gm"])
        gm_cost.patch_gm_cost(gamma=gm["gamma"], sigma=gm["sigma"], frame_w=common.FRAME_W, frame_h=common.FRAME_H)
        nsa_kalman.patch_nsa_kalman()
        undo += [nsa_kalman.unpatch_nsa_kalman, gm_cost.unpatch_gm_cost]
    if spec.get("item") == "adaptive":
        from src.tracking import adaptive_threshold
        undo.insert(0, adaptive_threshold.patch_adaptive_threshold(log_rows, state))
    elif spec.get("item") == "lqb":
        from src.tracking import lost_query_bank as lqb
        lqb.patch_lost_query_bank()
        undo.insert(0, lqb.unpatch_lost_query_bank)

    def _undo():
        for f in undo:
            f()
    return _undo, lqb


def frames_for(spec: dict, seq: str, max_frames: int | None):
    if spec.get("preprocess"):
        from src.tracking.preprocessing import METHODS
        return METHODS[spec["preprocess"]](str(common.video_path(seq)))
    return common.iter_frames(common.video_path(seq), max_frames)


def run(exp_id: str, sequences: list[str], out_dir: Path, max_frames: int | None, force: bool, device: str) -> pd.DataFrame | None:
    from ultralytics import YOLO

    spec = registry.get(exp_id)
    if spec["runner"] != "yolo":
        raise ValueError(f"'{exp_id}' is a '{spec['runner']}' experiment; use the matching runner")
    weights = common.WEIGHTS[spec["weights"]]
    tracker = str(registry.tracker_yaml(spec["tracker"]))
    todo = [s for s in sequences if force or not (out_dir / f"{s}.txt").exists()]
    if not todo:
        print(f"[{exp_id}] trajectories already exist, nothing to do (use --force to recompute)")
        return None
    log_rows: list = []
    state = {"seq": "", "frame": 0}
    undo, lqb = install_patches(spec, log_rows, state)
    t0 = time.time()
    try:
        for seq in todo:
            if lqb is not None:
                lqb._LOST_BANK.clear()
            model = YOLO(str(weights))                       # fresh model = fresh tracker state
            lines, n = [], 0
            state["seq"] = seq
            for idx, frame in frames_for(spec, seq, max_frames):
                if max_frames and idx > max_frames:
                    break
                state["frame"] = idx
                r = model.track(frame, persist=True, tracker=tracker, conf=CONF, device=device, verbose=False)[0]
                n = idx
                if r.boxes is None or r.boxes.id is None:
                    continue
                for (cx, cy, w, h), tid in zip(r.boxes.xywh.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int)):
                    lines.append(common.mot_line(idx, int(tid), cx - w / 2, cy - h / 2, w, h))
            common.write_atomic(out_dir / f"{seq}.txt", "\n".join(lines))
            print(f"[{exp_id}] {seq}: {n} frames ({time.time() - t0:.0f}s)", flush=True)
    finally:
        undo()
    if spec.get("item") == "adaptive":
        return pd.DataFrame(log_rows, columns=["sequence", "frame", "mean_conf", "effective_track_high_thresh", "n_dets"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="experiment id from configs/experiments.yaml")
    ap.add_argument("--sequences", nargs="*", default=None, help="default: the 30 held-out sequences")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="trajectory directory (default: trajectories/<id>)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    seqs = a.sequences or common.val_sequences()
    out_dir = a.out or common.TRAJ_ROOT / a.id
    log = run(a.id, seqs, out_dir, a.max_frames, a.force, a.device or common.device())
    if log is not None:                                         # adaptive threshold: prove the patch was active
        log_csv = common.OUT_ROOT / "adaptive_threshold_per_frame.csv"
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(log_csv, index=False)
        eff = log["effective_track_high_thresh"]
        n_low, n_high = int((eff < 0.25 - 1e-9).sum()), int((eff > 0.25 + 1e-9).sum())
        print(f"[{a.id}] frames at threshold 0.20/0.25/0.30 = {n_low}/{len(eff) - n_low - n_high}/{n_high}")
        if n_low + n_high == 0:
            print("[adaptive] FAILURE: the effective threshold never differed from the default; the patch was inactive", file=sys.stderr)
            return 7
    return 0


if __name__ == "__main__":
    sys.exit(main())
