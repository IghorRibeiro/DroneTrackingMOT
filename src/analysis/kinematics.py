"""Kinematic consistency check (Section 3.5, p = 0.207): do identity switches have a different kinematic signature
from correct associations?

For the HEM + SAHI512 + BoT-SORT trajectories every tracked box is matched to the ground truth frame by frame (IoU >= 0.5,
Hungarian assignment) and labelled TP, FP or ID_SWITCH (a TP whose track id differs from the last id matched to the same
ground-truth object). For every pair of consecutive frames of a track the following are measured: area ratio, width ratio,
height ratio, aspect-ratio ratio and the centre displacement normalised by the frame diagonal. Two-sided Mann-Whitney U tests
compare TP against FP and against ID_SWITCH. The reported value is the TP-vs-ID_SWITCH test on ``center_shift_normalized``
(about 464 thousand transitions): p = 0.207, i.e. the centroid displacement does not separate true associations from
identity switches.

Usage:
    python -m src.analysis.kinematics --traj-dir trajectories/hem_sahi512 --out outputs/kinematics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu

from src import common
from src.evaluation.mot_eval import load_mot

IOU_THRESHOLD = 0.5
FRAME_DIAG = float(np.sqrt(common.FRAME_W ** 2 + common.FRAME_H ** 2))
METRICS = ["area_ratio", "width_ratio", "height_ratio", "center_shift_normalized", "aspect_ratio_ratio"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of [x, y, w, h] boxes."""
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 0:1] + a[:, 2:3], a[:, 1:2] + a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match_frame(det_boxes: np.ndarray, gt_boxes: np.ndarray, threshold: float = IOU_THRESHOLD):
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(det_boxes)))
    cost = 1 - iou_matrix(det_boxes, gt_boxes)
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if 1 - cost[r, c] >= threshold]
    matched = {r for r, _ in matches}
    return matches, [i for i in range(len(det_boxes)) if i not in matched]


def label_events(det_path: Path, gt_path: Path) -> pd.DataFrame:
    det_all, gt_all = load_mot(det_path), load_mot(gt_path)
    gt_by_frame = {f: g for f, g in gt_all.groupby("frame")}
    gt_last_track: dict[int, int] = {}
    events = []
    for frame_num in sorted(det_all["frame"].unique()):
        det = det_all[det_all["frame"] == frame_num]
        gt = gt_by_frame.get(frame_num, gt_all.iloc[0:0])
        matches, unmatched = match_frame(det[["x", "y", "w", "h"]].to_numpy(float), gt[["x", "y", "w", "h"]].to_numpy(float))
        for i_det, i_gt in matches:
            d, g = det.iloc[i_det], gt.iloc[i_gt]
            track_id, gt_id = int(d["id"]), int(g["id"])
            event = "ID_SWITCH" if gt_id in gt_last_track and gt_last_track[gt_id] != track_id else "TP"
            events.append((frame_num, track_id, d["x"], d["y"], d["w"], d["h"], event))
            gt_last_track[gt_id] = track_id
        for i_det in unmatched:
            d = det.iloc[i_det]
            events.append((frame_num, int(d["id"]), d["x"], d["y"], d["w"], d["h"], "FP"))
    df = pd.DataFrame(events, columns=["frame", "track_id", "x", "y", "w", "h", "event"])
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True) if len(df) else df


def transitions(df_events: pd.DataFrame) -> list[dict]:
    rows = []
    for tid, t in df_events.groupby("track_id"):
        t = t.sort_values("frame")
        if len(t) < 2:
            continue
        fr, xs, ys, ws, hs, ev = (t[c].to_numpy() for c in ("frame", "x", "y", "w", "h", "event"))
        areas, cxs, cys = ws * hs, xs + ws / 2, ys + hs / 2
        for i in range(1, len(t)):
            if fr[i] - fr[i - 1] != 1 or areas[i - 1] <= 0:
                continue
            aspect_prev = ws[i - 1] / hs[i - 1] if hs[i - 1] > 0 else 0
            aspect_curr = ws[i] / hs[i] if hs[i] > 0 else 0
            shift = float(np.hypot(cxs[i] - cxs[i - 1], cys[i] - cys[i - 1]))
            rows.append(dict(track_id=tid, frame=fr[i], event=ev[i], area_ratio=areas[i] / areas[i - 1], width_ratio=ws[i] / ws[i - 1],
                             height_ratio=hs[i] / hs[i - 1], center_shift=shift, center_shift_normalized=shift / FRAME_DIAG,
                             aspect_ratio_ratio=aspect_curr / aspect_prev if aspect_prev > 0 else np.nan))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=common.TRAJ_ROOT / "hem_sahi512")
    ap.add_argument("--out", type=Path, default=common.OUT_ROOT / "kinematics")
    ap.add_argument("--sequences", nargs="*", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seq in a.sequences or common.val_sequences():
        events = label_events(a.traj_dir / f"{seq}.txt", common.LABEL_ROOT / seq / "gt" / "gt.txt")
        if len(events):
            rows += transitions(events)
        print(f"{seq}: done", flush=True)
    trans = pd.DataFrame(rows).dropna(subset=["aspect_ratio_ratio"])
    trans.to_csv(a.out / "tracking_failure_kinematics.csv", index=False)

    tp = trans[trans.event == "TP"]
    result = {"n_transitions": int(len(trans)), "n_TP": int(len(tp)), "tests": {}}
    for ev in ("FP", "ID_SWITCH"):
        other = trans[trans.event == ev]
        for m in METRICS:
            if len(tp) and len(other):
                u, p = mannwhitneyu(tp[m], other[m], alternative="two-sided")
                result["tests"][f"TP_vs_{ev}:{m}"] = dict(n_other=int(len(other)), U=float(u), p=float(p),
                                                          median_TP=float(tp[m].median()), median_other=float(other[m].median()))
    result["p_idswitch_center_shift"] = result["tests"].get("TP_vs_ID_SWITCH:center_shift_normalized", {}).get("p")
    (a.out / "kin_result.json").write_text(json.dumps(result, indent=2))
    print(f"p (center_shift_normalized, TP vs ID_SWITCH) = {result['p_idswitch_center_shift']} over {len(trans)} transitions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
