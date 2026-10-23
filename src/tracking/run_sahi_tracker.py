"""SAHI sliced detection + BoT-SORT (Table 3 and HIT-Tracker).

Pipeline per frame:  OpenCV decode -> SAHI tiles (YOLO26n-HEM) -> Ultralytics ``Results`` -> BoT-SORT -> MOT line.
A NEW ``BOTSORT`` object is created for every sequence (clean-run protocol). For HIT-Tracker the GM association cost
and NSA-Kalman patches are installed before the tracker is instantiated and removed afterwards.

Usage:
    python -m src.tracking.run_sahi_tracker --id hem_sahi512
    python -m src.tracking.run_sahi_tracker --id hit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from src import common
from src.tracking import registry


def build_sahi_detections(frame: np.ndarray, det_model, slice_size: int, overlap: float, conf_thresh: float,
                          match_threshold: float = 0.6) -> np.ndarray:
    """Run SAHI on one frame; returns an (N, 6) array ``[x1, y1, x2, y2, score, class_id]``."""
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(frame, detection_model=det_model, slice_height=slice_size, slice_width=slice_size,
                                   overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                   postprocess_match_threshold=match_threshold, verbose=0)
    height, width = frame.shape[:2]
    detections = []
    for obj in result.object_prediction_list:
        score = float(obj.score.value)
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        x1 = max(0.0, min(width - 1.0, float(x1)))
        y1 = max(0.0, min(height - 1.0, float(y1)))
        x2 = max(0.0, min(width - 1.0, float(x2)))
        y2 = max(0.0, min(height - 1.0, float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append([x1, y1, x2, y2, score, float(obj.category.id)])
    return np.asarray(detections, dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)


def load_tracker_args(tracker_name: str) -> SimpleNamespace:
    return SimpleNamespace(**yaml.safe_load(registry.tracker_yaml(tracker_name).read_text()))


def make_detector(weights: Path, device: str, conf: float):
    from sahi import AutoDetectionModel

    return AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                              confidence_threshold=conf, device=device)


def track_sequence(seq: str, det_model, tracker_args: SimpleNamespace, sahi_cfg: dict, device: str,
                   max_frames: int | None = None) -> tuple[list[str], int]:
    """Track one sequence with a fresh BoT-SORT; returns the MOT lines and the number of frames."""
    import torch
    from ultralytics.engine.results import Results
    from ultralytics.trackers import BOTSORT

    tdev = f"cuda:{device}" if device.isdigit() else device
    tracker = BOTSORT(args=tracker_args)
    lines, n = [], 0
    for idx, frame in common.iter_frames(common.video_path(seq), max_frames):
        n = idx
        det = build_sahi_detections(frame, det_model, sahi_cfg["slice"], sahi_cfg["overlap"], sahi_cfg["conf"], sahi_cfg["match_threshold"])
        res = Results(orig_img=frame, path="frame.jpg", names={0: "drone"}, boxes=torch.from_numpy(det).to(device=tdev, dtype=torch.float32))
        for t in tracker.update(res.boxes.cpu().numpy(), frame):
            x1, y1, x2, y2 = t[:4]
            lines.append(common.mot_line(idx, int(t[4]), x1, y1, x2 - x1, y2 - y1))
    return lines, n


def install_gm_nsa(gamma: float, tracker_name: str):
    from src.tracking import gm_cost, nsa_kalman

    gm = registry.gm_parameters(tracker_name, gamma)
    gm_cost.patch_gm_cost(gamma=gm["gamma"], sigma=gm["sigma"], frame_w=common.FRAME_W, frame_h=common.FRAME_H)
    nsa_kalman.patch_nsa_kalman()
    return lambda: (nsa_kalman.unpatch_nsa_kalman(), gm_cost.unpatch_gm_cost())


def run(exp_id: str, sequences: list[str], out_dir: Path, max_frames: int | None, force: bool, device: str) -> None:
    spec = registry.get(exp_id)
    if spec["runner"] != "sahi":
        raise ValueError(f"'{exp_id}' is a '{spec['runner']}' experiment; use the matching runner")
    sahi_cfg = registry.sahi_parameters(spec["tile"])
    todo = [s for s in sequences if force or not (out_dir / f"{s}.txt").exists()]
    if not todo:
        print(f"[{exp_id}] trajectories already exist, nothing to do (use --force to recompute)")
        return
    det_model = make_detector(common.WEIGHTS[spec["weights"]], device, sahi_cfg["conf"])
    tracker_name = "botsort_gm_nsa" if spec.get("gm") is not None else "botsort"
    undo = install_gm_nsa(spec["gm"], tracker_name) if spec.get("gm") is not None else (lambda: None)
    args = load_tracker_args(tracker_name)
    print(f"[{exp_id}] tile={sahi_cfg['slice']} overlap={sahi_cfg['overlap']} gm={spec.get('gm')} device={device}", flush=True)
    t0 = time.time()
    try:
        for seq in todo:
            lines, n = track_sequence(seq, det_model, args, sahi_cfg, device, max_frames)
            common.write_atomic(out_dir / f"{seq}.txt", "\n".join(lines))
            print(f"[{exp_id}] {seq}: {n} frames ({time.time() - t0:.0f}s)", flush=True)
    finally:
        undo()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    run(a.id, a.sequences or common.val_sequences(), a.out or common.TRAJ_ROOT / a.id, a.max_frames, a.force, a.device or common.device())
    return 0


if __name__ == "__main__":
    sys.exit(main())
