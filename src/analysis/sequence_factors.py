"""Per-sequence candidate factors for the root-cause analysis (Section 3.2).

For each of the 30 held-out sequences this script measures

  * swarm density          : ground-truth targets per frame (mean / max / min / std);
  * camera motion          : median displacement of Lucas-Kanade tracked background corners, averaged over frames
                             (sparse optical flow, robust to the few points that fall on drones);
  * image-quality metrics  : entropy, Canny edge density, Laplacian variance, block-wise local contrast (16 x 16 blocks)
                             and hot-pixel percentage, on 1 frame out of 25;
  * target-background contrast : |mean(target box) - mean(annulus)| / std(annulus), measured on ground-truth boxes of
                             1 frame out of 25 (the annulus is the immediate surrounding, not the whole scene).

Outputs (default: outputs/root_cause; the paper's tables are shipped in results/root_cause):
  density_per_sequence.csv, camera_motion_estimates.csv, frame_quality_metrics.csv,
  target_bg_contrast_per_object.csv, target_bg_contrast_per_sequence.csv

Usage:
    python -m src.analysis.sequence_factors --out outputs/root_cause
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import common
from src.evaluation.mot_eval import load_mot

FRAME_SAMPLE_STRIDE = 25   # ~30 sampled frames per 750-frame sequence
BLOCK_SIZE = 16            # block size of the local-contrast metric
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


# ---------------------------------------------------------------- density
def density_stats(seq: str) -> dict:
    df = load_mot(common.LABEL_ROOT / seq / "gt" / "gt.txt")
    counts = df.groupby("frame")["id"].count()
    return {"sequence": seq, "mean_density": counts.mean(), "max_density": counts.max(), "min_density": counts.min(), "std_density": counts.std()}


# ---------------------------------------------------------------- camera motion
def camera_motion(video: Path) -> dict | None:
    cap = cv2.VideoCapture(str(video))
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    medians = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
        if p0 is None or len(p0) < 10:
            prev_gray = gray
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
        if p1 is None:
            prev_gray = gray
            continue
        good_new, good_old = p1[st == 1], p0[st == 1]
        if len(good_new) < 10:
            prev_gray = gray
            continue
        medians.append(float(np.median(np.linalg.norm(good_new - good_old, axis=1))))
        prev_gray = gray
    cap.release()
    if not medians:
        return None
    return {"mean_motion": float(np.mean(medians)), "median_motion": float(np.median(medians)),
            "max_motion": float(np.max(medians)), "std_motion": float(np.std(medians))}


# ---------------------------------------------------------------- image quality
def frame_metrics(gray: np.ndarray) -> dict:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape
    contrasts = [gray[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE].std() for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE) for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE)]
    local_contrast = float(np.mean(contrasts)) if contrasts else 0.0
    hot_pixel_pct = float(np.mean(gray > gray.mean() + 2 * gray.std()) * 100)
    return {"entropy": entropy, "edge_density": edge_density, "laplacian_var": laplacian_var, "local_contrast": local_contrast, "hot_pixel_pct": hot_pixel_pct}


def sequence_quality(video: Path) -> dict | None:
    cap = cv2.VideoCapture(str(video))
    idx, samples = 0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % FRAME_SAMPLE_STRIDE == 0:
            samples.append(frame_metrics(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    cap.release()
    if not samples:
        return None
    df = pd.DataFrame(samples)
    return {f"mean_{c}": df[c].mean() for c in df.columns}


# ---------------------------------------------------------------- target-background contrast
def target_bg_contrast(gray: np.ndarray, x: int, y: int, w: int, h: int, pad_ratio: float = 0.75) -> float:
    H, W = gray.shape
    pad = max(int(max(w, h) * pad_ratio), 4)
    x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
    xin0, yin0, xin1, yin1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xin1 <= xin0 or yin1 <= yin0 or x1 <= x0 or y1 <= y0:
        return np.nan
    target = gray[yin0:yin1, xin0:xin1]
    outer = gray[y0:y1, x0:x1].astype(np.float64)
    mask = np.ones_like(outer, dtype=bool)
    mask[yin0 - y0:yin1 - y0, xin0 - x0:xin1 - x0] = False
    annulus = outer[mask]
    if annulus.size < 8 or target.size == 0:
        return np.nan
    return float(abs(target.mean() - annulus.mean()) / (annulus.std() + 1e-6))


def target_contrast_objects(seq: str) -> pd.DataFrame:
    """Per-object contrast on ground-truth boxes (TrainLabels/MultiUAV-<n>.txt: frame, id, left, top, width, height, ...)."""
    gt_path = common.DATA_ROOT / "TrainLabels" / f"MultiUAV-{''.join(ch for ch in seq if ch.isdigit())}.txt"
    video = common.video_path(seq)
    if not gt_path.exists() or not video.exists():
        return pd.DataFrame()
    cols = ["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class", "visibility"]
    gt = pd.read_csv(gt_path, header=None, names=cols)
    gt = gt[gt["frame"] % FRAME_SAMPLE_STRIDE == 0]
    cap = cv2.VideoCapture(str(video))
    rows = []
    for frame_num, group in gt.groupby("frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)   # GT frame numbers are 1-based
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for _, r in group.iterrows():
            rows.append({"sequence": seq, "frame": frame_num, "track_id": r.track_id, "bb_left": r.bb_left, "bb_top": r.bb_top,
                         "bb_width": r.bb_width, "bb_height": r.bb_height,
                         "target_bg_contrast": target_bg_contrast(gray, int(r.bb_left), int(r.bb_top), int(r.bb_width), int(r.bb_height))})
    cap.release()
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=common.OUT_ROOT / "root_cause")
    ap.add_argument("--sequences", nargs="*", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seqs = a.sequences or common.val_sequences()

    density, motion, quality, objects = [], [], [], []
    for seq in seqs:
        video = common.video_path(seq)
        density.append(density_stats(seq))
        m, q = camera_motion(video), sequence_quality(video)
        if m:
            motion.append({**m, "sequence": seq})
        if q:
            quality.append({**q, "sequence": seq})
        obj = target_contrast_objects(seq)
        if len(obj):
            objects.append(obj)
        print(f"{seq}: done", flush=True)

    pd.DataFrame(density).to_csv(a.out / "density_per_sequence.csv", index=False)
    pd.DataFrame(motion).to_csv(a.out / "camera_motion_estimates.csv", index=False)
    pd.DataFrame(quality).to_csv(a.out / "frame_quality_metrics.csv", index=False)
    objects_df = pd.concat(objects, ignore_index=True)
    objects_df.to_csv(a.out / "target_bg_contrast_per_object.csv", index=False)
    per_seq = objects_df.groupby("sequence")["target_bg_contrast"].mean().reset_index()
    per_seq.columns = ["sequence", "mean_target_bg_contrast"]
    per_seq.to_csv(a.out / "target_bg_contrast_per_sequence.csv", index=False)
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
