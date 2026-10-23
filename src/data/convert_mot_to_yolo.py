"""Convert MOTChallenge annotations (gt.txt) and the .mp4 videos of the 4th Anti-UAV Challenge (Track 3) to the YOLO
detection format.

Output:  <YOLO_ROOT>/all/images/<SEQ>_<frame>.jpg   and   <YOLO_ROOT>/all/labels/<SEQ>_<frame>.txt
Run ``src/data/split_train_val.py`` afterwards to split by sequence.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

from src import common

OUT_ROOT = common.YOLO_ROOT / "all"
OUT_IMAGES = OUT_ROOT / "images"
OUT_LABELS = OUT_ROOT / "labels"


def convert_sequence(video_path: Path, gt_path: Path, seq_name: str) -> int:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warning] could not open {video_path}, skipping {seq_name}")
        return 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_annots: dict[int, list[tuple[float, float, float, float]]] = {}
    with open(gt_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame = int(row[0])
            x, y, w, h = (float(v) for v in row[2:6])
            frame_annots.setdefault(frame, []).append((x, y, w, h))

    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_idx += 1  # gt.txt is 1-based
        stem = f"{seq_name}_{frame_idx:06d}"
        cv2.imwrite(str(OUT_IMAGES / f"{stem}.jpg"), img)
        lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}" for (x, y, w, h) in frame_annots.get(frame_idx, [])]
        (OUT_LABELS / f"{stem}.txt").write_text("\n".join(lines))
    cap.release()
    return frame_idx


def main() -> int:
    seq_dirs = sorted(p for p in common.LABEL_ROOT.iterdir() if p.is_dir())
    if not seq_dirs:
        raise RuntimeError(f"no sequences found in {common.LABEL_ROOT}; set ANTIUAV_ROOT (see README, 'Dataset')")
    print(f"Found {len(seq_dirs)} sequences")
    total = 0
    for seq_dir in seq_dirs:
        gt_path = seq_dir / "gt" / "gt.txt"
        video = common.video_path(seq_dir.name)
        if not gt_path.exists() or not video.exists():
            print(f"[warning] missing gt.txt or video for {seq_dir.name}, skipping")
            continue
        n = convert_sequence(video, gt_path, seq_dir.name)
        print(f"{seq_dir.name}: {n} frames")
        total += n
    print(f"\nTotal: {total} frames converted into {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
