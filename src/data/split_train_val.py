"""Split the converted dataset into train / validation at the SEQUENCE level (no temporal leakage between splits),
after temporal subsampling with stride 4 (frames 1, 5, 9, ... of every sequence).

Result (as in the paper): 170 training sequences with 32,330 images and 30 held-out sequences with 5,724 images.
The split rule is: sorted sequence names -> ``random.seed(42)`` -> shuffle -> first round(0.15 * 200) = 30 held out.
"""
from __future__ import annotations

import random
import shutil
import sys

from src import common

ALL_IMAGES = common.YOLO_ROOT / "all" / "images"
ALL_LABELS = common.YOLO_ROOT / "all" / "labels"
FRAME_STRIDE = 4  # keep 1 frame out of 4


def subsample_by_stride(label_files, stride: int):
    """Keep frames 1, 1 + stride, 1 + 2 * stride, ... of each sequence."""
    return [f for f in label_files if int(f.stem.rsplit("_", 1)[1]) % stride == 1]


def main() -> int:
    all_label_files = sorted(ALL_LABELS.glob("*.txt"))
    if not all_label_files:
        raise RuntimeError(f"no labels found in {ALL_LABELS}; run src/data/convert_mot_to_yolo.py first")
    label_files = subsample_by_stride(all_label_files, FRAME_STRIDE)
    print(f"Frames available: {len(all_label_files)} -> after stride {FRAME_STRIDE}: {len(label_files)}")

    sequences = sorted({f.stem.rsplit("_", 1)[0] for f in label_files})
    random.seed(common.SPLIT_SEED)
    random.shuffle(sequences)
    n_val = max(1, round(common.VAL_FRACTION * len(sequences)))
    val_seqs, train_seqs = set(sequences[:n_val]), set(sequences[n_val:])
    print(f"Train: {len(train_seqs)} sequences | validation: {len(val_seqs)} sequences")

    for split in ("train", "val"):
        (common.YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (common.YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    moved = {"train": 0, "val": 0}
    for label_path in label_files:
        split = "val" if label_path.stem.rsplit("_", 1)[0] in val_seqs else "train"
        img_path = ALL_IMAGES / f"{label_path.stem}.jpg"
        if not img_path.exists():
            print(f"[warning] image missing for {label_path.stem}, skipping")
            continue
        shutil.copy2(img_path, common.YOLO_ROOT / "images" / split / img_path.name)
        shutil.copy2(label_path, common.YOLO_ROOT / "labels" / split / label_path.name)
        moved[split] += 1
    print(f"Copied -> train: {moved['train']} images | validation: {moved['val']} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
