"""Build the mixed training set of the hard-example-mined (HEM) detector.

  * ALL stride-4 training images (and labels) of the 20 selected hard sequences  -> 3,846 images;
  * plus 5,000 randomly sampled stride-4 training images                          -> configs/splits/hem_easy_5000_images.txt;
  * validation = the 30 held-out sequences (unchanged).

The random component is deliberate: it limits over-specialisation to the hard subset. The file name list of the 5,000
images used for the released detector is shipped, so the exact mixture can be rebuilt (``--resample`` draws a new one).

Output: <HEM_ROOT>/images/{train,val}, <HEM_ROOT>/labels/{train,val}, <HEM_ROOT>/data.yaml
        (default HEM_ROOT = data/dataset_yolo_hem)

Usage:
    python -m src.detector.build_hem_dataset
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from src import common

HEM_ROOT = common.REPO_ROOT / "data" / "dataset_yolo_hem"


def copy_pair(stem: str, split_src: str, split_dst: str) -> bool:
    img = common.YOLO_ROOT / "images" / split_src / f"{stem}.jpg"
    lbl = common.YOLO_ROOT / "labels" / split_src / f"{stem}.txt"
    if not img.exists():
        return False
    for kind, src, ext in (("images", img, ".jpg"), ("labels", lbl, ".txt")):
        dst = HEM_ROOT / kind / split_dst
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst / f"{stem}{ext}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hard-list", type=Path, default=common.CONFIGS / "splits" / "hem_hard_sequences_20.txt")
    ap.add_argument("--easy-list", type=Path, default=common.CONFIGS / "splits" / "hem_easy_5000_images.txt")
    ap.add_argument("--resample", action="store_true", help="draw a new random sample of 5,000 training images instead of the shipped list")
    ap.add_argument("--seed", type=int, default=common.SPLIT_SEED)
    a = ap.parse_args()

    hard_seqs = set(common.read_list(a.hard_list))
    train_imgs = sorted((common.YOLO_ROOT / "images" / "train").glob("*.jpg"))
    if not train_imgs:
        raise RuntimeError(f"no training images in {common.YOLO_ROOT}/images/train; run src/data/split_train_val.py first")

    hard = [p.stem for p in train_imgs if p.stem.rsplit("_", 1)[0] in hard_seqs]
    if a.resample:
        pool = [p.stem for p in train_imgs if p.stem.rsplit("_", 1)[0] not in hard_seqs]
        random.seed(a.seed)
        easy = random.sample(pool, 5000)
    else:
        easy = [Path(n).stem for n in common.read_list(a.easy_list)]

    n_hard = sum(copy_pair(s, "train", "train") for s in hard)
    n_easy = sum(copy_pair(s, "train", "train") for s in easy)
    n_val = sum(copy_pair(p.stem, "val", "val") for p in (common.YOLO_ROOT / "images" / "val").glob("*.jpg"))
    (HEM_ROOT / "data.yaml").write_text(f"path: {HEM_ROOT.relative_to(common.REPO_ROOT)}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['drone']\n")
    print(f"hard images: {n_hard} | random images: {n_easy} | validation images: {n_val}")
    print(f"dataset written to {HEM_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
