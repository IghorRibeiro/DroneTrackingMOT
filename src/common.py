"""Shared paths, split definition and small I/O helpers.

All locations can be overridden with environment variables so that no absolute path is ever hard-coded:

    DRONE_MOT_ROOT root of all data directories below (default: the repository root)
    ANTIUAV_ROOT   directory ``MultiUAV_Train`` of the 4th Anti-UAV Challenge, Track 3 (default: $DRONE_MOT_ROOT/data/MultiUAV_Train)
    YOLO_ROOT      directory where the YOLO-format detector dataset is written (default: $DRONE_MOT_ROOT/data/dataset_yolo)
    WEIGHTS_DIR    detector checkpoints best.pt / best_hardmining.pt (default: $DRONE_MOT_ROOT/weights)
    TRAJ_ROOT      MOT-format trajectories, one directory per experiment (default: $DRONE_MOT_ROOT/trajectories)
    OUT_ROOT       scratch outputs of the analysis scripts (default: $DRONE_MOT_ROOT/outputs)
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_BASE = Path(os.environ.get("DRONE_MOT_ROOT", REPO_ROOT))
CONFIGS = REPO_ROOT / "configs"
RESULTS = REPO_ROOT / "results"

DATA_ROOT = Path(os.environ.get("ANTIUAV_ROOT", DATA_BASE / "data" / "MultiUAV_Train"))
LABEL_ROOT = DATA_ROOT / "original_label_file"      # UAV002/gt/gt.txt, ...
VIDEO_ROOT = DATA_ROOT / "TrainVideos"              # MultiUAV-002.mp4, ...
YOLO_ROOT = Path(os.environ.get("YOLO_ROOT", DATA_BASE / "data" / "dataset_yolo"))
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", DATA_BASE / "weights"))
TRAJ_ROOT = Path(os.environ.get("TRAJ_ROOT", DATA_BASE / "trajectories"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", DATA_BASE / "outputs"))

WEIGHTS = {
    "baseline": WEIGHTS_DIR / "best.pt",
    "hem": WEIGHTS_DIR / "best_hardmining.pt",
}

SPLIT_SEED = 42        # sequence-level split seed
VAL_FRACTION = 0.15    # 30 of the 200 labelled sequences are held out
FRAME_W, FRAME_H = 640, 512
SEED = 2026            # seed of every resampling procedure (bootstrap, 15/15 splits)


def seq_to_video_filename(seq: str) -> str:
    """UAV002 -> MultiUAV-002.mp4"""
    return f"MultiUAV-{re.search(r'[0-9]+', seq).group()}.mp4"


def video_path(seq: str) -> Path:
    return VIDEO_ROOT / seq_to_video_filename(seq)


def read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def split_from_rule() -> tuple[list[str], list[str]]:
    """Re-derive (train, held-out) from the annotation folders: sorted -> seed 42 -> shuffle -> first 15 % held out."""
    seqs = sorted(p.name for p in LABEL_ROOT.iterdir() if p.is_dir())
    shuffled = seqs.copy()
    random.seed(SPLIT_SEED)
    random.shuffle(shuffled)
    n_val = max(1, round(VAL_FRACTION * len(shuffled)))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def val_sequences() -> list[str]:
    """The 30 held-out sequences (configs/splits/heldout_30.txt). If the dataset is present the list is re-derived and checked."""
    listed = read_list(CONFIGS / "splits" / "heldout_30.txt")
    if LABEL_ROOT.exists():
        derived = split_from_rule()[1]
        if derived != listed:
            raise RuntimeError(f"configs/splits/heldout_30.txt differs from the seed-{SPLIT_SEED} rule: {sorted(set(derived) ^ set(listed))}")
    return listed


def train_sequences() -> list[str]:
    return read_list(CONFIGS / "splits" / "train_170.txt")


def device() -> str:
    """CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def iter_frames(video: Path, max_frames: int | None = None):
    """Yield (1-based frame index, BGR frame)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            yield i, frame
            if max_frames and i >= max_frames:
                break
    finally:
        cap.release()


def mot_line(frame: int, track_id: int, x: float, y: float, w: float, h: float) -> str:
    return f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"


def write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
