"""Fine-tune the YOLO26n detector (base detector or hard-example-mined detector).

Hyper-parameters are read from a YAML file (configs/base_train.yaml or configs/hem_train.yaml); they are the training
arguments recorded inside the released checkpoints. The paper's training runs used a single NVIDIA Tesla T4.

Usage:
    python -m src.detector.train_detector --config configs/base_train.yaml --name yolo26n_baseline
    python -m src.detector.train_detector --config configs/hem_train.yaml  --name yolo26n_hem
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src import common

NON_TRAINER_KEYS = {"hard_selection"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--name", required=True, help="run name (outputs go to runs/detect/<name>)")
    ap.add_argument("--device", default=None, help="override the device (default: CUDA > MPS > CPU)")
    a = ap.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(a.config.read_text())
    model_path = Path(cfg.pop("model"))
    model = YOLO(str(model_path if model_path.is_absolute() or model_path.exists() else model_path.name))
    for k in NON_TRAINER_KEYS:
        cfg.pop(k, None)
    cfg["data"] = str(common.REPO_ROOT / cfg["data"])
    cfg["device"] = a.device or common.device()
    model.train(name=a.name, project=str(common.REPO_ROOT / "runs" / "detect"), **cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
