#!/usr/bin/env bash
# Detector pipeline: dataset conversion, sequence-level split, base detector, hard-example selection, HEM dataset and HEM fine-tuning.
# The released checkpoints (weights/best.pt, weights/best_hardmining.pt) were trained on a single NVIDIA Tesla T4 GPU with the arguments
# in configs/base_train.yaml and configs/hem_train.yaml; retraining on other hardware gives close, not bit-identical, checkpoints.
# Usage: bash scripts/train_detectors.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
$PY -m src.data.convert_mot_to_yolo
$PY -m src.data.split_train_val
$PY -m src.detector.train_detector --config configs/base_train.yaml --name yolo26n_baseline
# Hard-example mining needs the base detector in weights/best.pt (copy runs/detect/yolo26n_baseline/weights/best.pt there).
# The list used for the released HEM detector is already in configs/splits/hem_hard_sequences_20.txt; to re-derive it run:
#   python -m src.detector.select_hard_examples --out outputs/hem_hard_sequences_20.txt
$PY -m src.detector.build_hem_dataset
$PY -m src.detector.train_detector --config configs/hem_train.yaml --name yolo26n_hem
