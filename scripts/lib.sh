#!/usr/bin/env bash
# Shared helpers for the reproduction scripts. Source it; do not run it.
#
# Environment (all optional):
#   DRONE_MOT_ROOT root of the data directories             (default: the repository root)
#   ANTIUAV_ROOT  MultiUAV_Train directory of the dataset      (default: data/MultiUAV_Train)
#   TRAJ_ROOT     MOT trajectories, one directory per run      (default: trajectories)
#   WEIGHTS_DIR   detector checkpoints                         (default: weights)
#   OUT_ROOT      outputs of the analysis scripts              (default: outputs)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONDONTWRITEBYTECODE=1
export OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs}"
PY="${PYTHON:-python}"

# MODE: "fast" re-evaluates saved trajectories (no detector); "full" re-runs detector + tracker first.
MODE="fast"
parse_mode() {
  for arg in "$@"; do
    case "$arg" in
      --fast) MODE="fast" ;;
      --full) MODE="full" ;;
      -h|--help) echo "usage: $0 [--fast|--full]   (default: --fast)"; exit 0 ;;
      *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
  done
}

# run_experiments ID...  : in full mode (re)compute the trajectories of every id
run_experiments() {
  if [[ "$MODE" == "full" ]]; then
    for id in "$@"; do
      echo ">>> running experiment '$id'"
      $PY -m src.tracking.run_experiment --id "$id"
    done
  fi
}

# verify_tables TABLE...  : evaluate saved trajectories and compare with the numbers printed in the paper
verify_tables() {
  $PY -m src.evaluation.verify_paper_numbers --tables "$@" --out "$OUT_ROOT/verify_$(echo "$@" | tr ' ' '_')"
}
