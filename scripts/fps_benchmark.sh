#!/usr/bin/env bash
# FPS of HEM+SAHI512, HIT-Tracker and the six trackers of Table 2, measured back to back in ONE session
# (about 2 hours on an Apple M2). FPS values are only comparable within the same session on the same machine:
# keep the computer plugged in, low-power mode off and every other heavy application closed.
# Usage: bash scripts/fps_benchmark.sh [--points ID ...]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
$PY -m src.evaluation.benchmark_fps --out "$OUT_ROOT/fps" "$@"
echo "Stored reference measurements: results/fps/fps_summary.csv"
