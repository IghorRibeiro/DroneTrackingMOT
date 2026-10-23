#!/usr/bin/env bash
# Section 3.6 - pixel-level preprocessing (MOG2, EMA, dense optical flow; full tracking evaluation). The detection-only mAP50 numbers are NOT produced by this repository.
# Usage: bash scripts/negative_preprocessing.sh [--fast|--full]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_mode "$@"
run_experiments botsort mog2 ema optflow
verify_tables neg_preproc
