#!/usr/bin/env bash
# Table 2 - native tracker comparison (YOLO26n baseline detector, default configurations). FPS: see fps_benchmark.sh.
# Usage: bash scripts/table2_trackers.sh [--fast|--full]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_mode "$@"
run_experiments botsort fasttrack bytetrack ocsort deepocsort tracktrack
verify_tables table2
