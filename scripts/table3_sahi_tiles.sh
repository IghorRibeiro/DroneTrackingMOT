#!/usr/bin/env bash
# Table 3 - SAHI tile-size sweep on the HEM detector with default BoT-SORT (relative FPS: value of the paper).
# Usage: bash scripts/table3_sahi_tiles.sh [--fast|--full]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_mode "$@"
run_experiments hem_sahi320 hem_sahi384 hem_sahi448 hem_sahi512 hem_sahi576
verify_tables table3
