#!/usr/bin/env bash
# Data of Fig. 3 / Section 4.1 - baseline, GM + NSA, HEM, HEM + SAHI512 and HIT-Tracker.
# Usage: bash scripts/fig3_pipeline.sh [--fast|--full]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_mode "$@"
run_experiments botsort gm_nsa_g055 hem hem_sahi512 hit
verify_tables fig3
