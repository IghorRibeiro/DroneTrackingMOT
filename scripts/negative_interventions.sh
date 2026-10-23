#!/usr/bin/env bash
# Section 3.6 - track-level interventions on top of GM + NSA (static filter, adaptive threshold, lost-query bank).
# Usage: bash scripts/negative_interventions.sh [--fast|--full]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_mode "$@"
run_experiments gm_nsa_g055 static adaptive lqb
verify_tables neg_interventions
