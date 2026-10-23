#!/usr/bin/env bash
# Data of Fig. 2 - gamma sweep of the GM association cost (8 values x 30 sequences).
# Usage: bash scripts/fig2_gamma_sweep.sh [--fast|--full]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_mode "$@"
run_experiments gm_nsa_g015 gm_nsa_g025 gm_nsa_g035 gm_nsa_g045 gm_nsa_g055 gm_nsa_g065 gm_nsa_g075 gm_nsa_g085
verify_tables fig2
