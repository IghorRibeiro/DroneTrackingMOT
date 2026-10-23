#!/usr/bin/env bash
# Regenerate the four figures of the paper at final print size -> figures/fig1.pdf ... fig4.pdf, plus 1600 px web PNGs fig1.png ... fig4.png.
#   fig1  root-cause correlations     (results/root_cause + results/per_sequence)
#   fig2  gamma sweep                 (results/per_sequence)
#   fig3  pipeline evolution          (results/per_sequence)
#   fig4  qualitative UAV240 f309     (needs the ORIGINAL dataset video + ground truth, which are not redistributed, and trajectories/)
# Usage: bash scripts/make_figures.sh [--only N ...]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
$PY -m src.figures.make_figures "$@"
