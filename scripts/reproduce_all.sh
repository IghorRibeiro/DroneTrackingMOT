#!/usr/bin/env bash
# Reproduce every table and figure of the paper.
#
#   --fast   (default) recompute all metrics from the SAVED trajectories - no detector, minutes on any laptop.
#            Needs: the dataset ground truth (ANTIUAV_ROOT) and trajectories/ (bash scripts/fetch_assets.sh).
#   --full   re-run detector + tracker for every experiment, then do everything --fast does.
#            About 6-7 hours on an Apple M2 (MPS); needs the dataset videos and weights/ (bash scripts/fetch_assets.sh).
#   --from-results   (with --fast) no dataset and no trajectories: recompute the totals from the shipped per-sequence
#            CSVs in results/per_sequence and compare them with the paper (checks the arithmetic, not the tracker).
#
# FPS is not recomputed here (it depends on the machine): see scripts/fps_benchmark.sh and results/fps/.
# Usage: bash scripts/reproduce_all.sh [--fast|--full] [--from-results]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
FROM_RESULTS=0
ARGS=()
for a in "$@"; do [[ "$a" == "--from-results" ]] && FROM_RESULTS=1 || ARGS+=("$a"); done
parse_mode "${ARGS[@]+"${ARGS[@]}"}"

if [[ $FROM_RESULTS -eq 1 ]]; then
  $PY -m src.evaluation.verify_paper_numbers --from-shipped-results --out "$OUT_ROOT/verify_all"
  $PY -m src.analysis.root_cause --out "$OUT_ROOT/root_cause"
  $PY -m src.figures.make_figures --only 1 2 3 --out "$OUT_ROOT/figures" --per-sequence "$REPO_ROOT/results/per_sequence"
  exit 0
fi

run_experiments botsort fasttrack bytetrack ocsort deepocsort tracktrack hem \
  gm_nsa_g015 gm_nsa_g025 gm_nsa_g035 gm_nsa_g045 gm_nsa_g055 gm_nsa_g065 gm_nsa_g075 gm_nsa_g085 \
  hem_sahi320 hem_sahi384 hem_sahi448 hem_sahi512 hem_sahi576 hit mog2 ema optflow static adaptive lqb

# every number of Tables 2-5, the main table and the gamma sweep (Fig. 2), checked against the paper (tolerance 0.05 pp, IDs exact)
$PY -m src.evaluation.verify_paper_numbers --out "$OUT_ROOT/verify_all"
bash scripts/statistics.sh
# Figures 1-4 (fig4 needs the dataset video; it is skipped with a message when it is not available)
$PY -m src.figures.make_figures --out "$OUT_ROOT/figures" --per-sequence "$OUT_ROOT/verify_all/per_sequence"
echo
echo "Done. Verification report: $OUT_ROOT/verify_all/verification_report.csv"
