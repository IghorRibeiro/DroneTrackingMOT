#!/usr/bin/env bash
# Statistics of the paper: paired bootstrap + Wilcoxon (HIT-Tracker vs baseline, ...), half/half selection simulations
# (gamma and SAHI tile, 1,000 splits), kinematic consistency check (p = 0.207).
# Per-sequence counts come from outputs/verify_*/per_sequence when a verification was run, otherwise from results/per_sequence.
# Finally every statistic quoted in the text of the paper is checked (src/analysis/check_paper_statistics.py).
# Usage: bash scripts/statistics.sh [--no-kinematics]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
PER_SEQ="$REPO_ROOT/results/per_sequence"
if [[ -d "$OUT_ROOT/verify_all/per_sequence" ]]; then PER_SEQ="$OUT_ROOT/verify_all/per_sequence"; fi
$PY -m src.analysis.paired_tests --per-sequence "$PER_SEQ" --out "$OUT_ROOT/statistics"
$PY -m src.analysis.selection_halves --per-sequence "$PER_SEQ" --out "$OUT_ROOT/statistics"
$PY -m src.analysis.root_cause --per-sequence "$PER_SEQ" --out "$OUT_ROOT/statistics" || true
if [[ "${1:-}" != "--no-kinematics" ]]; then
  $PY -m src.analysis.kinematics --out "$OUT_ROOT/statistics"
fi
$PY -m src.analysis.check_paper_statistics --statistics "$OUT_ROOT/statistics" --per-sequence "$PER_SEQ" || true
