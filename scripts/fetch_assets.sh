#!/usr/bin/env bash
# Download the saved MOT trajectories of every experiment from the GitHub Release and verify their SHA-256.
# (The detector checkpoints weights/best.pt and weights/best_hardmining.pt are part of the repository.)
# Usage: bash scripts/fetch_assets.sh [TAG]        (default tag: v1.0-lafusion2026)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
TAG="${1:-v1.0-lafusion2026}"
BASE="https://github.com/IghorRibeiro/DroneTrackingMOT/releases/download/$TAG"
mkdir -p trajectories
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
curl -fL "$BASE/SHA256SUMS.txt" -o "$tmp/SHA256SUMS.txt"
curl -fL "$BASE/trajectories.zip" -o "$tmp/trajectories.zip"
( cd "$tmp" && grep ' trajectories.zip$' SHA256SUMS.txt | shasum -a 256 -c - )
unzip -q -o "$tmp/trajectories.zip" -d trajectories
echo "trajectories/ is ready."
