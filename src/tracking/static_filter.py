"""Static-track filter (Table 5): post-tracking suppression of stationary tracks.

A track is suppressed, frame by frame, when the centre of its box stays almost still over a sustained window of frames
(low rolling standard deviation of the centre), which is characteristic of static background clutter (windows, doors,
facade structures) rather than of a drone. The filter operates directly on MOT-format tracker outputs; no detection or
tracking is re-run. The setting reported in the paper is ``window=45`` and ``std_threshold=2.0`` pixels.

Copyright (c) 2026 Ighor de Souza Ribeiro, Rigel Procópio Fernandes, Gabriel de Sapienza Luna.
Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import common

MOT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "a", "b", "c"]


def filter_static_tracks(df: pd.DataFrame, window: int = 30, min_periods_ratio: float = 0.6,
                         std_threshold: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(kept, suppressed)``.

    df: MOT rows with the columns frame, id, x, y, w, h.
    window: number of frames of the rolling window.
    min_periods_ratio: minimum fraction of ``window`` with valid detections for the statistic to be trusted
        (prevents short or sparse tracks from being flagged as static for lack of data).
    std_threshold: combined standard deviation of the centre (pixels) below which a track is static at that frame.
    """
    df = df.copy()
    df["cx"] = df["x"] + df["w"] / 2
    df["cy"] = df["y"] + df["h"] / 2
    suppressed = pd.Series(False, index=df.index)
    min_periods = max(2, int(window * min_periods_ratio))

    for _, group in df.groupby("id"):
        group = group.sort_values("frame")
        full_index = pd.RangeIndex(int(group["frame"].min()), int(group["frame"].max()) + 1)
        cx = pd.Series(index=full_index, dtype=float)
        cy = pd.Series(index=full_index, dtype=float)
        cx.loc[group["frame"].values] = group["cx"].values
        cy.loc[group["frame"].values] = group["cy"].values
        std_cx = cx.rolling(window, min_periods=min_periods).std()
        std_cy = cy.rolling(window, min_periods=min_periods).std()
        combined = np.sqrt(std_cx.fillna(np.inf) ** 2 + std_cy.fillna(np.inf) ** 2)
        is_static = combined < std_threshold
        for idx, frame in zip(group.index, group["frame"]):
            if is_static.get(frame, False):
                suppressed.loc[idx] = True

    return df[~suppressed].drop(columns=["cx", "cy"]), df[suppressed].drop(columns=["cx", "cy"])


def df_to_mot_lines(df: pd.DataFrame) -> list[str]:
    return [common.mot_line(int(r.frame), int(r.id), r.x, r.y, r.w, r.h) for r in df.itertuples(index=False)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply the static-track filter to MOT trajectories.")
    ap.add_argument("--src", type=Path, required=True, help="input trajectory directory (GM+NSA, gamma=0.55)")
    ap.add_argument("--dst", type=Path, required=True, help="output trajectory directory")
    ap.add_argument("--window", type=int, default=45)
    ap.add_argument("--std-threshold", type=float, default=2.0)
    a = ap.parse_args()
    for seq in common.val_sequences():
        df = pd.read_csv(a.src / f"{seq}.txt", header=None, names=MOT_COLS)
        kept, _ = filter_static_tracks(df, window=a.window, std_threshold=a.std_threshold)
        common.write_atomic(a.dst / f"{seq}.txt", "\n".join(df_to_mot_lines(kept)))
        print(f"{seq}: {len(df)} -> {len(kept)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
