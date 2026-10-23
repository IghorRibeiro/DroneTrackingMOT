"""Density-adaptive confidence threshold (Table 5).

Per frame, the first-stage association threshold ``track_high_thresh`` is set from the mean confidence of the frame's
detections:

    mean(conf) < 0.40  ->  0.20        mean(conf) > 0.70  ->  0.30        otherwise  ->  0.25 (default)

Ultralytics 8.4.113 exposes the detections to ``BYTETracker._split_detections(self, results)`` as a ``Results``-like
object (``results.conf``). The wrapper below uses exactly that signature. Every call is logged (frame, mean
confidence, effective threshold) so that a run can prove the patch was active; see ``run_yolo_tracker.py --item adaptive``.

Copyright (c) 2026 Ighor de Souza Ribeiro, Rigel Procópio Fernandes, Gabriel de Sapienza Luna.
Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE).
"""
from __future__ import annotations

import numpy as np

BASE_THRESH = 0.25
LOW_THRESH = 0.20     # mean confidence < 0.40
HIGH_THRESH = 0.30    # mean confidence > 0.70
LOW_CONF, HIGH_CONF = 0.40, 0.70


def effective_threshold(mean_conf: float) -> float:
    if mean_conf < LOW_CONF:
        return LOW_THRESH
    if mean_conf > HIGH_CONF:
        return HIGH_THRESH
    return BASE_THRESH


def patch_adaptive_threshold(log_rows: list | None = None, state: dict | None = None):
    """Install the wrapper; returns a function that restores the original method.

    ``log_rows`` collects ``(sequence, frame, mean_conf, effective_threshold, n_detections)`` tuples; ``state`` is a
    dict with the keys ``seq`` and ``frame`` updated by the caller before each frame.
    """
    from ultralytics.trackers.byte_tracker import BYTETracker

    original = BYTETracker._split_detections
    state = state if state is not None else {"seq": "", "frame": 0}

    def adaptive(self, results):
        scores = np.asarray(results.conf, dtype=float).reshape(-1)
        if scores.size:
            mean_conf = float(scores.mean())
            thr = effective_threshold(mean_conf)
        else:
            mean_conf, thr = float("nan"), BASE_THRESH
        previous = self.args.track_high_thresh
        self.args.track_high_thresh = thr
        if log_rows is not None:
            log_rows.append((state["seq"], state["frame"], mean_conf, thr, int(scores.size)))
        try:
            return original(self, results)
        finally:
            self.args.track_high_thresh = previous

    BYTETracker._split_detections = adaptive
    return lambda: setattr(BYTETracker, "_split_detections", original)
