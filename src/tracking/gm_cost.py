"""Geman-McClure (GM) association cost for BoT-SORT.

The IoU distance used by BoT-SORT is blended with a robust function of the normalised centre distance l
(L2 distance between track and detection centres divided by the frame diagonal):

    rho(l) = l^2 / (l^2 + sigma^2)                      (sigma = 0.15)
    C      = (1 - gamma) * C_IoU + gamma * rho(l)       (gamma = 0.55)

The patch replaces ``BOTSORT.get_dists`` at run time; the remainder of the function (score fusion and the optional
ReID branch) follows the Ultralytics implementation, from which this file is adapted
(Ultralytics, AGPL-3.0, https://github.com/ultralytics/ultralytics).

Copyright (c) 2026 Ighor de Souza Ribeiro, Rigel Procópio Fernandes, Gabriel de Sapienza Luna.
Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE).
"""
from __future__ import annotations

import numpy as np

_original_get_dists = None


def make_gm_get_dists(gamma: float = 0.55, sigma: float = 0.15, frame_w: int = 640, frame_h: int = 512):
    from ultralytics.trackers.utils import matching

    frame_diag = float(np.sqrt(frame_w ** 2 + frame_h ** 2))

    def get_dists(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)

        if len(tracks) and len(detections):
            track_centers = np.array([t.xywh[:2] for t in tracks])
            det_centers = np.array([d.xywh[:2] for d in detections])
            diff = track_centers[:, None, :] - det_centers[None, :, :]
            l2 = np.sqrt((diff ** 2).sum(axis=2))
            l2_norm = np.clip(l2 / frame_diag, 0.0, 1.0)
            robust = (l2_norm ** 2) / (l2_norm ** 2 + sigma ** 2)
            dists = (1 - gamma) * dists + gamma * robust

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)

        return dists

    return get_dists


def patch_gm_cost(gamma: float = 0.55, sigma: float = 0.15, frame_w: int = 640, frame_h: int = 512) -> None:
    global _original_get_dists
    from ultralytics.trackers.bot_sort import BOTSORT

    if _original_get_dists is None:
        _original_get_dists = BOTSORT.get_dists
    BOTSORT.get_dists = make_gm_get_dists(gamma, sigma, frame_w, frame_h)


def unpatch_gm_cost() -> None:
    global _original_get_dists
    from ultralytics.trackers.bot_sort import BOTSORT

    if _original_get_dists is not None:
        BOTSORT.get_dists = _original_get_dists
        _original_get_dists = None
