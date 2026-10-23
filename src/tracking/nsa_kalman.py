"""NSA-Kalman: confidence-scaled measurement noise for the BoT-SORT Kalman filter.

Ultralytics' Kalman filter already supports a per-measurement confidence in ``project``/``update`` (NSA formulation of
StrongSORT) but the default BoT-SORT track never passes it. This patch re-implements ``BOTrack.update`` and
``BOTrack.re_activate`` so that the detection confidence ``c`` reaches the filter, which scales the measurement noise as

    R' = max(1 - c, 0.05) * R

(the 0.05 floor is part of the Ultralytics implementation; see ``ultralytics/trackers/utils/kalman_filter.py``).

The update/re-activation bodies are adapted from Ultralytics (AGPL-3.0, https://github.com/ultralytics/ultralytics).

Copyright (c) 2026 Ighor de Souza Ribeiro, Rigel Procópio Fernandes, Gabriel de Sapienza Luna.
Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE).
"""
from __future__ import annotations

_originals: dict = {}


def _nsa_re_activate(self, new_track, frame_id, new_id: bool = False) -> None:
    from ultralytics.trackers.byte_tracker import TrackState

    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat)

    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_track.tlwh), confidence=new_track.score,
    )
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = True
    self.frame_id = frame_id
    if new_id:
        self.track_id = self.next_id()
    self.score = new_track.score
    self.cls = new_track.cls
    self.angle = new_track.angle
    self.idx = new_track.idx


def _nsa_update(self, new_track, frame_id) -> None:
    from ultralytics.trackers.byte_tracker import TrackState

    if new_track.curr_feat is not None:
        self.update_features(new_track.curr_feat)

    self.frame_id = frame_id
    self.tracklet_len += 1
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.convert_coords(new_track.tlwh), confidence=new_track.score,
    )
    self.state = TrackState.Tracked
    self.is_activated = True
    self.score = new_track.score
    self.cls = new_track.cls
    self.angle = new_track.angle
    self.idx = new_track.idx


def patch_nsa_kalman() -> None:
    from ultralytics.trackers.bot_sort import BOTrack

    if not _originals:
        _originals.update(update=BOTrack.update, re_activate=BOTrack.re_activate)
    BOTrack.update = _nsa_update
    BOTrack.re_activate = _nsa_re_activate


def unpatch_nsa_kalman() -> None:
    from ultralytics.trackers.bot_sort import BOTrack

    if _originals:
        BOTrack.update = _originals["update"]
        BOTrack.re_activate = _originals["re_activate"]
        _originals.clear()
