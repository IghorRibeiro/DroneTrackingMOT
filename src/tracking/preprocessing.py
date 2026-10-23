"""Pixel-level preprocessing used in the negative-result experiments (Table 4).

Three temporal background suppression / motion-emphasis strategies, applied before detection. Each function is a
generator yielding ``(frame_index, processed_frame)`` from a video. All of them attenuate (do not zero) the static
background, to limit the domain shift with respect to the raw frames used for detector training.

Copyright (c) 2026 Ighor de Souza Ribeiro, Rigel Procópio Fernandes, Gabriel de Sapienza Luna.
Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE).
"""
from __future__ import annotations

import cv2
import numpy as np

BG_SUPPRESSION_FACTOR = 0.25  # background keeps 25 % of its brightness (not 0 %)


def _blend_with_mask(frame: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """fg_mask: 0-255, where 255 = foreground (kept) and 0 = background (attenuated)."""
    mask_norm = fg_mask.astype(np.float32) / 255.0
    mask_norm = BG_SUPPRESSION_FACTOR + mask_norm * (1.0 - BG_SUPPRESSION_FACTOR)
    mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
    return (frame.astype(np.float32) * mask_3ch).astype(np.uint8)


def mog2_frames(video_path: str, history: int = 120, var_threshold: float = 16):
    bgsub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        fg_mask = cv2.dilate(bgsub.apply(frame), kernel, iterations=1)
        yield frame_idx, _blend_with_mask(frame, fg_mask)
    cap.release()


def ema_frames(video_path: str, alpha: float = 0.05, diff_threshold: float = 15.0):
    cap = cv2.VideoCapture(video_path)
    ret, first = cap.read()
    if not ret:
        cap.release()
        return
    bg = first.astype(np.float32)
    frame_idx = 1
    yield frame_idx, first  # first frame: no history yet, passed through raw

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        diff = cv2.absdiff(frame.astype(np.float32), bg)
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)
        bg = alpha * frame.astype(np.float32) + (1 - alpha) * bg
        yield frame_idx, _blend_with_mask(frame, fg_mask)
    cap.release()


def optical_flow_frames(video_path: str, motion_threshold_std_mult: float = 1.5):
    cap = cv2.VideoCapture(video_path)
    ret, first = cap.read()
    if not ret:
        cap.release()
        return
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    frame_idx = 1
    yield frame_idx, first  # first frame: no previous pair, passed through raw

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        threshold = mag.mean() + motion_threshold_std_mult * mag.std()
        fg_mask = ((mag > threshold) * 255).astype(np.uint8)
        fg_mask = cv2.dilate(fg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        prev_gray = gray
        yield frame_idx, _blend_with_mask(frame, fg_mask)
    cap.release()


METHODS = {"mog2": mog2_frames, "ema": ema_frames, "optflow": optical_flow_frames}
