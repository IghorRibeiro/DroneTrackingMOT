"""Fig. 4 - qualitative comparison on sequence UAV240, frame 309: Ground truth | BoT-SORT | HIT-Tracker.

The frame is the top-ranked one of an automatic scan of every ground-truth frame of the 30 held-out sequences (the ranking favours frames
where HIT-Tracker has a large, robust advantage over the six native trackers; see results/figure_selection/fig4_frame_ranking.csv), so it
is a best-case illustration, not a typical frame. Boxes: ground truth solid white (black halo), TP green solid, FP red dashed, FN orange
dotted; no text inside the boxes. TP / FP / FN come from matching predictions to the ground truth at IoU >= 0.5 (Hungarian assignment).
Print size 4.803 x 1.603 in.

REQUIRES the original dataset frame (video + ground truth are not redistributed; see README, "Dataset") and the trajectories of
``botsort`` and ``hit`` (trajectories/ or TRAJ_ROOT).

Usage:
    python -m src.figures.fig4_qualitative --out figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

from src import common
from src.figures import style as S

SEQUENCE, FRAME = "UAV240", 309
IOU_THRESHOLD = 0.5
EXPECTED = {"gt": 20, "botsort": (5, 14, 15), "hit": (20, 0, 0)}
GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]
PRED_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "a", "b", "c"]
TP_COLOR, FP_COLOR, FN_COLOR = "#17B15A", "#E53935", "#FFA000"


def load_frame_boxes(path: Path, columns: list[str], frame: int) -> np.ndarray:
    df = pd.read_csv(path, header=None, names=columns)
    return df[df["frame"] == frame][["x", "y", "w", "h"]].to_numpy(float)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    ix1, iy1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    ix2, iy2 = np.minimum((a[:, 0] + a[:, 2])[:, None], (b[:, 0] + b[:, 2])[None, :]), np.minimum((a[:, 1] + a[:, 3])[:, None], (b[:, 1] + b[:, 3])[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(gt: np.ndarray, pred: np.ndarray, threshold: float = IOU_THRESHOLD):
    """Hungarian matching on IoU; returns (matched gt indices, matched prediction indices)."""
    if len(gt) == 0 or len(pred) == 0:
        return set(), set()
    ious = iou_matrix(gt, pred)
    rows, cols = linear_sum_assignment(1.0 - ious)
    pairs = [(r, c) for r, c in zip(rows, cols) if ious[r, c] >= threshold]
    return {r for r, _ in pairs}, {c for _, c in pairs}


def add_box(ax, box, color, lw, ls="-", z=3):
    x, y, w, h = box
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw, linestyle=ls, joinstyle="miter", zorder=z))


def draw_gt(ax, gt, missed: set[int], scale: float = 1.0):
    """Ground truth: white boxes with a black halo; missed ones (FN) get an orange dotted outline. ``scale`` multiplies all line widths."""
    for i, box in enumerate(gt):
        add_box(ax, box, "black", 1.5 * scale, z=2)          # halo keeps the white box visible on bright background
        add_box(ax, box, "white", 0.75 * scale, z=3)
        if i in missed:
            add_box(ax, box, FN_COLOR, 0.9 * scale, ls=(0, (0.9, 1.1)), z=4)


def draw_pred(ax, pred, matched: set[int], scale: float = 1.0):
    for j, box in enumerate(pred):
        if j in matched:
            add_box(ax, box, TP_COLOR, 0.75 * scale, z=5)
        else:
            add_box(ax, box, FP_COLOR, 0.9 * scale, ls=(0, (2.2, 1.2)), z=5)


def counts(gt, pred) -> tuple[int, int, int]:
    mg, mp = match(gt, pred)
    return len(mp), len(pred) - len(mp), len(gt) - len(mg)


def build(traj_root: Path):
    cap = cv2.VideoCapture(str(common.video_path(SEQUENCE)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {FRAME} of {SEQUENCE}; the dataset video is required (README, 'Dataset')")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    gt = load_frame_boxes(common.LABEL_ROOT / SEQUENCE / "gt" / "gt.txt", GT_COLS, FRAME)
    preds = {k: load_frame_boxes(traj_root / k / f"{SEQUENCE}.txt", PRED_COLS, FRAME) for k in ("botsort", "hit")}
    shown = {"gt": len(gt), **{k: counts(gt, preds[k]) for k in preds}}

    fig = S.new_figure(4)
    W, H = S.FIGSIZE[4]
    gap, side = 0.02, 0.0
    pw = (W - 2 * side - 2 * gap) / 3
    ph = pw * height / width
    bottom = 0.215                                   # inches reserved below the images for the legend
    panels = [("Ground truth", None), ("BoT-SORT", "botsort"), ("HIT-Tracker", "hit")]
    axes = []
    for i, (name, key) in enumerate(panels):
        ax = fig.add_axes([(side + i * (pw + gap)) / W, bottom / H, pw / W, ph / H])
        ax.imshow(rgb, interpolation="none", aspect="auto")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        if key is None:
            draw_gt(ax, gt, set())
            title = f"{name} (n = {len(gt)})"
        else:
            mg, mp = match(gt, preds[key])
            draw_gt(ax, gt, set(range(len(gt))) - mg)
            draw_pred(ax, preds[key], mp)
            tp, fp, fn = shown[key]
            title = f"{name} — TP {tp} | FP {fp} | FN {fn}"
        ax.set_title(title, fontsize=S.TEXT_PT, pad=1.5)
        axes.append(ax)

    leg = fig.add_axes([0, 0, 1, (bottom - 0.01) / H])
    leg.set_xlim(0, W)
    leg.set_ylim(0, 1)
    leg.axis("off")
    entries = [("GT: ground truth", "white", "-", True), ("TP: matched", TP_COLOR, "-", False),
               ("FP: unmatched", FP_COLOR, (0, (2.2, 1.2)), False), ("FN: missed", FN_COLOR, (0, (0.9, 1.1)), False)]
    renderer = fig.canvas.get_renderer()
    inch_per_px = 1.0 / fig.dpi
    x = 0.0
    items = []
    for label, color, ls, halo in entries:                     # measure the texts, then centre the whole legend row
        t = leg.text(0, 0.5, label, fontsize=S.TEXT_PT, va="center", ha="left")
        width_in = t.get_window_extent(renderer).width * inch_per_px
        items.append((label, color, ls, halo, t, width_in))
    handle, gap_in, between = 0.28, 0.06, 0.32
    total = sum(handle + gap_in + it[5] for it in items) + between * (len(items) - 1)
    x = (W - total) / 2
    for label, color, ls, halo, t, width_in in items:
        xs = [x, x + handle]
        if halo:
            leg.plot(xs, [0.5, 0.5], color="black", linewidth=2.2, solid_capstyle="butt")
        leg.plot(xs, [0.5, 0.5], color=color, linewidth=1.0 if halo else 1.1, linestyle=ls, solid_capstyle="butt")
        t.set_position((xs[1] + gap_in, 0.5))
        x = xs[1] + gap_in + width_in + between
    return fig, shown


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-root", type=Path, default=common.TRAJ_ROOT)
    ap.add_argument("--out", type=Path, default=common.REPO_ROOT / "figures")
    ap.add_argument("--png", action="store_true", help="also write the 1600 px web PNG")
    a = ap.parse_args()
    fig, shown = build(a.traj_root)
    print(f"{SEQUENCE} f{FRAME}: GT n={shown['gt']} | BoT-SORT TP/FP/FN={shown['botsort']} | HIT-Tracker TP/FP/FN={shown['hit']}")
    if (shown["gt"], shown["botsort"], shown["hit"]) != (EXPECTED["gt"], EXPECTED["botsort"], EXPECTED["hit"]):
        print(f"ERROR: counts differ from the paper's figure {EXPECTED}", file=sys.stderr)
        return 4
    S.save_figure(fig, a.out / "fig4.pdf", a.out / "fig4.png" if a.png else None)
    print(f"written {a.out / 'fig4.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
