"""Presentation assets for the repository page, built from the Fig. 4 data (sequence UAV240, frame 309).

  assets/hero.png            Ground truth | BoT-SORT 5/20 | HIT-Tracker 20/20, redone for screen (2000 px wide, large titles)
  assets/social_preview.png  1280 x 640 social card: title + hero (upload under Settings > Social preview)
  assets/hero.gif            optional: the same sequence, BoT-SORT x HIT-Tracker side by side (about 5 s, 960 px wide, <= 8 MB)

Like Fig. 4 this needs the original dataset video and ground truth (not redistributed) and the trajectories of ``botsort`` and ``hit``.

Usage:
    python -m src.figures.hero --out assets            # hero.png + social_preview.png
    python -m src.figures.hero --out assets --gif      # + hero.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import common
from src.figures import fig4_qualitative as F4
from src.figures import style as S

TITLE_PT, LEGEND_PT = 15.5, 12.5
FRAME_RANGE = (249, 374)          # about 5 s at 25 FPS around frame 309


def read_frame(frame_number: int):
    cap = cv2.VideoCapture(str(common.video_path(F4.SEQUENCE)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_number}; the dataset video is required (README, 'Dataset')")
    return frame


def build_hero(traj_root: Path):
    S.apply_style()
    frame = read_frame(F4.FRAME)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    gt = F4.load_frame_boxes(common.LABEL_ROOT / F4.SEQUENCE / "gt" / "gt.txt", F4.GT_COLS, F4.FRAME)
    preds = {k: F4.load_frame_boxes(traj_root / k / f"{F4.SEQUENCE}.txt", F4.PRED_COLS, F4.FRAME) for k in ("botsort", "hit")}
    shown = {k: F4.counts(gt, preds[k]) for k in preds}
    W, H = 10.0, 3.52
    fig = plt.figure(figsize=(W, H), facecolor="white")
    gap, side = 0.1, 0.0
    pw = (W - 2 * gap) / 3
    ph = pw * height / width
    bottom = 0.5
    titles = [f"Ground truth: {len(gt)} drones", f"BoT-SORT: {shown['botsort'][0]}/{len(gt)} matched", f"HIT-Tracker: {shown['hit'][0]}/{len(gt)} matched"]
    for i, (key, title) in enumerate(zip((None, "botsort", "hit"), titles)):
        ax = fig.add_axes([(side + i * (pw + gap)) / W, bottom / H, pw / W, ph / H])
        ax.imshow(rgb, interpolation="none", aspect="auto")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        if key is None:
            F4.draw_gt(ax, gt, set(), scale=2.4)
        else:
            mg, mp = F4.match(gt, preds[key])
            F4.draw_gt(ax, gt, set(range(len(gt))) - mg, scale=2.4)
            F4.draw_pred(ax, preds[key], mp, scale=2.4)
        ax.set_title(title, fontsize=TITLE_PT, fontweight="bold", pad=5)
    leg = fig.add_axes([0, 0, 1, (bottom - 0.05) / H])
    leg.set_xlim(0, W)
    leg.set_ylim(0, 1)
    leg.axis("off")
    entries = [("GT: ground truth", "white", "-", True), ("TP: matched", F4.TP_COLOR, "-", False),
               ("FP: unmatched", F4.FP_COLOR, (0, (2.2, 1.2)), False), ("FN: missed", F4.FN_COLOR, (0, (0.9, 1.1)), False)]
    renderer = fig.canvas.get_renderer()
    texts = [leg.text(0, 0.5, label, fontsize=LEGEND_PT, va="center", ha="left") for label, *_ in entries]
    widths = [t.get_window_extent(renderer).width / fig.dpi for t in texts]
    handle, gap_in, between = 0.45, 0.1, 0.55
    x = (W - (sum(handle + gap_in + w for w in widths) + between * (len(entries) - 1))) / 2
    for (label, color, ls, halo), t, w in zip(entries, texts, widths):
        xs = [x, x + handle]
        if halo:
            leg.plot(xs, [0.5, 0.5], color="black", linewidth=4.2, solid_capstyle="butt")
        leg.plot(xs, [0.5, 0.5], color=color, linewidth=2.2, linestyle=ls, solid_capstyle="butt")
        t.set_position((xs[1] + gap_in, 0.5))
        x = xs[1] + gap_in + w + between
    return fig, shown


def build_social(hero_png: Path):
    """1280 x 640 card: title, subtitle and the hero image."""
    S.apply_style()
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="#101A27")
    fig.text(0.04, 0.905, "HIT-Tracker", fontsize=44, fontweight="bold", color="white", va="center")
    fig.text(0.04, 0.80, "Infrared multi-UAV tracking: hard-example mining, tiled inference, GM association cost and NSA-Kalman",
             fontsize=15.5, color="#C9D6E5", va="center")
    fig.text(0.96, 0.905, "LAFusion 2026", fontsize=20, color="#7FB7D9", va="center", ha="right", fontweight="bold")
    img = np.asarray(Image.open(hero_png).convert("RGB"))
    w = 0.92
    h = w * 12.8 * img.shape[0] / img.shape[1] / 6.4
    ax = fig.add_axes([0.04, 0.035, w, h])
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")
    return fig


def build_gif(traj_root: Path, out: Path, width_px: int = 960, step: int = 2, fps: float = 12.5) -> None:
    """Two panels side by side (BoT-SORT | HIT-Tracker) over about 5 s of UAV240, with per-frame match counts."""
    import pandas as pd
    gt_all = pd.read_csv(common.LABEL_ROOT / F4.SEQUENCE / "gt" / "gt.txt", header=None, names=F4.GT_COLS)
    pr = {k: pd.read_csv(traj_root / k / f"{F4.SEQUENCE}.txt", header=None, names=F4.PRED_COLS) for k in ("botsort", "hit")}
    cap = cv2.VideoCapture(str(common.video_path(F4.SEQUENCE)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_RANGE[0] - 1)
    panel_w = width_px // 2
    panel_h = int(panel_w * common.FRAME_H / common.FRAME_W)
    bar = 44
    scale = panel_w / common.FRAME_W
    frames = []
    for n in range(FRAME_RANGE[0], FRAME_RANGE[1]):
        ok, img = cap.read()
        if not ok:
            break
        if (n - FRAME_RANGE[0]) % step:
            continue
        gt = gt_all[gt_all.frame == n][["x", "y", "w", "h"]].to_numpy(float)
        canvas = np.full((panel_h + bar, panel_w * 2, 3), (39, 26, 16), np.uint8)
        for col, (key, name) in enumerate((("botsort", "BoT-SORT"), ("hit", "HIT-Tracker"))):
            pred = pr[key][pr[key].frame == n][["x", "y", "w", "h"]].to_numpy(float)
            mg, mp = F4.match(gt, pred)
            view = cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

            def rect(b, color, t):
                x0, y0 = int(round(b[0] * scale)), int(round(b[1] * scale))
                x1, y1 = int(round((b[0] + b[2]) * scale)), int(round((b[1] + b[3]) * scale))
                cv2.rectangle(view, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), color, t)
            for i, b in enumerate(gt):
                rect(b, (0, 0, 0), 3)
                rect(b, (255, 255, 255), 1)
                if i not in mg:
                    rect(b, (0, 160, 255), 2)                         # FN: orange (BGR)
            for j, b in enumerate(pred):
                rect(b, (90, 177, 23) if j in mp else (53, 57, 229), 2)   # TP green / FP red (BGR)
            canvas[bar:, col * panel_w:(col + 1) * panel_w] = view
            label = f"{name}: {len(mp)}/{len(gt)} matched"
            cv2.putText(canvas, label, (col * panel_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    cap.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(round(1000 / fps)), loop=0, optimize=True, disposal=1)
    print(f"written {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    if out.stat().st_size > 8e6:
        raise RuntimeError("hero.gif is larger than 8 MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-root", type=Path, default=common.TRAJ_ROOT)
    ap.add_argument("--out", type=Path, default=common.REPO_ROOT / "assets")
    ap.add_argument("--gif", action="store_true", help="also write hero.gif")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig, shown = build_hero(a.traj_root)
    assert shown["botsort"][0] == 5 and shown["hit"][0] == 20, shown
    fig.savefig(a.out / "hero.png", dpi=200, facecolor="white")
    plt.close(fig)
    build_social(a.out / "hero.png").savefig(a.out / "social_preview.png", dpi=100, facecolor="#101A27")
    plt.close("all")
    print(f"written {a.out / 'hero.png'} and {a.out / 'social_preview.png'}")
    if a.gif:
        build_gif(a.traj_root, a.out / "hero.gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
