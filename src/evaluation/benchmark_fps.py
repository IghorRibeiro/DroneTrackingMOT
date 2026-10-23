"""End-to-end FPS benchmark (Tables 2 and 3, Section 4.1).

FPS values are only comparable WITHIN one measurement session on one machine, so all points that are compared must be
measured in a single call of this script, back to back, with the machine otherwise idle.

Protocol (identical for every point):
  * end-to-end time = video decoding + detector (+ SAHI slicing/merging) + tracker + MOT-line formatting,
    measured per sequence with ``time.perf_counter()``; FPS = frames / seconds, aggregated as total frames / total seconds;
  * a warm-up (dummy frames through the detector) before each point and before each fresh model, excluded from timing;
  * a NEW model / tracker for every sequence (the same clean-run protocol used for the accuracy numbers);
  * the trajectories produced during timing are stored under ``<out>/trajectories/<point>`` and compared, count by count,
    with ``trajectories/<point>`` when it exists (timing must not change the tracking result);
  * system load (1-minute load average) and power state are logged per sequence; the script never aborts because of load,
    it records it. Keep the machine plugged in, with low-power mode off and no other heavy process running.

Usage:
    python -m src.evaluation.benchmark_fps --points hem_sahi512 hit botsort bytetrack fasttrack ocsort tracktrack deepocsort
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import common
from src.evaluation import mot_eval
from src.tracking import registry
from src.tracking import run_sahi_tracker as sahi_runner

DEFAULT_POINTS = ["hem_sahi512", "hit", "botsort", "bytetrack", "fasttrack", "ocsort", "tracktrack", "deepocsort"]


def power_info() -> dict:
    """macOS power state (informative only)."""
    if platform.system() != "Darwin":
        return {"platform": platform.platform()}
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    full = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    low = [ln for ln in full.splitlines() if "lowpowermode" in ln]
    return {"platform": platform.platform(), "on_ac": "AC Power" in batt, "battery": batt.strip().replace("\n", " | "),
            "low_power_mode": low[0].split()[-1] if low else "unknown"}


def loadavg1() -> float:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return float("nan")


def time_yolo_point(exp_id: str, spec: dict, seqs: list[str], tdir: Path, max_frames: int | None, device: str) -> list[dict]:
    from ultralytics import YOLO

    weights = common.WEIGHTS[spec["weights"]]
    tracker = str(registry.tracker_yaml(spec["tracker"]))
    blank = np.zeros((common.FRAME_H, common.FRAME_W, 3), dtype=np.uint8)
    warm = YOLO(str(weights))
    for _ in range(5):
        warm.predict(blank, device=device, verbose=False)
    del warm
    rows = []
    for seq in seqs:
        model = YOLO(str(weights))
        for _ in range(2):
            model.predict(blank, device=device, verbose=False)         # warm-up of the fresh model, outside the timer
        la0 = loadavg1()
        lines, n = [], 0
        t0 = time.perf_counter()
        for idx, r in enumerate(model.track(source=str(common.video_path(seq)), conf=0.25, tracker=tracker, persist=True,
                                            stream=True, device=device, verbose=False), start=1):
            n = idx
            if r.boxes.id is not None:
                for (cx, cy, w, h), tid in zip(r.boxes.xywh.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int)):
                    lines.append(common.mot_line(idx, int(tid), cx - w / 2, cy - h / 2, w, h))
            if max_frames and idx >= max_frames:
                break
        el = time.perf_counter() - t0
        common.write_atomic(tdir / f"{seq}.txt", "\n".join(lines))
        rows.append(dict(sequence=seq, n_frames=n, elapsed_seconds=el, fps=n / el, loadavg_start=la0, loadavg_end=loadavg1()))
        print(f"[{exp_id}] {seq}: {n} frames, {n / el:.1f} FPS | load {la0:.2f}->{loadavg1():.2f}", flush=True)
    return rows


def time_sahi_point(exp_id: str, spec: dict, seqs: list[str], tdir: Path, max_frames: int | None, device: str) -> list[dict]:
    sahi_cfg = registry.sahi_parameters(spec["tile"])
    det_model = sahi_runner.make_detector(common.WEIGHTS[spec["weights"]], device, sahi_cfg["conf"])
    blank = np.zeros((common.FRAME_H, common.FRAME_W, 3), dtype=np.uint8)
    for _ in range(3):
        sahi_runner.build_sahi_detections(blank, det_model, sahi_cfg["slice"], sahi_cfg["overlap"], sahi_cfg["conf"], sahi_cfg["match_threshold"])
    tracker_name = "botsort_gm_nsa" if spec.get("gm") is not None else "botsort"
    undo = sahi_runner.install_gm_nsa(spec["gm"], tracker_name) if spec.get("gm") is not None else (lambda: None)
    args = sahi_runner.load_tracker_args(tracker_name)
    rows = []
    try:
        for seq in seqs:
            la0 = loadavg1()
            t0 = time.perf_counter()
            lines, n = sahi_runner.track_sequence(seq, det_model, args, sahi_cfg, device, max_frames)
            el = time.perf_counter() - t0
            common.write_atomic(tdir / f"{seq}.txt", "\n".join(lines))
            rows.append(dict(sequence=seq, n_frames=n, elapsed_seconds=el, fps=n / el, loadavg_start=la0, loadavg_end=loadavg1()))
            print(f"[{exp_id}] {seq}: {n} frames, {n / el:.1f} FPS | load {la0:.2f}->{loadavg1():.2f}", flush=True)
    finally:
        undo()
    return rows


def crosscheck(exp_id: str, seqs: list[str], tdir: Path, max_frames: int | None) -> dict:
    ref = common.TRAJ_ROOT / exp_id
    if not ref.exists():
        return dict(point=exp_id, reference="(none)", identical=None, detail="no reference trajectories")
    keys = mot_eval.COUNT_COLS
    bad = []
    for s in seqs:
        a = mot_eval.eval_sequence(s, tdir / f"{s}.txt", max_frames)
        b = mot_eval.eval_sequence(s, ref / f"{s}.txt", max_frames)
        if {k: a[k] for k in keys} != {k: b[k] for k in keys}:
            bad.append(s)
    return dict(point=exp_id, reference=f"trajectories/{exp_id}", identical=not bad, detail=",".join(bad[:5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="*", default=DEFAULT_POINTS)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, default=common.OUT_ROOT / "fps")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    seqs = a.sequences or common.val_sequences()
    device = a.device or common.device()
    import ultralytics
    session = dict(date=dt.datetime.now().isoformat(timespec="seconds"), device=device, ultralytics=ultralytics.__version__,
                   points=a.points, power=power_info(), loadavg_start=loadavg1())
    print(f"session: {json.dumps(session, default=str)}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    summary, checks = [], []
    for exp_id in a.points:
        spec = registry.get(exp_id)
        tdir = a.out / "trajectories" / exp_id
        t0 = time.time()
        rows = (time_sahi_point if spec["runner"] == "sahi" else time_yolo_point)(exp_id, spec, seqs, tdir, a.max_frames, device)
        t = pd.DataFrame(rows)
        t.to_csv(a.out / f"fps_timing_{exp_id}.csv", index=False)
        cv = float(t.fps.std(ddof=0) / t.fps.mean() * 100) if len(t) > 1 else float("nan")
        summary.append(dict(point=exp_id, device=device, n_seq=len(t), frames=int(t.n_frames.sum()), seconds=float(t.elapsed_seconds.sum()),
                            fps_global=float(t.n_frames.sum() / t.elapsed_seconds.sum()), fps_median=float(t.fps.median()),
                            fps_min=float(t.fps.min()), fps_max=float(t.fps.max()), cv_pct=cv, loadavg_max=float(t.loadavg_end.max()),
                            wall_minutes=(time.time() - t0) / 60, date=dt.datetime.now().strftime("%Y-%m-%d %H:%M")))
        print(f"[{exp_id}] global FPS = {summary[-1]['fps_global']:.2f} (median {summary[-1]['fps_median']:.2f}, CV {cv:.1f}%)", flush=True)
        checks.append(crosscheck(exp_id, seqs, tdir, a.max_frames))
        print(f"[{exp_id}] trajectories vs reference: {checks[-1]}", flush=True)
        pd.DataFrame(summary).to_csv(a.out / "fps_summary.csv", index=False)
        pd.DataFrame(checks).to_csv(a.out / "fps_crosscheck.csv", index=False)
    (a.out / "fps_session.json").write_text(json.dumps(session, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
