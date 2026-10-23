"""Fast reproduction check: recompute every tracking number of the paper from saved MOT trajectories and compare it with
the value printed in the paper (``results/paper_numbers.csv``).

For each experiment the trajectories ``<traj-root>/<id>/<sequence>.txt`` are evaluated against the ground truth with
py-motmetrics (IoU >= 0.5, full sequences). Checks:

  * MOTA and IDF1 (unrounded) within ``--tol`` percentage points of the printed one-decimal value (default 0.05);
  * IDs exactly equal;
  * FPS (Table 2 and Section 4.1) cannot be recomputed without running the detector: the printed values are checked against the stored
    benchmark logs ``results/fps/fps_summary.csv``. Table 3 reports the throughput relative to the 512 px tile (``rel_fps``), measured within the
    tile-sweep session; those values are echoed, not recomputed;
  * the per-sequence counts equal the shipped ``results/per_sequence/<id>.csv``.

Exit status 0 when everything matches, 1 otherwise.

Usage:
    python -m src.evaluation.verify_paper_numbers --traj-root trajectories
    python -m src.evaluation.verify_paper_numbers --tables table2 table3         # a subset of the paper's tables / figures
    python -m src.evaluation.verify_paper_numbers --from-shipped-results      # no dataset needed: recompute from results/per_sequence
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import common
from src.evaluation import mot_eval

COUNT_COLS = mot_eval.COUNT_COLS


def evaluate_ids(ids: list[str], traj_root: Path | None, shipped: Path, out_dir: Path | None) -> dict[str, pd.DataFrame]:
    per_seq: dict[str, pd.DataFrame] = {}
    for exp_id in ids:
        if traj_root is None:
            per_seq[exp_id] = pd.read_csv(shipped / f"{exp_id}.csv")
            continue
        df = mot_eval.evaluate_dir(traj_root / exp_id)
        per_seq[exp_id] = df
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / f"{exp_id}.csv", index=False)
        s = mot_eval.summarize(df)
        print(f"  evaluated {exp_id:<14} MOTA={s['mota_pct']:.3f} IDF1={s['idf1_pct']:.3f} IDs={s['ids']}", flush=True)
    return per_seq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-root", type=Path, default=common.TRAJ_ROOT)
    ap.add_argument("--from-shipped-results", action="store_true", help="skip trajectory evaluation; use results/per_sequence")
    ap.add_argument("--tol", type=float, default=0.05, help="tolerance (percentage points) for MOTA / IDF1")
    ap.add_argument("--fps-tol", type=float, default=0.005)
    ap.add_argument("--out", type=Path, default=common.OUT_ROOT / "verify")
    ap.add_argument("--tables", nargs="*", default=None, help="only these rows of paper_numbers.csv (table2 table3 neg_preproc neg_interventions fig3 fig2)")
    a = ap.parse_args()

    shipped = common.RESULTS / "per_sequence"
    paper = pd.read_csv(common.RESULTS / "paper_numbers.csv")
    if a.tables:
        paper = paper[paper["table"].isin(a.tables)]
    ids = list(dict.fromkeys(paper["id"]))
    per_seq = evaluate_ids(ids, None if a.from_shipped_results else a.traj_root, shipped, a.out / "per_sequence")
    fps = pd.read_csv(common.RESULTS / "fps" / "fps_summary.csv").set_index("point")["fps_global"]

    rows, failures = [], 0
    for r in paper.itertuples(index=False):
        s = mot_eval.summarize(per_seq[r.id])
        checks = {"MOTA": abs(s["mota_pct"] - r.mota) <= a.tol + 1e-9, "IDF1": abs(s["idf1_pct"] - r.idf1) <= a.tol + 1e-9, "IDs": s["ids"] == r.ids}
        if not a.from_shipped_results:
            ref = pd.read_csv(shipped / f"{r.id}.csv").set_index("seq").sort_index()
            got = per_seq[r.id].set_index("seq").sort_index()
            checks["counts==shipped"] = bool((ref[COUNT_COLS].astype(int) == got[COUNT_COLS].astype(int)).all().all())
        fps_note = ""
        if pd.notna(r.fps):
            if r.table in ("table2", "fig3") and r.id in fps.index:
                checks["FPS"] = abs(float(fps[r.id]) - r.fps) <= a.fps_tol + 1e-9
                fps_note = f"{fps[r.id]:.2f} (stored benchmark)"
            else:
                fps_note = f"{r.fps:.2f} (value of the paper, not recomputable here)"
        if pd.notna(r.rel_fps):
            fps_note = f"rel. FPS {r.rel_fps:.2f} (value of the paper, not recomputable here)"
        ok = all(checks.values())
        failures += 0 if ok else 1
        rows.append(dict(table=r.table, id=r.id, paper=f"{r.mota:.1f}/{r.idf1:.1f}/{r.ids}", recomputed=f"{s['mota_pct']:.2f}/{s['idf1_pct']:.2f}/{s['ids']}",
                         fps=fps_note, status="OK" if ok else "DIFFERS " + ",".join(k for k, v in checks.items() if not v)))
    report = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(a.out / "verification_report.csv", index=False)
    print()
    print(report.to_string(index=False))
    print(f"\n{len(report) - failures}/{len(report)} rows match the paper (tolerance {a.tol} pp)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
