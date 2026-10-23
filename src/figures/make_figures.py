"""Generate the four figures of the paper at their final print size.

  fig1.pdf  root-cause correlations (IDF1 vs density, camera motion, edge density, local contrast)   4.803 x 1.658 in
  fig2.pdf  gamma sweep of the GM association cost                                                    3.362 x 1.700 in
  fig3.pdf  evolution from the baseline to HIT-Tracker                                                3.362 x 2.156 in
  fig4.pdf  qualitative comparison (UAV240, frame 309) - needs the dataset video, which is not redistributed   4.803 x 1.603 in

Data are read from results/ (figures 1-3) and from the dataset + trajectories/ (figure 4). The PDF page equals the figure size (no tight
bounding box). A report with the page size, the aspect-ratio error with respect to the paper layout, the font types and sizes, and every
text shown in each figure is written to <OUT_ROOT>/figures/figures_report.json.

Usage:
    python -m src.figures.make_figures                       # figures/fig1.pdf ... fig4.pdf and the 1600 px PNGs fig1.png ... fig4.png
    python -m src.figures.make_figures --only 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import common
from src.figures import check_pdf, fig1_root_cause, fig2_gamma_sweep, fig3_pipeline_evolution, fig4_qualitative
from src.figures import style as S

# aspect ratio (width / height) of the layout the figures must keep
TARGET_RATIO = {1: 2.89727, 2: 1.97793, 3: 1.55957, 4: 2.99668}


def build_figure(n: int, args):
    per_seq = args.per_sequence
    if n == 1:
        return fig1_root_cause.build(args.factors, per_seq)[0]
    if n == 2:
        sweep = fig2_gamma_sweep.load_sweep(per_seq)
        lo, hi, _ = fig2_gamma_sweep.stable_region(sweep, fig2_gamma_sweep.STABLE_DELTA_PP)
        if (round(lo, 2), round(hi, 2)) != (0.25, 0.65):
            raise RuntimeError(f"stable region [{lo}, {hi}] differs from the paper's [0.25, 0.65]")
        return fig2_gamma_sweep.build(sweep, fig2_gamma_sweep.STABLE_DELTA_PP, fig2_gamma_sweep.SELECTED_GAMMA)[0]
    if n == 3:
        return fig3_pipeline_evolution.build(fig3_pipeline_evolution.load_pipeline(per_seq))[0]
    fig, shown = fig4_qualitative.build(args.traj_root)
    if (shown["gt"], shown["botsort"], shown["hit"]) != (fig4_qualitative.EXPECTED["gt"], fig4_qualitative.EXPECTED["botsort"], fig4_qualitative.EXPECTED["hit"]):
        raise RuntimeError(f"fig4 counts {shown} differ from the paper")
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="figure numbers (default: 1 2 3, plus 4 when the dataset video of UAV240 is available)")
    ap.add_argument("--out", type=Path, default=common.REPO_ROOT / "figures")
    ap.add_argument("--per-sequence", type=Path, default=common.RESULTS / "per_sequence")
    ap.add_argument("--factors", type=Path, default=common.RESULTS / "root_cause")
    ap.add_argument("--traj-root", type=Path, default=common.TRAJ_ROOT)
    ap.add_argument("--no-png", action="store_true", help="do not write the 1600 px web PNG versions (figN.png)")
    args = ap.parse_args()
    if args.only is None:
        args.only = [1, 2, 3]
        if common.video_path(fig4_qualitative.SEQUENCE).exists():
            args.only.append(4)
        else:
            print("fig4 skipped: the dataset video of UAV240 is not available (ANTIUAV_ROOT); run with --only 4 once it is.")

    report, failures = {}, 0
    for n in args.only:
        try:
            fig = build_figure(n, args)
        except Exception as exc:  # noqa: BLE001
            print(f"fig{n}: FAILED - {exc}", file=sys.stderr)
            failures += 1
            continue
        fig.canvas.draw()
        texts = S.figure_texts(fig)
        pdf = args.out / f"fig{n}.pdf"
        S.save_figure(fig, pdf, None if args.no_png else args.out / f"fig{n}.png")
        info = check_pdf.inspect_pdf(pdf)
        ratio = info["width_in"] / info["height_in"]
        info.update(ratio=ratio, target_ratio=TARGET_RATIO[n], ratio_error_pct=100 * (ratio / TARGET_RATIO[n] - 1),
                    min_fontsize_pt=min(t["fontsize"] for t in texts), texts=[t["text"].replace("\n", " ") for t in texts])
        report[f"fig{n}"] = info
        print(f"fig{n}.pdf: {info['width_in']:.3f} x {info['height_in']:.3f} in | ratio error {info['ratio_error_pct']:+.3f}% | "
              f"Type 3 fonts: {info['type3_fonts']} | embedded {info['embedded_font_files']}/{info['font_descriptors']} | min font {info['min_fontsize_pt']:.1f} pt")
    out_json = common.OUT_ROOT / "figures" / "figures_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
