"""Common style of the four paper figures (print size, Springer requirements).

  * serif font (STIXGeneral, bundled with matplotlib) in all figures, embedded as TrueType (``pdf.fonttype`` 42, no Type 3);
  * every text >= 7 pt at the final print size, lines >= 0.6 pt;
  * series are also told apart by marker / line style / hatch (legible in black and white);
  * true minus sign (U+2212) in negative numbers; no internal titles (the LaTeX captions play that role);
  * the PDF page is exactly the figure size (nothing is saved with ``bbox_inches="tight"``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# final print sizes (inches) of the paper figures: width, height
FIGSIZE = {1: (4.803, 1.658), 2: (3.362, 1.700), 3: (3.362, 2.156), 4: (4.803, 1.603)}
TICK, LABEL, VALUE, TEXT_PT = 7.0, 7.5, 7.0, 7.0

NAVY = "#1F2E44"
BLUE = "#2F86AB"
ORANGE = "#C8701B"
RED = "#C0392B"
GREEN = "#1E8449"
GRAY = "#6B7785"
SOFT_BLUE = "#E6F2F8"
SOFT_GREEN = "#E7F3EA"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
        "font.size": TEXT_PT, "axes.labelsize": LABEL, "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": TEXT_PT,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "lines.linewidth": 0.9, "lines.markersize": 3.6, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
    })


def new_figure(number: int):
    apply_style()
    return plt.figure(figsize=FIGSIZE[number], facecolor="white")


WEB_WIDTH_PX = 1600


def save_figure(fig, pdf_path: Path, web_png: Path | None = None) -> None:
    """Save the PDF (page = figure size) and optionally a web PNG about 1600 px wide. No tight bounding box."""
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", metadata={"Title": pdf_path.stem, "Creator": "matplotlib", "Producer": "matplotlib"})
    if web_png is not None:
        Path(web_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(web_png, format="png", dpi=WEB_WIDTH_PX / fig.get_figwidth())


def figure_texts(fig) -> list[dict]:
    """Every text drawn in the figure: string, font size (pt)."""
    out = []
    for t in fig.findobj(match=lambda a: isinstance(a, matplotlib.text.Text)):
        s = t.get_text()
        if s and t.get_visible():
            out.append({"text": s, "fontsize": float(t.get_fontsize())})
    return out
