"""Shared matplotlib style for the paper figures (11 pt article, 6.5 in text width).

Figures are drawn at their final printed width in a Times-compatible serif (STIX, bundled with
matplotlib) at 8 pt, so figure text matches the body text and LaTeX never scales it. Fonts are
embedded as Type 42 (TrueType) rather than Type 3, which arXiv and journal PDF checkers accept.
Call `apply()` before importing pyplot; `save()` writes a timestamp-free PDF for LaTeX and a
300 dpi PNG for the README and the web reader.

Colour is assigned by the job it does. Ordered series (depth L, qubits n, QAOA depth p, the width
of a Gaussian initialisation) take a single-hue blue ramp, light to dark. Unordered series
(the four ansatz families, global versus local cost) take fixed categorical hues. References from
theory (Haar, Page, Gaussian, su(2^n)) are black dashed lines. All palettes were checked with the
colour-vision-deficiency validator of the dataviz skill; where two hues sit in its warning band
the series also differ by marker shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TEXT_IN = 6.5  # \textwidth for article, 11 pt, 1 in margins (letter)

# Ordinal ramps (one hue, light -> dark, interpolated in OKLab); index by the number of steps.
RAMPS = {
    3: ["#86b6ef", "#2a78d6", "#0d366b"],
    4: ["#86b6ef", "#5a84b8", "#315583", "#0b2a52"],
    5: ["#86b6ef", "#6590c5", "#466c9d", "#284a77", "#0b2a52"],
    6: ["#86b6ef", "#6b98ce", "#527bad", "#395e8e", "#22446f", "#0b2a52"],
}
# Categorical hues: blue, yellow, aqua, red (validated all-pairs; markers differ as well).
CAT = ["#2a78d6", "#eda100", "#1baf7a", "#e34948"]
CAT_MARKERS = ["o", "s", "^", "D"]
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
TEXT = "#333333"
MUTED = "#52514e"
LIGHT = "#898781"
GRID = "#e1e0d9"


def ramp(n: int) -> list[str]:
    if n not in RAMPS:
        raise ValueError(f"no validated ramp with {n} steps")
    return RAMPS[n]


def apply() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "axes.linewidth": 0.6,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlelocation": "left",
            "axes.titleweight": "normal",
            "axes.titlepad": 5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": TEXT,
            "ytick.labelcolor": TEXT,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.4,
            "lines.markeredgewidth": 0.5,
            "lines.markeredgecolor": "white",
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "legend.labelspacing": 0.3,
            "legend.borderaxespad": 0.3,
            "errorbar.capsize": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "bp-from-scratch",
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.w_pad": 0.04,
            "figure.constrained_layout.h_pad": 0.04,
            "figure.constrained_layout.wspace": 0.06,
        }
    )


# Keyword arguments for the common mark types.
LINE = {"linewidth": 1.0}
ERR = {"elinewidth": 0.6, "capthick": 0.6, "capsize": 1.5}
REF = {"color": INK, "linestyle": (0, (4, 2)), "linewidth": 0.9, "marker": None, "zorder": 1.5}


def panel(ax: Any, letter: str, title: str) -> None:
    """Bold panel letter followed by a short title, left-aligned above the axes."""
    ax.set_title(rf"$\bf{{({letter})}}$  {title}")


def refline(ax: Any, x: Any, y: Any, label: str | None = None, **kw: Any) -> None:
    ax.plot(x, y, label=label, **{**REF, **kw})


def save(fig: Any, pdf_dir: Path, png_dir: Path, stem: str) -> None:
    """PDF without timestamps (rebuilds are byte-stable) and a 300 dpi PNG."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png_dir / f"{stem}.png", dpi=300, metadata={"Software": None})
