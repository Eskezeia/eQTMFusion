"""
eqtmfusion.visualization.style
==================================
Publication-quality figure styling following Nature-family journal
conventions: sans-serif fonts, thin axis lines, no top/right spines,
outward ticks, colorblind-safe palette (Okabe-Ito), consistent panel
labeling (A, B, C...), and multi-format export (PNG at 300+ DPI, PDF, SVG)
so figures are usable both on-screen and as vector originals for print.

Reference conventions:
  - Nature Portfolio figure guidelines: https://www.nature.com/nature/for-authors/formatting-guide
  - Okabe & Ito (2008) colorblind-safe palette
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from contextlib import contextmanager

# Okabe-Ito colorblind-safe palette (Okabe & Ito, 2008) -- standard choice
# for genomics/genetics figures where color must remain distinguishable
# under the common forms of color vision deficiency.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
PALETTE = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"],
           OKABE_ITO["orange"], OKABE_ITO["reddish_purple"], OKABE_ITO["sky_blue"],
           OKABE_ITO["yellow"], OKABE_ITO["black"]]

MODALITY_COLORS = {
    "methylation": OKABE_ITO["blue"],
    "expression": OKABE_ITO["vermillion"],
    "snps": OKABE_ITO["bluish_green"],
    "mirna": OKABE_ITO["orange"],
    "proteomics": OKABE_ITO["reddish_purple"],
    "metabolomics": OKABE_ITO["sky_blue"],
}

PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "svg.fonttype": "none",  # keep text as text in SVG, not paths
    "pdf.fonttype": 42,      # embed as TrueType, editable in Illustrator
}


@contextmanager
def publication_style():
    """Context manager applying Nature-style rcParams for the enclosed block.

    Usage:
        with publication_style():
            fig, ax = plt.subplots()
            ...
    """
    with mpl.rc_context(rc=PUBLICATION_RCPARAMS):
        yield


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.08, fontsize: int = 11):
    """Add a bold panel label (A, B, C...) in the top-left corner of an axes,
    the standard convention for multi-panel journal figures."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="top", ha="left")


def style_axes(ax):
    """Apply standard axis styling (outward ticks, thin spines) to a single axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.6)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)


def save_publication_figure(fig, filepath_no_ext: str, formats=("png", "pdf"), dpi: int = 300):
    """
    Save a figure in multiple formats at publication resolution.
    filepath_no_ext: path without extension, e.g. 'figures/Figure1_model_diagnostics'
    Returns dict of {format: saved_path}.
    """
    import os
    os.makedirs(os.path.dirname(filepath_no_ext) or ".", exist_ok=True)
    saved = {}
    for fmt in formats:
        path = f"{filepath_no_ext}.{fmt}"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved[fmt] = path
    return saved
