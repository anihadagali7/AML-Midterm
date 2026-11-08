"""Shared matplotlib style for report figures (static PNG/PDF).

Categorical hues are assigned in a FIXED order (validated reference palette, light mode);
sequential = one blue ramp; thin marks, solid hairline grid, no dual axes.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#a3a29c"

# fixed identity colors for recurring entities so a method keeps its hue across figures
PROTOCOL_COLORS = {"A": SERIES[0], "B": SERIES[1], "C": SERIES[2]}
LABEL_COLORS = dict(zip(["Safety", "Funding", "Efficacy", "Recruitment", "Other"], SERIES[:5]))


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "font.family": "DejaVu Sans",
        "axes.prop_cycle": mpl.cycler(color=SERIES), "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, path):
    fig.savefig(path)
    if path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    return path
