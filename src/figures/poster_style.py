"""Shared look for the poster panels.

One place for the palette and the rcParams so every panel prints as one system.
Light surface only: these are figures for a printed poster, so there is no dark
mode to select steps for.

Palette roles (validated categorical slots 1 and 2, light mode):
  ACCENT_CNN     blue    - the sequence-CNN track
  ACCENT_GLUON   orange  - the IntaRNA-feature / LightGBM track, and the
                           shipped default wherever a panel emphasises it
Everything that is context rather than subject wears MUTED / GRID.
"""
from __future__ import annotations

import matplotlib as mpl

# -- ink -------------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e2e1dc"
FILL = "#f4f3f0"

# -- series ----------------------------------------------------------------
# The two accents are the categorical palette. Validated all-pairs on the light
# surface: CVD dE 24.7 (protan), normal-vision dE 33.6, both >= 3:1 contrast -
# every check passes. DEEMPH is context, not a series slot: it is deliberately
# gray and sits at 2.05:1, so every mark wearing it carries a direct label.
ACCENT_CNN = "#2a78d6"      # categorical slot 1
ACCENT_GLUON = "#eb6834"    # categorical slot 2
CNN_FILL = "#e8f0fc"
GLUON_FILL = "#fdeee7"
DEEMPH = "#b5b3ab"          # de-emphasised marks
DEEMPH_FILL = "#f0efec"

# Sequential blue ramp (steps 100-700), used for the pairing-matrix inset.
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]


def apply(base: float = 11.0) -> None:
    """Install the poster rcParams. `base` is the body font size in points."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": base,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.titlesize": base + 2,
        "axes.labelsize": base,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": False,
        "legend.fontsize": base - 1,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "svg.fonttype": "none",   # keep text editable in Illustrator/Inkscape
        "pdf.fonttype": 42,
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
