#!/usr/bin/env python3
"""
Panel 1 (icon version) - the method schematic with pictograms for its text.

Same two tracks and the same numbers as panel1_method_schematic.py, which it
imports from so nothing is retyped: the conv block count from the CNN
checkpoint, the bag size from the CatBoost fold directories, the feature counts
from the five groups that partition SELECTED_26, and a real CNN true positive
for the pairing matrix.

What changed is the ink. Lines of description are replaced by pictograms where
a shape carries the idea - stacked feature maps for the conv blocks, a small
network for the head, a paired duplex for IntaRNA, one glyph per feature family,
a bag of trees for CatBoost, a sigmoid for the probability output - leaving only
names, counts and the two APS values as text. It is also wider and shorter than
the original, because it is meant to span the full poster width.

The canvas is in inches with equal aspect, so every glyph keeps its drawn
proportions instead of being stretched by the figure's shape.

    python src/figures/panel1_icons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                Polygon, Rectangle)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402
from panel1_method_schematic import (BAG_DIR, FEATURE_GROUPS, SEED,  # noqa: E402
                                     check_feature_groups, load_model_args,
                                     pairing_matrix, pick_pair)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"

FIG_W, FIG_H = 16.0, 5.2
BLUE, ORANGE = ps.ACCENT_CNN, ps.ACCENT_GLUON

# Box bands (inches) for the two tracks, and the four shared stage columns.
A_Y0, A_Y1 = 2.85, 4.85
B_Y0, B_Y1 = 0.15, 2.15
COLS = [(2.55, 5.05), (5.55, 9.85), (10.35, 12.55), (13.05, 15.85)]
TITLE_DROP = 0.33   # from box top to title baseline region

# One-word names for the five feature families, in FEATURE_GROUPS order.
FAMILY_WORD = ["background", "conservation", "ΔG", "ensemble", "pairing"]


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def box(ax, x0, x1, y0, y1, title, accent, fill, title_size=11.5):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                lw=1.6, edgecolor=accent, facecolor=fill, zorder=2))
    ax.text((x0 + x1) / 2, y1 - 0.12, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=ps.INK, zorder=4)
    # content area below the title
    return x0 + 0.15, x1 - 0.15, y0 + 0.12, y1 - TITLE_DROP - 0.12


def arrow(ax, p0, p1, colour, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                                 lw=1.7, color=colour, shrinkA=0, shrinkB=0,
                                 zorder=5, connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------------------
# Pictograms - each draws inside the box (x0, x1, y0, y1) it is given
# ---------------------------------------------------------------------------

def strand(ax, x, y, length, colour, n_ticks, tick=0.07, lw=2.2):
    """An RNA strand: a backbone with one tick per few bases."""
    ax.plot([x, x + length], [y, y], color=colour, lw=lw, solid_capstyle="round",
            zorder=4)
    for t in np.linspace(x + 0.04, x + length - 0.04, n_ticks):
        ax.plot([t, t], [y, y - tick], color=colour, lw=1.1, zorder=4)


def icon_input(ax, x0, x1, y0, y1):
    cx = (x0 + x1) / 2
    mi_len, mre_len = 0.95, 1.55
    strand(ax, cx - mi_len / 2, y1 - 0.62, mi_len, ps.INK_2, 7)
    ax.text(cx, y1 - 0.46, "miRNA ≤ 30 nt", ha="center", va="bottom",
            fontsize=9, color=ps.INK_2, zorder=4)
    strand(ax, cx - mre_len / 2, y0 + 0.42, mre_len, ps.INK_2, 11)
    ax.text(cx, y0 + 0.58, "MRE 50 nt", ha="center", va="bottom", fontsize=9,
            color=ps.INK_2, zorder=4)


def icon_matrix(fig, ax_host, x0, x1, y0, y1, pair):
    cmap = LinearSegmentedColormap.from_list("mreblue", ps.BLUE_RAMP)
    m = pairing_matrix(pair.mirna_seq, pair.mre_seq)
    to_fig = ax_host.transData + fig.transFigure.inverted()
    (fx0, fy0), (fx1, fy1) = to_fig.transform([(x0 + 0.05, y0), (x1 - 0.05, y1)])
    ax = fig.add_axes((fx0, fy0, fx1 - fx0, fy1 - fy0))
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(ps.GRID)
    ax.add_patch(Rectangle((-0.5, SEED[0] - 1.5), m.shape[1], SEED[1] - SEED[0] + 1,
                           fill=False, edgecolor=ORANGE, lw=1.7, zorder=5))
    ax.text(m.shape[1] - 1.2, SEED[1] + 0.3, "seed", ha="right", va="top",
            fontsize=8.5, color=ORANGE, fontweight="bold", zorder=6,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.axhline(len(pair.mirna_seq) - 0.5, color=ps.MUTED, lw=0.9, ls=(0, (3, 2)))


def icon_conv_stack(ax, x0, x1, y0, y1, n_blocks):
    """Feature maps shrinking block by block, drawn with a little depth.

    Maps are spaced by the gap between them, not by their left edges: evenly
    spaced left edges with shrinking widths would leave the first maps touching
    and the gaps widening to the right.
    """
    cy = (y0 + y1) / 2
    depth = 0.10                       # back-layer offset, part of each map's width
    sizes = np.linspace(1.05, 0.30, 5)
    widths = sizes * 0.62 + depth
    left, right = x0 + 0.15, x1 - 0.15
    gap = (right - left - widths.sum()) / (len(sizes) - 1)
    xs = left + np.concatenate([[0.0], np.cumsum(widths[:-1] + gap)])
    shades = ps.BLUE_RAMP[2:7]
    for x, s, shade in zip(xs, sizes, shades):
        for k, dx in enumerate((depth, depth / 2, 0.0)):
            ax.add_patch(Rectangle((x + dx, cy - s / 2 + dx), s * 0.62, s,
                                   facecolor=shade, edgecolor=BLUE, lw=0.9,
                                   alpha=1.0 if k == 2 else 0.55, zorder=3 + k))
    # One arrow per gap, the same length each time, centred in the gap.
    pad = 0.05
    for xa, wa in zip(xs[:-1], widths[:-1]):
        arrow(ax, (xa + wa + pad, cy), (xa + wa + gap - pad, cy), BLUE)
    ax.text(x1 - 0.05, y0 + 0.02, f"× {n_blocks}", ha="right", va="bottom",
            fontsize=10.5, fontweight="bold", color=BLUE, zorder=6)


def icon_head(ax, x0, x1, y0, y1):
    """Global pooling funnel into a 2-layer head ending in one unit."""
    cy = (y0 + y1) / 2
    fx0, fx1 = x0 + 0.05, x0 + 0.55
    ax.add_patch(Polygon([(fx0, cy + 0.55), (fx1, cy + 0.14), (fx1, cy - 0.14),
                          (fx0, cy - 0.55)], closed=True, facecolor=ps.BLUE_RAMP[2],
                         edgecolor=BLUE, lw=1.0, zorder=3))
    layers = [4, 3, 1]
    lx = np.linspace(fx1 + 0.35, x1 - 0.12, len(layers))
    pts = [[(x, cy + (i - (n - 1) / 2) * 0.28) for i in range(n)]
           for x, n in zip(lx, layers)]
    for a, b in zip(pts[:-1], pts[1:]):
        for pa in a:
            for pb in b:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=ps.BLUE_RAMP[4],
                        lw=0.7, zorder=3)
    ax.plot([fx1, lx[0]], [cy, cy], color=BLUE, lw=1.0, zorder=3)
    for layer in pts:
        for (x, y) in layer:
            ax.add_patch(Circle((x, y), 0.075, facecolor="white", edgecolor=BLUE,
                                lw=1.2, zorder=4))


def icon_sigmoid(ax, x, y, w, h, colour):
    t = np.linspace(-6, 6, 60)
    ax.plot(x + (t + 6) / 12 * w, y + h / (1 + np.exp(-t)), color=colour, lw=2.2,
            zorder=4, solid_capstyle="round")
    ax.plot([x, x + w], [y, y], color=ps.MUTED, lw=0.8, zorder=3)
    ax.plot([x, x], [y, y + h], color=ps.MUTED, lw=0.8, zorder=3)


def icon_output(ax, x0, x1, y0, y1, aps, colour, explain=False):
    cy = (y0 + y1) / 2 + (0.18 if explain else 0.0)
    icon_sigmoid(ax, x0 + 0.1, cy - 0.34, 0.72, 0.68, colour)
    ax.text(x0 + 1.02, cy, f"APS {aps}", ha="left", va="center", fontsize=21,
            fontweight="bold", color=colour, zorder=4)
    if explain:
        mx, my = x0 + 1.12, y0 + 0.2
        ax.add_patch(Circle((mx, my + 0.05), 0.11, fill=False, edgecolor=ps.INK_2,
                            lw=1.6, zorder=4))
        ax.plot([mx + 0.08, mx + 0.19], [my - 0.03, my - 0.14], color=ps.INK_2,
                lw=2.0, solid_capstyle="round", zorder=4)
        ax.text(mx + 0.3, my + 0.02, "TreeSHAP", ha="left",
                va="center", fontsize=9.5, color=ps.INK_2, zorder=4)


def icon_duplex(ax, x0, x1, y0, y1):
    cy = (y0 + y1) / 2
    top_y, bot_y = cy + 0.22, cy - 0.22
    ax.plot([x0 + 0.35, x1 - 0.2], [top_y, top_y], color=ORANGE, lw=2.2,
            solid_capstyle="round", zorder=4)
    ax.plot([x0 + 0.1, x1 - 0.1], [bot_y, bot_y], color=ps.INK_2, lw=2.2,
            solid_capstyle="round", zorder=4)
    xs = np.linspace(x0 + 0.45, x1 - 0.3, 12)
    for k, x in enumerate(xs):
        if k in (6, 7):            # a small internal loop: unpaired bases
            continue
        ax.plot([x, x], [bot_y + 0.04, top_y - 0.04], color=ORANGE, lw=1.3,
                alpha=0.9 if k < 7 else 0.6, zorder=3)
    ax.text(x0 + 0.35, top_y + 0.1, "miRNA", ha="left", va="bottom", fontsize=8.5,
            color=ORANGE, zorder=4)
    ax.text(x0 + 0.1, bot_y - 0.1, "MRE", ha="left", va="top", fontsize=8.5,
            color=ps.INK_2, zorder=4)


# -- feature family glyphs, each inside a unit square at (x, y) of side s -------

def g_shuffle(ax, x, y, s, c):
    for (a, b) in (((0.1, 0.2), (0.85, 0.8)), ((0.1, 0.8), (0.85, 0.2))):
        arrow(ax, (x + a[0] * s, y + a[1] * s), (x + b[0] * s, y + b[1] * s), c)


def g_tree(ax, x, y, s, c):
    root = (x + 0.12 * s, y + 0.5 * s)
    ax.plot([root[0], x + 0.4 * s], [root[1], root[1]], color=c, lw=1.6, zorder=4)
    for yy in (0.18, 0.82):
        ax.plot([x + 0.4 * s, x + 0.4 * s], [y + 0.5 * s, y + yy * s], color=c,
                lw=1.6, zorder=4)
        ax.plot([x + 0.4 * s, x + 0.62 * s], [y + yy * s, y + yy * s], color=c,
                lw=1.6, zorder=4)
        for dy in (-0.14, 0.14):
            ax.plot([x + 0.62 * s, x + 0.62 * s], [y + yy * s, y + (yy + dy) * s],
                    color=c, lw=1.6, zorder=4)
            ax.plot([x + 0.62 * s, x + 0.9 * s], [y + (yy + dy) * s] * 2, color=c,
                    lw=1.6, zorder=4)


def g_energy(ax, x, y, s, c):
    t = np.linspace(-1, 1, 40)
    ax.plot(x + (t + 1) / 2 * s, y + (0.15 + 0.7 * t ** 2) * s, color=c, lw=2.0,
            zorder=4)
    ax.add_patch(Circle((x + 0.5 * s, y + 0.22 * s), 0.07 * s * 1.4, facecolor=c,
                        edgecolor="none", zorder=5))


def g_ensemble(ax, x, y, s, c):
    for k, yy in enumerate((0.78, 0.5, 0.22)):
        a = 1.0 - 0.3 * k
        ax.plot([x + 0.1 * s, x + 0.9 * s], [y + (yy + 0.06) * s] * 2, color=c,
                lw=1.6, alpha=a, zorder=4)
        ax.plot([x + 0.1 * s, x + 0.9 * s], [y + (yy - 0.06) * s] * 2, color=c,
                lw=1.6, alpha=a, zorder=4)


def g_pairing(ax, x, y, s, c):
    for k, xx in enumerate((0.2, 0.5, 0.8)):
        wobble = k == 1
        ax.add_patch(Circle((x + xx * s, y + 0.75 * s), 0.09 * s, facecolor=c,
                            edgecolor="none", zorder=4))
        ax.add_patch(Circle((x + xx * s, y + 0.25 * s), 0.09 * s,
                            facecolor="white" if wobble else c, edgecolor=c,
                            lw=1.2, zorder=4))
        ax.plot([x + xx * s] * 2, [y + 0.36 * s, y + 0.64 * s], color=c, lw=1.2,
                ls=(0, (1.5, 1.2)) if wobble else "-", zorder=3)


GLYPHS = [g_shuffle, g_tree, g_energy, g_ensemble, g_pairing]


def icon_features(ax, x0, x1, y0, y1):
    n = len(FEATURE_GROUPS)
    slot = (x1 - x0) / n
    s = 0.62
    for k, ((_, feats), word, glyph) in enumerate(zip(FEATURE_GROUPS, FAMILY_WORD,
                                                      GLYPHS)):
        cx = x0 + (k + 0.5) * slot
        glyph(ax, cx - s / 2, y1 - s - 0.02, s, ORANGE)
        ax.text(cx, y1 - s - 0.16, str(len(feats)), ha="center", va="top",
                fontsize=12, fontweight="bold", color=ORANGE, zorder=4)
        ax.text(cx, y0 + 0.02, word, ha="center", va="bottom", fontsize=8.2,
                color=ps.INK_2, zorder=4)


def icon_bag(ax, x0, x1, y0, y1, n_folds):
    """Three small decision trees standing for the bag of folds."""
    cy = (y0 + y1) / 2 + 0.1
    for k, dx in enumerate((-0.62, 0.0, 0.62)):
        tx = (x0 + x1) / 2 + dx - 0.10
        a = 0.45 + 0.275 * (2 - abs(k - 1) * 2) / 2
        root = (tx, cy + 0.33)
        kids = [(tx - 0.14, cy), (tx + 0.14, cy)]
        leaves = [(tx - 0.23, cy - 0.3), (tx - 0.06, cy - 0.3),
                  (tx + 0.06, cy - 0.3), (tx + 0.23, cy - 0.3)]
        for kd in kids:
            ax.plot([root[0], kd[0]], [root[1], kd[1]], color=ORANGE, lw=1.3,
                    alpha=a, zorder=3)
        for kd, lv in zip([kids[0], kids[0], kids[1], kids[1]], leaves):
            ax.plot([kd[0], lv[0]], [kd[1], lv[1]], color=ORANGE, lw=1.1, alpha=a,
                    zorder=3)
        for p, r in [(root, 0.07)] + [(kd, 0.058) for kd in kids] + \
                [(lv, 0.042) for lv in leaves]:
            ax.add_patch(Circle(p, r, facecolor=ORANGE if p != root else "white",
                                edgecolor=ORANGE, lw=1.1, alpha=a, zorder=4))
    ax.text(x1 - 0.02, y0 + 0.02, f"× {n_folds}", ha="right", va="bottom",
            fontsize=10.5, fontweight="bold", color=ORANGE, zorder=6)


# ---------------------------------------------------------------------------

def main():
    check_feature_groups()
    ma = load_model_args()
    n_folds = len([p for p in BAG_DIR.iterdir() if p.name.startswith("S1F")])
    pair = pick_pair()

    ps.apply(base=11)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.set_aspect("equal")
    ax.axis("off")

    a_mid, b_mid = (A_Y0 + A_Y1) / 2, (B_Y0 + B_Y1) / 2

    # -- shared input, between the tracks ---------------------------------
    ix = box(ax, 0.15, 2.05, 1.30, 3.70, "Input", ps.MUTED, ps.FILL)
    icon_input(ax, *ix)
    arrow(ax, (1.10, 3.72), (2.50, a_mid + 0.3), BLUE, rad=-0.35)
    arrow(ax, (1.10, 1.28), (2.50, b_mid - 0.3), ORANGE, rad=0.35)

    # -- track A: sequence CNN --------------------------------------------
    ax.text(COLS[0][0], A_Y1 + 0.08, "A   Sequence CNN", fontsize=13,
            fontweight="bold", color=BLUE, ha="left", va="bottom")
    titles_a = ["pairing matrix", "conv blocks", "pool → head", "P(bind)"]
    for k, ((x0, x1), title) in enumerate(zip(COLS, titles_a)):
        area = box(ax, x0, x1, A_Y0, A_Y1, title, BLUE, ps.CNN_FILL)
        if k == 0:
            icon_matrix(fig, ax, *area, pair)
        elif k == 1:
            icon_conv_stack(ax, *area, ma["n_conv_blocks"])
        elif k == 2:
            icon_head(ax, *area)
        else:
            icon_output(ax, *area, "0.87", BLUE)
        if k:
            arrow(ax, (COLS[k - 1][1] + 0.05, a_mid), (x0 - 0.05, a_mid), BLUE)

    # -- track B: features + CatBoost -------------------------------------
    ax.text(COLS[0][0], B_Y1 + 0.08, "B   Thermodynamic features → CatBoost",
            fontsize=13, fontweight="bold", color=ORANGE, ha="left", va="bottom")
    titles_b = ["IntaRNA duplex", "26 features", "CatBoost", "P(bind)"]
    for k, ((x0, x1), title) in enumerate(zip(COLS, titles_b)):
        area = box(ax, x0, x1, B_Y0, B_Y1, title, ORANGE, ps.GLUON_FILL)
        if k == 0:
            icon_duplex(ax, *area)
        elif k == 1:
            icon_features(ax, *area)
        elif k == 2:
            icon_bag(ax, *area, n_folds)
        else:
            icon_output(ax, *area, "0.84", ORANGE, explain=True)
        if k:
            arrow(ax, (COLS[k - 1][1] + 0.05, b_mid), (x0 - 0.05, b_mid), ORANGE)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=300)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"panel1_icons.{ext}", **kw)
    plt.close(fig)
    print("wrote", OUT / "panel1_icons.{png,svg,pdf}")


if __name__ == "__main__":
    main()
