#!/usr/bin/env python3
"""
Panel 2 - AutoGluon model selection as an accuracy/cost Pareto front.

Source: results/model_aps_manakov.csv, which carries APS and wall-clock seconds
for both evaluation sets from a single timing run (model_aps_leftout.csv is a
separate leftout-only re-run; mixing the two would put two different clocks on
one axis, so only the first file is read here).

Form: the accuracy-vs-cost convention used across the efficiency literature -
cost on x, quality on y, a staircase marking the Pareto front, and filled vs
hollow markers separating non-dominated from dominated models. That says more
than a plain scatter: it shows the shipped default is not merely a reasonable
compromise but the *fastest non-dominated* candidate in both sets, and it marks
two candidates as strictly worse than it on both axes at once.

The staircase is a step rather than a straight interpolation on purpose. There
is no model between two candidates, so the reachable claim is "to beat this
accuracy you must pay at least that much time", which is what a step draws.

    python src/figures/panel2_model_tradeoff.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "model_aps_manakov.csv"
OUT = REPO / "results" / "figures"

SHIPPED = "LightGBMLarge_BAG_L1"
REFERENCE = "WeightedEnsemble_L3"

FACETS = [
    ("test",    "APS_test",    "seconds_test",    "Manakov v7 test set"),
    ("leftout", "APS_leftout", "seconds_leftout", "Manakov leftout set"),
]

# Label placement per facet, hand-set so nothing collides: (dx pt, dy pt, ha, va).
OFFSETS = {
    "test": {
        "WeightedEnsemble_L3":  (-9,   0, "right",  "center"),
        "WeightedEnsemble_L2":  (-9,   0, "right",  "center"),
        # Right of the marker: centred below, the front's riser at ~106 s cuts
        # straight through the text.
        "LightGBM_BAG_L2":      (9,    0, "left",   "center"),
        "LightGBMLarge_BAG_L1": (0,   13, "center", "bottom"),
        "LightGBM_BAG_L1":      (9,   -4, "left",   "center"),
        "LightGBMXT_BAG_L1":    (9,    0, "left",   "center"),
    },
    "leftout": {
        "WeightedEnsemble_L3":  (0,   13, "center", "bottom"),
        "WeightedEnsemble_L2":  (-9,   0, "right",  "center"),
        "LightGBM_BAG_L2":      (0,  -13, "center", "top"),
        # Above the marker: this is the front's leftmost point, so its own
        # horizontal run leaves rightwards at exactly this height.
        "LightGBMLarge_BAG_L1": (0,   13, "center", "bottom"),
        "LightGBM_BAG_L1":      (9,    0, "left",   "center"),
        "LightGBMXT_BAG_L1":    (-9,   0, "right",  "center"),
    },
}

# The trade-off callout goes in the facet's empty lower-right corner, in axes
# fractions rather than data coordinates - anchoring it between the two models
# it describes drops it straight onto a neighbouring label in the leftout facet.
ANNOT_XY = (0.975, 0.045)

# Explicit log ticks - the default decade-only ticks leave the axis unreadable
# when every point sits inside one decade.
XTICKS = {"test": [10, 20, 50, 100, 200, 500], "leftout": [1, 2, 5, 10, 20, 50]}

# (left, right) multiplicative padding each facet needs to keep its labels inside
# the axis. The leftout facet needs more room on the right: its slowest model is
# labelled above-centre and would otherwise overrun. Floors, not final limits.
MIN_PAD = {"test": (2.6, 2.6), "leftout": (2.6, 5.0)}


def compute_xlims(df):
    """Limits that give every facet the same number of decades.

    A log axis reads as ratios, so a given horizontal distance has to mean the
    same factor in both facets or the eye compares them wrongly. Padding the two
    independently spanned 1.82 decades on the left against 2.07 on the right,
    which drew the 9.8x connector at 54% of its axis and the 9.0x one at 46% -
    a gap that was mostly axis scale rather than data. Each facet declares the
    padding it needs; the widest resulting span wins and the rest grow
    symmetrically to match.
    """
    need = {}
    for key, _, sec_col, _ in FACETS:
        lo, hi = df[sec_col].min(), df[sec_col].max()
        pad_lo, pad_hi = MIN_PAD[key]
        need[key] = (lo / pad_lo, hi * pad_hi)
    span = max(math.log10(hi / lo) for lo, hi in need.values())
    out = {}
    for key, (lo, hi) in need.items():
        grow = 10 ** ((span - math.log10(hi / lo)) / 2.0)
        out[key] = (lo / grow, hi * grow)
    return out


def pareto_front(df, aps_col, sec_col):
    """Models nothing else beats on both axes at once, fastest first.

    Sweeping cheapest-to-dearest, a model joins the front when it is the most
    accurate seen so far; anything that fails that test has a cheaper model
    which is also at least as accurate, so it is dominated.
    """
    best, front = -math.inf, []
    for _, r in df.sort_values(sec_col).iterrows():
        if r[aps_col] > best:
            front.append(r.model)
            best = r[aps_col]
    return front


def draw_facet(ax, df, aps_col, sec_col, title, key, xlim):
    ax.set_xscale("log")
    ax.grid(axis="both", lw=0.9, color=ps.GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(*xlim)

    lo, hi = df[aps_col].min(), df[aps_col].max()
    pad = (hi - lo) * 0.30
    ax.set_ylim(lo - pad, hi + pad)

    front = pareto_front(df, aps_col, sec_col)

    # The front itself, held flat to the next model's cost then stepped up. The
    # trailing point extends the last level to the axis edge: past the dearest
    # model nothing more accurate is on offer.
    fd = df[df.model.isin(front)].sort_values(sec_col)
    ax.step(list(fd[sec_col]) + [xlim[1]],
            list(fd[aps_col]) + [fd[aps_col].iloc[-1]],
            where="post", color=ps.MUTED, lw=1.5, zorder=2, clip_on=True)

    for _, r in df.iterrows():
        shipped = r.model == SHIPPED
        on_front = r.model in front
        if shipped:
            style = dict(ms=12, mfc=ps.ACCENT_GLUON, mec=ps.SURFACE, mew=2.0)
        elif on_front:
            style = dict(ms=9.5, mfc=ps.DEEMPH, mec=ps.SURFACE, mew=2.0)
        else:
            # Hollow = dominated. Fill carries the Pareto status, so the
            # distinction survives greyscale and colour-vision deficiency.
            style = dict(ms=9.5, mfc=ps.SURFACE, mec=ps.DEEMPH, mew=1.8)
        ax.plot(r[sec_col], r[aps_col], "o", zorder=5 if shipped else 4,
                clip_on=False, **style)

        dx, dy, ha, va = OFFSETS[key][r.model]
        ax.annotate(
            r.model, (r[sec_col], r[aps_col]),
            textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
            fontsize=9.5, color=ps.INK if shipped else ps.INK_2,
            fontweight="bold" if shipped else "normal", zorder=6,
        )

    # What walking the front from the shipped default to its top actually costs.
    a = df.loc[df.model == SHIPPED].iloc[0]
    b = df.loc[df.model == REFERENCE].iloc[0]
    ax.annotate(
        f"top of front vs shipped default:\n"
        f"+{b[aps_col] - a[aps_col]:.4f} APS for {b[sec_col] / a[sec_col]:.1f}× the time",
        ANNOT_XY, xycoords="axes fraction", ha="right", va="bottom",
        fontsize=9.5, color=ps.ACCENT_GLUON, fontweight="bold",
        linespacing=1.45, zorder=6,
    )

    ax.set_title(title, color=ps.INK, fontsize=12, fontweight="bold",
                 loc="left", pad=10)
    ax.set_xlabel("inference wall-clock (s, log scale)")
    ax.set_ylabel("average precision (APS)")
    ps.despine(ax)

    ax.set_xticks(XTICKS[key])
    ax.set_xticklabels([str(t) for t in XTICKS[key]])
    ax.minorticks_off()


def main():
    df = pd.read_csv(SRC)
    ps.apply(base=11)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.9))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.685, bottom=0.185,
                        wspace=0.26)

    xlims = compute_xlims(df)
    for ax, (key, aps_col, sec_col, title) in zip(axes, FACETS):
        draw_facet(ax, df, aps_col, sec_col, title, key, xlims[key])

    fig.text(0.075, 0.968,
             "The shipped default is the cheapest model on the Pareto front",
             fontsize=16, fontweight="bold", color=ps.INK, ha="left",
             va="top", transform=fig.transFigure)
    fig.text(0.075, 0.905,
             "Six AutoGluon candidates. Hollow points are dominated — another candidate is both "
             "faster and more accurate.\nThe shipped default is the fastest model on the front "
             "in both evaluation sets.",
             fontsize=10.5, color=ps.INK_2, ha="left", va="top",
             linespacing=1.5, transform=fig.transFigure)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=11, mfc=ps.ACCENT_GLUON,
                   mec=ps.SURFACE, mew=2.0, label="shipped default"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=ps.DEEMPH,
                   mec=ps.SURFACE, mew=2.0, label="Pareto-optimal"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=ps.SURFACE,
                   mec=ps.DEEMPH, mew=1.8, label="dominated"),
        plt.Line2D([], [], color=ps.MUTED, lw=1.5, label="Pareto front"),
    ]
    # One row under the subtitle, clear of both the headline and the facets.
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.072, 0.800),
               ncol=4, columnspacing=1.8, handletextpad=0.5,
               labelcolor=ps.INK_2)

    fig.text(0.075, 0.016,
             "Both facets: single timing run, results/model_aps_manakov.csv. Absolute values are "
             "not comparable across facets — the two sets differ in size.\nBoth x-axes span the "
             "same decades, so equal horizontal distance means equal speed ratio. The front is "
             "set-specific — LightGBMXT_BAG_L1 is on it for leftout only.\nThe shipped default is "
             "a 138 MB deployment clone of a 6.8 GB stack, and stays exactly explainable through "
             "TreeSHAP.",
             fontsize=8.5, color=ps.MUTED, ha="left", va="bottom",
             linespacing=1.5, transform=fig.transFigure)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"panel2_model_tradeoff.{ext}", **kw)
    plt.close(fig)
    print("wrote", OUT / "panel2_model_tradeoff.{png,svg,pdf}")
    for key, aps_col, sec_col, _ in FACETS:
        print(f"  {key:8s} front: {pareto_front(df, aps_col, sec_col)}")


if __name__ == "__main__":
    main()
