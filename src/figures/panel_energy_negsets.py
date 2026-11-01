"""Duplex free-energy (IntaRNA E) distributions: positives vs negatives, per negative set.

Form: a distribution comparison, so overlaid densities; small multiples by negative set
because the question is how the NEGATIVE distribution shifts while the positives stay put.
Positives are the same rows in all three panels, which makes them the anchor -- any
difference between panels is the negative definition, nothing else.

Colour: two categorical series only (positive / negative), taking poster_style's two
validated slots. Deliberately NOT three series with Type A and Type B as separate hues:
the A-vs-B comparison is carried by the new100 and old100 panels, which are pure A and
pure B respectively, so the small multiples answer it without a third hue.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/raid1/rafail/mreseeker/src/figures")
import poster_style as ps

TIDY = "/mnt/raid1/rafail/Hybridetector_pipeline/data/energy/tidy_energy.tsv"
OUT  = "/mnt/raid1/rafail/Hybridetector_pipeline/data/energy/energy_negsets.png"
SETS = [("old100", "old100  (100% original method)"),
        ("mix5050", "mix5050  (50/50)"),
        ("new100", "new100  (100% three-way)")]

d = pd.read_csv(TIDY, sep="\t")
grid = np.linspace(-34, 4, 500)   # full support: data spans -31.6 .. +1.6

def kde(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    h = 1.06 * v.std() * len(v) ** -0.2                      # Silverman
    z = (grid[:, None] - v[None, :]) / h
    return np.exp(-0.5 * z ** 2).sum(1) / (len(v) * h * np.sqrt(2 * np.pi))

LBL_Y = .172          # reserved label band, clear of every curve (peak ~.152)
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharex=True, sharey=True)
for ax, (s, title) in zip(axes, SETS):
    sub = d[d["set"] == s]
    pos, neg = sub[sub.label == 1].E.values, sub[sub.label == 0].E.values
    for v, col, fill, lab in ((pos, ps.ACCENT_CNN, ps.CNN_FILL, "positives"),
                              (neg, ps.ACCENT_GLUON, ps.GLUON_FILL, "negatives")):
        y = kde(v)
        ax.fill_between(grid, y, color=fill, zorder=1)
        ax.plot(grid, y, color=col, lw=2, zorder=3, label=lab)          # 2px lines
        m = np.median(v)
        ax.axvline(m, color=col, lw=1, ls=(0, (3, 3)), zorder=2, alpha=.9)
    # selective direct labels: medians only, never a number on every point
    ax.annotate(f"{np.median(pos):.1f}", (np.median(pos), LBL_Y), color=ps.ACCENT_CNN,
                ha="right", va="top", fontsize=9, xytext=(-3, 0), textcoords="offset points")
    ax.annotate(f"{np.median(neg):.1f}", (np.median(neg), LBL_Y), color=ps.ACCENT_GLUON,
                ha="left", va="top", fontsize=9, xytext=(3, 0), textcoords="offset points")
    ax.set_title(title, fontsize=10, color=ps.INK, loc="left", pad=8)
    ax.set_xlabel("IntaRNA duplex energy  E  (kcal/mol)", fontsize=9, color=ps.INK_2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(ps.GRID)
    ax.tick_params(colors=ps.INK_2, labelsize=9)
    ax.axvline(0, color=ps.DEEMPH, lw=.8, zorder=1)          # E>0 = unfavorable
    ax.grid(axis="y", color=ps.GRID, lw=.6, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_ylabel("density", fontsize=9, color=ps.INK_2)
axes[0].set_ylim(0, .19); axes[0].set_xlim(-30, 2)
axes[0].legend(frameon=False, fontsize=9, loc="upper left",
               labelcolor=ps.INK_2, handlelength=1.6)
fig.suptitle("Stronger duplexes in positives, and the gap barely moves with negative design",
             x=.008, ha="left", fontsize=12, color=ps.INK, y=1.02)
fig.text(.008, .955, "IntaRNA best-of-10 interaction energy; 20,000 sampled pairs per class per set. "
         "Positives are identical rows across panels. More negative = stronger duplex.",
         ha="left", fontsize=9, color=ps.MUTED)
fig.tight_layout(rect=(0, 0, 1, .93))
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=ps.SURFACE)
print(f"  -> {OUT}")
for s, _ in SETS:
    sub = d[d["set"] == s]
    print(f"  {s:9s} pos median {sub[sub.label==1].E.median():6.2f}   "
          f"neg median {sub[sub.label==0].E.median():6.2f}   "
          f"gap {sub[sub.label==1].E.median()-sub[sub.label==0].E.median():5.2f}")
