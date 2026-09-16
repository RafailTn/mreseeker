"""
Panel 8 - how much of SAEC's miRNA repertoire the models have already seen.

The generalisation panels claim the CNN holds up on an unseen cell line. That
claim is about unseen *sites*; this panel bounds it by showing how far the
miRNA *families* in SAEC (GSE304955) overlap those in HEK293T (Manakov 2022),
the cell line both models were trained on.

Two views, because they answer different questions
---------------------------------------------------
Left, an area-proportional Venn of family labels: how many distinct families
each experiment detects, and how many they share. Counts treat a family seen
once the same as one seen 20,000 times.

Right, the same split weighted by positive interactions: what share of each
cell line's binding events come from shared families. This is the number that
matters for a model, and it is much more lopsided than the counts suggest -
line-specific families are numerous but rare.

Definitions
-----------
* Positives only (label == 1): negatives are synthetic pairings drawn from the
  same miRNA pool, so they add no information about the repertoire.
* HEK293T = Manakov train + test + leftout, the corpus the models learned from.
* SAEC = gse304955_positives_merged_v7.tsv, the full positive set before the
  "novel" filter. That filter removed pairs overlapping miRBench, which would
  shrink SAEC's repertoire for reasons unrelated to biology.
* Family = noncodingRNA_fam. "unknown" is not a family and is left out of the
  Venn; its share is reported in the bars.

Usage
-----
python3 src/figures/panel8_family_venn.py [--bare]
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"
D = REPO.parent / "msc-thesis" / "data"

HEK_FILES = ["AGO2_eCLIP_Manakov2022_train_v7.tsv",
             "AGO2_eCLIP_Manakov2022_test_v7.tsv",
             "AGO2_eCLIP_Manakov2022_leftout_v7.tsv"]
SAEC_FILES = ["gse304955_positives_merged_v7.tsv"]
FAM, UNKNOWN = "noncodingRNA_fam", "unknown"

# Neutral tones on purpose: blue and orange mean the two models everywhere else
# on the poster, and neither cell line is a model.
HEK_EDGE, HEK_FILL = ps.INK_2, "#d9d7d1"
SAEC_EDGE, SAEC_FILL = ps.MUTED, "#efeeea"


def positives(files: list[str]) -> pd.Series:
    parts = []
    for f in files:
        d = pd.read_csv(D / f, sep="\t", usecols=[FAM, "label"], low_memory=False)
        parts.append(d.loc[d.label == 1, FAM].fillna(UNKNOWN))
    return pd.concat(parts, ignore_index=True)


def lens_area(r1: float, r2: float, d: float) -> float:
    """Area of intersection of two circles with centres d apart."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    k = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a1 + a2 - k


def centre_distance(r1: float, r2: float, target: float) -> float:
    """Bisection for the separation whose lens area equals `target`.

    Area-proportional, not schematic: a Venn whose overlap is drawn at a
    conventional half-way position would imply the two repertoires are about
    half shared, when here the smaller set sits almost entirely inside the
    larger one.
    """
    lo, hi = abs(r1 - r2), r1 + r2
    for _ in range(80):
        mid = (lo + hi) / 2
        if lens_area(r1, r2, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main(bare: bool = False, name: str | None = None) -> None:
    hek, saec = positives(HEK_FILES), positives(SAEC_FILES)
    H, S = set(hek) - {UNKNOWN}, set(saec) - {UNKNOWN}
    shared = H & S
    n_h_only, n_s_only, n_shared = len(H - S), len(S - H), len(shared)

    ps.apply(9.5)
    height = 4.9 if bare else 5.9
    fig = plt.figure(figsize=(12.6, height))
    top, bottom = 1 - 0.30 / height, (0.30 if bare else 1.15) / height
    axv = fig.add_axes([0.02, bottom, 0.50, top - bottom])
    axb = fig.add_axes([0.63, bottom + 0.20 * (top - bottom), 0.31,
                        0.55 * (top - bottom)])

    # ---- Venn ---------------------------------------------------------------
    rh, rs = math.sqrt(len(H) / math.pi), math.sqrt(len(S) / math.pi)
    d = centre_distance(rh, rs, n_shared)
    ch, cs = (0.0, 0.0), (d, 0.0)
    axv.add_patch(Circle(ch, rh, facecolor=HEK_FILL, edgecolor=HEK_EDGE, lw=2.0,
                         alpha=0.95, zorder=1))
    axv.add_patch(Circle(cs, rs, facecolor=SAEC_FILL, edgecolor=SAEC_EDGE, lw=2.0,
                         alpha=0.55, zorder=2))

    # Region label positions: the HEK-only crescent left of the SAEC circle,
    # the lens between the two, the SAEC-only sliver to the right of HEK's edge.
    x_h_only = (-rh + (cs[0] - rs)) / 2
    x_shared = ((cs[0] - rs) + rh) / 2
    axv.text(x_h_only, 0, f"{n_h_only}", ha="center", va="center", fontsize=17,
             fontweight="bold", color=ps.INK_2, zorder=4)
    axv.text(x_shared, 0, f"{n_shared}", ha="center", va="center", fontsize=22,
             fontweight="bold", color=ps.INK, zorder=4)
    axv.text(x_shared, -2.1, "shared", ha="center", va="top", fontsize=9.5,
             color=ps.INK_2, zorder=4)

    # The SAEC-only crescent is too thin to hold its number, so it goes outside
    # with a leader - squeezing it in would make the smallest region look full.
    sliver_x = (rh + cs[0] + rs) / 2
    axv.annotate(f"{n_s_only}\nSAEC only", xy=(sliver_x, 0.3 * rs),
                 xytext=(cs[0] + rs + 2.8, rs * 0.95), ha="left", va="center",
                 fontsize=10.5, fontweight="bold", color=ps.INK_2,
                 linespacing=1.3, zorder=4,
                 arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=1.0,
                                 shrinkA=2, shrinkB=2))
    top_s = saec[saec.isin(S - H)].value_counts()
    if len(top_s):
        axv.text(cs[0] + rs + 2.8, rs * 0.95 - 2.9,
                 f"largest: {top_s.index[0]} ({top_s.iloc[0]} rows)",
                 ha="left", va="top", fontsize=8.5, color=ps.MUTED, zorder=4)

    # Set names sit outside both circles: inside, each one crossed an outline,
    # which reads as the label belonging to whichever region it lands in.
    axv.text(-rh * 0.30, rh + 0.7, f"HEK293T   {len(H)} families", ha="center",
             va="bottom", fontsize=11, fontweight="bold", color=HEK_EDGE,
             zorder=4)
    axv.text(cs[0] + rs * 0.35, -rh - 0.9, f"SAEC   {len(S)} families",
             ha="center", va="top", fontsize=11, fontweight="bold",
             color=ps.MUTED, zorder=4)
    axv.set_xlim(-rh - 0.8, cs[0] + rs + 12.5)
    axv.set_ylim(-rh - 3.6, rh + 3.0)
    axv.set_aspect("equal")
    axv.axis("off")
    axv.set_title("miRNA families detected", loc="left", fontsize=12,
                  fontweight="bold", color=ps.INK, pad=6)

    # ---- Row-weighted bars --------------------------------------------------
    def split(series: pd.Series, own_only: set) -> list[float]:
        return [series.isin(shared).mean(), series.isin(own_only).mean(),
                (series == UNKNOWN).mean()]

    bars = [("SAEC", split(saec, S - H), len(saec)),
            ("HEK293T", split(hek, H - S), len(hek))]
    tones = [ps.INK_2, "#b5b3ab", "#e2e1dc"]
    for i, (label, fracs, n) in enumerate(bars):
        y = len(bars) - 1 - i
        left = 0.0
        for frac, tone in zip(fracs, tones):
            axb.barh(y, frac, left=left, height=0.52, color=tone,
                     edgecolor=ps.SURFACE, lw=1.0, zorder=3)
            left += frac
        axb.text(fracs[0] / 2, y, f"{fracs[0]:.1%}", ha="center", va="center",
                 fontsize=11, fontweight="bold", color=ps.SURFACE, zorder=4)
        rest = 1 - fracs[0]
        axb.text(1.015, y, f"{rest:.1%} other", ha="left", va="center",
                 fontsize=9, color=ps.INK_2)
        axb.text(-0.015, y, f"{label}\n{n:,} positives", ha="right",
                 va="center", fontsize=9.5, color=ps.INK, linespacing=1.35)
    axb.set_xlim(0, 1)
    axb.set_ylim(-0.6, len(bars) - 0.4)
    axb.set_yticks([])
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    axb.set_xlabel("share of positive interactions")
    ps.despine(axb, keep=("bottom",))
    axb.set_title("weighted by binding events", loc="left", fontsize=12,
                  fontweight="bold", color=ps.INK, pad=26)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t) for t in tones]
    axb.legend(handles, ["shared family", "line-specific family", "unknown family"],
               loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
               labelcolor=ps.INK_2, handlelength=1.2, columnspacing=1.2,
               fontsize=8.5)

    if not bare:
        legend = (
            "miRNA family repertoire of the SAEC cell line (GSE304955, all positive "
            "interactions before novelty filtering) against HEK293T (Manakov 2022 "
            "train, test and leftout positives), the cell line both models were "
            f"trained on. Left: distinct families, area-proportional; "
            f"'{UNKNOWN}' family labels are excluded. Right: the share of each "
            "cell line's positive interactions that come from families shared "
            "between the two, specific to that line, or unannotated.")
        fig.text(0.030, 0.20 / height, "\n".join(textwrap.wrap(legend, width=176)),
                 fontsize=8.5, color=ps.INK_2, va="bottom", linespacing=1.5)

    name = name or ("panel8_bare" if bare else "panel8_family_venn")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.{{png,svg,pdf}}")
    print(f"families: HEK {len(H)}  SAEC {len(S)}  shared {n_shared}  "
          f"HEK-only {n_h_only}  SAEC-only {n_s_only}")
    print(f"lens check: area {lens_area(rh, rs, d):.2f} vs shared {n_shared}")
    for label, fracs, n in bars:
        print(f"{label:<8} shared {fracs[0]:.2%}  specific {fracs[1]:.2%}  "
              f"unknown {fracs[2]:.2%}  (n={n:,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bare", action="store_true",
                    help="Drop the legend paragraph, for the poster.")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    main(bare=a.bare, name=a.name)
