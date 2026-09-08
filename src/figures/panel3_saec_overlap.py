"""
Panel 3 - is the SAEC (GSE304955) result actually out-of-distribution?

Reads the two tables written by src/benchmark/saec_overlap.py and makes the two
statements a reviewer will ask for, side by side:

  (a) how much of the set the model has already seen, once "seen" is measured by
      genomic overlap rather than string equality, and
  (b) whether that familiarity buys the model anything.

Colour note: the poster-wide palette reads blue as the CNN track and orange as
the IntaRNA/LightGBM track. Every number on this panel is the CNN, so orange
would be a false cue. The two series here are separated as ink vs blue instead,
and the panel keeps the poster's meaning of blue intact.

Usage
-----
python3 src/figures/panel3_saec_overlap.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import poster_style as ps

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"

# The bin the strict claim rests on.
STRICT_BIN = "none"


def main(bare: bool = False) -> None:
    """`bare` drops the figure legend. On the poster the caption beneath the
    panel is the explanatory text, and repeating it inside the image both
    duplicates it and shrinks the plotting area."""
    curve = pd.read_csv(RESULTS / "saec_mirbench_overlap.csv")
    table = pd.read_csv(RESULTS / "saec_aps_by_overlap.csv")

    ps.apply(9.5)
    height = 4.45 if bare else 5.76
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.6, height), gridspec_kw=dict(
            width_ratios=[1.05, 1.0], wspace=0.24, left=0.055, right=0.985,
            # Panel titles need a fixed 0.40 in of headroom, so the top margin
            # is a length converted to a fraction, not a fixed fraction.
            top=1 - 0.40 / height,
            bottom=(0.68 if bare else 1.68) / height))

    # ---- (a) how much overlap, as a function of how strict "overlap" is ------
    # Log y: the two series are two orders of magnitude apart, and the whole
    # point is the size of that gap, not the shape of the upper curve.
    axL.set_yscale("log")
    for col, colour, label, lw in (
            ("frac_any_mirna", ps.INK, "any miRNA", 2.2),
            ("frac_same_mirna", ps.ACCENT_CNN, "same miRNA", 2.2)):
        axL.plot(curve.shared_bp, curve[col], color=colour, lw=lw,
                 solid_capstyle="round", zorder=3)

    one = curve[curve.shared_bp == 1].iloc[0]

    axL.text(2.0, one.frac_any_mirna * 1.22, f"any miRNA — {one.frac_any_mirna:.1%}",
             fontsize=10, fontweight="bold", color=ps.INK, va="bottom")
    axL.text(2.0, one.frac_same_mirna * 0.62,
             f"same miRNA — {one.frac_same_mirna:.2%}",
             fontsize=10, fontweight="bold", color=ps.ACCENT_CNN, va="top")

    axL.set_xlim(0, 51)
    axL.set_ylim(0.0001, 1.4)
    axL.set_xticks([0, 10, 20, 30, 40, 50])
    axL.set_yticks([0.0001, 0.001, 0.01, 0.1, 1.0])
    axL.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    axL.set_xlabel("shared bp with the nearest miRBench window  (of 50)")
    axL.set_ylabel("SAEC positives")
    axL.grid(axis="y", zorder=0)
    ps.despine(axL)
    axL.set_title("a   miRBench vs SAEC: Locus-Level overlap",
                  loc="left", fontsize=11.5, fontweight="bold", color=ps.INK,
                  pad=10)

    # ---- (b) does that familiarity help? ------------------------------------
    # APS against its own random baseline, which is the positive rate. Plotting
    # both on one axis is what makes the panel honest: APS drifts down across
    # the bins, but so does prevalence, and the gap between them is the signal.
    sub = table[table.match == "any miRNA"].reset_index(drop=True)
    x = range(len(sub))

    axR.plot(x, sub.pos_rate, "-", color=ps.DEEMPH, lw=1.8, zorder=2)
    axR.plot(x, sub.pos_rate, "o", ms=7, color=ps.DEEMPH, zorder=3)
    axR.plot(x, sub.aps, "-", color=ps.ACCENT_CNN, lw=2.2, zorder=4)
    axR.plot(x, sub.aps, "o", ms=10, color=ps.ACCENT_CNN, zorder=5)

    for i, r in sub.iterrows():
        axR.annotate("", xy=(i, r.aps), xytext=(i, r.pos_rate),
                     arrowprops=dict(arrowstyle="-", color=ps.CNN_FILL, lw=7),
                     zorder=1)
        axR.text(i, r.aps + 0.018, f"{r.aps:.3f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=ps.ACCENT_CNN, zorder=6)
        axR.text(i, (r.aps + r.pos_rate) / 2, f"lift\n+{r.aps - r.pos_rate:.3f}",
                 ha="center", va="center", fontsize=9, color=ps.INK_2,
                 linespacing=1.35, zorder=6)
        axR.text(i, 0.325, f"n = {r.n:,}", ha="center", va="bottom",
                 fontsize=8.5, color=ps.MUTED)

    # Centred under an interior marker: at the last one the string would run
    # off the axis, and the lift labels own the space above the grey line.
    axR.text(len(sub) - 2, sub.pos_rate.iloc[-2] - 0.032,
             "random baseline (= positive rate)",
             fontsize=8.5, color=ps.MUTED, ha="center", va="top")

    axR.set_xlim(-0.42, len(sub) - 0.58)
    axR.set_ylim(0.30, 0.96)
    axR.set_xticks(list(x))
    axR.set_xticklabels(sub.bin.replace({"none": "none\n(0 bp)"}))
    axR.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axR.set_xlabel("shared bp with the nearest miRBench window (any miRNA)")
    axR.set_ylabel("average precision")
    axR.grid(axis="y", zorder=0)
    ps.despine(axR)
    axR.set_title("b   Familiarity with locus does not help the model",
                  loc="left", fontsize=11.5, fontweight="bold", color=ps.INK,
                  pad=10)

    if not bare:
        # Flush to the figure edge and wrapped to its full width: the legend
        # describes the whole panel, not the left axis it used to sit under.
        # Flush to the figure edge and wrapped to its full width: the legend
        # describes the whole panel, not the left axis it used to sit under.
        # Wrapped by measure rather than by hand so a figsize change cannot push
        # a line off the canvas.
        legend = (
            "Overlap between the SAEC (GSE304955) evaluation set and the miRBench v7 "
            "corpus, and its effect on measured performance. 104,844 positive sites and "
            "an equal number of sampled negatives were scored with the Manakov-trained "
            "sequence CNN. Sites were matched to the union of the six miRBench v7 sets by "
            "genomic interval on the same chromosome and strand; all windows are 50 nt in "
            "both corpora, so shared length is 50 - |\u0394start|. "
            "(a) Share of SAEC positives whose nearest miRBench window shares at least the "
            "given number of bases, for a window bound by any miRNA (black) and for one "
            "bound by the same miRNA (blue). Complete overlap, at 50 bases, is the only "
            "case an exact sequence match detects. "
            "(b) Average precision within disjoint bins of that overlap, any miRNA (blue), "
            "against the random-classifier baseline, which equals the positive rate of "
            "each bin (grey); shaded bars give the difference. 99.8% of rows belong to a "
            "miRNA family present in the training corpus.")
        fig.text(
            0.030, 0.020, "\n".join(textwrap.wrap(legend, width=176)),
            fontsize=8.5, color=ps.INK_2, va="bottom", linespacing=1.5)

    # The bare cut is a separate file: the standalone figure still wants its
    # legend, and only the poster copy goes without one.
    name = "panel3_bare" if bare else "panel3_saec_overlap"
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)
    print("wrote", OUT / f"{name}.{{png,svg,pdf}}")


if __name__ == "__main__":
    import sys
    main(bare="--bare" in sys.argv)
