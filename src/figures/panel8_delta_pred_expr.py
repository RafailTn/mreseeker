"""
Panel 8 - how much a GTEx eQTL variant moves the predicted binding probability,
restricted to variants whose miRNA is among the top 100 expressed in the tissue
the eQTL was called in.

Why the expression filter is the population worth plotting
----------------------------------------------------------
The unfiltered eQTL x miRBench join asserts a repression mechanism for every
miRNA in the repertoire, including ones a tissue barely transcribes. Keeping
only rows whose miRNA ranks in that tissue's top 100 (by median CPM,
src/eqtl_analysis/gtex_top_mirnas.py) drops the rows where the mechanism cannot
run. `--key expr_in_interaction` narrows it once more, to variants that land on
a base actually paired to the miRNA in the IntaRNA duplex.

Why the two PIP strata are separate series
------------------------------------------
Fine-mapped PIP > 0.9 is the closest thing here to a causal variant and
PIP < 0.01 is the matched background, so pooling them hides the one comparison
the panel can make. They cannot share a count axis - the background outnumbers
the causal set ~16:1 - so each series is drawn as a share of its own n. The
counts are printed in the legend instead, because a percentage off 40 pairs and
one off 731 do not deserve the same credence.

Why the y axis is logarithmic
-----------------------------
63% of tissue-expressed pairs sit inside |delta| < 0.01 - the allele swap
changes one nucleotide in a 50-nt fragment and usually the model does not care.
On a linear axis that central spike is the only thing visible and the tails,
which are the interesting part, are a flat line at zero.

p(ref) and p(alt) are the sequence CNN's interaction probabilities for the
reference and alternate 50-nt fragment against the same miRNA, written by
src/eqtl_analysis/eqtl_analysis_cnn.py. delta_pred = p(alt) - p(ref), so
negative means the alternate allele is predicted to bind worse - the direction
a repression-loss eQTL implies. The score is sequence-only, so it is identical
across tissues for the same (variant, fragment) pair; the tissue enters this
panel only through the expression filter.

    pixi run -m dependencies/cnn python src/figures/panel8_delta_pred_expr.py
    pixi run -m dependencies/cnn python src/figures/panel8_delta_pred_expr.py \
        --key expr_in_interaction
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "eqtl_intarna_union"
OUT = REPO / "results" / "figures"

# (file suffix, legend label, colour, fill) - high PIP is the subject, the
# low-PIP background is context and wears the de-emphasis grey.
STRATA = [("gt_0_9",  "PIP > 0.9",   ps.ACCENT_CNN, ps.CNN_FILL),
          ("lt_0_01", "PIP < 0.01",  ps.DEEMPH,     ps.DEEMPH_FILL)]
# On the count axis each row gets the tick step its own range can carry: the
# two rows differ ~18x in n, so one shared step would be either unreadably
# dense on the background or a single gridline on the causal set.
COUNT_TICK = {"gt_0_9": 5, "lt_0_01": 50}

TITLES = {
    "expr_ALL": "Most eQTL alleles leave predicted binding untouched; a tail does not",
    "expr_in_interaction": "Inside the IntaRNA duplex, the tail is where most "
                           "eQTL alleles live",
}
SUBSET = {
    "expr_ALL": "",
    "expr_in_interaction":
        " and the variant hits a base paired to the miRNA in the IntaRNA duplex "
        "of a CNN true positive",
}
LIMIT = 0.5          # delta_pred is a probability difference, so it cannot exceed this


def load(src: Path, key: str, stratum: str) -> pd.DataFrame:
    """Unique (variant, fragment) pairs of one PIP stratum.

    Same de-duplication key as summarise_eqtl_configs.py. A pair can occur in
    both strata - high PIP for one tissue/gene, low for another - and since
    delta_pred is sequence-only it is then literally the same number in both
    series; that is a property of the data, not double counting.
    """
    d = pd.read_csv(src / f"delta_pred_{key}_pip_{stratum}.tsv", sep="\t",
                    low_memory=False).dropna(subset=["delta_pred"])
    return d.drop_duplicates(["variant", "unique_key"])


def main(src: Path = SRC, key: str = "expr_ALL", bare: bool = False,
         name: str | None = None, width: float = 7.4,
         binw: float | None = None, scale: str | None = None) -> None:
    series = [(s, lab, col, fil, load(src, key, s)["delta_pred"].to_numpy())
              for s, lab, col, fil in STRATA]
    # Bin width follows the *smaller* series: on a log axis an empty bin drops
    # the step line to the floor, so too-fine bins turn the high-PIP series into
    # a comb of single-pair spikes rather than a distribution.
    if binw is None:
        n_min = min(len(a) for *_, a in series)
        binw = 0.01 if n_min >= 1000 else 0.02 if n_min >= 200 else 0.05
    edges = np.arange(-LIMIT, LIMIT + binw, binw)
    # Shares by default: with 5,901 background pairs against 367 causal ones,
    # raw counts would flatten the causal row to nothing. The interaction subset
    # is small enough that the counts are the more honest reading - a bar of 3
    # pairs should look like 3 pairs, not like 7.5% of a stratum.
    if scale is None:
        scale = "count" if key == "expr_in_interaction" else "share"

    height = 5.4 if bare else 6.6
    # Counts get one linear axis per row, each with its own tick step, so the
    # rows cannot share y; shares are comparable and do.
    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True,
                             sharey=(scale == "share"))
    ps.apply()

    # Small multiples rather than one overlaid axes: the two distributions cover
    # the same x range and the causal set is the minority, so overlaid outlines
    # spend most of their length hidden behind the background. The y axis is
    # shared either way - on shares the rows are directly comparable, on counts
    # the difference in row height *is* the difference in n.
    for ax, (stratum, lab, col, fil, a) in zip(axes, series):
        w = (None if scale == "count"
             else np.full(len(a), 100.0 / len(a)))   # share of that stratum, %
        ax.hist(a, bins=edges, weights=w, color=fil, zorder=2)
        ax.hist(a, bins=edges, weights=w, histtype="step", color=col,
                linewidth=2.0, zorder=3)
        if scale == "count":
            # Linear, ticked per COUNT_TICK, with one step of headroom for the
            # labels; a log axis would make a 1-pair bin look like a finding.
            step = COUNT_TICK[stratum]
            top = (int(np.histogram(a, bins=edges)[0].max() // step) + 2) * step
            ax.set_ylim(0, top)
            ax.yaxis.set_major_locator(MultipleLocator(step))
        else:
            ax.set_yscale("log")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ps.despine(ax)
        # Direct labels, so identity never rests on colour alone and no legend
        # box has to sit on top of the central spike.
        box = dict(facecolor=ps.SURFACE, edgecolor="none", pad=2.5)
        ax.text(0.012, 0.92, f"{lab}   n = {len(a):,}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, color=col, fontweight="bold",
                bbox=box, zorder=6)
        ax.text(0.988, 0.92,
                f"median |Δ| {np.median(np.abs(a)):.4f}    "
                f"|Δ|>0.10 {(np.abs(a) > 0.10).mean():.0%}    "
                f"Δ<0 {(a < 0).mean():.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=ps.INK_2, bbox=box, zorder=6)

    axes[0].set_xlim(-LIMIT, LIMIT)
    if scale == "share":
        # Headroom above the tallest bar, so the labels never sit on a mark.
        lo, hi = axes[0].get_ylim()
        axes[0].set_ylim(lo, hi * 4.0)
    axes[1].set_xlabel("Δ predicted binding probability   (p$_{alt}$ − p$_{ref}$)")
    fig.supylabel("variant × fragment pairs" if scale == "count"
                  else "share of that stratum's pairs (%)",
                  fontsize=11, color=ps.INK_2, x=0.022)

    if not bare:
        axes[0].set_title(TITLES[key], loc="left", pad=10)
        legend = (
            "GTEx fine-mapped cis-eQTLs joined to miRBench and SAEC AGO2 fragments, kept where the "
            "assigned miRNA is in the top 100 expressed miRNAs of that eQTL's own tissue (median CPM, "
            f"GTEx v11 small-RNA){SUBSET[key]}. Reference and alternate fragments scored by the "
            "sequence CNN (cnn_branches_mirbind_embed16_restruct.pt); one row per unique variant x "
            f"fragment pair, {binw:g}-wide bins. "
            + (f"Counts are absolute and linear, ticked every {COUNT_TICK['gt_0_9']} pairs on the "
               f"PIP>0.9 row and every {COUNT_TICK['lt_0_01']} on the PIP<0.01 row, which holds "
               f"{len(series[1][4]) / len(series[0][4]):.0f}x as many pairs."
               if scale == "count" else
               "Each stratum is normalised to its own n, since the PIP<0.01 background outnumbers "
               "the PIP>0.9 set; the share axis is logarithmic."))
        # The caption runs a line longer for the narrower configurations, so the
        # bottom margin follows the wrap rather than a constant that fits one key.
        lines = textwrap.wrap(legend, width=96)
        fig.subplots_adjust(left=0.115, right=0.975, top=0.925, hspace=0.12,
                            bottom=0.17 + 0.020 * len(lines))
        fig.text(0.030, 0.035, "\n".join(lines),
                 fontsize=8.5, color=ps.INK_2, va="bottom", linespacing=1.5)
    else:
        fig.subplots_adjust(left=0.115, right=0.975, top=0.985, bottom=0.115,
                            hspace=0.12)

    name = name or (f"panel8_{key}_bare" if bare else f"panel8_delta_pred_{key}")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)

    for _, lab, _, _, a in series:
        print(f"{key:20s} {lab:11s} {len(a):5,} pairs  mean {a.mean():+.4f}  "
              f"median |Δ| {np.median(np.abs(a)):.4f}  "
              f"|Δ|>0.10 {(np.abs(a) > 0.10).mean():5.1%}  Δ<0 {(a < 0).mean():5.1%}")
    print(f"wrote {OUT}/{name}.{{png,svg,pdf}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC,
                    help="Directory holding delta_pred_<key>_pip_*.tsv.")
    ap.add_argument("--key", default="expr_ALL", choices=sorted(TITLES),
                    help="expr_ALL (top-100 expressed) or expr_in_interaction "
                         "(also inside the IntaRNA footprint).")
    ap.add_argument("--bare", action="store_true",
                    help="Drop the title and legend paragraph, for the poster.")
    ap.add_argument("--scale", choices=("share", "count"), default=None,
                    help="y axis: share of each stratum, or absolute pair "
                         "counts. Default: counts for expr_in_interaction, "
                         "shares for expr_ALL.")
    ap.add_argument("--bin", dest="binw", type=float, default=None,
                    help="Bin width; default 0.01, or 0.025 for small subsets.")
    ap.add_argument("--width", type=float, default=7.4)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    main(src=a.src, key=a.key, bare=a.bare, name=a.name, width=a.width,
         binw=a.binw, scale=a.scale)
