"""
Panel 9 - does a variant's effect on predicted binding depend on how well the
reference allele bound in the first place?

The histogram panel (panel8_delta_pred_expr.py) shows how big the allele effect
is; it cannot show where in the model's range that effect happens. A shift of
-0.3 starting from p(ref) = 0.90 leaves the site predicted bound; the same shift
from 0.55 carries it across the decision threshold. Plotting delta_pred against
p(ref) puts both facts on one axes.

What the wedge is
-----------------
p(alt) = p(ref) + delta is a probability, so every point must satisfy
0 <= p(ref) + delta <= 1. That confines the cloud to a wedge whose edges are
drawn: no scoring quirk, just arithmetic. Points are far from those edges here
because the fragments in this set are CNN true positives, whose p(ref) sits
between 0.27 and 0.95.

The 0.5 line is the threshold panel5_pr_curves.py reports the operating point
at; pairs whose p(ref) is above it and whose p(alt) is below have their call
flipped by the variant, and they are counted per stratum in the legend.

    pixi run -m dependencies/cnn python src/figures/panel9_delta_vs_p_ref.py
    pixi run -m dependencies/cnn python src/figures/panel9_delta_vs_p_ref.py \
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
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402
from panel8_delta_pred_expr import LIMIT, SRC, STRATA, SUBSET, load  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "figures"
THRESHOLD = 0.5          # the operating point panel5_pr_curves.py reports at

TITLES = {
    "expr_ALL": "Large allele effects start from mid-range predictions",
    "expr_in_interaction": "Inside the duplex, the losses start from bound sites",
}


def main(src: Path = SRC, key: str = "expr_ALL", bare: bool = False,
         name: str | None = None, width: float = 7.4,
         zoom_y: bool = False) -> None:
    series = [(lab, col, load(src, key, s)) for s, lab, col, _ in STRATA]

    height = 5.0 if bare else 6.2
    fig, ax = plt.subplots(figsize=(width, height))
    ps.apply()

    # The wedge: p(alt) = p(ref) + delta has to stay inside [0, 1].
    xs = np.array([-LIMIT, 0.0, LIMIT])
    ax.plot(xs, np.clip(-xs, 0, 1), color=ps.GRID, linewidth=1.0, zorder=1)
    ax.plot(xs, np.clip(1 - xs, 0, 1), color=ps.GRID, linewidth=1.0, zorder=1)
    ax.axhline(THRESHOLD, color=ps.MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=2)
    ax.text(-LIMIT + 0.012, THRESHOLD + 0.012, f"threshold {THRESHOLD}",
            fontsize=8.5, color=ps.MUTED, va="bottom")

    # Background first, causal set on top: 5,901 grey points would otherwise
    # bury the 367 that the panel is about.
    for i, (lab, col, d) in enumerate(reversed(series)):
        x, y = d["delta_pred"].to_numpy(), d["p_ref"].to_numpy()
        flip = int(((y >= THRESHOLD) & (y + x < THRESHOLD)).sum())
        ax.scatter(x, y, s=13 if i else 9, c=col, alpha=0.75 if i else 0.30,
                   linewidths=0.6 if i else 0.0,
                   edgecolors=ps.SURFACE if i else "none", zorder=4 + i,
                   label=f"{lab}   n = {len(d):,}   call flipped: {flip} "
                         f"({flip / len(d):.1%})")

    ax.set_xlim(-LIMIT, LIMIT)
    # The full probability axis by default: the empty bottom half is the fact
    # that the CNN never scores these fragments below ~0.27, which --zoom-y
    # hides in exchange for resolution on the occupied band.
    if zoom_y:
        lo = min(d["p_ref"].min() for _, _, d in series)
        ax.set_ylim(max(0.0, np.floor(lo * 20) / 20), 1)
    else:
        ax.set_ylim(0, 1)
    ax.set_xlabel("Δ predicted binding probability   (p$_{alt}$ − p$_{ref}$)")
    ax.set_ylabel("p(ref): predicted binding of the reference allele")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ps.despine(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left",
              bbox_to_anchor=(0.0, 1.005), ncol=1, fontsize=9,
              labelcolor=ps.INK_2, borderaxespad=0.0, scatterpoints=3)

    if not bare:
        ax.set_title(TITLES[key], loc="left", pad=44)
        legend = (
            "One point per unique variant x fragment pair: the sequence CNN's probability for the "
            "reference fragment against the vertical, its change under the alternate allele against "
            "the horizontal. GTEx fine-mapped cis-eQTLs on miRBench and SAEC AGO2 fragments, kept "
            "where the assigned miRNA is in the top 100 expressed miRNAs of that eQTL's own tissue"
            f"{SUBSET[key]}. The pale guides are where p(alt) would leave [0, 1]; the dashed line is "
            "the 0.5 operating point, and 'call flipped' counts pairs carried from above it to below.")
        lines = textwrap.wrap(legend, width=96)
        fig.subplots_adjust(left=0.105, right=0.975, top=0.845,
                            bottom=0.17 + 0.021 * len(lines))
        fig.text(0.030, 0.035, "\n".join(lines),
                 fontsize=8.5, color=ps.INK_2, va="bottom", linespacing=1.5)
    else:
        fig.tight_layout()

    name = name or (f"panel9_{key}_bare" if bare else f"panel9_delta_vs_p_ref_{key}")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)

    for lab, _, d in series:
        x, y = d["delta_pred"].to_numpy(), d["p_ref"].to_numpy()
        r = np.corrcoef(np.abs(x), y)[0, 1]
        print(f"{key:20s} {lab:11s} {len(d):5,} pairs  p(ref) median {np.median(y):.3f}  "
              f"corr(|Δ|, p_ref) {r:+.3f}  "
              f"flipped {((y >= THRESHOLD) & (y + x < THRESHOLD)).sum():4d}")
    print(f"wrote {OUT}/{name}.{{png,svg,pdf}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--key", default="expr_ALL", choices=sorted(TITLES))
    ap.add_argument("--bare", action="store_true")
    ap.add_argument("--zoom-y", action="store_true",
                    help="Clip the y axis to the occupied p(ref) band.")
    ap.add_argument("--width", type=float, default=7.4)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    main(src=a.src, key=a.key, bare=a.bare, name=a.name, width=a.width,
         zoom_y=a.zoom_y)
