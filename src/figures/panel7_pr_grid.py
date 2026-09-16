"""
Panel 7 - precision-recall curves for every evaluation set, on one grid.

Panel 6 shows the two Manakov sets large; this is the same comparison across all
of them, so the reader can see whether the shape of the CNN's advantage on the
training distribution survives on independent AGO2 data and an unseen cell line.
Facets read left to right, top to bottom, outwards from the training
distribution - the same order as the head-to-head panel.

Axes are shared on purpose: with a common precision scale, a facet whose curves
sit lower is a genuinely harder set, not an artefact of per-facet limits.

HCT116, Hejret test and Klimentova test are left out; panel 4 carries all of
them with intervals, and the two small test sets (~1,000 rows each) give curves
too stepped to read at grid size.

Label recovery and its row-order checks are borrowed from
src/benchmark/head_to_head.py, since predict_target.py drops the label column.

Usage
-----
python3 src/figures/panel7_pr_grid.py [--bare]
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
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmark"))
import poster_style as ps  # noqa: E402
from panel6_pr_curves import THRESHOLD, operating_point, thin  # noqa: E402
from head_to_head import load as load_scores  # noqa: E402

REPO = HERE.parents[1]
OUT = REPO / "results" / "figures"
MSC = REPO.parent / "msc-thesis"
D, P = MSC / "data", REPO / "results" / "preds"

# (title, CNN predictions, feature-model predictions, truth table if the
# feature-model file carries no label column)
FACETS = [
    ("Manakov test", D / "manakov_test_errors_v7_restructure.tsv",
     P / "gluon_manakov_test.tsv", None),
    ("Manakov leftout", D / "manakov_leftout_errors_v7_restructure.tsv",
     P / "gluon_manakov_leftout.tsv", None),
    ("Hejret train", D / "hejret_train_errors_v7_restructure.tsv",
     P / "gluon_hejret_train.tsv", D / "AGO2_CLASH_Hejret2023_train_v7.tsv"),
    ("SAEC", MSC / "results" / "novel" / "gse304955_labelled_pred.tsv",
     P / "gluon_gse304955.tsv", D / "gse304955_novel_labelled_v7.tsv"),
]
CNN_NAME, GLU_NAME = "sequence CNN", "IntaRNA features + CatBoost"
# Square-ish grid for whatever sets are listed: 4 facets -> 2x2, 6 -> 2x3.
NCOLS = min(3, math.ceil(math.sqrt(len(FACETS))))


def main(bare: bool = False, name: str | None = None) -> None:
    ps.apply(9.5)
    nrows = math.ceil(len(FACETS) / NCOLS)
    # Row height follows facet width, so a 2-column grid is not squashed flat.
    height = (12.6 / NCOLS) * 0.78 * nrows + (0.75 if bare else 1.65)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(12.6, height),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.08, hspace=0.30))
    fig.subplots_adjust(left=0.065, right=0.985, top=1 - 0.75 / height,
                        bottom=(0.55 if bare else 1.45) / height)
    flat = list(axes.flat)

    rows = []
    for ax, (title, cnn_path, glu_path, truth) in zip(flat, FACETS):
        labels = {title: truth} if truth is not None else {}
        texts, prevalence, n = [], None, None
        for series, colour, path in ((CNN_NAME, ps.ACCENT_CNN, cnn_path),
                                     (GLU_NAME, ps.ACCENT_GLUON, glu_path)):
            _, y, s = load_scores(f"{title}:{path}", labels)
            prevalence, n = y.mean(), len(y)
            precision, recall, _ = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            r, p = thin(recall, precision)
            ax.plot(r, p, color=colour, lw=1.9, zorder=4, solid_capstyle="round")
            orx, ory = operating_point(y, s, THRESHOLD)
            ax.plot([orx], [ory], "o", ms=6.5, mfc=ps.SURFACE, mec=colour,
                    mew=1.7, zorder=5)
            texts.append((f"AP {ap:.3f}", colour))
            rows.append((title, series, ap, orx, ory, n))

        ax.axhline(prevalence, color=ps.MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)
        # Per-facet AP values sit in the lower-left interior, which every curve
        # leaves empty: at low recall they all run along precision ~1.
        for k, (t, colour) in enumerate(texts):
            ax.text(0.04, 0.30 - 0.11 * k, t, transform=ax.transAxes,
                    fontsize=9.5, fontweight="bold", color=colour, va="bottom")
        ax.set_title(f"{title}   n = {n:,}", loc="left", fontsize=10.5,
                     fontweight="bold", color=ps.INK, pad=7)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0.42, 1.005)
        ax.grid(axis="both", zorder=0)
        ps.despine(ax)

    for ax in flat[len(FACETS):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("recall")
    for ax in axes[:, 0]:
        ax.set_ylabel("precision")

    # One shared key: repeating it in six facets would be six copies of ink.
    handles = [
        plt.Line2D([], [], color=ps.ACCENT_CNN, lw=2.2, label=CNN_NAME),
        plt.Line2D([], [], color=ps.ACCENT_GLUON, lw=2.2, label=GLU_NAME),
        plt.Line2D([], [], marker="o", ls="", ms=7, mfc=ps.SURFACE,
                   mec=ps.INK_2, mew=1.7, label=f"threshold {THRESHOLD}"),
        plt.Line2D([], [], color=ps.MUTED, lw=1.3, ls=(0, (4, 3)),
                   label="random ranking (positive rate)"),
    ]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.060, 1 - 0.08 / height), ncol=4,
               frameon=False, labelcolor=ps.INK_2, handlelength=1.8,
               columnspacing=2.0)

    if not bare:
        legend = (
            "Precision-recall curves for both models on the same rows of each "
            "evaluation set, with shared axes so facet heights are comparable. "
            "AP is the area under each curve. Open circles mark the 0.5 "
            "probability threshold the shipped pipelines use for their binary "
            "prediction; the dashed line is the positive rate, the precision of a "
            "random ranking. Facets run outwards from the training distribution: "
            "Manakov, then the independent Hejret AGO2 dataset, then the unseen "
            "SAEC cell line.")
        fig.text(0.030, 0.18 / height, "\n".join(textwrap.wrap(legend, width=176)),
                 fontsize=8.5, color=ps.INK_2, va="bottom", linespacing=1.5)

    name = name or ("panel7_bare" if bare else "panel7_pr_grid")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.{{png,svg,pdf}}")
    print(f"\n{'set':<17}{'model':<30}{'AP':>7}{'recall@0.5':>12}{'prec@0.5':>10}")
    for t, m, ap, r, p, _ in rows:
        print(f"{t:<17}{m:<30}{ap:>7.4f}{r:>12.3f}{p:>10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bare", action="store_true",
                    help="Drop the legend paragraph, for the poster.")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    main(bare=a.bare, name=a.name)
