"""
Panel 5 - precision-recall curves behind the Manakov APS numbers.

Average precision is the area under the precision-recall curve, so a single APS
value hides where along the ranking two models differ. Two models can reach the
same APS with very different shapes: one precise at the top of its ranking and
weak in the tail, the other uniformly middling. The curves show which.

Each facet also marks the 0.5 threshold that `predict_target.py` and
`predict_cnn.py` use for their `prediction` column, because that is the one
point on the curve a user of the shipped pipeline actually operates at.

The horizontal dashed line is the positive rate: the precision of a random
ranking, and the floor APS is measured against.

Inputs are per-row prediction files with `label` and `interaction_probability`,
the same files src/benchmark/head_to_head.py reads.

Usage
-----
python3 src/figures/panel5_pr_curves.py [--bare]
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
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"
MSC = REPO.parent / "msc-thesis"

THRESHOLD = 0.5
MAX_POINTS = 2500     # per curve; the raw curve has one vertex per distinct score

FACETS = [
    ("Manakov v7 test",
     MSC / "data" / "manakov_test_errors_v7_restructure.tsv",
     REPO / "results" / "preds" / "gluon_manakov_test.tsv"),
    ("Manakov leftout",
     MSC / "data" / "manakov_leftout_errors_v7_restructure.tsv",
     REPO / "results" / "preds" / "gluon_manakov_leftout.tsv"),
]
SERIES = [("sequence CNN", ps.ACCENT_CNN), ("IntaRNA features + CatBoost", ps.ACCENT_GLUON)]


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"ERROR: missing prediction file {path}")
    df = pd.read_csv(path, sep="\t", usecols=["label", "interaction_probability"])
    df = df.dropna()
    return df["label"].to_numpy(), df["interaction_probability"].to_numpy(float)


def thin(recall: np.ndarray, precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep ~MAX_POINTS vertices spread evenly in recall.

    The full curve for 324k rows has hundreds of thousands of vertices, which is
    invisible detail at poster size and makes the vector outputs enormous.
    Sampling by recall rather than by index keeps the steep low-recall end, where
    few rows move the curve a long way, as well resolved as the flat middle.
    """
    if len(recall) <= MAX_POINTS:
        return recall, precision
    order = np.argsort(recall)
    r, p = recall[order], precision[order]
    targets = np.linspace(r[0], r[-1], MAX_POINTS)
    idx = np.unique(np.clip(np.searchsorted(r, targets), 0, len(r) - 1))
    return r[idx], p[idx]


def operating_point(y: np.ndarray, s: np.ndarray, thr: float) -> tuple[float, float]:
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    return tp / max(int(y.sum()), 1), tp / max(int(pred.sum()), 1)


def main(bare: bool = False, name: str | None = None) -> None:
    ps.apply(9.5)
    height = 4.7 if bare else 5.9
    fig, axes = plt.subplots(1, 2, figsize=(12.6, height), sharey=True,
                             gridspec_kw=dict(wspace=0.10))
    fig.subplots_adjust(left=0.065, right=0.985, top=1 - 0.40 / height,
                        bottom=(0.62 if bare else 1.62) / height)

    summary = []
    for ax, (title, cnn_path, glu_path) in zip(axes, FACETS):
        prevalence = None
        handles = []
        for (label, colour), path in zip(SERIES, (cnn_path, glu_path)):
            y, s = load(path)
            prevalence = y.mean()
            precision, recall, _ = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            r, p = thin(recall, precision)
            line, = ax.plot(r, p, color=colour, lw=2.2, zorder=4,
                            solid_capstyle="round",
                            label=f"{label}   AP {ap:.3f}")
            handles.append(line)
            orx, ory = operating_point(y, s, THRESHOLD)
            ax.plot([orx], [ory], "o", ms=8, mfc=ps.SURFACE, mec=colour,
                    mew=2.0, zorder=5)
            summary.append((title, label, ap, orx, ory, len(y)))

        ax.axhline(prevalence, color=ps.MUTED, lw=1.3, ls=(0, (4, 3)), zorder=2)
        # Left end of the line: the right end is where every curve converges on
        # the baseline at recall 1.0, so a label there sits on top of both.
        ax.text(0.015, prevalence + 0.008, f"random ranking = {prevalence:.3f}",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=8.5, color=ps.MUTED)

        handles.append(plt.Line2D([], [], marker="o", ls="", ms=8, mfc=ps.SURFACE,
                                  mec=ps.INK_2, mew=2.0,
                                  label=f"threshold {THRESHOLD}"))
        # Above the baseline label, in the band both curves leave empty: at low
        # recall they run along precision 1.0, so the lower-left interior is clear.
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.20),
                  frameon=False, labelcolor=ps.INK_2, handlelength=1.8)

        ax.set_title(title, loc="left", fontsize=12, fontweight="bold",
                     color=ps.INK, pad=10)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0.44, 1.005)
        ax.set_xlabel("recall")
        ax.grid(axis="both", zorder=0)
        ps.despine(ax)
    axes[0].set_ylabel("precision")

    if not bare:
        legend = (
            "Precision-recall curves for both models on the same rows of each "
            "Manakov evaluation set; the average precision (AP) in each key is the "
            "area under that curve. Open circles mark the 0.5 probability "
            "threshold the shipped pipelines use for their binary prediction. The "
            "dashed line is the positive rate, the precision of a random ranking.")
        fig.text(0.030, 0.20 / height, "\n".join(textwrap.wrap(legend, width=176)),
                 fontsize=8.5, color=ps.INK_2, va="bottom", linespacing=1.5)

    name = name or ("panel5_bare" if bare else "panel5_pr_curves")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.{{png,svg,pdf}}")
    print(f"\n{'set':<18}{'model':<30}{'AP':>7}{'recall@0.5':>12}{'prec@0.5':>10}{'n':>9}")
    for t, m, ap, r, p, n in summary:
        print(f"{t:<18}{m:<30}{ap:>7.4f}{r:>12.3f}{p:>10.3f}{n:>9,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bare", action="store_true",
                    help="Drop the legend paragraph, for the poster.")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    main(bare=a.bare, name=a.name)
