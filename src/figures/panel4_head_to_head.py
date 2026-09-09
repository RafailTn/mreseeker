"""
Panel 4 - sequence CNN against the feature model, on every evaluation set.

The poster asks whether sequence alone matches engineered thermodynamics, but
until this panel nothing put the two models on one axis: the Pareto panel is
AutoGluon-only and the generalisation panel is CNN-only.

Why the intervals are not decoration
------------------------------------
The evaluation sets run from 954 rows to 324,171. Drawn as bare dots, a 0.004
difference on Klimentova and the same difference on Manakov test would look
equally solid, when one is well inside its sampling error and the other is many
times outside it. The bars are percentile bootstrap over rows, computed by
src/benchmark/head_to_head.py - APS is not a mean, so it has no closed-form
standard error and resampling is the honest option.

The grey tick on each row is that set's positive rate, which is exactly the APS
a random classifier scores there. It is drawn because the sets are not all
balanced identically (Manakov test is 51.5% positive, the rest ~50%), so the
distance above it - not the absolute height - is what the two models should be
compared on.

Usage
-----
python3 src/figures/panel4_head_to_head.py [--bare]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "head_to_head.csv"
OUT = REPO / "results" / "figures"

CNN = "sequence CNN"
GLUON = "IntaRNA features + CatBoost"

# Top to bottom: what the model was trained on, then other AGO2 datasets, then
# cell lines it has never seen. Reading down the panel is reading outwards from
# the training distribution, which is the claim the panel exists to make.
ORDER = ["Manakov test", "Manakov leftout",
         "Hejret test", "Hejret train", "Klimentova test",
         "SAEC", "HCT116"]

# Which group each set belongs to, for the sentence in the legend. Built from
# the sets actually present rather than hard-coded, so a run that omits one
# does not describe a row the reader cannot see.
TRAINED_ON = {"Manakov test", "Manakov leftout"}
CELL_LINES = {"SAEC", "HCT116"}

# Vertical offset of each model from its row centre.
DODGE = 0.19


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def describe(datasets: list[str]) -> str:
    """The ordering sentence, naming only sets that are on the figure."""
    external = [d for d in datasets
                if d not in TRAINED_ON and d not in CELL_LINES]
    lines = [d for d in datasets if d in CELL_LINES]
    parts = ["Sets are ordered outwards from the training distribution"]
    if any(d in TRAINED_ON for d in datasets):
        parts.append("Manakov is what both models were trained on")
    if external:
        parts.append(f"{_join(external)} are independent AGO2 datasets")
    if lines:
        verb = "is a cell line" if len(lines) == 1 else "are cell lines"
        parts.append(f"{_join(lines)} {verb} neither model has seen")
    return parts[0] + ": " + ", ".join(parts[1:]) + "."


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} not found. Build it with:\n"
            f"  python3 src/benchmark/head_to_head.py --cnn ... --gluon ... "
            f"-o {path}")
    df = pd.read_csv(path)
    # Results files written before the shipped model changed name the series
    # after LightGBM. Normalise rather than fail: the rows are still valid.
    df["model"] = df.model.replace(
        {"IntaRNA features + LightGBM": GLUON}, regex=False)
    seen = [d for d in ORDER if d in set(df.dataset)]
    rest = [d for d in df.dataset.unique() if d not in ORDER]
    df["_rank"] = df.dataset.map({d: i for i, d in enumerate(seen + rest)})
    return df.sort_values("_rank")


def main(bare: bool = False, src: Path = SRC, name: str | None = None) -> None:
    df = load(src)
    datasets = list(dict.fromkeys(df.dataset))
    ps.apply(9.5)

    height = 0.78 * len(datasets) + (1.25 if bare else 2.65)
    fig, ax = plt.subplots(figsize=(12.6, height))
    fig.subplots_adjust(left=0.165, right=0.885, top=1 - 0.35 / height,
                        bottom=(0.62 if bare else 1.55) / height)

    counts: dict[str, int] = {}
    for i, ds in enumerate(datasets):
        y = len(datasets) - 1 - i
        rows = df[df.dataset == ds]
        pair = {r.model: r for r in rows.itertuples()}
        cnn, glu = pair.get(CNN), pair.get(GLUON)

        # Random baseline for this set: APS of a coin weighted to its prevalence.
        base = (cnn or glu).pos_rate
        ax.plot([base, base], [y - DODGE - 0.12, y + DODGE + 0.12], "-",
                color=ps.MUTED, lw=1.6, zorder=2)

        # The two models get their own sub-row. Sharing one line made the
        # interval, the connector and the two markers a single stripe with no
        # readable endpoints - on the small sets the two intervals overlap each
        # other as well, which is exactly where the reader needs to see them.
        for r, colour, dy in ((cnn, ps.ACCENT_CNN, DODGE),
                              (glu, ps.ACCENT_GLUON, -DODGE)):
            if r is None:
                continue
            yy = y + dy
            if pd.notna(r.ci_lo):
                # Capped bar: the caps are what make the extent legible when two
                # intervals overlap, which a plain line cannot show.
                ax.plot([r.ci_lo, r.ci_hi], [yy, yy], "-", color=colour, lw=1.4,
                        alpha=0.9, zorder=3, solid_capstyle="butt")
                for x_end in (r.ci_lo, r.ci_hi):
                    ax.plot([x_end, x_end], [yy - 0.075, yy + 0.075], "-",
                            color=colour, lw=1.4, alpha=0.9, zorder=3)
            ax.plot([r.aps], [yy], "o", ms=9, mfc=colour, mec=ps.SURFACE,
                    mew=1.8, zorder=5)

        if cnn is not None and glu is not None:
            d = cnn.aps - glu.aps
            ax.annotate(f"{d:+.3f}", (1.008, y), xycoords=("axes fraction", "data"),
                        ha="left", va="center", fontsize=9.5,
                        color=ps.ACCENT_CNN if d > 0 else ps.ACCENT_GLUON,
                        fontweight="bold")
        counts[ds] = int((cnn or glu).n)

    ax.annotate("Δ", (1.008, len(datasets) - 0.42),
                xycoords=("axes fraction", "data"), ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=ps.INK)

    ax.set_yticks(range(len(datasets)))
    # Row size goes in the tick label rather than a separate gutter column,
    # which would sit exactly where the dataset name already is.
    ax.set_yticklabels([f"{d}\nn = {counts[d]:,}" for d in datasets[::-1]],
                       fontsize=10.5, linespacing=1.45)
    ax.set_ylim(-0.6, len(datasets) - 0.4)
    ax.set_xlabel("average precision")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", zorder=0)
    ps.despine(ax, keep=("bottom",))

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=10, mfc=ps.ACCENT_CNN,
                   mec=ps.SURFACE, mew=1.8, label=CNN),
        plt.Line2D([], [], marker="o", ls="", ms=10, mfc=ps.ACCENT_GLUON,
                   mec=ps.SURFACE, mew=1.8, label=GLUON),
        plt.Line2D([], [], marker="|", ls="", ms=11, color=ps.MUTED, mew=1.6,
                   label="random baseline (= positive rate)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=3, frameon=False, labelcolor=ps.INK_2, handletextpad=0.5,
              columnspacing=1.8)

    if not bare:
        legend = (
            "Average precision of both models on the same rows of each evaluation "
            "set, one sub-row per model. Capped bars are 95% percentile bootstrap "
            "intervals over rows; on the largest sets the interval is narrower "
            "than the marker, so only the caps show. The grey line is the set's "
            "positive rate, which is the average precision a random classifier "
            "achieves there, so the distance to the right of it is the signal. "
            "\u0394 is CNN minus feature model. " + describe(datasets))
        fig.text(0.030, 0.20 / height,
                 "\n".join(textwrap.wrap(legend, width=176)),
                 fontsize=8.5, color=ps.INK_2, va="bottom", linespacing=1.5)

    name = name or ("panel4_bare" if bare else "panel4_head_to_head")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.{{png,svg,pdf}}")

    both = df.pivot_table(index="dataset", columns="model", values="aps")
    if both.shape[1] == 2:
        both = both.reindex([d for d in datasets if d in both.index])
        both["delta"] = both[CNN] - both[GLUON]
        print("\n" + both.to_string(float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=SRC)
    ap.add_argument("--bare", action="store_true",
                    help="Drop the legend paragraph, for the poster.")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    main(bare=a.bare, src=a.csv, name=a.name)
