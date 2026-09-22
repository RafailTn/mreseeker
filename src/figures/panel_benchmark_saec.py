#!/usr/bin/env python3
"""
Published miRBench predictors against the two models in this repository, with
SAEC added as a fifth evaluation set.

This is panel_benchmark.py extended along the dataset axis. The original panel
retyped the four-column benchmark table from the miRBench README; that table has
no SAEC column, because SAEC is not one of the miRBench sets. So the reference
numbers here are measured rather than quoted: every predictor was re-run from
the miRBench package over all five sets, and the four miRBench columns came back
identical to the published table to four decimals, which is what licenses
putting the measured SAEC column beside them.

Both inputs are read from results/, never retyped:
  results/mirbench_predictor_evaluation.csv  - the published predictors
  results/head_to_head.csv                   - the two models in this repository

The model list is a cut of panel_benchmark.py's: two published predictors and
the chance line, against the two models in this repository.

    python3 src/figures/panel_benchmark_saec.py [--width 6.2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"
H2H = REPO / "results" / "head_to_head.csv"
MEASURED = REPO / "results" / "mirbench_predictor_evaluation.csv"

# A cut of panel_benchmark.py's model list: the strongest published predictor
# (miRBenchCNN_Manakov), the strongest non-neural one (TargetScanCnn), and the
# chance line. Order here is only the lookup order; bars are sorted by mean APS.
MIRBENCH = [
    "miRBenchCNN_Manakov",
    "TargetScanCnn_McGeary2019",
    "Random",
]

# SAEC last in every group, and darkest: the eye reads the ramp left to right
# and lands on the out-of-distribution set, which is the comparison the panel
# exists to make. No new hue is spent, so the poster's two accent slots stay
# with the two model tracks.
SETS = ["Klimentova test", "Hejret test", "Manakov test", "Manakov leftout", "SAEC"]
TONES = [ps.BLUE_RAMP[1], ps.BLUE_RAMP[3], ps.BLUE_RAMP[5], "#1d5490", "#0b2f57"]

# results/mirbench_predictor_evaluation.csv names the sets after their files.
DATASET_ALIAS = {
    "AGO2_eCLIP_Klimentova2022_test": "Klimentova test",
    "AGO2_CLASH_Hejret2023_test": "Hejret test",
    "AGO2_eCLIP_Manakov2022_test": "Manakov test",
    "AGO2_eCLIP_Manakov2022_leftout": "Manakov leftout",
    "GSE304955_novel_labelled": "SAEC",
}

OURS = {"this work: sequence CNN": "sequence CNN",
        "this work: IntaRNA + CatBoost": "IntaRNA features + CatBoost"}


def measured() -> dict[str, tuple[float, ...]]:
    """The published predictors, as re-run over all five sets."""
    if not MEASURED.exists():
        raise SystemExit(f"{MEASURED} not found; run the miRBench evaluation first")
    df = pd.read_csv(MEASURED)
    df["dataset"] = df["dataset"].map(DATASET_ALIAS).fillna(df["dataset"])
    out = {}
    for name in MIRBENCH:
        sub = df[df.predictor == name]
        row = dict(zip(sub.dataset, sub.average_precision))
        missing = [s for s in SETS if s not in row]
        if missing:
            raise SystemExit(f"{name}: no average precision for {missing} in {MEASURED}")
        out[name] = tuple(float(row[s]) for s in SETS)
    return out


def ours_from_head_to_head() -> dict[str, tuple[float, ...]]:
    if not H2H.exists():
        raise SystemExit(f"{H2H} not found; run src/benchmark/head_to_head.py first")
    df = pd.read_csv(H2H)
    out = {}
    for label, model in OURS.items():
        row = {r.dataset: r.aps for r in df[df.model == model].itertuples()}
        missing = [s for s in SETS if s not in row]
        if missing:
            raise SystemExit(f"{model}: no APS for {missing} in {H2H}")
        out[label] = tuple(row[s] for s in SETS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=float, default=6.2)
    ap.add_argument("--name", default="panel_benchmark_saec")
    a = ap.parse_args()

    data = {**ours_from_head_to_head(), **measured()}
    # Best mean first, so the reader's eye starts where our models sit.
    order = sorted(data, key=lambda k: -float(np.mean(data[k])))

    ps.apply(9.5)
    height = 4.2
    fig, ax = plt.subplots(figsize=(a.width, height))
    fig.subplots_adjust(left=0.62 / a.width, right=1 - 0.62 / a.width,
                        top=1 - 0.42 / height, bottom=1.30 / height)

    n = len(SETS)
    bw = 0.76 / n
    for j, (s, tone) in enumerate(zip(SETS, TONES)):
        xs = np.arange(len(order)) + (j - (n - 1) / 2) * bw
        ax.bar(xs, [data[k][j] for k in order], width=bw * 0.94, color=tone,
               edgecolor="none", zorder=3, label=s)

    # Ours are marked by the label, not by a colour: the ramp is spent on the
    # evaluation sets.
    for i, k in enumerate(order):
        if k in OURS:
            ax.axvspan(i - 0.5, i + 0.5, color=ps.GLUON_FILL, zorder=1)

    ax.axhline(0.5, color=ps.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(1.015, 0.5, "chance", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=8, color=ps.MUTED, clip_on=False)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([k.replace("this work: ", "") for k in order], rotation=24,
                       ha="right", fontsize=9)
    for t, k in zip(ax.get_xticklabels(), order):
        if k in OURS:
            t.set_fontweight("bold")
            t.set_color(ps.ACCENT_GLUON)
    ax.set_ylim(0.45, 0.93)
    ax.set_ylabel("average precision")
    ax.grid(axis="y", zorder=0)
    ps.despine(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=5, frameon=False,
              labelcolor=ps.INK_2, handlelength=1.1, columnspacing=1.4, fontsize=8.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{a.name}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {OUT}/{a.name}.{{png,svg,pdf}}")
    head = "  ".join(f"{s:>16}" for s in SETS)
    print(f"  {'':<32}{head}")
    for k in order:
        print(f"  {k:<32}" + "  ".join(f"{v:>16.3f}" for v in data[k]))


if __name__ == "__main__":
    main()
