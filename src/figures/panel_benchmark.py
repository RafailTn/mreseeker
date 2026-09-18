#!/usr/bin/env python3
"""
Published miRBench predictors against the two models in this repository.

The reference numbers are the benchmark table in the miRBench repository
(github.com/katarinagresova/miRBench), average precision on four evaluation
sets. Our two models are not retyped: they are read from
results/head_to_head.csv, which src/benchmark/head_to_head.py writes from the
prediction files, so the bars cannot drift from the head-to-head panel.

    python3 src/figures/panel_benchmark.py [--width 8.0]
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

# Average precision, from the miRBench README benchmark table.
# columns: Klimentova test, Hejret test, Manakov test, Manakov leftout
MIRBENCH = {
    "miRBenchCNN_Manakov": (0.8730, 0.8694, 0.8699, 0.8603),
    "miRBenchCNN_HejretCorrected": (0.8009, 0.8878, 0.8073, 0.8179),
    "TargetScanCnn_McGeary2019": (0.7873, 0.7423, 0.8004, 0.8066),
    "miRBind_Klimentova2022": (0.7769, 0.7993, 0.7319, 0.7448),
    "miRNA_CNN_Hejret2023": (0.7579, 0.7949, 0.7310, 0.7379),
    "InteractionAwareModel_Yang2024": (0.7020, 0.7412, 0.7117, 0.6435),
    "RNACofold": (0.6847, 0.7420, 0.6375, 0.6691),
    "TargetNet_Min2021": (0.5465, 0.5533, 0.5762, 0.5959),
    "CnnMirTarget_Zheng2020": (0.5151, 0.5109, 0.5264, 0.5083),
    "Random": (0.5338, 0.5263, 0.5149, 0.5045),
}
SETS = ["Klimentova test", "Hejret test", "Manakov test", "Manakov leftout"]
OURS = {"this work: sequence CNN": "sequence CNN",
        "this work: IntaRNA + CatBoost": "IntaRNA features + CatBoost"}
TONES = [ps.BLUE_RAMP[2], ps.BLUE_RAMP[4], ps.BLUE_RAMP[6], "#0b2f57"]


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
    ap.add_argument("--width", type=float, default=8.0)
    ap.add_argument("--name", default="panel_benchmark")
    a = ap.parse_args()

    data = {**ours_from_head_to_head(), **MIRBENCH}
    # Best mean first, so the reader's eye starts where our models sit.
    order = sorted(data, key=lambda k: -float(np.mean(data[k])))

    ps.apply(9.5)
    height = 4.3
    fig, ax = plt.subplots(figsize=(a.width, height))
    fig.subplots_adjust(left=0.62 / a.width, right=1 - 0.62 / a.width,
                        top=1 - 0.42 / height, bottom=1.62 / height)

    n = len(SETS)
    bw = 0.78 / n
    for j, (s, tone) in enumerate(zip(SETS, TONES)):
        xs = np.arange(len(order)) + (j - (n - 1) / 2) * bw
        ax.bar(xs, [data[k][j] for k in order], width=bw * 0.94, color=tone,
               edgecolor="none", zorder=3, label=s)

    # Ours are marked by the label, not by a fifth colour: the four tones are
    # already spent on the evaluation sets.
    for i, k in enumerate(order):
        if k in OURS:
            ax.axvspan(i - 0.5, i + 0.5, color=ps.GLUON_FILL, zorder=1)

    ax.axhline(0.5, color=ps.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    # Outside the axes, on the line itself: inside, it sat on the right-hand
    # bars, which are the ones closest to chance.
    ax.text(1.015, 0.5, "chance", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=8, color=ps.MUTED, clip_on=False)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([k.replace("this work: ", "") for k in order], rotation=32,
                       ha="right", fontsize=8.5)
    for t, k in zip(ax.get_xticklabels(), order):
        if k in OURS:
            t.set_fontweight("bold")
            t.set_color(ps.ACCENT_GLUON)
    ax.set_ylim(0.45, 0.93)
    ax.set_ylabel("average precision")
    ax.grid(axis="y", zorder=0)
    ps.despine(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=4, frameon=False,
              labelcolor=ps.INK_2, handlelength=1.1, columnspacing=1.4, fontsize=8.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{a.name}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {OUT}/{a.name}.{{png,svg,pdf}}")
    for k in order[:4]:
        print(f"  {k:<32} " + "  ".join(f"{v:.3f}" for v in data[k]))


if __name__ == "__main__":
    main()
