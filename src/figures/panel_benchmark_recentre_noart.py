#!/usr/bin/env python3
"""
panel_benchmark_saec.py on the re-centred, ARTIFACT-dropped test sets, with the four new cell-line sets.

Test condition (data/flank_recentre/, src/cnn/flank_recentre_noart.py): flank positives re-centred on their seed
(column gene_rc) and the ARTIFACT cell removed at edge 0.25 (positives with no seed in 300 nt, subclass none,
not comp3p, base150 score 0.1 to < 0.25), with as many random negatives per miRNA (mask drop_art in
data/flank_recentre/test_noart_hi025/). The ARTIFACT cell is picked by the base 150-nt CNN's own score, so the
removed positives are ones a CNN scores low: every predictor gains, a CNN that ranks like base150 most.

Inputs:
  results/mirbench_recentre_noart_evaluation.csv          published predictors (mirbench_recentre_noart_eval.py)
  data/flank_recentre/test/pred_recentre_noart_hi025_gene_rc_<set>.tsv
                                                          sequence CNN = cnn_checkpoints/cnn_mre50_recentre_noart_hi025.pt

    python3 src/figures/panel_benchmark_recentre_noart.py [--width 7.6]
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
from sklearn.metrics import average_precision_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"
MEASURED = REPO / "results" / "mirbench_recentre_noart_evaluation.csv"
FR = REPO / "data" / "flank_recentre"
TABLE = REPO / "results" / "benchmark_recentre_noart.csv"

MIRBENCH = ["miRBenchCNN_Manakov", "TargetScanCnn_McGeary2019", "Random"]
OURS = "sequence CNN"

# file stem -> label; the miRBench sets first, then the four cell lines that are not in miRBench
SETS = {
    "klimentova_test": "Klimentova test",
    "hejret_test": "Hejret test",
    "manakov_test": "Manakov test",
    "manakov_leftout": "Manakov leftout",
    "gse304955": "SAEC",
    "gse263552": "SUM159PT",
    "gse180129_HSPC": "HSPC",
    "gse180129_Kasumi1": "Kasumi-1",
}
TONES = ps.BLUE_RAMP[1:] + ["#081f3d"]


def measured() -> dict[str, tuple[float, ...]]:
    if not MEASURED.exists():
        raise SystemExit(f"{MEASURED} not found; run src/figures/mirbench_recentre_noart_eval.py first")
    df = pd.read_csv(MEASURED)
    out = {}
    for name in MIRBENCH:
        row = dict(zip(df[df.predictor == name].dataset, df[df.predictor == name].average_precision))
        missing = [s for s in SETS if not np.isfinite(row.get(s, np.nan))]
        if missing:
            raise SystemExit(f"{name}: no average precision for {missing} in {MEASURED}")
        out[name] = tuple(float(row[s]) for s in SETS)
    return out


def ours() -> tuple[float, ...]:
    aps = []
    for s in SETS:
        p = pd.read_csv(FR / "test" / f"pred_recentre_noart_hi025_gene_rc_{s}.tsv", sep="\t",
                        usecols=["gene_rc", "label", "interaction_probability"])
        keep = ~pd.read_csv(FR / "test_noart_hi025" / f"{s}.tsv", sep="\t").drop_art.to_numpy()
        assert len(keep) == len(p), s
        keep &= ~p.gene_rc.str.upper().str.contains("[^ACGTU]").to_numpy()  # as mirbench_recentre_noart_eval.py
        aps.append(average_precision_score(p.label[keep], p.interaction_probability[keep]))
    return tuple(aps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=float, default=7.6)
    ap.add_argument("--name", default="panel_benchmark_recentre_noart")
    a = ap.parse_args()

    data = {OURS: ours(), **measured()}
    order = sorted(data, key=lambda k: -float(np.mean(data[k])))
    pd.DataFrame([dict(model=k, dataset=lab, aps=data[k][j]) for k in order for j, lab in enumerate(SETS.values())]
                 ).to_csv(TABLE, index=False)

    ps.apply(9.5)
    height = 4.4
    fig, ax = plt.subplots(figsize=(a.width, height))
    fig.subplots_adjust(left=0.62 / a.width, right=1 - 0.62 / a.width, top=1 - 0.62 / height, bottom=1.30 / height)

    n = len(SETS)
    bw = 0.80 / n
    for j, (lab, tone) in enumerate(zip(SETS.values(), TONES)):
        xs = np.arange(len(order)) + (j - (n - 1) / 2) * bw
        ax.bar(xs, [data[k][j] for k in order], width=bw * 0.92, color=tone, edgecolor="none", zorder=3, label=lab)
    for i, k in enumerate(order):
        if k == OURS:
            ax.axvspan(i - 0.5, i + 0.5, color=ps.GLUON_FILL, zorder=1)

    ax.axhline(0.5, color=ps.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(1.015, 0.5, "chance", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8,
            color=ps.MUTED, clip_on=False)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=24, ha="right", fontsize=9)
    for t, k in zip(ax.get_xticklabels(), order):
        if k == OURS:
            t.set_fontweight("bold")
            t.set_color(ps.ACCENT_GLUON)
    lo = min(min(v) for v in data.values())
    ax.set_ylim(min(0.45, lo - 0.02), 1.0)
    ax.set_ylabel("average precision")
    ax.grid(axis="y", zorder=0)
    ps.despine(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=4, frameon=False, labelcolor=ps.INK_2,
              handlelength=1.1, columnspacing=1.4, fontsize=8.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{a.name}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {OUT}/{a.name}.{{png,svg,pdf}} and {TABLE}")
    print(f"  {'':<28}" + "".join(f"{s:>16}" for s in SETS.values()))
    for k in order:
        print(f"  {k:<28}" + "".join(f"{v:>16.4f}" for v in data[k]))


if __name__ == "__main__":
    main()
