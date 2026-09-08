"""
Panel 4 - accuracy against end-to-end cost, every model on one axis.

Panel 2 plots `predict_proba` time, which is the wrong axis for comparing the
two families: it charges the LightGBM models for their trees and charges the CNN
for nothing, when in reality the feature models cannot run at all until IntaRNA
has folded every duplex. On a 400-pair smoke test IntaRNA alone was ~70% of the
gluon pipeline, which is enough to invert the ranking. This panel uses the cost
of going from a (miRNA, MRE) pair to a score.

Composing the number
--------------------
Timing all 25 gluon models end-to-end would re-run IntaRNA 25 times for no
reason: every gluon model shares one feature-extraction front end and differs
only in the final predict step. So the cost is composed rather than measured
per model:

    end_to_end(m) = (gluon TOTAL - gluon predict) + predict(m)

The first term comes from one `pipeline_timing.py --pipeline gluon` run, the
second from `compare_models.py`. The CNN needs no composition - its TOTAL is
already end-to-end.

Everything is normalised to seconds per 1,000 pairs before being combined,
because the two scripts do not have to have been run on the same number of rows.
They *should* have been run on the same rows, though, or the APS values are not
comparable; --n-test exists to state what compare_models.py was run on and the
script refuses to guess silently.

Inputs
------
results/model_aps_manakov.csv    compare_models.py --repeats 5 --warmup 1
results/timing_gluon.csv(+meta)  pipeline_timing.py --pipeline gluon
results/timing_cnn.csv(+meta)    pipeline_timing.py --pipeline cnn

Usage
-----
python3 src/figures/panel4_endtoend_pareto.py --n-test 324171
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import poster_style as ps

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"

PER = 1000.0                      # cost is quoted per 1,000 pairs
PREDICT_STAGE = "predict"         # the gluon stage that is model-specific
SHIPPED = "LightGBMLarge_BAG_L1"
CNN_LABEL = "sequence CNN"


def load_timing(path: Path) -> tuple[pd.DataFrame, dict]:
    meta_path = path.with_suffix(".meta.json")
    if not path.exists() or not meta_path.exists():
        raise SystemExit(
            f"ERROR: missing {path} or {meta_path}.\n"
            f"Run src/benchmark/pipeline_timing.py to produce them."
        )
    return pd.read_csv(path), json.loads(meta_path.read_text())


def stage_seconds(df: pd.DataFrame, stage: str, col: str = "seconds") -> float:
    row = df[df.stage == stage]
    if row.empty:
        raise SystemExit(f"ERROR: no '{stage}' stage in the timing CSV. "
                         f"Stages present: {list(df.stage)}")
    return float(row.iloc[0][col])


def build(aps_csv: Path, gluon_csv: Path, cnn_csv: Path,
          n_test: int | None, aps_col: str, sec_col: str,
          cnn_aps: float | None = None, cost: str = "seconds") -> pd.DataFrame:
    gl, gl_meta = load_timing(gluon_csv)
    cnn, cnn_meta = load_timing(cnn_csv)

    if n_test is None:
        n_test = int(gl_meta["n_pairs"])
        print(f"--n-test not given; assuming compare_models.py ran on the same "
              f"{n_test} rows as the gluon timing run.")

    # Per-1000-pair front end: everything the feature models share.
    gl_n = int(gl_meta["n_pairs"])
    shared = ((stage_seconds(gl, "TOTAL", cost)
               - stage_seconds(gl, PREDICT_STAGE, cost)) / gl_n * PER)

    tab = pd.read_csv(aps_csv)
    if sec_col not in tab.columns or aps_col not in tab.columns:
        raise SystemExit(f"ERROR: {aps_csv} lacks {aps_col}/{sec_col}. "
                         f"Columns: {list(tab.columns)}")
    rows = []
    for _, r in tab.iterrows():
        predict = r[sec_col] / n_test * PER
        row = {"model": r["model"], "family": "gluon", "aps": r[aps_col],
               "seconds": shared + predict, "predict": predict}
        # Quartiles only exist when compare_models.py ran with --repeats > 1.
        for q, out in (("_p25", "lo"), ("_p75", "hi")):
            if sec_col + q in tab.columns:
                row[out] = shared + r[sec_col + q] / n_test * PER
        rows.append(row)

    cnn_n = int(cnn_meta["n_pairs"])
    # Cost scales with row count, so the timing runs may use a subsample; APS
    # does not, so it should come from the full set. --cnn-aps is how those two
    # different row counts are reconciled.
    cnn_row = {"model": CNN_LABEL, "family": "cnn",
               "aps": cnn_aps if cnn_aps is not None else cnn_meta["aps"],
               "seconds": stage_seconds(cnn, "TOTAL", cost) / cnn_n * PER,
               "predict": stage_seconds(cnn, "forward", cost) / cnn_n * PER
               if "forward" in set(cnn.stage) else float("nan")}
    for stat, out in ((cost + "_p25", "lo"), (cost + "_p75", "hi")):
        cnn_row[out] = stage_seconds(cnn, "TOTAL", stat) / cnn_n * PER
    if cnn_row["aps"] is None:
        raise SystemExit(
            "ERROR: the CNN timing run recorded no APS, so it has no y value. "
            "Either re-run pipeline_timing.py --pipeline cnn on an input "
            "carrying a 'label' column, or pass --cnn-aps."
        )
    rows.append(cnn_row)
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> list[str]:
    """Models with no other model both faster and at least as accurate."""
    best, front = -math.inf, []
    for _, r in df.sort_values("seconds").iterrows():
        if r.aps > best:
            front.append(r.model)
            best = r.aps
    return front


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aps-csv", type=Path, default=RESULTS / "model_aps_manakov.csv")
    ap.add_argument("--gluon-timing", type=Path, default=RESULTS / "timing_gluon.csv")
    ap.add_argument("--cnn-timing", type=Path, default=RESULTS / "timing_cnn.csv")
    ap.add_argument("--aps-col", default="APS_test")
    ap.add_argument("--seconds-col", default="seconds_test")
    ap.add_argument("--n-test", type=int, default=None,
                    help="Rows compare_models.py scored, used to turn its "
                         "per-set times into per-pair times. Defaults to the "
                         "gluon timing run's row count.")
    ap.add_argument("--cost", choices=["seconds", "cpu_seconds"],
                    default="cpu_seconds",
                    help="Which currency the x-axis spends. cpu_seconds is the "
                         "default because it does not move with the machine's "
                         "core count; seconds is wall-clock latency.")
    ap.add_argument("--cnn-aps", type=float, default=None,
                    help="Override the CNN APS from the timing run's metadata. "
                         "Use when the timing run used a cost subsample but APS "
                         "was scored on the full set.")
    ap.add_argument("--name", default="panel4_endtoend_pareto")
    args = ap.parse_args()

    df = build(args.aps_csv, args.gluon_timing, args.cnn_timing,
               args.n_test, args.aps_col, args.seconds_col, args.cnn_aps,
               args.cost)
    front = pareto_front(df)
    df.to_csv(RESULTS / f"{args.name}.csv", index=False)

    ps.apply(9.5)
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.94, bottom=0.30)

    ax.set_xscale("log")
    fd = df[df.model.isin(front)].sort_values("seconds")
    ax.step(list(fd.seconds) + [df.seconds.max() * 3],
            list(fd.aps) + [fd.aps.iloc[-1]],
            where="post", color=ps.MUTED, lw=1.5, zorder=2)

    for _, r in df.iterrows():
        on = r.model in front
        cnn = r.family == "cnn"
        colour = ps.ACCENT_CNN if cnn else ps.ACCENT_GLUON
        if "lo" in df.columns and pd.notna(r.get("lo")):
            ax.plot([r.lo, r.hi], [r.aps, r.aps],
                    color=colour if on else ps.DEEMPH, lw=1.4, zorder=3,
                    alpha=0.9 if on else 0.6)
        style = (dict(mfc=colour, mec=colour, ms=13 if cnn or r.model == SHIPPED else 9)
                 if on else dict(mfc=ps.SURFACE, mec=ps.DEEMPH, mew=1.8, ms=9))
        ax.plot([r.seconds], [r.aps], "o", zorder=5, **style)

    for _, r in df[df.model.isin(front)].iterrows():
        ax.annotate(r.model, (r.seconds, r.aps), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9,
                    fontweight="bold" if r.family == "cnn" or r.model == SHIPPED
                    else "normal",
                    color=ps.ACCENT_CNN if r.family == "cnn" else ps.INK)

    unit = "CPU-seconds" if args.cost == "cpu_seconds" else "wall-clock seconds"
    ax.set_xlabel(f"end-to-end cost — {unit} per {int(PER):,} pairs  (log scale)")
    ax.set_ylabel("average precision")
    ax.grid(axis="y", zorder=0)
    ps.despine(ax)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=10, color=ps.ACCENT_CNN,
                   label="sequence CNN"),
        plt.Line2D([], [], marker="o", ls="", ms=10, color=ps.ACCENT_GLUON,
                   label="IntaRNA features + LightGBM (on the front)"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=ps.SURFACE,
                   mec=ps.DEEMPH, mew=1.8, label="dominated"),
        plt.Line2D([], [], color=ps.MUTED, lw=1.5, label="Pareto front"),
    ]
    ax.legend(handles=handles, loc="lower right", labelcolor=ps.INK_2)

    fig.text(0.095, 0.020,
             "Accuracy against the cost of scoring a (miRNA, MRE) pair from "
             "sequence, on the same held-out set. Cost for a feature model is "
             "its shared IntaRNA\nfront end plus its own predict step; for the "
             "CNN it is the measured total. Bars are the interquartile range "
             "over timed repeats. Equal horizontal\ndistance is equal speed "
             "ratio.",
             fontsize=8.5, color=ps.INK_2, va="bottom", linespacing=1.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{args.name}.{ext}", **kw)
    plt.close(fig)

    print(f"\nPareto front ({len(front)}): {front}")
    print(df.sort_values("seconds").to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {OUT}/{args.name}.{{png,svg,pdf}} and "
          f"{RESULTS}/{args.name}.csv")


if __name__ == "__main__":
    main()
