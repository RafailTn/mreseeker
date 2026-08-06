#!/usr/bin/env python3
"""
Collate every eQTL filtering configuration into one comparison table.

Each `eqtl_pip_separation.py` run writes its own separation_stats.json, but the
question being asked is comparative - does filtering on the IntaRNA interaction
footprint, on CNN confidence, or on tissue expression change the separation? -
and that only reads off a table with all configurations side by side.

Two blocks are emitted per configuration:
  separation  - n, Mann-Whitney p and rank-biserial for signed and |delta_pred|,
                and the delta_pred-vs-beta_marginal correlation.
  magnitude   - the delta_pred distribution over unique (variant, fragment)
                pairs, which is where the interaction filter shows its effect
                even though the separation test does not move.

Rank-biserial sign convention (from eqtl_pip_separation.py): negative means the
high-PIP group ranks larger, i.e. the direction the mechanism predicts for
|delta_pred|. The column is reported as-is; `abs_direction_ok` flags it.

    pixi run -m dependencies/cnn python src/eqtl_analysis/summarise_eqtl_configs.py \
        --results-dir results/eqtl_intarna -o results/eqtl_intarna/summary_all_configs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# (key, human label) - key names both separation_<key>/ and delta_pred_<key>_*.
CONFIGS = [
    ("ALL",                 "unfiltered"),
    ("in_interaction",      "in_interaction (TP)"),
    ("p70",                 "in_interaction (TP prob>=0.7)"),
    ("expr_ALL",            "tissue-expressed only"),
    ("expr_in_interaction", "tissue-expressed + in_interaction"),
]


def load_separation(d: Path) -> dict:
    with open(d / "separation_stats.json") as fh:
        blob = json.load(fh)
    run = blob["run"]
    by = {}
    for s in blob["stats"]:
        if s.get("test") == "mann_whitney_u":
            by[s["metric"]] = s
        elif s.get("test") == "correlation":
            by[f"corr::{s['group']}"] = s
    signed, absd = by.get("signed delta_pred", {}), by.get("|delta_pred|", {})
    comb = by.get("corr::combined", {})
    return {
        "n_high": signed.get("n_high"), "n_low": signed.get("n_low"),
        "signed_p": signed.get("p_value"), "signed_rbc": signed.get("rank_biserial"),
        "signed_median_high": signed.get("median_high"),
        "signed_median_low": signed.get("median_low"),
        "abs_p": absd.get("p_value"), "abs_rbc": absd.get("rank_biserial"),
        "abs_median_high": absd.get("median_high"),
        "abs_median_low": absd.get("median_low"),
        "spearman_r": comb.get("spearman_r"), "spearman_p": comb.get("spearman_p"),
        "pearson_r": comb.get("pearson_r"), "pearson_p": comb.get("pearson_p"),
        "n_high_rows_in": run.get("n_high_rows_in"),
        "n_low_rows_in": run.get("n_low_rows_in"),
        "dedup_by": run.get("dedup_by"),
    }


def load_magnitude(results: Path, key: str) -> dict:
    """delta_pred distribution over unique (variant, fragment) pairs."""
    frames = []
    for g in ("gt_0_9", "lt_0_01"):
        p = results / f"delta_pred_{key}_pip_{g}.tsv"
        if p.exists():
            frames.append(pd.read_csv(p, sep="\t", low_memory=False))
    if not frames:
        return {}
    d = pd.concat(frames).dropna(subset=["delta_pred"])
    d = d.drop_duplicates(["variant", "unique_key"])
    a = d["delta_pred"].to_numpy()
    return {
        "n_unique_pairs": len(a),
        "mean_delta": float(a.mean()),
        "median_abs_delta": float(np.median(np.abs(a))),
        "frac_abs_gt_0.10": float((np.abs(a) > 0.10).mean()),
        "frac_negative": float((a < 0).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("-o", required=True, help="Output path stem (.tsv and .md).")
    args = ap.parse_args()

    results = Path(args.results_dir)
    rows = []
    for key, label in CONFIGS:
        d = results / f"separation_{key}"
        if not (d / "separation_stats.json").exists():
            print(f"  skipping {key}: no separation_stats.json")
            continue
        row = {"config": key, "label": label}
        row.update(load_separation(d))
        row.update(load_magnitude(results, key))
        # Predicted direction: causal (high-PIP) variants disrupt binding more.
        row["abs_direction_ok"] = (row.get("abs_rbc") is not None
                                   and row["abs_rbc"] < 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.o)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".tsv"), sep="\t", index=False, float_format="%.6g")

    # Markdown mirror - the table is meant to be read, not just parsed.
    lines = ["# eQTL filtering configurations", "",
             "`rbc` = rank-biserial; **negative = high-PIP ranks larger**, the",
             "direction the mechanism predicts for `|delta_pred|`.", "",
             "## Separation (delta_pred: pip>0.9 vs pip<0.01)", "",
             "| config | high n | low n | signed p | \\|Δ\\| p | \\|Δ\\| rbc | dir | Spearman vs β |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['n_high']:,} | {r['n_low']:,} | "
            f"{r['signed_p']:.3g} | {r['abs_p']:.3g} | {r['abs_rbc']:+.4f} | "
            f"{'ok' if r['abs_direction_ok'] else 'x'} | "
            f"{r['spearman_r']:+.4f} (p={r['spearman_p']:.3g}) |")
    lines += ["", "## delta_pred magnitude (unique variant x fragment pairs)", "",
              "| config | pairs | mean Δ | median \\|Δ\\| | \\|Δ\\|>0.10 | Δ<0 |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if "n_unique_pairs" not in r:
            continue
        lines.append(
            f"| {r['label']} | {r['n_unique_pairs']:,} | {r['mean_delta']:+.4f} | "
            f"{r['median_abs_delta']:.4f} | {r['frac_abs_gt_0.10']:.1%} | "
            f"{r['frac_negative']:.1%} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")

    print(df[["label", "n_high", "n_low", "signed_p", "abs_p", "abs_rbc",
              "median_abs_delta", "abs_direction_ok"]].to_string(index=False))
    print(f"\nWritten: {out.with_suffix('.tsv')}\n         {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
