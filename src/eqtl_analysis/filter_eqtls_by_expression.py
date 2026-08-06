#!/usr/bin/env python3
"""
Keep only the eQTL rows whose assigned miRNA is actually expressed in the tissue
the effect was measured in.

The merged GTEx x miRBench files pair each fragment with the one miRNA an AGO2
eCLIP captured in a single cell type, then join it against betas from up to 49
GTEx tissues. Only ~40% of the resulting rows have that miRNA among the top 100
expressed in their own tissue; the rest assert a repression mechanism that
cannot be operating there.

This filters on the existing assignment rather than proposing new miRNAs - the
CNN cannot rank arbitrary candidates (see mirna_repertoire_diagnostic.py: 0.86x
enrichment over shuffled sites), so the assignment is all there is to work with.
Because delta_pred is purely sequence-driven it is unchanged by this filter, so
already-scored delta_pred TSVs can be filtered directly with no re-scoring.

`noncodingRNA_name` is pipe-separated when one sequence carries several miRBase
aliases; a row is kept if *any* alias is expressed, since they are the same
molecule.

    pixi run -m dependencies/cnn python src/eqtl_analysis/filter_eqtls_by_expression.py \
        --input results/eqtl_intarna/delta_pred_in_interaction_pip_gt_0_9.tsv \
        --top-mirnas results/gtex_mirna_expression/top100_by_subtissue.tsv \
        -o results/eqtl_intarna/delta_pred_expr_in_interaction_pip_gt_0_9.tsv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def norm_tissue(x: str) -> str:
    """GTEx v8 eQTL and v11 miRNA tables differ only in runs of underscores."""
    return re.sub(r"_+", "_", str(x)).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Merged or delta_pred eQTL TSV.")
    ap.add_argument("--top-mirnas", required=True,
                    help="top100_by_subtissue.tsv from gtex_top_mirnas.py.")
    ap.add_argument("-o", required=True, help="Filtered output TSV.")
    ap.add_argument("--annotated-out", help="Optional: all rows + annotation columns.")
    ap.add_argument("--top-n", type=int, default=100,
                    help="Use only the top N expressed miRNAs per tissue (default 100).")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t", low_memory=False)
    df["_t"] = df["tissue"].map(norm_tissue)

    top = pd.read_csv(args.top_mirnas, sep="\t")
    top["_t"] = top["tissue"].map(norm_tissue)
    top = top[top["rank"] <= args.top_n]

    mir_by_tissue = top.groupby("_t")["miRNA"].agg(set).to_dict()
    # Best (lowest) expression rank and its CPM, for the surviving alias.
    rank_by = {(t, m): (r, c) for t, m, r, c in
               zip(top["_t"], top["miRNA"], top["rank"], top["median_cpm"])}

    missing_tissues = sorted(set(df["_t"]) - set(mir_by_tissue))
    if missing_tissues:
        print(f"  WARNING: {len(missing_tissues)} tissue(s) absent from the "
              f"expression table, their rows cannot be kept: "
              f"{missing_tissues[:5]}", file=sys.stderr)

    expressed, ranks, cpms = [], [], []
    for tis, name in zip(df["_t"], df["noncodingRNA_name"].fillna("")):
        pool = mir_by_tissue.get(tis, set())
        hits = [a.strip() for a in str(name).split("|")
                if a.strip() and a.strip() in pool]
        if hits:
            best = min(hits, key=lambda a: rank_by[(tis, a)][0])
            r, c = rank_by[(tis, best)]
            expressed.append(True); ranks.append(r); cpms.append(c)
        else:
            expressed.append(False); ranks.append(np.nan); cpms.append(np.nan)

    ann = df.drop(columns="_t").copy()
    ann["mirna_expressed"] = expressed
    ann["mirna_expr_rank"] = ranks
    ann["mirna_median_cpm"] = cpms

    kept = ann[ann["mirna_expressed"]].copy()
    n, k = len(ann), len(kept)
    print(f"\n  {Path(args.input).name}")
    print(f"    rows                     : {n:,} -> {k:,} ({k / n:.1%})")
    print(f"    unique variants          : {ann['variant'].nunique():,} -> "
          f"{kept['variant'].nunique():,}")
    print(f"    unique tissues           : {ann['tissue'].nunique():,} -> "
          f"{kept['tissue'].nunique():,}")
    if k:
        print(f"    median expression rank   : {kept['mirna_expr_rank'].median():.0f}"
              f"  (median CPM {kept['mirna_median_cpm'].median():,.0f})")

    out = Path(args.o)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, sep="\t", index=False)
    print(f"    written                  : {out}")
    if args.annotated_out:
        Path(args.annotated_out).parent.mkdir(parents=True, exist_ok=True)
        ann.to_csv(args.annotated_out, sep="\t", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
