#!/usr/bin/env python3
"""
Union the per-source GTEx-eQTL x MRE-fragment tables into one non-redundant file.

The eQTL analysis was first run on the miRBench (Manakov 2022) fragments alone.
Adding a second source - the SAEC GSE304955 fragments - means the same eQTL can
now be reported against the same target site twice, once per source, and those
duplicates would be counted twice by every downstream test.

Redundancy is keyed on (variant, gene, tissue, gene_b, noncodingRNA): the MRE
*sequence* plus the miRNA sequence, not the reported coordinates. `delta_pred`
is a pure function of those two sequences, so two rows agreeing on them are
interchangeable for this analysis even when the two experiments called slightly
different fragment boundaries - which they do. Keying on coordinates instead
would miss those (23 such rows in the pip>0.9 set).

`unique_key` is namespaced as `<source>:<key>` because the two sources number
their fragments independently and the raw ids genuinely collide (22 in the
pip<0.01 set), which would otherwise silently merge unrelated fragments.

The first --add file wins ties, so listing miRBench first keeps the
already-analysed rows (and their error-TSV provenance) as the retained copy.

    pixi run -m dependencies/gluon python src/eqtl_analysis/union_eqtl_fragments.py \
        --add miRBench=data/..._merged_with_miRBench_fragments.tsv \
        --add gse304955=data/..._merged_with_gse304955_fragments.tsv \
        -o data/GTEx_v8_eQTLs_pip_gt_0_9_union_fragments.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Sequence identity, not locus identity - see module docstring.
DEDUP_COLS = ["variant", "gene", "tissue", "gene_b", "noncodingRNA"]
UPPER_COLS = ["gene_b", "noncodingRNA"]


def load(tag: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [c for c in DEDUP_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {path} lacks {missing}")
    df.insert(0, "source", tag)
    if "unique_key" in df.columns:
        df["unique_key_orig"] = df["unique_key"]
        df["unique_key"] = tag + ":" + df["unique_key"].astype(str)
    print(f"  {tag:<12} {len(df):>7,} rows  {df.shape[1]:>3} cols  {Path(path).name}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", action="append", required=True, metavar="TAG=PATH",
                    help="Source-tagged merged fragment TSV (repeatable). The "
                         "first listed source wins on duplicates.")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    specs = []
    for spec in args.add:
        if "=" not in spec:
            raise SystemExit(f"ERROR: --add needs TAG=PATH, got {spec!r}")
        tag, path = spec.split("=", 1)
        specs.append((tag, path))

    print("Loading sources...")
    frames = [load(tag, path) for tag, path in specs]
    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n  concatenated: {len(df):,} rows, {df.shape[1]} columns "
          f"(union of per-source schemas)")

    # Normalise the sequence half of the key so case differences cannot split
    # what is the same pair.
    key = df[DEDUP_COLS].copy()
    for c in UPPER_COLS:
        key[c] = key[c].astype(str).str.upper()
    key_t = pd.Series(list(map(tuple, key.itertuples(index=False, name=None))),
                      index=df.index)

    # Record every source that contributed a given key before collapsing, so a
    # row retained from source A still shows it was corroborated by B.
    contributors = (df.groupby(key_t)["source"]
                      .agg(lambda s: ",".join(sorted(set(s)))))
    df["sources"] = key_t.map(contributors)
    df["n_sources"] = df["sources"].str.count(",") + 1

    dup = key_t.duplicated(keep="first")
    out = df[~dup].copy()

    print(f"\n  dropped {int(dup.sum()):,} redundant rows "
          f"({', '.join(DEDUP_COLS)})")
    print(f"  union: {len(out):,} rows")
    print("\n  retained rows by source:")
    for tag, n in out["source"].value_counts().items():
        print(f"    {tag:<12} {n:>7,}")
    shared = int((out["n_sources"] > 1).sum())
    print(f"  corroborated by >1 source: {shared:,}")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, sep="\t", index=False)
    print(f"\nWritten: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
