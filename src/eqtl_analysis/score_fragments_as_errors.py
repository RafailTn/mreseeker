#!/usr/bin/env python3
"""
Give a fragment source without a CNN error TSV one, so it can feed the
IntaRNA interaction filter.

`tp_intarna_pairs.py` identifies the interaction footprint from CNN *true
positives*, which it reads from the error TSVs (`error_type` == 'TP'). The
Manakov/miRBench fragments have those; the GSE304955 (SAEC) fragments do not -
they ship as labelled positives only, and their `.cnncache.npz` holds encoded
model inputs, not predictions. Without this step every GSE304955 row would fall
out of the `in_interaction` configuration purely for want of a label.

Every row in these files is a labelled positive (label == 1), so the error type
is decided by the prediction alone: TP at or above the threshold, FN below. That
is the same definition the Manakov error TSVs use, which is what makes the two
sources poolable.

Scoring is out-of-sample: the checkpoint is trained on miRBench/Manakov, and
GSE304955 is an independent cell line. The miRBench pairs are deliberately *not*
rescored here - their existing labels come from out-of-fold predictions, and
re-scoring training rows in-sample would not be comparable.

Output is shaped like an error TSV (`gene` = MRE sequence, `noncodingRNA` =
miRNA sequence, plus `error_type` / `interaction_probability`) so it drops
straight into `tp_intarna_pairs.py --errors`.

    pixi run -m dependencies/cnn python src/eqtl_analysis/score_fragments_as_errors.py \
        --input data/GTEx_..._pip_gt_0_9_union_fragments.tsv \
        --input data/GTEx_..._pip_lt_0_01_union_fragments.tsv \
        --source gse304955 \
        --checkpoint cnn_checkpoints/cnn_branches_mirbind_embed16_restruct.pt \
        -o errors/gse304955_eqtl_pairs_errors.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_CNN_DIR = _HERE.parent / "cnn"
if str(_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_DIR))

from predict_cnn import score_dataframe  # noqa: E402

# `gene`/`start`/`end` mean different things on the two sides: in the merged
# file they are the eQTL gene and the variant position, in an error TSV they are
# the MRE sequence and the fragment span. Move the eQTL ones aside first,
# otherwise the rename below produces duplicate column names.
_RENAME_EQTL = {"gene": "eqtl_gene", "start": "variant_start", "end": "variant_end"}
_RENAME = {"gene_b": "gene", "start_b": "start", "end_b": "end"}
_CARRY = ["noncodingRNA_name", "noncodingRNA_fam", "feature", "chr",
          "start", "end", "strand", "dominant_region", "label"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True,
                    help="Merged/union eQTL x fragment TSV (repeatable).")
    ap.add_argument("--source", default=None,
                    help="If given, keep only rows whose `source` column "
                         "matches (e.g. gse304955).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold separating TP from FN (default 0.5, "
                         "matching the error TSVs).")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    frames = []
    for p in args.input:
        d = pd.read_csv(p, sep="\t", low_memory=False)
        if args.source is not None:
            if "source" not in d.columns:
                raise SystemExit(f"ERROR: --source given but {p} has no `source` column")
            d = d[d["source"] == args.source]
        print(f"  {Path(p).name}: {len(d):,} rows"
              + (f" (source={args.source})" if args.source else ""))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    if df.empty:
        raise SystemExit("ERROR: no rows selected.")

    df = df.rename(columns=_RENAME_EQTL).rename(columns=_RENAME)
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise SystemExit(f"ERROR: duplicate columns after rename: {dupes}")
    for col in ("gene", "noncodingRNA"):
        if col not in df.columns:
            raise SystemExit(f"ERROR: input lacks {col}")
        df[col] = df[col].astype(str).str.upper()

    # One score per distinct (MRE, miRNA) pair - the CNN sees nothing else, so
    # scoring per eQTL row would just repeat identical work.
    pairs = df.drop_duplicates(["gene", "noncodingRNA"]).reset_index(drop=True)
    keep = [c for c in _CARRY if c in pairs.columns]
    pairs = pairs[["gene", "noncodingRNA"] + keep]
    print(f"\n  {len(df):,} rows -> {len(pairs):,} distinct (MRE, miRNA) pairs")

    if "label" in pairs.columns:
        vals = sorted(pairs["label"].dropna().unique().tolist())
        print(f"  label values present: {vals}")
        if set(vals) - {1, "1"}:
            print("  WARNING: non-positive labels present; error_type below "
                  "assumes every row is a labelled positive.", file=sys.stderr)

    print(f"\nScoring with {args.checkpoint} on {args.device}...")
    scored, ckpt = score_dataframe(
        args.checkpoint, pairs, device=args.device,
        mre_col="gene", mirna_col="noncodingRNA",
        batch_size=args.batch_size, num_workers=0, threshold=args.threshold)

    p = scored["interaction_probability"]
    scored["error_type"] = p.ge(args.threshold).map({True: "TP", False: "FN"})
    n_tp = int((scored["error_type"] == "TP").sum())
    print(f"\n  TP: {n_tp:,} / {len(scored):,} ({n_tp / len(scored):.1%}) "
          f"at threshold {args.threshold}")
    print(f"  interaction_probability: min={p.min():.4f} median={p.median():.4f} "
          f"max={p.max():.4f}")
    print(f"  >=0.7: {int(p.ge(0.7).sum()):,}")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(dest, sep="\t", index=False)
    print(f"\nWritten: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
