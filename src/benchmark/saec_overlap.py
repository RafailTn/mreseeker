"""
How much of the SAEC (GSE304955) evaluation set does the model already know?

The set is billed as an independent cell line, and a generalisation claim rests
on that. The file shipped as `_positives_novel_` was already filtered so that no
(MRE sequence, miRNA sequence) pair appears in any miRBench v7 set - but "no
identical pair" is a weaker guarantee than it sounds. Every MRE here is a 50-nt
window cut around a read pile-up, and two experiments never cut the same site at
the same base. Shift the window one nucleotide and the string differs, the
genomic start differs, and an exact-match audit calls the site novel.

So this script matches on genomic *interval overlap* instead. Every window in
both corpora is exactly 50 nt, which makes the test cheap: two windows on the
same chromosome and strand overlap iff their starts differ by less than 50, and
the shared length is 50 - |dstart|. That reduces the whole comparison to a
sorted-array range query per (chr, strand) group.

Two overlap keys are reported, and the gap between them is the point:

  ov_any   same chr/strand, any miRNA. "This locus was bound in miRBench."
  ov_mir   same chr/strand *and* the same miRNA. "This interaction was seen."

Only the second is leakage in the sense that matters. The first is a much
weaker familiarity - the model may have learned the region is AGO-accessible -
and the stratified APS table says whether it actually buys the model anything.

Outputs
-------
results/saec_mirbench_overlap.csv   survival curve: share of rows sharing >= k bp
results/saec_aps_by_overlap.csv     APS within disjoint shared-bp bins

Example
-------
python3 src/benchmark/saec_overlap.py \
    --predictions ../msc-thesis/results/novel/gse304955_labelled_pred.tsv \
    --reference-dir ../msc-thesis/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

WINDOW = 50           # every MRE in both corpora is a 50-nt window; asserted below
KEY_COLS = ["gene", "noncodingRNA_name", "chr", "start", "strand"]

# The six miRBench v7 sets the checkpoint could have seen, train and test alike:
# a site the model was tested on during development is still a site it has met.
REFERENCE_FILES = [
    "AGO2_eCLIP_Manakov2022_train_v7.tsv",
    "AGO2_eCLIP_Manakov2022_test_v7.tsv",
    "AGO2_eCLIP_Manakov2022_leftout_v7.tsv",
    "AGO2_CLASH_Hejret2023_train_v7.tsv",
    "AGO2_CLASH_Hejret2023_test_v7.tsv",
    "AGO2_eCLIP_Klimentova2022_test_v7.tsv",
]

# Disjoint, not cumulative. Nested strata share rows, so their APS values are
# not independent readings and a trend across them is partly autocorrelation.
BINS = [(0, 0, "none"), (1, 29, "1-29 bp"), (30, 44, "30-44 bp"), (45, 50, "45-50 bp")]

THRESHOLDS = list(range(0, WINDOW + 1))


def load_coords(path: Path, extra: List[str] = ()) -> pd.DataFrame:
    """Read the columns needed to place a window on the genome, dropping unplaced rows.

    `chr` and `strand` are read as strings because the reference files mix int and
    str chromosome names across sets, and a groupby key that changes dtype between
    corpora silently matches nothing.
    """
    df = pd.read_csv(path, sep="\t", usecols=list(KEY_COLS) + list(extra),
                     low_memory=False)
    before = len(df)
    df = df.dropna(subset=["chr", "start", "strand"]).copy()
    df["chr"] = df["chr"].astype(str)
    df["strand"] = df["strand"].astype(str)
    df["start"] = df["start"].astype(np.int64)
    if before != len(df):
        print(f"  {path.name}: dropped {before - len(df)} rows without coordinates")
    lengths = df["gene"].str.len().unique()
    if list(lengths) != [WINDOW]:
        raise SystemExit(
            f"ERROR: {path.name} has MRE lengths {sorted(lengths)}, expected only "
            f"{WINDOW}. The equal-length shortcut in best_overlap() would be wrong."
        )
    return df.reset_index(drop=True)


def best_overlap(query: pd.DataFrame, ref: pd.DataFrame, keys: List[str]) -> np.ndarray:
    """Shared bp between each query window and its closest reference window.

    Equal window lengths make this a distance problem rather than an interval
    problem: overlap = WINDOW - |dstart| when that is positive, 0 otherwise. So
    per group it is a sorted array and two searchsorted bounds, not an interval
    tree. Groups absent from the reference score 0 without being visited.
    """
    out = np.zeros(len(query), dtype=np.int64)
    index: Dict[tuple, np.ndarray] = {
        k: np.sort(g["start"].values) for k, g in ref.groupby(keys, sort=False)
    }
    for key, grp in query.groupby(keys, sort=False):
        starts = index.get(key)
        if starts is None:
            continue
        q_start = grp["start"].values
        rows = query.index.get_indexer(grp.index)
        lo = np.searchsorted(starts, q_start - WINDOW + 1, "left")
        hi = np.searchsorted(starts, q_start + WINDOW - 1, "right")
        for i, (a, b, s) in enumerate(zip(lo, hi, q_start)):
            if b > a:
                out[rows[i]] = WINDOW - np.abs(starts[a:b] - s).min()
    return out


def survival(df: pd.DataFrame, col: str) -> List[float]:
    """Share of rows sharing at least k bp, for every k. Positives only.

    Restricted to positives because that is the claim under audit: a negative
    here is a synthetic mispairing, and whether its window was seen before says
    nothing about whether a real interaction leaked.
    """
    pos = df[df.label == 1]
    return [float((pos[col] >= k).mean()) for k in THRESHOLDS]


def aps_by_bin(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for lo, hi, name in BINS:
        sub = df[(df[col] >= lo) & (df[col] <= hi)]
        if sub.label.nunique() < 2:
            continue
        rows.append({
            "bin": name, "lo": lo, "hi": hi, "n": len(sub),
            "n_pos": int(sub.label.sum()), "pos_rate": float(sub.label.mean()),
            "aps": float(average_precision_score(
                sub.label, sub.interaction_probability)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path,
                    help="Labelled SAEC prediction TSV from predict_cnn.py "
                         "(needs label + interaction_probability).")
    ap.add_argument("--reference-dir", required=True, type=Path,
                    help="Directory holding the six miRBench v7 TSVs.")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    args = ap.parse_args()

    print("Reading SAEC predictions ...")
    q = load_coords(args.predictions, extra=["label", "interaction_probability"])
    print(f"  {len(q)} rows, {int(q.label.sum())} positives")

    print("Reading miRBench reference sets ...")
    refs = []
    for name in REFERENCE_FILES:
        path = args.reference_dir / name
        if not path.exists():
            raise SystemExit(f"ERROR: missing reference set {path}")
        refs.append(load_coords(path))
    ref = pd.concat(refs, ignore_index=True)
    print(f"  {len(ref)} reference windows")

    print("Matching windows (any miRNA) ...")
    q["ov_any"] = best_overlap(q, ref, ["chr", "strand"])
    print("Matching windows (same miRNA) ...")
    q["ov_mir"] = best_overlap(q, ref, ["chr", "strand", "noncodingRNA_name"])

    args.outdir.mkdir(parents=True, exist_ok=True)

    curve = pd.DataFrame({
        "shared_bp": THRESHOLDS,
        "frac_any_mirna": survival(q, "ov_any"),
        "frac_same_mirna": survival(q, "ov_mir"),
    })
    curve.to_csv(args.outdir / "saec_mirbench_overlap.csv", index=False)

    tables = []
    for col, label in (("ov_any", "any miRNA"), ("ov_mir", "same miRNA")):
        t = aps_by_bin(q, col)
        t.insert(0, "match", label)
        tables.append(t)
    overall = pd.DataFrame([{
        "match": "all", "bin": "all", "lo": -1, "hi": -1, "n": len(q),
        "n_pos": int(q.label.sum()), "pos_rate": float(q.label.mean()),
        "aps": float(average_precision_score(q.label, q.interaction_probability)),
    }])
    table = pd.concat([overall] + tables, ignore_index=True)
    table.to_csv(args.outdir / "saec_aps_by_overlap.csv", index=False)

    print("\nOverlap of SAEC positives with the miRBench union:")
    for k in (1, 10, 20, 30, 40, 45, 50):
        r = curve[curve.shared_bp == k].iloc[0]
        print(f"  >={k:>2} bp   any miRNA {r.frac_any_mirna:6.2%}   "
              f"same miRNA {r.frac_same_mirna:6.2%}")
    print("\nAPS within disjoint shared-bp bins:")
    print(table.to_string(index=False,
                          float_format=lambda v: f"{v:.4f}"))
    print(f"\nWritten to: {args.outdir}/saec_mirbench_overlap.csv, "
          f"{args.outdir}/saec_aps_by_overlap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
