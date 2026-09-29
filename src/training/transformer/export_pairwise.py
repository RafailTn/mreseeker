#!/usr/bin/env python3
"""
Pairwise training tables for the existing CNN, from the per-3'UTR labels.

v1 of the transcript-level plan: keep cnn_branches_mirbind.py exactly as it is
and change only what it trains on.  Rows are v7-shaped (gene, noncodingRNA,
noncodingRNA_fam, label) so `cnn_branches_mirbind.py train --mre-col gene
--mirna-col noncodingRNA --family-col noncodingRNA_fam` reads them unchanged.

Three kinds of row, all inside the chosen 3'UTRs:

  positive     a chimeric v7 window and its miRNA
  old          a v7 label-0 row: same kind of window, a different miRNA.
               Teaches *which* miRNA binds.
  none_seed    a 50-nt window lying wholly in NONE (expressed, AGO2 not
               enriched), paired with every expressed miRNA that has a 7mer or
               8mer seed match in it.  Seed matching alone would call these
               bound; the labels say nothing bound.
  none_random  the same windows, paired with --random-per-window miRNAs drawn
               by chimera abundance, so the new negatives are not all
               seed-matched.
  wrong_mirna  (test only) a held-out positive window re-paired with an
               expressed miRNA of another family and no 7mer+ match in it.

Positives come only from inside the chosen 3'UTRs, because NONE exists only
there: mixing CDS/intron positives with UTR-only negatives would let the model
learn "UTR-like sequence = negative".  Leftout-family rows are dropped (the
leftout benchmark keeps them), as are `external` tables.

Outputs (in --out-dir)
----------------------
A_train.tsv, A_val.tsv   positives + old negatives: the current recipe, UTR-only
B_train.tsv, B_val.tsv   positives + the same number of negatives, a
                         --new-frac share of them none_seed/none_random
test_old.tsv             chr1 positives + chr1 old negatives
test_none.tsv            chr1 positives + chr1 none_seed/none_random negatives
test_wrong_mirna.tsv     chr1 positives + one wrong-miRNA pair each (pair_id
                         links them): if B's score drops much less than A's,
                         B has stopped reading the miRNA

Train A and B on identical positives and splits; they differ only in negatives.

Family balance (train and val, on by default): each miRNA family keeps
n = min(positives, A negatives, B negatives) of each, sampled once for the
positives and shared by A and B.  Every family is then exactly 50/50, so a
family's identity says nothing about the label, and the overall ratio is
50/50 too.  Families with only positives or only negatives drop out.  Test
files keep their natural ratios; both recipes are scored on the same ones.

Example
-------
python3 src/training/transformer/export_pairwise.py --labels-dir results/utr_labels --out-dir results/pairwise_v1
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_utr_labels import NONE, canonical_site, dna, revcomp
from site_spacing import kmer_code, kmer_codes, read_fasta

COLS = ["gene", "noncodingRNA", "noncodingRNA_fam", "label", "neg_type", "split",
        "transcript_id", "tx_start", "site_type"]


def none_windows(lab: np.ndarray, W: int, stride: int) -> list:
    """Starts of W-nt windows lying wholly in NONE, at least `stride` apart."""
    ok = np.concatenate(([0], np.cumsum(lab == NONE)))
    starts = np.arange(max(len(lab) - W + 1, 0))
    full = np.nonzero(ok[starts + W] - ok[starts] == W)[0]
    picks, last = [], -stride
    for s in full:
        if s - last >= stride:
            picks.append(int(s))
            last = s
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--stride", type=int, default=25,
                    help="minimum spacing of NONE windows; below 50 they overlap")
    ap.add_argument("--random-per-window", type=int, default=1)
    ap.add_argument("--new-frac", type=float, default=0.5,
                    help="recipe B: share of negatives taken from NONE windows")
    ap.add_argument("--min-sites", type=int, default=50,
                    help="chimeras a miRNA needs (train+val) to count as expressed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-family-balance", dest="family_balance", action="store_false")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d, out = Path(args.labels_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    W = args.window

    sites = pd.read_csv(d / "sites.tsv", sep="\t",
                        usecols=["transcript_id", "split", "role", "label", "tx_start",
                                 "mirna_seq", "family", "heldout_family", "site_type",
                                 "target_seq"])
    sites = sites[(sites["role"] != "external") & ~sites["heldout_family"]]
    sites = sites.drop_duplicates(["target_seq", "mirna_seq", "label"])
    sites = sites.rename(columns={"target_seq": "gene", "mirna_seq": "noncodingRNA",
                                  "family": "noncodingRNA_fam"})
    sites["neg_type"] = np.where(sites["label"] == 1, "positive", "old")
    pos, old = sites[sites["label"] == 1], sites[sites["label"] == 0]

    # Expressed repertoire and abundance, from training chimeras only.
    fit = pos[pos["split"].isin(["train", "val"])]
    counts = fit["noncodingRNA"].value_counts()
    rep = counts[counts >= args.min_sites]
    rep_seq = rep.index.tolist()
    rep_fam = fit.drop_duplicates("noncodingRNA").set_index("noncodingRNA")["noncodingRNA_fam"]
    rep_fam = [rep_fam[s] for s in rep_seq]
    weights = rep.to_numpy(dtype=float) / rep.sum()
    targets = defaultdict(set)  # 7mer site code -> repertoire indices (7mer-m8, 7mer-A1)
    for i, s in enumerate(rep_seq):
        m = dna(s)
        targets[kmer_code(revcomp(m[1:8]))].add(i)
        targets[kmer_code(revcomp(m[1:7]) + "A")].add(i)
    print(f"[repertoire] {len(rep_seq)} miRNAs with >= {args.min_sites} chimeras")

    def seed_matched(window: str) -> set:
        hits = set()
        for c in kmer_codes(window):
            hits |= targets.get(int(c), set())
        return hits

    def draw(exclude_idx: set, exclude_fam=None):
        for _ in range(100):
            i = int(rng.choice(len(rep_seq), p=weights))
            if i not in exclude_idx and rep_fam[i] != exclude_fam:
                return i
        return None

    # NONE windows -> new negatives.
    labels = np.load(d / "labels.npz")
    utrs = pd.read_csv(d / "utrs.tsv", sep="\t", usecols=["transcript_id", "split"])
    split_of = dict(zip(utrs["transcript_id"], utrs["split"]))
    seqs = read_fasta(d / "utr3.fa")
    new_rows = []
    for t in labels.files:
        seq = seqs.get(t)
        if seq is None:
            continue
        for s in none_windows(labels[t], W, args.stride):
            win = seq[s:s + W]
            matched = seed_matched(win)
            pairs = [(i, "none_seed") for i in sorted(matched)]
            for _ in range(args.random_per_window):
                i = draw(matched)
                if i is not None:
                    pairs.append((i, "none_random"))
            for i, kind in pairs:
                new_rows.append((win, rep_seq[i], rep_fam[i], 0, kind, split_of[t], t, s,
                                 canonical_site(rep_seq[i], win)[0] or "none"))
    new = pd.DataFrame(new_rows, columns=COLS)
    print(f"[none] {len(new):,} new negatives "
          + str(new.groupby(["split", "neg_type"]).size().to_dict()))

    def write(name, df):
        df[COLS + [c for c in df.columns if c not in COLS]].to_csv(out / name, sep="\t", index=False)
        print(f"[write] {name}: {len(df):,} rows, "
              + ", ".join(f"{k} {v:,}" for k, v in df["neg_type"].value_counts().items()))

    def balance(p, a_neg, b_neg):
        """Per family, n = min of the three counts; one positive sample for both."""
        fam = "noncodingRNA_fam"
        n = pd.concat([x[fam].value_counts() for x in (p, a_neg, b_neg)], axis=1).fillna(0).min(axis=1)
        n = n[n > 0].astype(int)

        def take(df):
            return pd.concat([g.sample(n[f], random_state=args.seed)
                              for f, g in df.groupby(fam) if f in n.index])
        dropped = sorted(set(p[fam]) - set(n.index))
        return take(p), take(a_neg), take(b_neg), len(n), dropped

    for sp in ("train", "val"):
        p, o, n = (x[x["split"] == sp] for x in (pos, old, new))
        n_neg = len(o)
        n_new = min(int(round(args.new_frac * n_neg)), len(n))
        if n_new < args.new_frac * n_neg:
            print(f"[B] {sp}: only {len(n):,} NONE negatives for a {args.new_frac:.0%} share "
                  f"of {n_neg:,}; using all of them (lower --stride or raise "
                  f"--random-per-window for more)")
        b_neg = pd.concat([o.sample(n_neg - n_new, random_state=args.seed),
                           n.sample(n_new, random_state=args.seed)])
        a_neg = o
        if args.family_balance:
            before = len(p)
            p, a_neg, b_neg, kept, dropped = balance(p, a_neg, b_neg)
            print(f"[balance] {sp}: {kept} families kept, {len(dropped)} dropped; positives "
                  f"{before:,} -> {len(p):,}")
        write(f"A_{sp}.tsv", pd.concat([p, a_neg])[COLS])
        write(f"B_{sp}.tsv", pd.concat([p, b_neg])[COLS])

    p, o, n = (x[x["split"] == "test"] for x in (pos, old, new))
    write("test_old.tsv", pd.concat([p, o])[COLS])
    write("test_none.tsv", pd.concat([p, n])[COLS])

    wrong = []
    for k, r in enumerate(p.itertuples(index=False)):
        i = draw(seed_matched(r.gene), exclude_fam=r.noncodingRNA_fam)
        if i is None:
            continue
        wrong.append({**{c: getattr(r, c) for c in COLS}, "pair_id": k})
        wrong.append({"gene": r.gene, "noncodingRNA": rep_seq[i], "noncodingRNA_fam": rep_fam[i],
                      "label": 0, "neg_type": "wrong_mirna", "split": "test",
                      "transcript_id": r.transcript_id, "tx_start": r.tx_start,
                      "site_type": canonical_site(rep_seq[i], r.gene)[0] or "none", "pair_id": k})
    write("test_wrong_mirna.tsv", pd.DataFrame(wrong))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
