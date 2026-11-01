#!/usr/bin/env python3
"""
Keep only the GTEx eQTLs that land on a base actually paired to the miRNA in
the CNN duplex of a CNN true-positive (MRE, miRNA) pair.

The duplex is the one the model relied on: `cnn_branches_mirbind.py explain
--duplex` decodes it from the Integrated-Gradients map of the pairing matrix
(see duplex_decode.py). Inputs are produced by

    src/eqtl_analysis/tp_pairs.py                       ->  data/eqtl_tp_pairs.tsv
    src/training/cnn/cnn_branches_mirbind.py explain \
        --input data/eqtl_tp_pairs.tsv --mre-col mre_seq --mirna-col mirna_seq \
        --duplex --output results/eqtl_duplex/eqtl_tp_explain.tsv

"Inside the interaction" is the strict reading: the variant must hit a target
nucleotide that is WC or G-U paired in the decoded duplex, not merely sit
somewhere in its span. A variant on a mismatch or bulge breaks no hydrogen
bond, so the permissive span would wave through variants with no mechanism.
The permissive call is still reported per row (`in_duplex_span`) so the two can
be compared without a rerun. Each surviving eQTL is labelled with the miRNA
position it disrupts - and hence whether it falls in the seed (miRNA nt 2-8) -
and with the IG attribution at its MRE position (`snp_attr`, logits, summed
over miRNA rows; positive = the reference base supports binding).

Coordinate conventions, matching eqtl_analysis_cnn.py:
  `gene_b` is transcript-oriented, so stored index i (0-based) maps to genomic
  `start_b + i` on '+' and `end_b - i` on '-'. The SNP position is `start`
  (1-based; == `end` for SNPs).

    pixi run -m dependencies/cnn python src/eqtl_analysis/filter_eqtls_by_interaction.py \
        --eqtl data/GTEx_..._pip_gt_0_9_union_fragments.tsv \
        --explain results/eqtl_duplex/eqtl_tp_explain.tsv \
        -o results/eqtl_duplex/eqtl_in_interaction_pip_gt_0_9.tsv \
        --annotated-out results/eqtl_duplex/eqtl_annotated_pip_gt_0_9.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Merged GTEx x miRBench fragment columns (same names as eqtl_analysis_cnn.py).
COL_SNP_POS   = 'start'
COL_MRE_SEQ   = 'gene_b'
COL_MIRNA     = 'noncodingRNA'
COL_MRE_START = 'start_b'
COL_MRE_END   = 'end_b'
COL_STRAND    = 'strand'
COL_REF       = 'allele1'
COL_ALT       = 'allele2'

SEED_RANGE = (2, 8)   # miRNA nt 2-8, 1-based inclusive
MRE_LEN = 50

_COMPLEMENT = str.maketrans('ACGT', 'TGCA')


def parse_duplex_pairs(duplex_pairs: str) -> tuple[dict[int, int], int, int]:
    """Map paired target position -> partner miRNA position, both 1-based.

    `duplex_pairs` is explain's "pos:j:kind:attr;..." (pos 1-based miRNA, j
    0-based MRE, empty for unpaired / bulged miRNA bases). Only WC / GU count
    as paired; the span runs over every aligned target base, mismatches included.
    """
    paired, aligned = {}, []
    if not isinstance(duplex_pairs, str):
        return {}, 0, 0
    for tok in duplex_pairs.split(';'):
        if not tok:
            continue
        pos, j, kind, _ = tok.split(':')
        if not j or int(j) < 0:
            continue
        aligned.append(int(j) + 1)
        if kind in ('WC', 'GU'):
            paired[int(j) + 1] = int(pos)
    if not aligned:
        return {}, 0, 0
    return paired, min(aligned), max(aligned)


def build_footprints(explain_tsv: str, mre_col: str = 'mre_seq',
                     mirna_col: str = 'mirna_seq') -> dict:
    """(MRE, miRNA) sequence pair -> footprint dict, keyed as the eQTL file is."""
    ex = pd.read_csv(explain_tsv, sep='\t', low_memory=False)
    if 'duplex_pairs' not in ex.columns:
        raise SystemExit(f"ERROR: {explain_tsv} has no duplex_pairs column; "
                         f"rerun explain with --duplex.")
    attr_cols = [f'mre_pos_{j:02d}' for j in range(MRE_LEN)]
    attr = ex[attr_cols].to_numpy(dtype=float)

    footprints, n_bad, n_paired_tot = {}, 0, 0
    for i, r in enumerate(ex.itertuples(index=False)):
        paired, lo, hi = parse_duplex_pairs(r.duplex_pairs)
        if not paired:
            n_bad += 1
            continue
        n_paired_tot += len(paired)
        footprints[(str(getattr(r, mre_col)).upper(),
                    str(getattr(r, mirna_col)).upper())] = {
            'paired': paired,
            'span': (lo, hi),
            'duplex_type': r.duplex_type,
            'dlogit_duplex': float(r.dlogit_duplex),
            'attr': attr[i],
        }

    print(f"  pairs with a usable duplex : {len(footprints):,}")
    if n_bad:
        print(f"  pairs dropped (no paired base) : {n_bad:,}")
    if footprints:
        print(f"  mean paired target bases   : {n_paired_tot / len(footprints):.1f}")
    return footprints


def annotate(df: pd.DataFrame, footprints: dict) -> pd.DataFrame:
    """Add per-row interaction columns; `in_interaction` is the filter column."""
    out = {k: [] for k in (
        'is_cnn_tp', 'snp_offset', 'in_interaction', 'in_duplex_span',
        'mirna_pair_pos', 'in_seed', 'dist_to_nearest_paired',
        'n_paired_bases', 'snp_attr', 'duplex_type', 'dlogit_duplex',
        'mre_span_start', 'mre_span_end',
        'variant_class', 'ref_matches_mre')}

    def push(**kw):
        for k in out:
            out[k].append(kw.get(k, np.nan))

    for r in df.itertuples(index=False):
        key = (str(getattr(r, COL_MRE_SEQ)).upper(),
               str(getattr(r, COL_MIRNA)).upper())
        fp = footprints.get(key)
        if fp is None:
            push(is_cnn_tp=False)
            continue

        snp_pos = int(float(getattr(r, COL_SNP_POS)))
        mb_start = int(float(getattr(r, COL_MRE_START)))
        mb_end = int(float(getattr(r, COL_MRE_END)))
        strand = str(getattr(r, COL_STRAND))

        # Transcript-oriented offset, 0-based into gene_b.
        offset = (snp_pos - mb_start) if strand == '+' else (mb_end - snp_pos)
        if not 0 <= offset < len(key[0]):
            push(is_cnn_tp=True, snp_offset=offset, in_interaction=False,
                 in_duplex_span=False)
            continue

        tpos = offset + 1                      # 1-based, matches the footprint
        paired = fp['paired']
        span_lo, span_hi = fp['span']
        mirna_pos = paired.get(tpos)
        nearest = min(abs(tpos - p) for p in paired) if paired else np.nan

        # The footprint call is purely positional, so indels and fragments that
        # happen to carry the ALT base are still legitimately "in the
        # interaction". But eqtl_analysis_cnn.py skips both when building
        # delta_pred, so flag them rather than silently diverging from it -
        # filter on `variant_class == 'SNP' & ref_matches_mre` to line the two
        # analyses up row for row.
        ref = str(getattr(r, COL_REF, '')).upper()
        alt = str(getattr(r, COL_ALT, '')).upper()
        is_snp = len(ref) == 1 and len(alt) == 1
        if is_snp:
            expect = ref if strand == '+' else ref.translate(_COMPLEMENT)
            ref_ok = key[0][offset] == expect
        else:
            ref_ok = np.nan

        push(is_cnn_tp=True,
             snp_offset=offset,
             variant_class='SNP' if is_snp else 'indel',
             ref_matches_mre=ref_ok,
             in_interaction=mirna_pos is not None,
             in_duplex_span=span_lo <= tpos <= span_hi,
             mirna_pair_pos=mirna_pos if mirna_pos is not None else np.nan,
             in_seed=(SEED_RANGE[0] <= mirna_pos <= SEED_RANGE[1])
                     if mirna_pos is not None else np.nan,
             dist_to_nearest_paired=nearest,
             n_paired_bases=len(paired),
             snp_attr=float(fp['attr'][offset]),
             duplex_type=fp['duplex_type'], dlogit_duplex=fp['dlogit_duplex'],
             mre_span_start=span_lo, mre_span_end=span_hi)

    ann = df.copy()
    for k, v in out.items():
        ann[k] = v
    # `.eq(True)` rather than `.fillna(False).astype(bool)`: the columns are
    # object dtype (True/False/NaN), which fillna downcasts with a warning.
    ann['is_cnn_tp'] = ann['is_cnn_tp'].eq(True)
    ann['in_interaction'] = ann['in_interaction'].eq(True)
    return ann


def report(ann: pd.DataFrame, label: str) -> None:
    n = len(ann)
    tp = ann['is_cnn_tp']
    hit = ann['in_interaction']
    span = ann['in_duplex_span'].eq(True)
    seed = ann['in_seed'].eq(True)

    def pct(x, d):
        return f"{x:,} ({x / d:.1%})" if d else f"{x:,}"

    print(f"\n{'=' * 64}\n  {label}\n{'=' * 64}")
    print(f"  input rows                        : {n:,}")
    print(f"  ...on a CNN-TP pair               : {pct(int(tp.sum()), n)}")
    print(f"  ...inside the duplex span         : {pct(int(span.sum()), n)}")
    print(f"  ...on a paired base (KEPT)        : {pct(int(hit.sum()), n)}")
    print(f"  ...of those, in miRNA seed 2-8    : {pct(int((hit & seed).sum()), int(hit.sum()))}")

    nk = int(hit.sum())
    indel = hit & ann['variant_class'].eq('indel')
    refbad = hit & ann['variant_class'].eq('SNP') & ann['ref_matches_mre'].eq(False)
    clean = hit & ann['variant_class'].eq('SNP') & ann['ref_matches_mre'].eq(True)
    print(f"\n  of the kept rows -- indel (no delta_pred) : {pct(int(indel.sum()), nk)}")
    print(f"                      MRE carries ALT base  : {pct(int(refbad.sum()), nk)}")
    print(f"                      clean SNP, REF agrees : {pct(int(clean.sum()), nk)}")
    print(f"\n  unique variants  in : {ann.loc[hit, 'variant'].nunique():,}"
          f"  / total {ann['variant'].nunique():,}")
    if 'unique_key' in ann.columns:
        print(f"  unique fragments in : {ann.loc[hit, 'unique_key'].nunique():,}"
              f"  / total {ann['unique_key'].nunique():,}")
    miss = ann.loc[tp & ~hit, 'dist_to_nearest_paired'].dropna()
    if len(miss):
        print(f"\n  near-misses on TP pairs: median {miss.median():.0f} nt from the "
              f"nearest paired base (n={len(miss):,})")
    print('=' * 64)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eqtl', required=True, help='Merged GTEx eQTL x fragment TSV.')
    ap.add_argument('--explain', required=True,
                    help='TSV from cnn_branches_mirbind.py explain --duplex.')
    ap.add_argument('--mre-col', default='mre_seq', help='MRE column of --explain.')
    ap.add_argument('--mirna-col', default='mirna_seq', help='miRNA column of --explain.')
    ap.add_argument('-o', required=True, help='Filtered output TSV (in_interaction only).')
    ap.add_argument('--annotated-out', help='Optional: every input row + annotations.')
    args = ap.parse_args()

    print('Building CNN duplex footprints...')
    footprints = build_footprints(args.explain, args.mre_col, args.mirna_col)
    if not footprints:
        print('No usable footprints. Exiting.', file=sys.stderr)
        return 1

    print(f'\nLoading {Path(args.eqtl).name}...')
    df = pd.read_csv(args.eqtl, sep='\t', low_memory=False)
    ann = annotate(df, footprints)
    report(ann, Path(args.eqtl).name)

    kept = ann[ann['in_interaction']].copy()
    out = Path(args.o)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, sep='\t', index=False)
    print(f'\nFiltered eQTLs written to: {out}  ({len(kept):,} rows)')

    if args.annotated_out:
        ap_out = Path(args.annotated_out)
        ap_out.parent.mkdir(parents=True, exist_ok=True)
        ann.to_csv(ap_out, sep='\t', index=False)
        print(f'Annotated (all rows)     : {ap_out}  ({len(ann):,} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
