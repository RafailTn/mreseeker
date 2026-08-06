#!/usr/bin/env python3
"""
Keep only the GTEx eQTLs that land on a base actually paired to the miRNA in
the IntaRNA duplex of a CNN true-positive (MRE, miRNA) pair.

Inputs are the pairs TSV + IntaRNA TSV produced by

    src/eqtl_analysis/tp_intarna_pairs.py   ->  data/eqtl_tp_pairs.tsv
    src/gluon/intarna_parallel.py -n 1      ->  results/.../eqtl_tp_intarna.tsv

"Inside the interaction" is the strict reading: the variant must hit a target
nucleotide carrying a bracket in the target half of `hybrid_dp`, not merely sit
somewhere in the start_target..end_target span. A variant in an internal bulge
or loop breaks no hydrogen bond, so the permissive span would wave through
variants with no mechanism. The permissive call is still reported per row
(`in_mfe_span`) so the two can be compared without a rerun.

hybrid_dp is `<target 5'->3'>&<query 5'->3'>`; brackets nest across the `&`, so
the first '(' in the target half pairs with the *last* ')' in the query half.
Walking it with a stack recovers the exact partner, which is what lets each
surviving eQTL be labelled with the miRNA position it disrupts - and hence
whether it falls in the seed (miRNA nt 2-8).

Coordinate conventions, matching eqtl_analysis_cnn.py:
  `gene_b` is transcript-oriented, so stored index i (0-based) maps to genomic
  `start_b + i` on '+' and `end_b - i` on '-'. The SNP position is `start`
  (1-based; == `end` for SNPs).

    pixi run -m dependencies/gluon python src/eqtl_analysis/filter_eqtls_by_interaction.py \
        --eqtl data/GTEx_..._pip_gt_0_9_..._merged_with_miRBench_fragments.tsv \
        --pairs data/eqtl_tp_pairs.tsv \
        --intarna results/eqtl_intarna/eqtl_tp_intarna.tsv \
        -o results/eqtl_intarna/eqtl_in_interaction_pip_gt_0_9.tsv \
        --annotated-out results/eqtl_intarna/eqtl_annotated_pip_gt_0_9.tsv
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

_COMPLEMENT = str.maketrans('ACGT', 'TGCA')


def parse_hybrid(hybrid_dp: str, start_target: int, start_query: int) -> dict[int, int]:
    """Map paired target position -> partner miRNA position, both 1-based.

    Positions are absolute within the full 50-nt MRE / full miRNA, i.e. the
    offsets inside `hybrid_dp` shifted by start_target / start_query.
    """
    if not isinstance(hybrid_dp, str) or '&' not in hybrid_dp:
        return {}
    t_part, q_part = hybrid_dp.split('&', 1)

    stack = [start_target + i for i, c in enumerate(t_part) if c == '(']
    pairs: dict[int, int] = {}
    # Query half is 5'->3' while the brackets nest, so each ')' closes the
    # innermost open target base - pop from the end.
    for i, c in enumerate(q_part):
        if c != ')':
            continue
        if not stack:
            return {}            # malformed structure, drop the pair
        pairs[stack.pop()] = start_query + i
    return {} if stack else pairs


def build_footprints(pairs_tsv: str, intarna_tsv: str) -> tuple[dict, pd.DataFrame]:
    """(MRE, miRNA) sequence pair -> footprint dict, keyed as the eQTL file is.

    Joins on `pair_index`, the row order both files were built from.
    """
    pairs = pd.read_csv(pairs_tsv, sep='\t')
    inta = pd.read_csv(intarna_tsv, sep='\t')
    inta = inta[inta['interaction_rank'] == 1]

    merged = pairs.merge(inta, on='pair_index', how='inner',
                         suffixes=('', '_intarna'))
    if len(merged) != len(pairs):
        print(f"  WARNING: {len(pairs) - len(merged):,} pairs had no rank-1 "
              f"IntaRNA interaction", file=sys.stderr)

    footprints, n_bad, n_paired_tot = {}, 0, 0
    for r in merged.itertuples(index=False):
        if getattr(r, 'status', 'success') != 'success':
            n_bad += 1
            continue
        bp = parse_hybrid(r.hybrid_dp, int(r.start_target), int(r.start_query))
        if not bp:
            n_bad += 1
            continue
        n_paired_tot += len(bp)
        footprints[(r.mre_seq.upper(), r.mirna_seq.upper())] = {
            'paired': bp,
            'span': (int(r.start_target), int(r.end_target)),
            'E': float(r.E),
            'E_hybrid': float(r.E_hybrid),
            'query_span': (int(r.start_query), int(r.end_query)),
            'hybrid_dp': r.hybrid_dp,
        }

    print(f"  pairs with a usable duplex : {len(footprints):,}")
    if n_bad:
        print(f"  pairs dropped (no/malformed duplex) : {n_bad:,}")
    if footprints:
        print(f"  mean paired target bases   : {n_paired_tot / len(footprints):.1f}")
    return footprints, merged


def annotate(df: pd.DataFrame, footprints: dict) -> pd.DataFrame:
    """Add per-row interaction columns; `in_interaction` is the filter column."""
    out = {k: [] for k in (
        'is_cnn_tp', 'snp_offset', 'in_interaction', 'in_mfe_span',
        'mirna_pair_pos', 'in_seed', 'dist_to_nearest_paired',
        'n_paired_bases', 'intarna_E', 'intarna_E_hybrid',
        'mre_span_start', 'mre_span_end', 'hybrid_dp',
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
                 in_mfe_span=False)
            continue

        tpos = offset + 1                      # 1-based, matches IntaRNA
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
             in_mfe_span=span_lo <= tpos <= span_hi,
             mirna_pair_pos=mirna_pos if mirna_pos is not None else np.nan,
             in_seed=(SEED_RANGE[0] <= mirna_pos <= SEED_RANGE[1])
                     if mirna_pos is not None else np.nan,
             dist_to_nearest_paired=nearest,
             n_paired_bases=len(paired),
             intarna_E=fp['E'], intarna_E_hybrid=fp['E_hybrid'],
             mre_span_start=span_lo, mre_span_end=span_hi,
             hybrid_dp=fp['hybrid_dp'])

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
    span = ann['in_mfe_span'].eq(True)
    seed = ann['in_seed'].eq(True)

    def pct(x, d):
        return f"{x:,} ({x / d:.1%})" if d else f"{x:,}"

    print(f"\n{'=' * 64}\n  {label}\n{'=' * 64}")
    print(f"  input rows                        : {n:,}")
    print(f"  ...on a CNN-TP pair               : {pct(int(tp.sum()), n)}")
    print(f"  ...inside the MFE span            : {pct(int(span.sum()), n)}")
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
    ap.add_argument('--pairs', required=True, help='Pairs TSV from tp_intarna_pairs.py.')
    ap.add_argument('--intarna', required=True, help='IntaRNA TSV from intarna_parallel.py.')
    ap.add_argument('-o', required=True, help='Filtered output TSV (in_interaction only).')
    ap.add_argument('--annotated-out', help='Optional: every input row + annotations.')
    args = ap.parse_args()

    print('Building IntaRNA footprints...')
    footprints, _ = build_footprints(args.pairs, args.intarna)
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
