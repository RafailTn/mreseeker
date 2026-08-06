#!/usr/bin/env python3
"""
Collect the CNN true-positive (MRE, miRNA) pairs that an eQTL actually sits on,
and emit the row-aligned FASTA pair `intarna_parallel.py` expects.

Only pairs present in the merged GTEx-eQTL x miRBench fragment files can ever
contribute to the downstream eQTL filter, so there is no point pushing all
~840k CNN TPs through IntaRNA - the eQTL-overlapping subset (~7.3k pairs)
yields an identical filter for a fraction of the runtime.

A pair is keyed on (MRE sequence, miRNA sequence), the only identity the error
TSVs and the eQTL files share: the error TSVs carry no `unique_key`, and the
same 50-nt fragment can recur at several loci. Keying on sequence also matches
what the CNN actually saw, and it keeps the IntaRNA footprint (a pure function
of the two sequences) reusable across every locus the pair occurs at.

    pixi run -m dependencies/gluon python src/eqtl_analysis/tp_intarna_pairs.py \
        --eqtl data/GTEx_..._pip_gt_0_9_..._merged_with_miRBench_fragments.tsv \
        --eqtl data/GTEx_..._pip_lt_0_01_..._merged_with_miRBench_fragments.tsv \
        --errors errors/AGO2_eCLIP_Manakov2022_train_v7_oof_errtype.tsv \
        --errors errors/manakov_test_errors_v7_restructure.tsv \
        --errors errors/manakov_leftout_errors_v7_restructure.tsv \
        --out-pairs data/eqtl_tp_pairs.tsv \
        --mre-fasta data/eqtl_tp_mre.fa --mirna-fasta data/eqtl_tp_mirna.fa

The FASTAs are written one record per pairs-TSV row, in file order, with the
row index baked into every ID - `intarna_parallel.py` zips the two files, so a
misalignment has to be visible rather than silent.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

_ID_SAFE = re.compile(r'[^A-Za-z0-9._-]')

# Columns carried from the error TSV onto each retained pair. `gene` is the
# 50-nt MRE, already transcript-oriented on both strands.
_CARRY = ['noncodingRNA_name', 'noncodingRNA_fam', 'feature', 'chr', 'start',
          'end', 'strand', 'dominant_region', 'interaction_probability']


def sanitize(name: str) -> str:
    """IntaRNA takes the ID as a bare command-line token, so keep it boring."""
    return _ID_SAFE.sub('_', str(name)) if name and str(name) != 'nan' else ''


def load_eqtl_pairs(paths: list[str]) -> set[tuple[str, str]]:
    """(MRE, miRNA) sequence pairs that at least one eQTL row sits on."""
    want: set[tuple[str, str]] = set()
    for p in paths:
        d = pd.read_csv(p, sep='\t', low_memory=False,
                        usecols=['gene_b', 'noncodingRNA'])
        want.update(zip(d['gene_b'].astype(str).str.upper(),
                        d['noncodingRNA'].astype(str).str.upper()))
        print(f"  {Path(p).name}: {len(d):,} rows -> {len(want):,} pairs so far")
    return want


def scan_errors(paths: list[str], want: set[tuple[str, str]],
                min_prob: float | None = None) -> dict:
    """Stream the error TSVs, keeping TP rows whose pair an eQTL sits on.

    The train OOF file is ~2 GB, so this reads row-by-row rather than loading a
    frame. First TP occurrence of a pair wins; later splits only append their
    tag to `tp_sources` so it stays visible when a pair is a TP in more than
    one split.

    `min_prob` additionally requires interaction_probability >= that value, so
    the set can be narrowed to high-confidence TPs. Every TP already clears the
    0.5 decision threshold, so this only tightens an existing cut - it never
    admits a row that `error_type` had rejected. The threshold is applied per
    row *before* dedup, and a pair is kept if any of its TP rows clears it; the
    retained row is the first such one.
    """
    kept: dict[tuple[str, str], dict] = {}
    for path in paths:
        tag = Path(path).stem
        rows = matched = tp = new = below = 0
        with open(path, newline='') as fh:
            reader = csv.reader(fh, delimiter='\t')
            hdr = next(reader)
            idx = {c: i for i, c in enumerate(hdr)}
            for col in ('gene', 'noncodingRNA', 'error_type'):
                if col not in idx:
                    raise SystemExit(f"ERROR: column '{col}' not in {path}")
            i_gene, i_mir, i_err = idx['gene'], idx['noncodingRNA'], idx['error_type']
            carry = [(c, idx[c]) for c in _CARRY if c in idx]
            if min_prob is not None and 'interaction_probability' not in idx:
                raise SystemExit(f"ERROR: --min-prob needs 'interaction_probability' "
                                 f"in {path}")
            i_prob = idx.get('interaction_probability')

            for row in reader:
                rows += 1
                key = (row[i_gene].upper(), row[i_mir].upper())
                if key not in want:
                    continue
                matched += 1
                if row[i_err] != 'TP':
                    continue
                tp += 1
                if min_prob is not None:
                    try:
                        if float(row[i_prob]) < min_prob:
                            below += 1
                            continue
                    except (TypeError, ValueError):
                        below += 1
                        continue
                if key in kept:
                    kept[key]['tp_sources'].append(tag)
                    continue
                new += 1
                rec = {c: row[i] for c, i in carry}
                rec['mre_seq'] = row[i_gene].upper()
                rec['mirna_seq'] = row[i_mir].upper()
                rec['tp_sources'] = [tag]
                kept[key] = rec
        extra = (f", {below:,} TP below the prob cut" if min_prob is not None else "")
        print(f"  {Path(path).name}: {rows:,} rows, {matched:,} on an eQTL pair, "
              f"{tp:,} of those TP{extra} ({new:,} newly retained)")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eqtl', action='append', required=True,
                    help='Merged GTEx eQTL x miRBench fragment TSV (repeatable).')
    ap.add_argument('--errors', action='append', required=True,
                    help='Error TSV carrying gene/noncodingRNA/error_type (repeatable).')
    ap.add_argument('--out-pairs', required=True, help='Deduplicated pairs TSV.')
    ap.add_argument('--mre-fasta', required=True)
    ap.add_argument('--mirna-fasta', required=True)
    ap.add_argument('--min-prob', type=float, default=None,
                    help='Additionally require interaction_probability >= this '
                         '(e.g. 0.7 for high-confidence TPs only). TPs already '
                         'clear 0.5, so this only tightens that cut.')
    args = ap.parse_args()

    print('Loading eQTL-overlapping (MRE, miRNA) pairs...')
    want = load_eqtl_pairs(args.eqtl)
    print(f'  unique pairs an eQTL sits on: {len(want):,}\n')

    cut = ('' if args.min_prob is None
           else f' (interaction_probability >= {args.min_prob})')
    print(f'Scanning error TSVs for CNN true positives{cut}...')
    kept = scan_errors(args.errors, want, args.min_prob)
    if not kept:
        print('No eQTL-overlapping TP pairs found. Exiting.', file=sys.stderr)
        return 1
    print(f'\n  distinct TP pairs retained: {len(kept):,} '
          f'({len(kept) / len(want):.1%} of eQTL pairs)')

    df = pd.DataFrame(list(kept.values()))
    df['tp_sources'] = df['tp_sources'].apply(lambda s: ','.join(sorted(set(s))))
    df.insert(0, 'pair_index', range(1, len(df) + 1))

    out = Path(args.out_pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)

    with open(args.mre_fasta, 'w') as fm, open(args.mirna_fasta, 'w') as fq:
        for i, mre, mirna, name in zip(df['pair_index'], df['mre_seq'],
                                       df['mirna_seq'],
                                       df.get('noncodingRNA_name', [''] * len(df))):
            mir_id = sanitize(name)
            fm.write(f'>mre_{i}\n{mre}\n')
            fq.write(f">{mir_id + '_' if mir_id else 'mir_'}{i}\n{mirna}\n")

    print(f'\nWrote {len(df):,} pairs to {out}')
    print(f'Wrote {len(df):,} records each to {args.mre_fasta} and {args.mirna_fasta}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
