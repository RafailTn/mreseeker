#!/usr/bin/env python3
"""
Extract mature miRNA sequences from a miRBase GFF3 + genome FASTA.

The GTEx miRNA expression tables (results/gtex_mirna_expression/) name miRNAs
but carry no sequence, and the miRBench-derived FASTAs in data/ use anonymous
ids (mir1, mir2, ...). This bridges the two so a tissue's expressed repertoire
can actually be scored against a target site.

Sequences are pulled from the `miRNA` (mature) rows of the GFF3 — not
`miRNA_primary_transcript` — reverse-complemented on the minus strand, and
emitted as DNA (A/C/G/T) to match the `noncodingRNA` column of the v7 TSVs.

    pixi run -m dependencies/cnn python src/eqtl_analysis/mirna_sequences_from_gff.py \
        --gff  ~/Downloads/msc-thesis/data/hsa.gff3 \
        --fasta ~/Downloads/hg38/GRCh38.primary_assembly.genome.fa \
        -o data/mirbase_mature_hg38.tsv \
        --validate-against ~/Downloads/msc-thesis/data/GTEx_..._fragments.tsv

`--validate-against` cross-checks the extracted sequences against the
(noncodingRNA_name, noncodingRNA) pairs of a v7-style TSV. Any disagreement
means the GFF3 build and the dataset's miRBase version have diverged, which
would silently corrupt every downstream score, so it is reported loudly.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_ATTR = re.compile(r'(\w+)=([^;]+)')
_COMPLEMENT = str.maketrans('ACGTN', 'TGCAN')


def revcomp(seq: str) -> str:
    return seq.upper().translate(_COMPLEMENT)[::-1]


def parse_gff(path: str) -> pd.DataFrame:
    """Mature miRNA records: name, chrom, 1-based inclusive start/end, strand."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9 or f[2] != 'miRNA':
                continue
            attrs = dict(_ATTR.findall(f[8]))
            rows.append({
                'name'  : attrs.get('Name', ''),
                'mimat' : attrs.get('ID', ''),
                'chrom' : f[0],
                'start' : int(f[3]),
                'end'   : int(f[4]),
                'strand': f[6],
            })
    df = pd.DataFrame(rows)
    print(f"  {len(df):,} mature miRNA records, {df['name'].nunique():,} unique names")
    return df


def extract(df: pd.DataFrame, fasta_path: str) -> pd.DataFrame:
    from pyfaidx import Fasta
    fa = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    present = set(fa.keys())

    seqs, n_missing_chrom = [], 0
    for r in df.itertuples(index=False):
        if r.chrom not in present:
            n_missing_chrom += 1
            seqs.append(None)
            continue
        s = fa[r.chrom][r.start - 1:r.end]      # GFF3 is 1-based inclusive
        seqs.append(revcomp(s) if r.strand == '-' else s.upper())

    out = df.assign(sequence=seqs).dropna(subset=['sequence'])
    if n_missing_chrom:
        print(f"  WARNING: {n_missing_chrom:,} records on contigs absent from the "
              f"FASTA", file=sys.stderr)
    print(f"  {len(out):,} sequences extracted "
          f"(length {out['sequence'].str.len().min()}-{out['sequence'].str.len().max()} nt)")
    return out


def validate(seq_by_name: dict[str, str], tsv: str) -> None:
    """Compare against (noncodingRNA_name, noncodingRNA) pairs of a v7 TSV.

    Names in that column are pipe-separated when one sequence has several
    miRBase aliases, so each alias is checked independently.
    """
    df = pd.read_csv(tsv, sep='\t', low_memory=False,
                     usecols=['noncodingRNA_name', 'noncodingRNA']).drop_duplicates()
    n_check = n_match = n_absent = 0
    mismatches = []
    for name, seq in zip(df['noncodingRNA_name'].fillna(''), df['noncodingRNA']):
        for alias in str(name).split('|'):
            alias = alias.strip()
            if not alias:
                continue
            ref = seq_by_name.get(alias)
            if ref is None:
                n_absent += 1
                continue
            n_check += 1
            if ref == str(seq).upper():
                n_match += 1
            elif len(mismatches) < 5:
                mismatches.append((alias, str(seq).upper(), ref))

    print(f"\n  Validation against {Path(tsv).name}:")
    print(f"    names checked        : {n_check:,}")
    print(f"    exact sequence match : {n_match:,} "
          f"({n_match / n_check:.1%})" if n_check else "    (nothing to check)")
    print(f"    names not in GFF3    : {n_absent:,}")
    for alias, got, ref in mismatches:
        print(f"      MISMATCH {alias}\n        dataset: {got}\n        gff3   : {ref}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gff', required=True)
    ap.add_argument('--fasta', required=True)
    ap.add_argument('-o', required=True, help='Output TSV: name, mimat, sequence, ...')
    ap.add_argument('--validate-against', help='v7-style TSV to cross-check against.')
    args = ap.parse_args()

    print(f'Parsing {Path(args.gff).name}...')
    gff = parse_gff(args.gff)

    print(f'Extracting from {Path(args.fasta).name}...')
    out = extract(gff, args.fasta)

    seq_by_name = dict(zip(out['name'], out['sequence']))
    if args.validate_against:
        validate(seq_by_name, args.validate_against)

    dest = Path(args.o)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, sep='\t', index=False)
    print(f'\nWrote {len(out):,} mature miRNA sequences to {dest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
