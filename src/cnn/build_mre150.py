#!/usr/bin/env python3
"""Build 150-nt MRE versions of the v7 tables: the 50-nt window (`gene`) plus 50 nt of genomic
flank on each side, strand-aware, as column `gene150`. The `gene` column is kept, so one file
trains/evaluates both the 50-nt and the 150-nt model on identical rows.

    python3 src/cnn/build_mre150.py [--flank 50]
    python3 src/cnn/build_mre150.py --up 20 --down 100 --col gene170 --out data/mre170   # asymmetric

--up / --down are the 5' / 3' flanks in mRNA sense (default: --flank on both sides).

Inputs (defaults are this machine's paths; override them elsewhere):
  --genome FA       GRCh38 primary assembly FASTA with a samtools .fai index (Ensembl names, chr without "chr")
  --src-dir DIR     folder with the miRBench v7 tables (Manakov / Hejret / Klimentova)
  --novel-dir DIR   folder with the gse*_novel_labelled_v7.tsv tables
  --set NAME=PATH   add a table, or replace one of the default sets (repeatable); with --only, build just those

    python3 src/cnn/build_mre150.py --genome GRCh38.fa --src-dir v7/ --only manakov_train manakov_test
    python3 src/cnn/build_mre150.py --genome GRCh38.fa --set mysites=my_v7.tsv --only mysites --out data/mine

Coordinates are 1-based inclusive on the Ensembl GRCh38 primary assembly (chr column without
"chr"). Every fetched window is checked against `gene`; rows without coordinates, or whose
centre does not match, get N flanks (counted in the log). Positions past a chromosome end are N.

Writes data/mre150/:
  manakov_train_nochr2.tsv, manakov_val_chr2.tsv   (train split by chromosome: chr2 = val;
                                                    test is chr1, so no other chromosome fits)
  manakov_test.tsv, manakov_leftout.tsv, gse304955.tsv, gse263552.tsv,
  gse180129_HSPC.tsv, gse180129_Kasumi1.tsv, hejret_test.tsv, hejret_train.tsv, klimentova_test.tsv
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# defaults = this machine's paths; --genome / --src-dir / --novel-dir override them (FA is read by fetch(),
# and gsf_features.py swaps it for its own genome)
FA = "/mnt/raid1/rafail/Hybridetector_pipeline/external/HybriDetector/DBs/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
SRC_DIR = "/mnt/raid1/rafail/microCLIP2/data"
NOVEL_DIR = f"{ROOT}/data"
# set name -> (folder key, file name)
DEFAULT_SETS = {
    "manakov_train": ("src", "AGO2_eCLIP_Manakov2022_train_v7.tsv"),
    "manakov_test": ("src", "AGO2_eCLIP_Manakov2022_test_v7.tsv"),
    "manakov_leftout": ("src", "AGO2_eCLIP_Manakov2022_leftout_v7.tsv"),
    "gse304955": ("novel", "gse304955_novel_labelled_v7.tsv"),
    "gse263552": ("novel", "gse263552_novel_labelled_v7.tsv"),
    "gse180129_HSPC": ("novel", "gse180129_HSPC_novel_labelled_v7.tsv"),
    "gse180129_Kasumi1": ("novel", "gse180129_Kasumi1_novel_labelled_v7.tsv"),
    "hejret_test": ("src", "AGO2_CLASH_Hejret2023_test_v7.tsv"),
    "hejret_train": ("src", "AGO2_CLASH_Hejret2023_train_v7.tsv"),
    "klimentova_test": ("src", "AGO2_eCLIP_Klimentova2022_test_v7.tsv"),
}
SETS = {n: os.path.join(SRC_DIR if k == "src" else NOVEL_DIR, f) for n, (k, f) in DEFAULT_SETS.items()}  # default paths
KEEP = ["gene", "gene150", "noncodingRNA", "noncodingRNA_name", "noncodingRNA_fam", "feature",
        "label", "chr", "start", "end", "strand"]
OUT = f"{ROOT}/data/mre150"
RCT = str.maketrans("ACGTN", "TGCAN")
rc = lambda s: s.translate(RCT)[::-1]


def fetch(regions):
    """'chr:start-end' (1-based inclusive) -> + strand sequence"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(dict.fromkeys(regions)) + "\n")
    out = subprocess.run(["samtools", "faidx", FA, "-r", fh.name], capture_output=True, text=True,
                         check=True).stdout
    os.unlink(fh.name)
    seqs, cur = {}, None
    for line in out.splitlines():
        if line.startswith(">"):
            cur = line[1:]; seqs[cur] = []
        else:
            seqs[cur].append(line)
    return {k: "".join(v).upper() for k, v in seqs.items()}


def extend(d, U, Dn):
    """U nt 5' and Dn nt 3' of the window, in mRNA sense"""
    gene = d.gene.astype(str).str.upper().str.replace("U", "T")
    ok = d[["chr", "start", "end", "strand"]].notna().all(axis=1) & d.strand.isin(["+", "-"])
    out = [("N" * U + g + "N" * Dn) for g in gene]
    sub = d[ok]
    chrs = sub.chr.astype(str).str.replace("^chr", "", regex=True).replace({"M": "MT"})
    st, en = sub.start.astype(float).astype(int), sub.end.astype(float).astype(int)
    plus = (sub.strand == "+").to_numpy()
    fl = np.where(plus, U, Dn); fr = np.where(plus, Dn, U)     # genomic left / right flank
    want_lo, hi = st - fl, en + fr
    lo = want_lo.clip(lower=1)
    reg = (chrs + ":" + lo.astype(str) + "-" + hi.astype(str)).tolist()
    seqs = fetch(reg)
    n_ok = 0
    for i, r, w0, l0, h0, sd, g in zip(sub.index, reg, want_lo, lo, hi, sub.strand, gene[ok]):
        s = seqs.get(r, "")
        left_missing = l0 - w0                       # nt cut at chromosome start
        right_missing = h0 - (l0 + len(s) - 1)       # nt past chromosome end
        s = "N" * left_missing + s + "N" * max(0, right_missing)
        if sd == "-":
            s = rc(s)
        if len(s) == len(g) + U + Dn and s[U:U + len(g)] == g:
            out[i] = s; n_ok += 1
    return out, int(ok.sum()), n_ok


def main():
    global FA
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", default=FA, help="GRCh38 primary assembly FASTA (samtools faidx-indexed)")
    ap.add_argument("--src-dir", default=SRC_DIR, help="folder with the miRBench v7 tables")
    ap.add_argument("--novel-dir", default=NOVEL_DIR, help="folder with the gse*_novel_labelled_v7.tsv tables")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=PATH",
                    help="add or replace an input table (repeatable)")
    ap.add_argument("--flank", type=int, default=50)
    ap.add_argument("--up", type=int, default=None, help="5' flank (default --flank)")
    ap.add_argument("--down", type=int, default=None, help="3' flank (default --flank)")
    ap.add_argument("--col", default="gene150", help="output sequence column")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", nargs="+", default=None, help="build only these sets")
    a = ap.parse_args()
    FA = a.genome
    if not os.path.exists(FA + ".fai"):
        ap.error(f"--genome {FA}: no .fai index next to it (run: samtools faidx {FA})")
    dirs = {"src": a.src_dir, "novel": a.novel_dir}
    sets = {n: os.path.join(dirs[k], f) for n, (k, f) in DEFAULT_SETS.items()}
    for kv in a.set:
        if "=" not in kv:
            ap.error(f"--set {kv}: expected NAME=PATH")
        n, pth = kv.split("=", 1)
        sets[n] = pth
    todo = {n: pth for n, pth in sets.items() if not a.only or n in a.only}
    if a.only and set(a.only) - set(sets):
        ap.error(f"--only: unknown set(s) {sorted(set(a.only) - set(sets))}")
    missing = [pth for pth in todo.values() if not os.path.exists(pth)]
    if missing:
        ap.error("input table(s) not found: " + ", ".join(missing))
    U = a.flank if a.up is None else a.up
    Dn = a.flank if a.down is None else a.down
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    for name, path in todo.items():
        d = pd.read_csv(path, sep="\t", low_memory=False)
        d = d.reset_index(drop=True)
        d[a.col], n_coord, n_ok = extend(d, U, Dn)
        d = d[list(dict.fromkeys(c for c in KEEP + [a.col] if c in d.columns and (c != "gene150" or a.col == "gene150")))]
        print(f"{name}: {len(d)} rows, {n_coord} with coordinates, {n_ok} genome-matched "
              f"({n_ok / len(d):.4%}); N-flanked {len(d) - n_ok}", flush=True)
        if name == "manakov_train":
            val = d.chr.astype(str).str.replace("^chr", "", regex=True) == "2"
            d[~val].to_csv(f"{out}/manakov_train_nochr2.tsv", sep="\t", index=False)
            d[val].to_csv(f"{out}/manakov_val_chr2.tsv", sep="\t", index=False)
            print(f"  train {int((~val).sum())} / val(chr2) {int(val.sum())}", flush=True)
        else:
            d.to_csv(f"{out}/{name}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
