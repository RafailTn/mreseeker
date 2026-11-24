#!/usr/bin/env python3
"""Label-free readout windows for the artifact check: human/rat mixing chimeras and Expt8 occupancy sites.

Trimmed from src/cnn/eval_window_clean.py + eval_noise.build_inputs (mixing and occupancy only; same rows,
same columns, same sequences). Windows are extended from the central 50 nt in mRNA sense:
50 = core, 150 = +-50, 300 = +-125. A row is kept only if every window fits on the chromosome, so the
three tables are row-aligned.

  mixing     Expt9/10 293TC9mix chimeras of human-only / rat-only miRNAs with a 50-nt target, coordinates on
             the dual-species genome; real = same-species target, else post-lysis artifact
  occupancy  Manakov Expt8 miR-1 / miR-124 sites (validate_overexpression): site defined in one replicate,
             AGO2 occupancy gain (own-oe minus other-oe) measured in the other

Inputs (DATA, not code) from $HYBRIDETECTOR_DIR: data/mixing_artifacts/chimeras.tsv,
data/validate_overexpression/{sites,occupancy}.tsv, the HybriDetector hg38 and hg38+rn genome FASTAs
(indexed). Needs samtools on PATH.

    python3 src/artifact/eval_windows.py build   [--out DIR]   # -> DIR/{50,150,300}/{mixing,occupancy}.tsv
    MRESEEKER_MRE_LEN=150 pixi run ... python3 src/artifact/eval_windows.py predict [--out DIR] [--checkpoint PT]
                                                       # -> DIR/150/pred_w150_{mixing,occupancy}.tsv
DIR defaults to data/relabel/noise/eval_win_v2, where the other scripts read them. Existing files are kept.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_common import HYB_DIR, ROOT, rc  # noqa: E402

OUT = f"{ROOT}/data/relabel/noise/eval_win_v2"
VO = f"{HYB_DIR}/data/validate_overexpression"
CHIM = f"{HYB_DIR}/data/mixing_artifacts/chimeras.tsv"
FA_HSRN = f"{HYB_DIR}/external/HybriDetector_hsrn/DBs/genome_hsrn.fa"
FA_HS = f"{HYB_DIR}/external/HybriDetector/DBs/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
WIN = {"50": (0, 0), "150": (50, 50), "300": (125, 125)}    # (5' flank, 3' flank) in mRNA sense
L = 50
CKPT = f"{ROOT}/cnn_checkpoints/cnn_mre150.pt"
PX = ["pixi", "run", "--frozen", "--manifest-path", f"{ROOT}/dependencies/cnn/pixi.toml"]


def fetch(fa, chrom, start, end, strand, out):
    """1-based inclusive core [start, end]; returns {win: [seq or None]} in mRNA sense."""
    regs, keys = [], []
    for i, (c, s, e, st) in enumerate(zip(chrom, start, end, strand)):
        for w, (f5, f3) in WIN.items():
            lo, hi = (s - f5, e + f3) if st == "+" else (s - f3, e + f5)
            if lo >= 1:
                regs.append(f"{c}:{lo}-{hi}"); keys.append((i, w, st))
    tmp = f"{out}/regions.txt"
    with open(tmp, "w") as fh:
        fh.write("\n".join(dict.fromkeys(regs)) + "\n")
    txt = subprocess.run(["samtools", "faidx", fa, "-r", tmp], capture_output=True, text=True, check=True).stdout
    seqs, cur = {}, None
    for line in txt.splitlines():
        if line.startswith(">"):
            cur = line[1:]; seqs[cur] = []
        else:
            seqs[cur].append(line)
    seqs = {k: "".join(v).upper() for k, v in seqs.items()}
    res = {w: [None] * len(chrom) for w in WIN}
    for r, (i, w, st) in zip(regs, keys):
        q = seqs.get(r, "")
        q = q if st == "+" else rc(q)
        f5, f3 = WIN[w]
        if len(q) == L + f5 + f3:
            res[w][i] = q
    return res


def write_windows(x, s, core, name, out):
    ok = np.array([all(s[w][i] is not None for w in WIN) for i in range(len(x))])
    assert np.mean([s["50"][i] == t for i, t in enumerate(core) if ok[i]]) > 0.99, "core window mismatch"
    for w in WIN:
        os.makedirs(f"{out}/{w}", exist_ok=True)
        y = x[ok].copy(); y["gene"] = [q for q, k in zip(s[w], ok) if k]
        y.to_csv(f"{out}/{w}/{name}.tsv", sep="\t", index=False)
    print(f"{name}: {ok.sum():,} of {len(x):,} rows have every window", flush=True)


def build(out):
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{out}/150/mixing.tsv"):
        c = pd.read_csv(CHIM, sep="\t", low_memory=False,
                        usecols=["cells", "mirna_species", "mirna_seq", "target_species", "target", "canonical",
                                 "chr.g", "start.g", "end.g", "strand.g"])
        c = c[(c.cells == "293TC9mix") & c.mirna_species.isin(["human_only", "rat_only"]) & (c.target.str.len() == L)].copy()
        c["real"] = ((c.mirna_species == "human_only") & (c.target_species == "human")) | \
                    ((c.mirna_species == "rat_only") & (c.target_species == "rat"))
        s = fetch(FA_HSRN, c["chr.g"].astype(str), c["start.g"], c["end.g"], c["strand.g"], out)
        write_windows(c.rename(columns={"mirna_seq": "noncodingRNA"}), s, c.target.tolist(), "mixing", out)
    if not os.path.exists(f"{out}/150/occupancy.tsv"):
        s = pd.read_csv(f"{VO}/sites.tsv", sep="\t", dtype={"chr": str})
        o = pd.read_csv(f"{VO}/occupancy.tsv", sep="\t")
        s = s[s.which.isin(["miR1", "miR124"])][["site_id", "which", "rep", "gene", "mirna_seq", "site_class"]]
        s = s.drop_duplicates(["site_id", "which", "rep"]).rename(columns={"rep": "defined_in"})
        o = o[o.which.isin(["miR1", "miR124"])]
        x = o.merge(s, on=["site_id", "which", "defined_in", "site_class"], how="inner")
        x["gain"] = np.where(x.which == "miR1", x.delta_miR1_minus_miR124, -x.delta_miR1_minus_miR124)
        x = x[x.gene.str.len() == L].rename(columns={"mirna_seq": "noncodingRNA"}).reset_index(drop=True)
        p = x.site_id.str.extract(r"^(.+):(\d+)-(\d+):([+-])$")
        sq = fetch(FA_HS, p[0], p[1].astype(int), p[2].astype(int), p[3], out)
        write_windows(x, sq, x.gene.tolist(), "occupancy", out)


def predict(out, ckpt):
    assert os.environ.get("MRESEEKER_MRE_LEN") == "150", "set MRESEEKER_MRE_LEN=150"
    for s in ["mixing", "occupancy"]:
        o = f"{out}/150/pred_w150_{s}.tsv"
        if not os.path.exists(o):
            subprocess.run([sys.executable, f"{ROOT}/src/cnn/predict_cnn.py", "--checkpoint", ckpt, "--input",
                            f"{out}/150/{s}.tsv", "-o", o, "--mre-col", "gene", "--mirna-col", "noncodingRNA",
                            "--no-dedup"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
            print("wrote", o, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "predict"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--checkpoint", default=CKPT)
    a = ap.parse_args()
    build(a.out) if a.cmd == "build" else predict(a.out, a.checkpoint)
