#!/usr/bin/env python3
"""Shared code for the artifact check and removal (src/artifact). Everything here used to be imported from outside
this folder; it is copied verbatim so the folder runs on its own:

  site_class / has_seed / subclass     canonical seed classes and partial-pairing subclasses
                                        (was Hybridetector_pipeline/audit_noncanonical.site_class and
                                        src/cnn/noseed_tier1.subclass)
  dinuc_shuffle / rand_p                decoy shuffles and randomised-tie empirical p (audit_noncanonical)
  mixing_estimate                       post-lysis share A_h / A_r of one human/rat mix block
                                        (Hybridetector_pipeline/mixing_artifacts.estimate)
  SAMPLES / load_sample                 HybriDetector chimera reads for the ligation-geometry test
                                        (src/cnn/relabel_chimeras.py)

Code that stays outside: src/cnn/build_mre150.py (window tables; tracked) and src/training/cnn (model; tracked).
External DATA (not code) still comes from the Hybridetector pipeline folder; point HYBRIDETECTOR_DIR at it.
IntaRNA comes from PATH, else from the repo's gluon Pixi environment.
"""
import os
import random
import re
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CNN_SRC = f"{ROOT}/src/cnn"                         # build_mre150.py (tracked)
TRAIN_SRC = f"{ROOT}/src/training/cnn"              # cnn_branches_mirbind.py (tracked)
HYB_DIR = os.environ.get("HYBRIDETECTOR_DIR", "/mnt/raid1/rafail/Hybridetector_pipeline")
# GRCh38 primary assembly (Ensembl names, .fai next to it): GRCH38_FA, else the fetch_data.sh copy, else HybriDetector's
GENOME = os.environ.get("GRCH38_FA") or next(
    (f for f in (f"{ROOT}/data/external/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                 f"{HYB_DIR}/external/HybriDetector/DBs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
     if os.path.exists(f + ".fai")),
    f"{HYB_DIR}/external/HybriDetector/DBs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
INTARNA = shutil.which("IntaRNA") or f"{ROOT}/dependencies/gluon/.pixi/envs/default/bin/IntaRNA"

RCT = str.maketrans("ACGTN", "TGCAN")
rc = lambda s: s.translate(RCT)[::-1]


def norm(s):
    return s.astype(str).str.upper().str.replace("U", "T")


# ── seed classes ─────────────────────────────────────────────────────────────
def site_class(t, m):
    """best canonical site of miRNA m (5'->3', DNA letters) in target t (mRNA sense), first match wins"""
    m8, m7, off = rc(m[1:8]), rc(m[1:7]), rc(m[2:8])
    if m8 + "A" in t:
        return "8mer"
    if m8 in t:
        return "7mer-m8"
    if m7 + "A" in t:
        return "7mer-A1"
    if m7 in t:
        return "6mer"
    if off in t:
        return "offset-6mer"
    return "noncanonical"


def has_seed(t, m):
    return site_class(t, m) != "noncanonical"


def subclass(t, m):
    """Partial-pairing subclass of target t (mRNA sense) for miRNA m (5'->3'), first match wins."""
    # seed nt2-7 / nt3-8 with exactly one G:U wobble (miRNA G opposite target U, miRNA U opposite target G)
    for a, b in [(1, 7), (2, 8)]:
        s = m[a:b]
        for i, x in enumerate(s):
            if x in "GT":
                v = s[:i] + ("A" if x == "G" else "C") + s[i + 1:]   # rc partner swap: G->U wobble, U->G wobble
                if rc(v) in t:
                    return "seed_GU"
    # 7mer nt2-8 with exactly one mismatch (any substitution)
    s = rc(m[1:8])
    for i in range(7):
        for x in "ACGT":
            if x != s[i] and s[:i] + x + s[i + 1:] in t:
                return "seed7_1mm"
    # nt2-8 paired with one extra (bulged) target nt between miRNA nt k and k+1 (k = 3..6)
    for k in range(3, 7):
        a, b = rc(m[k:8]), rc(m[1:k])
        for x in "ACGT":
            if a + x + b in t:
                return "seed_bulge"
    if rc(m[11:17]) in t:
        return "3p_6mer"
    r = rc(m[:20])
    if any(r[i:i + 8] in t for i in range(len(r) - 7)):
        return "contig8"
    return "none"


# ── decoys and empirical p ───────────────────────────────────────────────────
def dinuc_shuffle(s: str, rng: random.Random) -> str:
    """Altschul-Erickson dinucleotide-preserving shuffle (random Eulerian path).

    Retries until the walk uses every edge; returns `s` unchanged if it never does."""
    if len(s) < 3:
        return s
    for _ in range(100):
        try:
            out = _dinuc_shuffle_once(s, rng)
        except (KeyError, IndexError):
            continue
        if len(out) == len(s):
            return out
    return s


def _dinuc_shuffle_once(s: str, rng: random.Random) -> str:
    edges: dict[str, list[str]] = {}
    for a, b in zip(s, s[1:]):
        edges.setdefault(a, []).append(b)
    last = s[-1]
    for _ in range(100):
        # random arborescence into `last`: each non-last vertex picks a random last exit
        last_exit = {}
        for v in edges:
            if v == last:
                continue
            last_exit[v] = rng.choice(edges[v])
        # check every vertex reaches `last` via last_exit edges
        ok = True
        for v in last_exit:
            seen, u = set(), v
            while u != last:
                if u in seen or u not in last_exit:
                    ok = False
                    break
                seen.add(u)
                u = last_exit[u]
            if not ok:
                break
        if ok:
            break
    lists = {}
    for v, outs in edges.items():
        outs = list(outs)
        if v in last_exit:
            outs.remove(last_exit[v])
            rng.shuffle(outs)
            outs.append(last_exit[v])
        else:
            rng.shuffle(outs)
        lists[v] = outs
    res, u = [s[0]], s[0]
    for _ in range(len(s) - 1):
        u = lists[u].pop(0)
        res.append(u)
    return "".join(res)


def rand_p(real: float, decoys: np.ndarray, rng: np.random.Generator, lower_better=True) -> float:
    d = decoys[~np.isnan(decoys)]
    if np.isnan(real) or len(d) == 0:
        return np.nan
    if lower_better:
        better, ties = (d < real).sum(), (d == real).sum()
    else:
        better, ties = (d > real).sum(), (d == real).sum()
    return (better + rng.uniform() * (ties + 1)) / (len(d) + 1)


# ── human/rat mixing ─────────────────────────────────────────────────────────
def mixing_estimate(d: pd.DataFrame, q_r: float, f: dict) -> dict:
    """A_h, A_r from one block of mix hybrids (q_r: read-weighted rat share of the block)"""
    hh = d[d.mirna_species == "human_only"]
    rr = d[d.mirna_species == "rat_only"]
    n_h, c_h = len(hh), int((hh.target_species == "rat").sum())
    miscall = f["m_h"] * n_h                                   # expected human reads called rat-only
    n_r = max(len(rr) - miscall, 0.0)
    c_r = max(float((rr.target_species == "human").sum()) - miscall, 0.0)
    x_h = c_h / n_h if n_h else np.nan
    x_r = c_r / n_r if n_r > 0 else np.nan
    return dict(n_h=n_h, x_h=x_h, A_h=(x_h - f["f_h"]) / q_r if q_r > 0 else np.nan,
                n_r=round(n_r, 1), x_r=x_r, A_r=(x_r - f["f_r"]) / (1 - q_r) if q_r < 1 else np.nan)


# ── HybriDetector chimera reads (ligation geometry) ──────────────────────────
HD = f"{HYB_DIR}/external/HybriDetector"
HYB = f"{HD}/hyb_pairs"
SAMPLES = ["Expt1_293T_NoGelChimeCLIP_rep1", "Expt1_293T_NoGelChimeCLIP_rep2",
           "Expt1_293T_WithGelChimeCLIP_rep1", "Expt1_293T_WithGelChimeCLIP_rep2",
           "Expt4_293T_NoGelChimeCLIP_rep1", "Expt4_293T_WithGelChimeCLIP_rep1",
           "Expt9_293T_NoGelChimeCLIP_rep1", "Expt9_293T_NoGelChimeCLIP_rep2",
           "Expt9_293T_WithGelChimeCLIP_rep1", "Expt9_293T_WithGelChimeCLIP_rep2"]
CHR = set([str(i) for i in range(1, 23)] + ["MT", "X", "Y"])
CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def cigar_parts(c):
    """(left softclip, right softclip, reference length)"""
    ops = CIG.findall(c)
    ls = int(ops[0][0]) if ops and ops[0][1] == "S" else 0
    rs = int(ops[-1][0]) if len(ops) > 1 and ops[-1][1] == "S" else 0
    ref = sum(int(n) for n, o in ops if o in "MDN=X")
    return ls, rs, ref


def load_sample(s):
    d = pd.read_csv(f"{HYB}/{s}.hyb_pairs_deduplicated.tsv", sep="\t", low_memory=False,
                    dtype={"chr.g": str})
    n0 = len(d)
    # HybriDetector filters (filter_and_collapse.R) + high-confidence (nomm) + unique miRNA
    d = d[d.smallrna.astype(str).str.startswith("hsa") & ~d.smallrna.astype(str).str.contains("|", regex=False)]
    d = d[(d.mrna_length == True) & (d.mrna_aln == True)  # noqa: E712
          & ~d.repeatmasker.isin(["Helitron1Na_Mam", "(G)n"])
          & ~((d.A_perc > 50) | (d.C_perc > 50) | (d.G_perc > 50) | (d.T_perc > 50))
          & (d["seq.m"].str.len() <= 30) & (d.smallrna_nomm == True) & d["chr.g"].isin(CHR)]  # noqa: E712
    # pos.m per (miRNA, read miRNA sequence) from the per-read table
    a = pd.read_csv(f"{HYB}/{s}.all_hyb_pairs.tsv", sep="\t", usecols=["smallrna", "seq.m", "pos.m"])
    pm = a.groupby(["smallrna", "seq.m"])["pos.m"].agg(lambda v: v.value_counts().index[0])
    d = d.merge(pm.rename("pos_m").reset_index(), on=["smallrna", "seq.m"], how="left")
    d["pos_m"] = d.pos_m.fillna(-1).astype(int)
    cp = np.array([cigar_parts(c) for c in d["cigar.g"]])
    pos = d["pos.g"].to_numpy(int)
    left, right = pos, pos + cp[:, 2] - 1
    plus = (d.strand == "+").to_numpy()
    lig5 = np.where(plus, left, right)
    lig3 = np.where(plus, right, left)
    fwd = d.forward_dir.astype(str).str.upper().eq("TRUE").to_numpy()
    d["anchor"] = np.where(fwd, lig3, lig5)
    d["fwd"] = fwd
    d["hd_end"] = pos + d["seq.g"].str.len().to_numpy()      # HybriDetector's end (pos + clipped length)
    d["sample"] = s
    print(f"{s}: {n0} dedup rows -> {len(d)} after HD filters; pos.m>1 share (Nunique-weighted) "
          f"{np.average(d.pos_m > 1, weights=d.Nunique):.3f}; target-first {np.average(fwd, weights=d.Nunique):.3f}",
          flush=True)
    return d
