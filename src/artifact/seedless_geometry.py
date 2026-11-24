#!/usr/bin/env python3
"""Stage 1.2 of the seed-less plan: ligation geometry (per read, independent of the CNN and of energy ranks).

For a real AGO-bound duplex the ligation junction (target 5' end of a miRNA-first read, target 3' end of a
target-first read) sits at a characteristic offset from the duplex. For a chance ligation the own miRNA's best
duplex is no closer to that offset than a decoy miRNA's best duplex (exchangeable under random ligation).

  reads   10 HEK293T chimeric eCLIP libraries (Expt1/4/9), HybriDetector filters + junction anchors exactly as
          artifact_common.load_sample; reads of the row's miRNA family on the same strand whose
          junction lies in the central 150 nt of the row's 300-nt window (mRNA-sense coordinates)
  anchor  target 3' end (end1) of the IntaRNA duplex from Stage 1.1, for the own miRNA and for every decoy
  f(d)    smoothed offset density d = junction - anchor per read orientation, fitted on core-canonical
          positives (own duplex), Nunique-weighted
  score   per row and query: mean Nunique-weighted log f(d); p_geom = randomised rank of the own score among
          the 10 expressed decoys (higher = better); R per cell from Storey pi0 (seedless_cells.py)

    python3 src/artifact/seedless_geometry.py   -> data/seedless/geometry/geometry.tsv.gz
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_common import ROOT, SAMPLES, load_sample, rand_p  # noqa: E402

PAIR = f"{ROOT}/data/seedless/pairing"
OUT = f"{ROOT}/data/seedless/geometry"
N_EXPR = 10
DMAX = 150
N_SWAP = 20


def reads():
    f = f"{OUT}/reads.tsv.gz"
    if os.path.exists(f):
        return pd.read_csv(f, sep="\t", dtype={"chr": str})
    from multiprocessing import Pool
    with Pool(5) as p:
        parts = p.map(load_sample, SAMPLES)
    d = pd.concat(parts, ignore_index=True)
    d = d.rename(columns={"chr.g": "chr", "smallRNA_fam": "fam"})[["chr", "strand", "fam", "smallrna", "anchor", "fwd",
                                                                     "Nunique", "sample"]]
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(f, sep="\t", index=False)
    return d


def name_to_family():
    """miRNA name -> v7 family, from the noncodingRNA_name lists of the 150-nt tables"""
    m = {}
    for f in ["manakov_train_nochr2", "manakov_val_chr2", "manakov_test", "manakov_leftout"]:
        d = pd.read_csv(f"{ROOT}/data/mre150/{f}.tsv", sep="\t", usecols=["noncodingRNA_name", "noncodingRNA_fam"]).drop_duplicates()
        for names, fam in zip(d.noncodingRNA_name.astype(str), d.noncodingRNA_fam.astype(str)):
            for n in names.split("|"):
                m.setdefault(n, fam)
    return m


def load_pairs():
    rows = pd.read_csv(f"{PAIR}/rows.tsv.gz", sep="\t", low_memory=False)
    R, U = [], []
    for f in sorted(glob.glob(f"{PAIR}/chunks/*.npz")):
        z = np.load(f, allow_pickle=True)
        R.append(z["R"]); U.append(z["uid"])
    R, U = np.concatenate(R), np.concatenate(U)
    rows = rows.set_index("uid").loc[U].reset_index()
    return rows, R


def collect(rows, rd):
    """per row: arrays of (x junction in 300-nt mRNA coords, orientation fwd, weight)"""
    rd = rd.copy()
    rd["chr"] = rd["chr"].astype(str)
    key = rd["chr"] + "|" + rd.strand + "|" + rd.fam.astype(str)
    groups = {k: (g.anchor.to_numpy(), g.fwd.to_numpy(bool), g.Nunique.to_numpy(float))
              for k, g in rd.assign(key=key).sort_values("anchor").groupby("key")}
    out = []
    for chr_, st, en, sd, fam in zip(rows.chr.astype(str), rows.start, rows.end, rows.strand, rows.noncodingRNA_fam.astype(str)):
        g = groups.get(f"{chr_.replace('chr', '')}|{sd}|{fam}")
        if g is None or not np.isfinite(st):
            out.append(None); continue
        st, en = int(st), int(en)
        lo, hi = st - 125, en + 125
        a, fw, w = g
        i, j = np.searchsorted(a, lo), np.searchsorted(a, hi, side="right")
        x = (a[i:j] - lo + 1) if sd == "+" else (hi - a[i:j] + 1)
        k = (x >= 76) & (x <= 225)
        out.append((x[k], fw[i:j][k], w[i:j][k]) if k.any() else None)
    return out


def fit_density(J, A):
    """J: list of junction tuples, A: own-duplex anchors (end1) -> log density per orientation over d"""
    dens = {}
    for o in (False, True):
        h = np.zeros(2 * DMAX + 1)
        for jt, a in zip(J, A):
            if jt is None or not np.isfinite(a):
                continue
            x, fw, w = jt
            m = fw == o
            d = np.clip((x[m] - a).astype(int), -DMAX, DMAX) + DMAX
            np.add.at(h, d, w[m])
        k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
        h = np.convolve(h, k / k.sum(), mode="same") + 1e-3 * h.sum() / len(h) + 1e-9
        dens[o] = np.log(h / h.sum())
    return dens


def score(J, A, dens):
    """A: (n, Q) anchors -> (n, Q) mean log-likelihood; NaN where no reads or no duplex"""
    S = np.full(A.shape, np.nan)
    for i, jt in enumerate(J):
        if jt is None:
            continue
        x, fw, w = jt
        for q in range(A.shape[1]):
            a = A[i, q]
            if not np.isfinite(a):
                continue
            d = np.clip((x - a).astype(int), -DMAX, DMAX) + DMAX
            ll = np.where(fw, dens[True][d], dens[False][d])
            S[i, q] = np.average(ll, weights=w)
    return S


def main():
    rows, R = load_pairs()
    keep = rows.set.isin(["manakov_test", "manakov_leftout"]).to_numpy() & (rows.label.to_numpy() == 1)
    rows, R = rows[keep].reset_index(drop=True), R[keep]
    rows["end"] = rows.start + 49
    rd = reads()
    fmap = name_to_family()
    rd["fam"] = rd.smallrna.map(fmap)
    print(f"read miRNAs mapped to a v7 family: {np.average(rd.fam.notna(), weights=rd.Nunique):.3f} of unique reads")
    rd = rd[rd.fam.notna()]
    print(f"reads: {len(rd):,} rows, {rd.Nunique.sum():,.0f} unique reads; target-first share "
          f"{np.average(rd.fwd, weights=rd.Nunique):.3f}", flush=True)
    J = collect(rows, rd)
    A = R[:, 4, :]                                # end1 of each query's duplex (300-nt coords)
    core = (rows.stratum == "core").to_numpy() & (rows.set == "manakov_test").to_numpy()
    half = np.random.default_rng(0).random(len(rows)) < 0.5
    dens = fit_density([J[i] for i in np.where(core & half)[0]], A[core & half, 0])
    S = score(J, A[:, :1 + N_EXPR], dens)
    nrng = np.random.default_rng(3)
    rows["n_reads"] = [0 if j is None else int(j[2].sum()) for j in J]
    rows["S_own"] = S[:, 0]
    rows["p_geom"] = [rand_p(s[0], s[1:], nrng, lower_better=False) for s in S]
    # anchor-free null: the same rows with junctions of a random other covered row. Windows are centred on the
    # reads, so this keeps the centrality part of R_geom; K permutations (a single one is noisy in small cells),
    # p_geom_swap = the first (continuity), p_geom_swap0..K-1 for the per-cell mean in seedless_cells.py
    cov = np.array([j is not None for j in J])
    for k in range(N_SWAP):
        perm = np.where(cov)[0].copy()
        nrng.shuffle(perm)
        Jp = list(J)
        for i, kk in zip(np.where(cov)[0], perm):
            Jp[i] = J[kk]
        Sp = score(Jp, A[:, :1 + N_EXPR], dens)
        rows[f"p_geom_swap{k}"] = [rand_p(s[0], s[1:], nrng, lower_better=False) for s in Sp]
        print(f"swap permutation {k + 1}/{N_SWAP}", flush=True)
    rows["p_geom_swap"] = rows["p_geom_swap0"]
    rows["fit_half"] = core & half
    os.makedirs(OUT, exist_ok=True)
    rows.drop(columns=["target"], errors="ignore").to_csv(f"{OUT}/geometry.tsv.gz", sep="\t", index=False)
    for o in (False, True):
        d = np.arange(-DMAX, DMAX + 1)
        print(f"orientation target_first={o}: mode d = {d[dens[o].argmax()]}, "
              f"50% mass within {np.sort(np.exp(dens[o]))[::-1].cumsum().searchsorted(0.5)} bins")
    t = rows[~rows.fit_half & (rows.n_reads > 0)]
    for (s, st), g in t.groupby(["set", "stratum"]):
        p = g.p_geom.dropna()
        q = g.p_geom_swap.dropna()
        print(f"{s} {st}: n {len(p):,}  pi0(lambda=.5) {min(1, (p > .5).mean() / .5):.3f}  "
              f"swap pi0 {min(1, (q > .5).mean() / .5):.3f}  coverage {(rows[(rows.set == s) & (rows.stratum == st)].n_reads > 0).mean():.3f}")


if __name__ == "__main__":
    main()
