#!/usr/bin/env python3
"""Seed strata for every row of the 150-nt tables (seed-less plan, Definitions).

  stratum  core / flank150 / flank300 / noseed300   (artifact_common.has_seed on the 50/150/300-nt windows)
  sub      partial-pairing subclass of the 50-nt core for noseed300 rows (artifact_common.subclass)
  comp3p   noseed300 row whose core holds a partial seed (G:U, 1 mismatch or 1-nt bulge, nt 2-8) AND
           >= 4 contiguous WC pairs of miRNA nt 13-16 or 14-17 placed 0-4 extra target nt 5' of the
           contiguous-duplex position (3'-compensatory geometry)
  p150     base150 probability where a prediction file exists (row-aligned: data/mre150/pred_mre150_<set>.tsv
           from --no-dedup, else data/seedless/base150/pred_<set>.npy from seedless_predict.py)

    python3 src/artifact/seedless_strata.py [set ...]   -> data/seedless/strata/<set>.tsv.gz
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_common import ROOT, has_seed, subclass, norm, rc  # noqa: E402

OUT = f"{ROOT}/data/seedless/strata"
SETS = ["manakov_train_nochr2", "manakov_val_chr2", "manakov_test", "manakov_leftout", "gse304955",
        "gse263552", "gse180129_HSPC", "gse180129_Kasumi1", "hejret_test", "klimentova_test"]
P = "interaction_probability"


def partial_seed_starts(t, m):
    """Target index pairing miRNA nt 8 for every partial seed (G:U / 1-mm / 1-nt bulge) in t."""
    out = set()
    for a, b, shift in [(1, 7, 1), (2, 8, 0)]:          # nt2-7 (nt8 sits one 5' of the match), nt3-8
        s = m[a:b]
        for i, x in enumerate(s):
            if x in "GT":
                v = rc(s[:i] + ("A" if x == "G" else "C") + s[i + 1:])
                j = t.find(v)
                while j >= 0:
                    out.add(j - shift)
                    j = t.find(v, j + 1)
    s = rc(m[1:8])
    for i in range(7):
        for x in "ACGT":
            if x != s[i]:
                v = s[:i] + x + s[i + 1:]
                j = t.find(v)
                while j >= 0:
                    out.add(j)
                    j = t.find(v, j + 1)
    for k in range(3, 7):
        a, b = rc(m[k:8]), rc(m[1:k])
        for x in "ACGT":
            v = a + x + b
            j = t.find(v)
            while j >= 0:
                out.add(j)
                j = t.find(v, j + 1)
    return out


def comp3p(t, m):
    if len(m) < 17:
        return False
    s1316, s1417 = rc(m[12:16]), rc(m[13:17])
    for s in partial_seed_starts(t, m):
        for o in range(0, 5):
            a = s - 8 - o
            if a >= 0 and t[a:a + 4] == s1316:
                return True
            a = s - 9 - o
            if a >= 0 and t[a:a + 4] == s1417:
                return True
    return False


def classify(args):
    g50, g150, g300, m = args
    if has_seed(g50, m):
        return "core", "", False
    if has_seed(g150, m):
        return "flank150", "", False
    if has_seed(g300, m):
        return "flank300", "", False
    return "noseed300", subclass(g50, m), comp3p(g50, m)


def build(name, workers=24):
    a = pd.read_csv(f"{ROOT}/data/mre150/{name}.tsv", sep="\t", low_memory=False)
    b = pd.read_csv(f"{ROOT}/data/mre300/{name}.tsv", sep="\t", usecols=["gene", "gene300"], low_memory=False)
    assert len(a) == len(b) and (a.gene.values == b.gene.values).all(), name
    m, g50, g150, g300 = norm(a.noncodingRNA), norm(a.gene), norm(a.gene150), norm(b.gene300)
    with Pool(workers) as pool:
        r = pool.map(classify, zip(g50, g150, g300, m), chunksize=4096)
    keep = [c for c in ["gene", "noncodingRNA", "noncodingRNA_fam", "feature", "label", "chr", "start", "end",
                        "strand"] if c in a.columns]
    o = a[keep].copy()
    o.insert(0, "row", np.arange(len(a)))
    o["stratum"] = [x[0] for x in r]
    o["sub"] = [x[1] for x in r]
    o["comp3p"] = [x[2] for x in r]
    o["mirna_gc"] = m.str.count("[GC]").values
    f = f"{ROOT}/data/mre150/pred_mre150_{name}.tsv"
    fn = f"{ROOT}/data/seedless/base150/pred_{name}.npy"     # seedless_predict.py output (row-aligned)
    if os.path.exists(f):
        p = pd.read_csv(f, sep="\t", low_memory=False)
        assert len(p) == len(a), f
        if "gene150" in p.columns:
            assert (p.gene150.values == a.gene150.values).all(), f
        o["p150"] = p[P].values
    elif os.path.exists(fn):
        p = np.load(fn)
        assert len(p) == len(a), fn
        o["p150"] = p.astype(float)
    os.makedirs(OUT, exist_ok=True)
    o.to_csv(f"{OUT}/{name}.tsv.gz", sep="\t", index=False)
    ns = o[o.stratum == "noseed300"]
    print(f"{name}: {len(o):,} rows; strata(pos) {o[o.label == 1].stratum.value_counts(normalize=True).round(3).to_dict()}; "
          f"noseed300 pos {int(ns.label.sum()):,} (none {int(((ns['sub'] == 'none') & (ns.label == 1)).sum()):,}); "
          f"comp3p pos/neg rate {ns[ns.label == 1].comp3p.mean():.4f}/{ns[ns.label == 0].comp3p.mean():.4f}",
          flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or SETS):
        build(s)
