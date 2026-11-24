#!/usr/bin/env python3
"""Compare a rerun of the artifact check / removal with the reference outputs in src/artifact/expected/.

  strata      data/seedless/strata/*.tsv.gz: rows per set x label x stratum x subclass (+ comp3p). Deterministic
              given the window tables, so these should match exactly.
  lowcomp     results/lowcomplexity_rnas.tsv: the flagged small RNAs (low complexity, self-enriched or ambiguous)
              and their signature. Deterministic; should match exactly.
  cells       results/seedless_cells.tsv: cell labels (G1, G1_es, G1_final, G1_es_final) per set x group x bin, and
              the row counts. The bins use the base150 scores (deterministic) for test sets, but the pairing null uses
              IntaRNA and seeded decoys, so labels should agree; R values may move in the last digits.
  arms        data/seedless/arms/diag_<arm>.tsv: share of positives / negatives set to weight 0. The training rows
              are binned on out-of-fold scores, and GPU training is not bit-reproducible, so expect small
              differences (a few 0.001).
Only the parts whose rerun output exists are compared.

    python3 src/artifact/check_expected.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_common import ROOT  # noqa: E402

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected")
LAB = ["G1", "G1_es", "G1_final", "G1_es_final"]


def strata():
    e = pd.read_csv(f"{EXP}/strata_counts.tsv", sep="\t", keep_default_na=False)
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/data/seedless/strata/*.tsv.gz")):
        x = pd.read_csv(f, sep="\t", usecols=["label", "stratum", "sub", "comp3p"], low_memory=False)
        x["sub"] = x["sub"].where(x.stratum == "noseed300", "")
        g = x.groupby(["label", "stratum", "sub"]).agg(n=("comp3p", "size"), n_comp3p=("comp3p", "sum")).reset_index()
        rows.append(g.assign(set=os.path.basename(f)[:-7]))
    if not rows:
        return None
    o = pd.concat(rows)
    k = ["set", "label", "stratum", "sub"]
    m = e.merge(o, on=k, how="outer", suffixes=("_exp", "_new"))
    m = m[m.set.isin(o.set.unique())]
    bad = m[(m.n_exp != m.n_new) | (m.n_comp3p_exp != m.n_comp3p_new)]
    print(f"strata: {len(m)} cells over {o.set.nunique()} sets, {len(bad)} differ")
    if len(bad):
        print(bad.head(20).to_string(index=False))
    return len(bad) == 0


def lowcomp():
    f = f"{ROOT}/results/lowcomplexity_rnas.tsv"
    if not os.path.exists(f):
        return None
    e = pd.read_csv(f"{EXP}/lowcomplexity_flagged.tsv", sep="\t")
    u = pd.read_csv(f, sep="\t")
    u = u[u.low_complexity.astype(str).eq("True") | u.signature.isin(["self", "ambiguous"])]
    m = e.merge(u[["seq", "signature", "low_complexity"]], on="seq", how="outer", suffixes=("_exp", "_new"))
    bad = m[(m.signature_exp != m.signature_new) | (m.low_complexity_exp.astype(str) != m.low_complexity_new.astype(str))]
    print(f"lowcomp: {len(e)} flagged RNAs expected, {len(u)} found, {len(bad)} differ")
    if len(bad):
        print(bad[["seq", "signature_exp", "signature_new", "low_complexity_exp", "low_complexity_new"]].to_string(index=False))
    return len(bad) == 0


def cells():
    f = f"{ROOT}/results/seedless_cells.tsv"
    if not os.path.exists(f):
        return None
    k = ["set", "group", "bin"]
    e = pd.read_csv(f"{EXP}/seedless_cells.tsv", sep="\t")
    o = pd.read_csv(f, sep="\t")
    m = e.merge(o, on=k, how="outer", suffixes=("_exp", "_new"), indicator=True)
    miss = (m._merge != "both").sum()
    m = m[m._merge == "both"]
    ok = True
    for c in LAB:
        d = m[m[f"{c}_exp"].astype(str) != m[f"{c}_new"].astype(str)]
        print(f"cells {c}: {len(m) - len(d)}/{len(m)} labels agree")
        if len(d):
            ok = False
            print(d[k + [f"{c}_exp", f"{c}_new"]].to_string(index=False))
    for c in ["n_pos", "n_neg"]:
        print(f"cells {c}: {int((m[f'{c}_exp'] != m[f'{c}_new']).sum())} cells differ, "
              f"max |diff| {np.nanmax(np.abs(m[f'{c}_exp'] - m[f'{c}_new'])):.0f}")
    for c in ["R_emp", "R_geom"]:
        print(f"cells {c}: max |diff| {np.nanmax(np.abs(m[f'{c}_exp'] - m[f'{c}_new'])):.4f}")
    if miss:
        print(f"cells: {miss} cells only in one table")
    return ok and not miss


def arms():
    e = pd.read_csv(f"{EXP}/arm_diag.tsv", sep="\t").set_index("arm")
    fs = sorted(glob.glob(f"{ROOT}/data/seedless/arms/diag_*.tsv"), key=os.path.getmtime)   # newest wins
    new = pd.concat([pd.read_csv(f, sep="\t") for f in fs]).drop_duplicates("arm", keep="last").set_index("arm") \
        if fs else pd.DataFrame()
    res = []
    for a in e.index:
        if a in new.index:
            o = new.loc[a]
            res.append(dict(arm=a, **{f"{c}_exp": e.at[a, c] for c in ["pos_zero_share", "neg_zero_share"]},
                            **{f"{c}_new": o[c] for c in ["pos_zero_share", "neg_zero_share"]}))
    if not res:
        return None
    r = pd.DataFrame(res)
    r["max_abs_diff"] = np.maximum(abs(r.pos_zero_share_exp - r.pos_zero_share_new),
                                   abs(r.neg_zero_share_exp - r.neg_zero_share_new))
    print("arms (share of rows set to weight 0):")
    print(r.round(5).to_string(index=False))
    return bool((r.max_abs_diff < 0.01).all())


if __name__ == "__main__":
    out = {}
    for name, fn in [("strata", strata), ("lowcomp", lowcomp), ("cells", cells), ("arms", arms)]:
        out[name] = fn()
        print()
    print("summary: " + ", ".join(f"{k} {'not run' if v is None else 'OK' if v else 'DIFFERS'}" for k, v in out.items()))
