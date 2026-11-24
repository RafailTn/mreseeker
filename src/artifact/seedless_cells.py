#!/usr/bin/env python3
"""Seed-less plan, Gate G1: cell table (dataset x group x 150-nt score bin) and labels.

Realness from the Stage 1.1 decoy p-values (Storey pi0, lambda = 0.5, unclipped; expressed-decoy p primary,
shuffle p as sensitivity):
  R_emp = 1 - pi0(pos) / pi0(neg)   same-cell negatives as the null (primary in missed bins)
  R_abs = 1 - pi0(pos)              decoys only
  s     = R_abs of called (p150 >= 0.5) seed_bulge positives, Manakov chr1 (efficiency); R/s reported
Locus-cluster bootstrap CIs. Geometry (Stage 1.2): R_geom = 1 - pi0(p_geom), positives with reads;
R_geom_swap = the same with junctions of random other rows (mean over the swap permutations), the centrality
part, so R_geom - R_geom_swap is the row-specific lower bound (not used by the labels).
In-cell readouts (Stage 1.3, per group x bin, all datasets): Expt8 pooled condition-matched response (median,
CI) and mean response as a fraction of core >= 0.8; mixing same-species share per direction and A_h / A_r.

G1 (pre-registered, plan file): ARTIFACT = R_emp <= 0.10 & CI_hi <= 0.20 & R/s <= 0.25 & R_geom <= 0.10 (where
reads) & Expt8 not above null (CI_lo <= 0) & mixing same-species share not above the 'none' level in either
direction. REAL-MISSED = R_emp >= 0.30 or R_geom >= 0.30 or Expt8 CI_lo > 0. MIXED otherwise (weight R_emp).
Leftout and SAEC must reproduce the chr1 label, else MIXED.
Sensitivity label G1_es: the Expt8 clause uses effect size instead (mean response >= 0.10 of core for REAL,
< 0.10 for ARTIFACT).

    python3 src/artifact/seedless_cells.py        -> results/seedless_cells.tsv
    python3 src/artifact/seedless_cells.py fine   -> results/seedless_cells_fine.tsv (finer score bins, none/partial)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_common import ROOT  # noqa: E402

RES = f"{ROOT}/results"
os.makedirs(RES, exist_ok=True)   # results/ is gitignored, so a fresh clone lacks it
BINS = [0, 0.1, 0.3, 0.5, 0.8, 1.0001]
BLAB = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.8", ">=0.8"]
PARTIAL = ["seed_GU", "seed7_1mm", "seed_bulge", "3p_6mer", "contig8"]
GROUPS = ["none", *PARTIAL, "comp3p", "partial", "noseed300", "core", "flank150"]


def pi0(p, lam=0.5):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return np.nan if len(p) == 0 else (p > lam).mean() / (1 - lam)


def member(d, g):
    if g == "comp3p":
        return d.comp3p.fillna(False).astype(bool).to_numpy() & (d.stratum == "noseed300").to_numpy()
    if g == "partial":
        return d["sub"].isin(PARTIAL).to_numpy() & (d.stratum == "noseed300").to_numpy()
    if g == "noseed300":
        return (d.stratum == "noseed300").to_numpy()
    if g in ("core", "flank150"):
        return (d.stratum == g).to_numpy()
    return (d["sub"] == g).to_numpy() & (d.stratum == "noseed300").to_numpy()


def boot_R(pp, pn, lp, ln, col, B=300, seed=0):
    """locus-cluster bootstrap of R_emp and R_abs"""
    rng = np.random.default_rng(seed)
    gp = pd.Series(np.arange(len(pp))).groupby(lp).apply(np.array).to_list()
    gn = pd.Series(np.arange(len(pn))).groupby(ln).apply(np.array).to_list() if len(pn) else []
    re, ra = [], []
    for _ in range(B):
        ip = np.concatenate([gp[i] for i in rng.integers(0, len(gp), len(gp))])
        a = pi0(pp[ip])
        ra.append(1 - a)
        if gn:
            jn = np.concatenate([gn[i] for i in rng.integers(0, len(gn), len(gn))])
            re.append(1 - a / pi0(pn[jn]))
    q = lambda v: np.nanpercentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
    return q(re), q(ra)


def pairing_cells(col="p_expr"):
    d = pd.read_csv(f"{ROOT}/data/seedless/pairing/pairing.tsv.gz", sep="\t", low_memory=False)
    d = d[d.set.isin(["manakov_test", "manakov_leftout", "gse304955"])].copy()
    d["bin"] = pd.cut(d.p150, BINS, labels=BLAB, right=False).astype(str)
    d["locus"] = d.chr.astype(str) + ":" + d.start.astype(str) + ":" + d.strand.astype(str)
    rows = []
    for ds, x in d.groupby("set"):
        for g in GROUPS:
            mg = member(x, g)
            for b in BLAB + ["missed", "called", "all"]:
                if b == "missed":
                    mb = (x.p150 < 0.5).to_numpy()
                elif b == "called":
                    mb = (x.p150 >= 0.5).to_numpy()
                elif b == "all":
                    mb = np.ones(len(x), bool)
                else:
                    mb = (x.bin == b).to_numpy()
                c = x[mg & mb]
                pos, neg = c[c.label == 1], c[c.label == 0]
                if len(pos) < 30:
                    continue
                r = {"set": ds, "group": g, "bin": b, "n_pos": len(pos), "n_neg": len(neg),
                     "pi0_pos": pi0(pos[col]), "pi0_neg": pi0(neg[col]) if len(neg) >= 30 else np.nan,
                     "pi0_pos_shuf": pi0(pos.p_shuf), "pi0_neg_shuf": pi0(neg.p_shuf) if len(neg) >= 30 else np.nan}
                r["R_abs"] = 1 - r["pi0_pos"]
                r["R_emp"] = 1 - r["pi0_pos"] / r["pi0_neg"] if np.isfinite(r["pi0_neg"]) else np.nan
                r["R_emp_shuf"] = 1 - r["pi0_pos_shuf"] / r["pi0_neg_shuf"] if np.isfinite(r["pi0_neg_shuf"]) else np.nan
                (r["R_emp_lo"], r["R_emp_hi"]), (r["R_abs_lo"], r["R_abs_hi"]) = boot_R(
                    pos[col].to_numpy(), neg[col].to_numpy() if len(neg) >= 30 else np.array([]),
                    pos.locus.to_numpy(), neg.locus.to_numpy(), col)
                rows.append(r)
    t = pd.DataFrame(rows)
    s = t[(t.set == "manakov_test") & (t.group == "seed_bulge") & (t.bin == "called")].R_abs
    s = float(s.iloc[0]) if len(s) else np.nan
    t["s_eff"] = s
    t["R_over_s"] = t.R_emp / s
    return t


def geometry_cells():
    try:
        g = pd.read_csv(f"{ROOT}/data/seedless/geometry/geometry.tsv.gz", sep="\t", low_memory=False)
    except FileNotFoundError:
        return None
    g = g[~g.fit_half.astype(bool) & (g.n_reads > 0)].copy()
    g["bin"] = pd.cut(g.p150, BINS, labels=BLAB, right=False).astype(str)
    rows = []
    for ds, x in g.groupby("set"):
        for grp in GROUPS:
            mg = member(x, grp)
            for b in BLAB + ["missed", "called", "all"]:
                mb = {"missed": x.p150 < 0.5, "called": x.p150 >= 0.5, "all": x.p150 >= 0}.get(b, x.bin == b).to_numpy()
                c = x[mg & mb]
                if len(c) < 30:
                    continue
                sw = [k for k in c.columns if k.startswith("p_geom_swap") and k != "p_geom_swap"] or ["p_geom_swap"]
                rows.append({"set": ds, "group": grp, "bin": b, "n_geom": len(c), "R_geom": 1 - pi0(c.p_geom),
                             "R_geom_swap": float(np.mean([1 - pi0(c[k]) for k in sw]))})
    return pd.DataFrame(rows)


def main():
    t = pairing_cells()
    gm = geometry_cells()
    if gm is not None:
        t = t.merge(gm, on=["set", "group", "bin"], how="left")
    occ = pd.read_csv(f"{RES}/seedless_readouts_occupancy_base150.tsv", sep="\t")
    mix = pd.read_csv(f"{RES}/seedless_readouts_mixing_base150.tsv", sep="\t")
    occ = occ.rename(columns={"group": "group"})[["group", "bin", "resp_pooled", "resp_pooled_lo", "resp_pooled_hi",
                                                  "frac_core_pooled", "n_miR1", "n_miR124"]]
    mix = mix[["group", "bin", "same_human_only", "same_rat_only", "A_h", "A_r", "n"]].rename(columns={"n": "n_mix"})
    t = t.merge(occ, on=["group", "bin"], how="left").merge(mix, on=["group", "bin"], how="left")
    none_lvl = mix[(mix.group == "none") & (mix.bin == "all")].iloc[0]

    def label(r, es=False):
        geom_ok = (not np.isfinite(r.get("R_geom", np.nan))) or r["R_geom"] <= 0.10
        if es:
            e8_null = not (r.frac_core_pooled >= 0.10) if np.isfinite(r.frac_core_pooled) else True
            e8_real = bool(r.frac_core_pooled >= 0.10) if np.isfinite(r.frac_core_pooled) else False
        else:
            e8_null = not (r.resp_pooled_lo > 0) if np.isfinite(r.resp_pooled_lo) else True
            e8_real = bool(r.resp_pooled_lo > 0) if np.isfinite(r.resp_pooled_lo) else False
        mix_ok = not ((r.same_human_only > none_lvl.same_human_only) or (r.same_rat_only > none_lvl.same_rat_only))
        if (np.isfinite(r.R_emp) and r.R_emp <= 0.10 and r.R_emp_hi <= 0.20 and r.R_over_s <= 0.25 and geom_ok
                and e8_null and mix_ok):
            return "ARTIFACT"
        if (np.isfinite(r.R_emp) and r.R_emp >= 0.30) or (np.isfinite(r.get("R_geom", np.nan)) and r["R_geom"] >= 0.30) \
                or e8_real:
            return "REAL-MISSED"
        return "MIXED"
    t["G1"] = [label(r) for _, r in t.iterrows()]
    t["G1_es"] = [label(r, es=True) for _, r in t.iterrows()]
    # replication: leftout / SAEC must reproduce the chr1 label of the same group x bin
    ref = t[t.set == "manakov_test"].set_index(["group", "bin"])
    for lab in ["G1", "G1_es"]:
        final = {}
        for (g, b), r in ref.iterrows():
            others = t[(t.group == g) & (t.bin == b) & (t.set != "manakov_test")][lab]
            final[(g, b)] = r[lab] if (len(others) == 0 or (others == r[lab]).all()) else "MIXED"
        t[f"{lab}_final"] = [final.get((g, b), np.nan) for g, b in zip(t.group, t.bin)]
    t.to_csv(f"{RES}/seedless_cells.tsv", sep="\t", index=False, float_format="%.4f")
    show = t[(t.set == "manakov_test") & t.bin.isin(BLAB + ["missed", "called"])]
    cols = ["group", "bin", "n_pos", "n_neg", "R_emp", "R_emp_lo", "R_emp_hi", "R_emp_shuf", "R_abs", "R_over_s",
            "R_geom", "resp_pooled", "frac_core_pooled", "A_h", "A_r", "G1", "G1_es", "G1_final", "G1_es_final"]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(f"efficiency s (R_abs, called seed_bulge, chr1) = {t.s_eff.iloc[0]:.3f}")
        print(show[[c for c in cols if c in show]].round(3).to_string(index=False))


def fine_bins():
    """Finer CNN-score bins for 'none' (and partial) rows: pairing R_emp / R_abs (pos, neg) and geometry R_geom,
    R_geom_swap (mean over swap permutations) and their difference (row-specific lower bound), per set."""
    edges = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0001]
    lab = [f"{a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]
    d = pd.read_csv(f"{ROOT}/data/seedless/pairing/pairing.tsv.gz", sep="\t", low_memory=False)
    d = d[d.set.isin(["manakov_test", "manakov_leftout", "gse304955"]) & (d.stratum == "noseed300")].copy()
    d["locus"] = d.chr.astype(str) + ":" + d.start.astype(str) + ":" + d.strand.astype(str)
    d["fbin"] = pd.cut(d.p150, edges, labels=lab, right=False).astype(str)
    g = pd.read_csv(f"{ROOT}/data/seedless/geometry/geometry.tsv.gz", sep="\t", low_memory=False)
    g = g[~g.fit_half.astype(bool) & (g.n_reads > 0) & (g.stratum == "noseed300")].copy()
    g["fbin"] = pd.cut(g.p150, edges, labels=lab, right=False).astype(str)
    sw = [k for k in g.columns if k.startswith("p_geom_swap") and k != "p_geom_swap"] or ["p_geom_swap"]
    rows = []
    for grp, gm, dm in [("none", g["sub"] == "none", d["sub"] == "none"),
                        ("partial", g["sub"].isin(PARTIAL), d["sub"].isin(PARTIAL))]:
        for (ds, b), c in d[dm].groupby(["set", "fbin"]):
            pos, neg = c[c.label == 1], c[c.label == 0]
            if len(pos) < 30:
                continue
            r = {"set": ds, "group": grp, "fbin": b, "n_pos": len(pos), "n_neg": len(neg),
                 "R_abs_pos": 1 - pi0(pos.p_expr), "R_abs_neg": 1 - pi0(neg.p_expr) if len(neg) >= 30 else np.nan}
            r["R_emp"] = 1 - pi0(pos.p_expr) / pi0(neg.p_expr) if len(neg) >= 30 else np.nan
            if len(neg) >= 30:
                (r["R_emp_lo"], r["R_emp_hi"]), _ = boot_R(pos.p_expr.to_numpy(), neg.p_expr.to_numpy(),
                                                           pos.locus.to_numpy(), neg.locus.to_numpy(), "p_expr", B=200)
            x = g[gm & (g.set == ds) & (g.fbin == b)]
            if len(x) >= 30:
                r["n_geom"] = len(x)
                r["R_geom"] = 1 - pi0(x.p_geom)
                r["R_geom_swap"] = float(np.mean([1 - pi0(x[k]) for k in sw]))
                r["d_geom"] = r["R_geom"] - r["R_geom_swap"]
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(f"{RES}/seedless_cells_fine.tsv", sep="\t", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    fine_bins() if sys.argv[1:] == ["fine"] else main()
