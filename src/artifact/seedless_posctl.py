#!/usr/bin/env python3
"""Positive control for the artifact check: how often do the pairing and geometry tests detect REAL seed-less sites?

The cell rules scale the pairing signal by an efficiency s (seedless_cells.py: R_abs of called seed_bulge positives)
and use an absolute geometry bar. Both choices need a set of seed-less sites known to be real. Expt8 provides one:
miR-1 / miR-124 sites with no canonical seed of their miRNA within 300 nt whose occupancy rises when that miRNA is
overexpressed.

  sites       seed-less (noseed300), input-gated Expt8 miR-1 / miR-124 sites with their Stage 1.1 pairing p-values
              (the 'occupancy' rows of data/seedless/pairing) and their response
                response = sign * (delta - median delta of endogenous sites defined in the same condition)
              sign = +1 miR-1, -1 miR-124 (as seedless_readouts.py occupancy); endogenous sites, which do not carry
              the overexpressed miRNA, under the same transform give the null response distribution
  geometry    ligation geometry of those sites from the 4 Expt8 chimeric libraries: density f(d) exactly as
              seedless_geometry.py (Manakov chr1 core positives, same half split), own duplex vs the 10 expressed
              decoys, 20 swap permutations
  efficiency  per response band: pairing R and geometry R_geom - R_geom_swap. Responders = response >= q90 of the
              null; their real share = 1 - (sites expected there if all were null) / (sites there). Efficiency =
              (responders - low responders) / real share, with bootstrap CIs. The low responders (< null median)
              still hold some real sites, so both efficiencies are biased low (real shares derived from them, high)
  cell        applies the efficiencies to the ARTIFACT cell (none, not comp3p, 0.1 <= p150 < edge) of the test sets:
              real share = R_emp / s_pair and (R_geom - R_geom_swap) / s_geom, locus-cluster bootstrap CIs.
              The geometry number assumes R_geom - R_geom_swap = 0 for a fully null cell, which is NOT verified:
              low-responding Expt8 sites already sit near 0.09, so read it as an upper bound

Caveats: two overexpressed miRNAs (most sites miR-124); the endogenous null does not match the X sites exactly in
the far tail (fewer miR-124 sites above null q97.5 than it predicts), so the real shares are approximate.

    python3 src/artifact/seedless_posctl.py sites        -> data/seedless/posctl/sites.tsv.gz
    python3 src/artifact/seedless_posctl.py geometry     -> data/seedless/posctl/sites_geom.tsv.gz (reads cached)
    python3 src/artifact/seedless_posctl.py efficiency   -> results/seedless_posctl_bands.tsv, _efficiency.tsv
    python3 src/artifact/seedless_posctl.py cell [edge]  -> results/seedless_posctl_cell.tsv (edge default 0.25)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seedless_geometry as G  # noqa: E402
from artifact_common import HYB, ROOT, load_sample, rand_p  # noqa: E402

OUT = f"{ROOT}/data/seedless/posctl"
RES = f"{ROOT}/results"
EXPT8 = f"{ROOT}/data/seedless/readouts/expt8_sites150.tsv"
PAIRING = f"{ROOT}/data/seedless/pairing/pairing.tsv.gz"
SIGN = {"miR1": 1.0, "miR124": -1.0}
N_SWAP = 20
N_BOOT = 500
SEED = 0


def pi0(p, lam=0.5):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return np.nan if len(p) == 0 else (p > lam).mean() / (1 - lam)


def _endo():
    e = pd.read_csv(EXPT8, sep="\t", low_memory=False)
    endo = e[(e.which == "endogenous") & (e.input_gate.astype(str) == "True")]
    med = endo.groupby("cond").delta_miR1_minus_miR124.median()
    return e, endo, med


def null_response(endo, med):
    return np.concatenate([SIGN[c] * (endo[endo.cond == c].delta_miR1_minus_miR124.values - med[c])
                           for c in sorted(endo.cond.unique())])


# ── sites ───────────────────────────────────────────────────────────────────
def sites():
    e, endo, med = _endo()
    print(f"gated endogenous sites {len(endo):,}; median delta by defining condition {med.round(3).to_dict()}")
    # the X rows of expt8_sites150 are row-aligned with eval_win_v2 occupancy.tsv, which the pairing 'row' indexes
    x = e[e.which != "endogenous"].reset_index(drop=True)
    x["row"] = np.arange(len(x))
    x["resp"] = [SIGN[w] * (d - med[c]) for w, d, c in zip(x.which, x.delta_miR1_minus_miR124, x.cond)]
    p = pd.read_csv(PAIRING, sep="\t", low_memory=False)
    p = p[p.set == "occupancy"][["uid", "row", "p_expr", "p_shuf", "E_real", "E_expr_mean", "p150"]]
    d = x.drop(columns=["p_expr"]).merge(p, on="row", how="inner")        # expt8 'p_expr' = occupancy p, not pairing
    assert (d.stratum == "noseed300").all(), "pairing occupancy rows should all be seed-less within 300 nt"
    d = d[d.input_gate.astype(str) == "True"].copy()
    d["grp"] = np.where(d["sub"].fillna("") == "none", "none", "partial")
    d.loc[d.comp3p.astype(str) == "True", "grp"] = "comp3p"
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(f"{OUT}/sites.tsv.gz", sep="\t", index=False)
    print(f"seed-less gated miR-1 / miR-124 sites {len(d):,}")
    print(d.groupby(["which", "grp"]).size().unstack().to_string())


# ── geometry ────────────────────────────────────────────────────────────────
def manakov_density():
    """f(d) exactly as seedless_geometry.main; checked against its stored S_own"""
    rows, R = G.load_pairs()
    fmap = G.name_to_family()
    keep = rows.set.isin(["manakov_test", "manakov_leftout"]).to_numpy() & (rows.label.to_numpy() == 1)
    mr, MR = rows[keep].reset_index(drop=True), R[keep]
    mr["end"] = mr.start + 49
    rd = G.reads()
    rd["fam"] = rd.smallrna.map(fmap)
    rd = rd[rd.fam.notna()]
    J = G.collect(mr, rd)
    core = (mr.stratum == "core").to_numpy() & (mr.set == "manakov_test").to_numpy()
    half = np.random.default_rng(0).random(len(mr)) < 0.5
    dens = G.fit_density([J[i] for i in np.where(core & half)[0]], MR[core & half, 4, 0])
    f = f"{G.OUT}/geometry.tsv.gz"
    if os.path.exists(f):
        s = G.score(J, MR[:, 4, :1 + G.N_EXPR], dens)[:, 0]
        ref = pd.read_csv(f, sep="\t", usecols=["S_own"]).S_own.to_numpy()
        gap = np.nanmax(np.abs(s - ref))
        print(f"density check vs seedless_geometry S_own: max |diff| {gap:.1e}")
        assert gap < 1e-6, "density differs from seedless_geometry.py"
    return dens, rows, R, fmap


def expt8_reads(fmap):
    f = f"{OUT}/reads_expt8.tsv.gz"
    if not os.path.exists(f):
        libs = sorted(x[:-len(".hyb_pairs_deduplicated.tsv")] for x in os.listdir(HYB)
                      if x.startswith("Expt8") and x.endswith(".hyb_pairs_deduplicated.tsv"))
        print(f"Expt8 libraries: {libs}", flush=True)
        assert len(libs) == 4, libs
        with Pool(4) as pool:
            d = pd.concat(pool.map(load_sample, libs), ignore_index=True)
        d = d.rename(columns={"chr.g": "chr"})
        d = d.loc[:, ~d.columns.duplicated()][["chr", "strand", "smallrna", "anchor", "fwd", "Nunique", "sample"]]
        d.to_csv(f, sep="\t", index=False)
    d = pd.read_csv(f, sep="\t", dtype={"chr": str})
    d["fam"] = d.smallrna.map(fmap)
    return d[d.fam.notna()]


def geometry():
    dens, rows, R, fmap = manakov_density()
    s = pd.read_csv(f"{OUT}/sites.tsv.gz", sep="\t", low_memory=False)
    occ = rows.set.eq("occupancy").to_numpy()
    idx = pd.Series(np.arange(occ.sum()), index=rows[occ].uid.values)
    A = R[occ][idx.loc[s.uid].values, 4, :1 + G.N_EXPR]                     # end1 of own + decoy duplexes
    s["chr"] = s.chr.astype(str)
    s["end"] = s.start + 49
    J = G.collect(s, expt8_reads(fmap))
    S = G.score(J, A, dens)
    rng = np.random.default_rng(3)
    s["n_reads"] = [0 if j is None else int(j[2].sum()) for j in J]
    s["p_geom"] = [rand_p(v[0], v[1:], rng, lower_better=False) for v in S]
    cov = np.where([j is not None for j in J])[0]
    for k in range(N_SWAP):
        perm = cov.copy()
        rng.shuffle(perm)
        Jp = list(J)
        for i, kk in zip(cov, perm):
            Jp[i] = J[kk]
        s[f"p_geom_swap{k}"] = [rand_p(v[0], v[1:], rng, lower_better=False) for v in G.score(Jp, A, dens)]
    s.to_csv(f"{OUT}/sites_geom.tsv.gz", sep="\t", index=False)
    print(f"sites {len(s):,}, with Expt8 reads {len(cov):,}")


# ── efficiency ──────────────────────────────────────────────────────────────
def _swaps(d):
    return [c for c in d.columns if c.startswith("p_geom_swap")]


def _stats(x, sw):
    if not sw:                                       # geometry not run (no Expt8 reads): pairing only
        return dict(R_pair=1 - pi0(x.p_expr), R_geom=np.nan, R_geom_swap=np.nan, d_geom=np.nan)
    rg = 1 - pi0(x.p_geom)
    rs = float(np.mean([1 - pi0(x[k]) for k in sw]))
    return dict(R_pair=1 - pi0(x.p_expr), R_geom=rg, R_geom_swap=rs, d_geom=rg - rs)


def _eff(d, null, sw, lo_q=0.9, hi_q=1.0):
    lo = np.quantile(null, lo_q)
    hi = np.quantile(null, hi_q) if hi_q < 1 else np.inf
    m = (d.resp >= lo) & (d.resp < hi)
    low = d[d.resp < np.quantile(null, 0.5)]
    real = 1 - len(d) * ((null >= lo) & (null < hi)).mean() / m.sum()
    a, b = _stats(d[m], sw), _stats(low, sw)
    pair = 1 - pi0(d.p_expr[m]) / pi0(low.p_expr)
    geom = a["d_geom"] - b["d_geom"]
    return dict(n=int(m.sum()), real_share=real, R_pair_rel=pair, s_pair=pair / real,
                d_geom_rel=geom, s_geom=geom / real)


def efficiency():
    _, endo, med = _endo()
    null = null_response(endo, med)
    f = f"{OUT}/sites_geom.tsv.gz"
    if not os.path.exists(f):
        print("no sites_geom.tsv.gz (geometry step needs the Expt8 hyb_pairs): pairing efficiency only")
        f = f"{OUT}/sites.tsv.gz"
    d = pd.read_csv(f, sep="\t", low_memory=False)
    sw = _swaps(d)
    q = [np.quantile(null, v) for v in (0.5, 0.9, 0.975)]
    bands = [("<null_q50", -np.inf, q[0]), ("null_q50-q90", q[0], q[1]), ("null_q90-q975", q[1], q[2]),
             (">null_q975", q[2], np.inf)]
    rows, eff = [], []
    rng = np.random.default_rng(SEED)
    for g in ["none", "partial", "all"]:
        dg = d if g == "all" else d[d.grp == g]
        for lab, lo, hi in bands:
            x = dg[(dg.resp >= lo) & (dg.resp < hi)]
            rows.append(dict(grp=g, band=lab, n=len(x), n_if_null=len(dg) * ((null >= lo) & (null < hi)).mean(),
                             med_resp=x.resp.median(), **_stats(x, sw)))
        for lo_q, hi_q in [(0.9, 1.0), (0.9, 0.975)]:
            pt = _eff(dg, null, sw, lo_q, hi_q)
            bs = pd.DataFrame([_eff(dg.iloc[rng.integers(0, len(dg), len(dg))], null[rng.integers(0, len(null), len(null))],
                                    sw, lo_q, hi_q) for _ in range(N_BOOT)])
            r = dict(grp=g, responders=f"null_q{lo_q:g}-q{hi_q:g}", **pt)
            for k in ["real_share", "R_pair_rel", "s_pair", "d_geom_rel", "s_geom"]:
                r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(bs[k], [2.5, 97.5])
            eff.append(r)
    b, e = pd.DataFrame(rows), pd.DataFrame(eff)
    os.makedirs(RES, exist_ok=True)
    b.to_csv(f"{RES}/seedless_posctl_bands.tsv", sep="\t", index=False, float_format="%.4f")
    e.to_csv(f"{RES}/seedless_posctl_efficiency.tsv", sep="\t", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250):
        print(b.round(3).to_string(index=False), "\n")
        print(e[["grp", "responders", "n", "real_share", "s_pair", "s_pair_lo", "s_pair_hi", "s_geom", "s_geom_lo",
                 "s_geom_hi"]].round(3).to_string(index=False))


# ── cell ────────────────────────────────────────────────────────────────────
def _in_cell(x, edge):
    return ((x.stratum == "noseed300") & (x["sub"] == "none") & ~x.comp3p.fillna(False).astype(bool)
            & (x.p150 >= 0.1) & (x.p150 < edge))


def _clusters(x):
    loc = (x.chr.astype(str) + ":" + x.start.astype(str) + ":" + x.strand.astype(str)).values
    return pd.Series(np.arange(len(x))).groupby(loc).apply(np.array).to_list()


def _rs(cl, rng):
    return np.concatenate([cl[i] for i in rng.integers(0, len(cl), len(cl))])


def cell(edge=0.25):
    edge = float(edge)
    e = pd.read_csv(f"{RES}/seedless_posctl_efficiency.tsv", sep="\t")
    e = e[(e.grp == "none") & (e.responders == "null_q0.9-q1")].iloc[0]
    print(f"efficiencies ('none', responders >= null q90): s_pair {e.s_pair:.3f} [{e.s_pair_lo:.3f}, {e.s_pair_hi:.3f}], "
          f"s_geom {e.s_geom:.3f} [{e.s_geom_lo:.3f}, {e.s_geom_hi:.3f}]")
    p = pd.read_csv(PAIRING, sep="\t", low_memory=False)
    p = p[p.set.isin(["manakov_test", "manakov_leftout", "gse304955"]) & (p.stratum == "noseed300")]
    p = p[_in_cell(p, edge)]
    g = pd.read_csv(f"{G.OUT}/geometry.tsv.gz", sep="\t", low_memory=False)
    g = g[~g.fit_half.astype(bool) & (g.n_reads > 0) & _in_cell(g, edge)]
    sw = [c for c in g.columns if c.startswith("p_geom_swap") and c != "p_geom_swap"]
    rng = np.random.default_rng(SEED)
    out = []
    for ds in ["manakov_test", "manakov_leftout", "gse304955"]:
        x = p[p.set == ds]
        pos, neg = x[x.label == 1], x[x.label == 0]
        cp, cn = _clusters(pos), _clusters(neg)
        r_emp = 1 - pi0(pos.p_expr) / pi0(neg.p_expr)
        bs = [1 - pi0(pos.p_expr.values[_rs(cp, rng)]) / pi0(neg.p_expr.values[_rs(cn, rng)]) for _ in range(N_BOOT)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        r = dict(set=ds, edge=edge, n_pos=len(pos), n_neg=len(neg), R_emp=r_emp, R_emp_lo=lo, R_emp_hi=hi,
                 real_pair=r_emp / e.s_pair, real_pair_lo=max(lo, 0) / e.s_pair_hi, real_pair_hi=hi / e.s_pair_lo)
        y = g[g.set == ds]
        if len(y) >= 30 and np.isfinite(e.s_geom):
            cy = _clusters(y)
            dg = lambda ii: (1 - pi0(y.p_geom.values[ii])) - np.mean([1 - pi0(y[k].values[ii]) for k in sw])
            d0 = dg(np.arange(len(y)))
            lo2, hi2 = np.percentile([dg(_rs(cy, rng)) for _ in range(N_BOOT)], [2.5, 97.5])
            r.update(n_geom=len(y), d_geom=d0, d_geom_lo=lo2, d_geom_hi=hi2, real_geom_if_null0=d0 / e.s_geom,
                     real_geom_lo=max(lo2, 0) / e.s_geom_hi, real_geom_hi=hi2 / e.s_geom_lo)
        out.append(r)
    t = pd.DataFrame(out)
    t.to_csv(f"{RES}/seedless_posctl_cell.tsv", sep="\t", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250):
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fns = {"sites": sites, "geometry": geometry, "efficiency": efficiency, "cell": cell}
    if cmd not in fns:
        sys.exit(__doc__)
    fns[cmd](*sys.argv[2:])
