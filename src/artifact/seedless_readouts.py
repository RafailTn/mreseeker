#!/usr/bin/env python3
"""Stage 1.3 of the seed-less plan: in-cell readouts per cell (group x 150-nt score bin).

Groups: core / flank150 / flank300 / noseed300 split by artifact_common.subclass (none, seed_GU, seed7_1mm,
seed_bulge, 3p_6mer, contig8) and comp3p. Bins of the 150-nt score: <0.1, 0.1-0.3, 0.3-0.5, 0.5-0.8, >=0.8.

  build      Expt8 sites from Hybridetector validate_overexpression (miR-1 / miR-124 sites AND the endogenous-
             miRNA background sites, with the condition whose library defined them) + 150/300-nt windows
             (build_mre150.extend) + strata -> data/seedless/readouts/expt8_sites150.tsv
  occupancy  input-gated sites. delta = log2 occupancy miR-1-oe minus miR-124-oe (cross-replicate). The raw
             delta of endogenous sites is ~ +0.35..0.59, not 0 (library offset), so each X site is compared
             with endogenous sites of the SAME cell and the SAME defining condition:
               response = sign_X * (delta - median delta of matched endogenous sites), sign = +1 miR-1, -1 miR-124
             plus the offset-free contrast (median delta miR-1 sites - median delta miR-124 sites) / 2.
  mixing     human/rat mixed lysates: same-species share per direction and post-lysis share A_h / A_r per cell
             (mixing_artifacts.estimate per expt x gel block, block q_r and floors from artifact_rate.tsv,
             pooled by hybrid count).

    python3 src/artifact/seedless_readouts.py build|occupancy|mixing|transport [model]   -> results/seedless_readouts_*.tsv
model = base150 (default; data/seedless/base150/pred_*.npy) or an arm dir such as arms/A0/s42.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedless_strata import classify, ROOT  # noqa: E402
from artifact_common import CNN_SRC, HYB_DIR, mixing_estimate  # noqa: E402

VO = f"{HYB_DIR}/data/validate_overexpression"
MIX = f"{HYB_DIR}/data/mixing_artifacts"
EW = f"{ROOT}/data/relabel/noise/eval_win_v2"
OUT = f"{ROOT}/data/seedless/readouts"
RES = f"{ROOT}/results"
os.makedirs(RES, exist_ok=True)   # results/ is gitignored, so a fresh clone lacks it
BINS = [0, 0.1, 0.3, 0.5, 0.8, 1.0001]
BLAB = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.8", ">=0.8"]


def norm(s):
    return s.astype(str).str.upper().str.replace("U", "T")


def groups(stratum, sub, comp):
    g = np.where(stratum == "noseed300", sub, stratum).astype(object)
    return g, np.asarray(comp, bool) & (stratum == "noseed300")


def score_bin(p):
    return pd.cut(p, BINS, labels=BLAB, right=False).astype(str)


def boot(fn, arrays, n=400, seed=0, paired=None):
    """Percentile CI. paired=(i, j, k, l): arrays i,j share a resample index, as do k,l (row-wise pairs)."""
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        if paired:
            i, j, k, l = paired
            a = rng.integers(0, len(arrays[i]), len(arrays[i]))
            b = rng.integers(0, len(arrays[k]), len(arrays[k]))
            v.append(fn(arrays[i][a], arrays[j][a], arrays[k][b], arrays[l][b]))
        else:
            v.append(fn(*[x[rng.integers(0, len(x), len(x))] for x in arrays]))
    return np.nanpercentile(v, [2.5, 97.5])


def matched_resp(xv, xcnd, ev, ecnd, sign, stat):
    """sign * (X delta - stat of endogenous delta with the same defining condition), stat over all X sites."""
    out = []
    for c in np.unique(xcnd):
        em = ecnd == c
        ref = ev[em] if em.sum() >= 20 else ev
        out.append(sign * (xv[xcnd == c] - stat(ref)))
    return stat(np.concatenate(out)) if out else np.nan


# ── Expt8 build ─────────────────────────────────────────────────────────────
def build():
    sys.path.insert(0, CNN_SRC)
    from build_mre150 import extend  # noqa: E402
    o = pd.read_csv(f"{VO}/occupancy.tsv", sep="\t")
    s = pd.read_csv(f"{VO}/sites.tsv", sep="\t")
    s = s.drop_duplicates(["site_id", "which", "rep"]).rename(columns={"rep": "defined_in"})
    s = s[["site_id", "which", "defined_in", "site_class", "cond", "chr", "start", "end", "strand", "feature",
           "gene", "mirna_seq", "noncodingRNA_fam"]]
    x = o.merge(s, on=["site_id", "which", "defined_in", "site_class"], how="inner").reset_index(drop=True)
    x["gene"] = norm(x.gene)
    x["mirna_seq"] = norm(x.mirna_seq)
    x["gene150"], n_ok, n150 = extend(x, 50, 50)
    x["gene300"], _, n300 = extend(x, 125, 125)
    print(f"Expt8 rows {len(x):,} ({x.which.value_counts().to_dict()}); windows ok 150 {n150:,} / 300 {n300:,} of {n_ok:,}")
    from multiprocessing import Pool
    with Pool(16) as pool:
        r = pool.map(classify, zip(x.gene, x.gene150, x.gene300, x.mirna_seq), chunksize=2048)
    x["stratum"] = [a[0] for a in r]
    x["sub"] = [a[1] for a in r]
    x["comp3p"] = [a[2] for a in r]
    os.makedirs(OUT, exist_ok=True)
    x.to_csv(f"{OUT}/expt8_sites150.tsv", sep="\t", index=False)
    print(x.groupby(["which", "stratum"]).size().unstack().to_string())


def pred(model, name):
    d = f"{ROOT}/data/seedless/{model}"
    return np.load(f"{d}/pred_{name}.npy")


# ── occupancy ───────────────────────────────────────────────────────────────
def occupancy(model="base150"):
    x = pd.read_csv(f"{OUT}/expt8_sites150.tsv", sep="\t", low_memory=False)
    x["p"] = pred(model, "expt8_all")
    x = x[x.input_gate.astype(str) == "True"].copy()
    x["grp"], x["c3"] = groups(x.stratum.values, x["sub"].fillna("").values, x.comp3p.values)
    x["bin"] = score_bin(x.p.values)
    endo = x[x.which == "endogenous"]
    rows = []
    cells = [(g, b) for g in ["core", "flank150", "flank300", "seed_bulge", "seed_GU", "seed7_1mm", "3p_6mer",
                               "contig8", "none", "comp3p", "noseed300", "missed_none", "missed_partial"]
             for b in BLAB + ["all"]]
    for g, b in cells:
        def sel(d):
            if g == "comp3p":
                m = d.c3.values
            elif g == "noseed300":
                m = d.stratum.values == "noseed300"
            elif g == "missed_none":
                m = (d.grp.values == "none") & (d.p.values < 0.5)
            elif g == "missed_partial":
                m = np.isin(d.grp.values, ["seed_bulge", "seed_GU", "seed7_1mm", "3p_6mer", "contig8"]) & (d.p.values < 0.5)
            else:
                m = d.grp.values == g
            if b != "all":
                m = m & (d.bin.values == b)
            return d[m]
        e = sel(endo)
        r = {"group": g, "bin": b, "n_endo": len(e)}
        for X, sign in [("miR1", 1.0), ("miR124", -1.0)]:
            xs = sel(x[x.which == X])
            r[f"n_{X}"] = len(xs)
            if len(xs) < 20 or len(e) < 20:
                continue
            # endogenous null matched on the defining condition (selection control), pooled if too few
            xv, xcnd = xs.delta_miR1_minus_miR124.values, xs.cond.values
            ev, ecnd = e.delta_miR1_minus_miR124.values, e.cond.values
            r[f"resp_{X}"] = matched_resp(xv, xcnd, ev, ecnd, sign, np.median)
            r[f"resp_{X}_lo"], r[f"resp_{X}_hi"] = boot(
                lambda a, b_, c_, d_: matched_resp(a, b_, c_, d_, sign, np.median), [xv, xcnd, ev, ecnd], paired=(0, 1, 2, 3))
            r[f"mean_resp_{X}"] = matched_resp(xv, xcnd, ev, ecnd, sign, np.mean)
            r[f"gain_raw_{X}"] = np.median(sign * xv)
        # pooled over both transfected miRNAs: condition-matched signed responses of all X sites
        x1, x2 = sel(x[x.which == "miR1"]), sel(x[x.which == "miR124"])
        if len(x1) + len(x2) >= 20 and len(e) >= 20:
            xv = np.concatenate([x1.delta_miR1_minus_miR124.values, x2.delta_miR1_minus_miR124.values])
            sg = np.concatenate([np.ones(len(x1)), -np.ones(len(x2))])
            xc = np.concatenate([x1.cond.values, x2.cond.values])
            ev, ecnd = e.delta_miR1_minus_miR124.values, e.cond.values

            def pooled(xv_, sg_, xc_, ev_, ec_, stat=np.median):
                out = []
                for sgn in (1.0, -1.0):
                    m_ = sg_ == sgn
                    if m_.any():
                        for cnd in np.unique(xc_[m_]):
                            em = ec_ == cnd
                            ref = ev_[em] if em.sum() >= 20 else ev_
                            out.append(sgn * (xv_[m_ & (xc_ == cnd)] - stat(ref)))
                return stat(np.concatenate(out))
            r["resp_pooled"] = pooled(xv, sg, xc, ev, ecnd)
            r["mean_resp_pooled"] = pooled(xv, sg, xc, ev, ecnd, np.mean)
            rng_ = np.random.default_rng(1)
            bs = []
            for _ in range(400):
                ia = rng_.integers(0, len(xv), len(xv)); ib = rng_.integers(0, len(ev), len(ev))
                bs.append(pooled(xv[ia], sg[ia], xc[ia], ev[ib], ecnd[ib]))
            r["resp_pooled_lo"], r["resp_pooled_hi"] = np.nanpercentile(bs, [2.5, 97.5])
        a = x1.delta_miR1_minus_miR124.values
        c = x2.delta_miR1_minus_miR124.values
        if len(a) >= 20 and len(c) >= 20:
            r["contrast"] = (np.median(a) - np.median(c)) / 2
            r["contrast_lo"], r["contrast_hi"] = boot(lambda u, v: (np.median(u) - np.median(v)) / 2, [a, c])
        r["endo_delta_med"] = np.median(e.delta_miR1_minus_miR124) if len(e) else np.nan
        for cond in ["miR1", "miR124"]:
            ec = e[e.cond == cond]
            r[f"endo_delta_{cond}def"] = np.median(ec.delta_miR1_minus_miR124) if len(ec) >= 10 else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    ref = t[(t.group == "core") & (t.bin == ">=0.8")].iloc[0]
    for X in ["miR1", "miR124", "pooled"]:
        t[f"frac_core_{X}"] = t[f"mean_resp_{X}"] / ref[f"mean_resp_{X}"]
    f = f"{RES}/seedless_readouts_occupancy_{model.replace('/', '_')}.tsv"
    t.to_csv(f, sep="\t", index=False, float_format="%.4f")
    show = t[t.bin == "all"][["group", "n_miR1", "n_miR124", "n_endo", "resp_miR1", "resp_miR1_lo", "resp_miR1_hi",
                              "resp_miR124", "resp_miR124_lo", "resp_miR124_hi", "resp_pooled", "resp_pooled_lo", "resp_pooled_hi",
                              "frac_core_pooled"]]
    print(show.round(3).to_string(index=False))
    return t


# ── mixing ──────────────────────────────────────────────────────────────────
def _mix_prep(model="base150"):
    """mixing sites (150-nt windows, strata, score bins) + per-chimera table joined to them + block rates"""
    estimate = mixing_estimate
    m = pd.read_csv(f"{EW}/150/mixing.tsv", sep="\t", low_memory=False)
    m300 = pd.read_csv(f"{EW}/300/mixing.tsv", sep="\t", usecols=["gene"], low_memory=False)
    m["p"] = pred(model, "mixing") if model != "base150" or os.path.exists(f"{ROOT}/data/seedless/base150/pred_mixing.npy") \
        else pd.read_csv(f"{EW}/150/pred_w150_mixing.tsv", sep="\t", usecols=["interaction_probability"]).iloc[:, 0].values
    mi = norm(m.noncodingRNA)
    from multiprocessing import Pool
    with Pool(16) as pool:
        r = pool.map(classify, zip(norm(m.target), norm(m.gene), norm(m300.gene), mi), chunksize=1024)
    m["stratum"], m["sub"], m["comp3p"] = [a[0] for a in r], [a[1] for a in r], [a[2] for a in r]
    m["grp"], m["c3"] = groups(m.stratum.values, m["sub"].values, m.comp3p.values)
    m["bin"] = score_bin(m.p.values)
    k = ["chr.g", "start.g", "end.g", "strand.g"]
    m["key"] = m[k].astype(str).agg(":".join, axis=1) + "|" + mi
    c = pd.read_csv(f"{MIX}/chimeras.tsv", sep="\t", low_memory=False,
                    usecols=["sample", "expt", "gel", *k, "mirna_species", "mirna_seq", "target_species"])
    c = c[c.mirna_species.isin(["human_only", "rat_only"])].copy()
    c["key"] = c[k].astype(str).agg(":".join, axis=1) + "|" + norm(c.mirna_seq)
    lab = m.drop_duplicates("key").set_index("key")[["grp", "c3", "bin", "p", "stratum"]]
    c = c.join(lab, on="key", how="inner")
    rate = pd.read_csv(f"{MIX}/artifact_rate.tsv", sep="\t")
    rate = rate[(rate.unit == "pooled") & (rate.stratum == "all")].set_index(["expt", "gel"])
    return m, c, rate, estimate


def _A(xc, rate, estimate):
    """post-lysis share per direction, mixing_artifacts.estimate per expt x gel block, pooled by hybrid count"""
    ah, nh, ar, nr = [], [], [], []
    for (expt, gel), d in xc.groupby(["expt", "gel"]):
        if (expt, gel) not in rate.index:
            continue
        q = rate.loc[(expt, gel)]
        e = estimate(d, float(q.q_r), {"f_h": float(q.f_h), "f_r": float(q.f_r), "m_h": float(q.m_h)})
        if e["n_h"] and np.isfinite(e["A_h"]):
            ah.append(e["A_h"]); nh.append(e["n_h"])
        if e["n_r"] and np.isfinite(e["A_r"]):
            ar.append(e["A_r"]); nr.append(e["n_r"])
    return (np.average(ah, weights=nh) if nh else np.nan), (np.average(ar, weights=nr) if nr else np.nan)


def mixing(model="base150"):
    m, c, rate, estimate = _mix_prep(model)
    rows = []
    for g in ["core", "flank150", "flank300", "seed_bulge", "seed_GU", "seed7_1mm", "3p_6mer", "contig8", "none",
              "comp3p", "noseed300", "missed_none", "missed_partial"]:
        for b in BLAB + ["all"]:
            def sel(d):
                if g == "comp3p":
                    s_ = d.c3.values
                elif g == "noseed300":
                    s_ = d.stratum.values == "noseed300"
                elif g == "missed_none":
                    s_ = (d.grp.values == "none") & (d.p.values < 0.5)
                elif g == "missed_partial":
                    s_ = np.isin(d.grp.values, ["seed_bulge", "seed_GU", "seed7_1mm", "3p_6mer", "contig8"]) & (d.p.values < 0.5)
                else:
                    s_ = d.grp.values == g
                return d[s_ & (d.bin.values == b)] if b != "all" else d[s_]
            xm, xc = sel(m), sel(c)
            r = {"group": g, "bin": b, "n": len(xm)}
            for sp in ["human_only", "rat_only"]:
                q = xm[xm.mirna_species == sp]
                r[f"same_{sp}"] = q.real.mean() if len(q) else np.nan
                r[f"n_{sp}"] = len(q)
            r["A_h"], r["A_r"] = _A(xc, rate, estimate)        # per block, pooled by hybrid counts
            rows.append(r)
    t = pd.DataFrame(rows)
    f = f"{RES}/seedless_readouts_mixing_{model.replace('/', '_')}.tsv"
    t.to_csv(f, sep="\t", index=False, float_format="%.4f")
    print(t[t.bin == "all"].round(3).to_string(index=False))
    # known-artifact reference: score distribution of cross- vs same-species 'none' chimeras
    nn = m[m.grp == "none"]
    for sp in ["human_only", "rat_only"]:
        q = nn[nn.mirna_species == sp]
        print(f"none {sp}: median p150 same-species {q[q.real].p.median():.3f} (n {int(q.real.sum())}) vs "
              f"cross-species {q[~q.real].p.median():.3f} (n {int((~q.real).sum())})")
    return t


def transport(model="base150"):
    """Stage 1.4: mixing sees only species-specific miRNAs. Readouts within miRNA-GC tertiles (cut points from the
    training positives' miRNA GC count), then reweighted to the tertile mix of the chr1 population being judged."""
    m, c, rate, estimate = _mix_prep(model)
    tr = pd.read_csv(f"{ROOT}/data/seedless/strata/manakov_train_nochr2.tsv.gz", sep="\t", usecols=["label", "mirna_gc"])
    lo, hi = np.quantile(tr.mirna_gc[tr.label == 1], [1 / 3, 2 / 3])
    tert = lambda g: np.where(g <= lo, "lowGC", np.where(g <= hi, "midGC", "highGC"))
    gc = lambda s: norm(s).str.count("[GC]").to_numpy()
    m["tert"] = tert(gc(m.noncodingRNA))
    c["tert"] = tert(gc(c.mirna_seq))
    te = pd.read_csv(f"{ROOT}/data/seedless/strata/manakov_test.tsv.gz", sep="\t",
                     usecols=["label", "stratum", "sub", "p150", "mirna_gc"])
    te = te[te.label == 1]
    te["tert"] = tert(te.mirna_gc.to_numpy())
    sel_te = {"none": (te.stratum == "noseed300") & (te["sub"] == "none"),
              "missed_none": (te.stratum == "noseed300") & (te["sub"] == "none") & (te.p150 < 0.5),
              "missed_partial": (te.stratum == "noseed300") & te["sub"].isin(["seed_bulge", "seed_GU", "seed7_1mm",
                                                                              "3p_6mer", "contig8"]) & (te.p150 < 0.5),
              "core": te.stratum == "core"}

    def sel(d, g):
        if g == "missed_none":
            return d[(d.grp.values == "none") & (d.p.values < 0.5)]
        if g == "missed_partial":
            return d[np.isin(d.grp.values, ["seed_bulge", "seed_GU", "seed7_1mm", "3p_6mer", "contig8"]) & (d.p.values < 0.5)]
        return d[d.grp.values == g]
    rows = []
    for g in ["none", "missed_none", "missed_partial", "core"]:
        mix = te[sel_te[g]].tert.value_counts(normalize=True)
        for t in ["lowGC", "midGC", "highGC"]:
            xm, xc = sel(m[m.tert == t], g), sel(c[c.tert == t], g)
            r = {"group": g, "tert": t, "chr1_share": mix.get(t, 0.0), "n": len(xm)}
            for sp in ["human_only", "rat_only"]:
                q = xm[xm.mirna_species == sp]
                r[f"same_{sp}"] = q.real.mean() if len(q) else np.nan
                r[f"n_{sp}"] = len(q)
            r["A_h"], r["A_r"] = _A(xc, rate, estimate)
            rows.append(r)
    t = pd.DataFrame(rows)
    out = []
    for g, x in t.groupby("group", sort=False):
        r = {"group": g, "tert": "chr1_reweighted", "chr1_share": 1.0, "n": x.n.sum()}
        for k in ["same_human_only", "same_rat_only", "A_h", "A_r"]:
            ok = np.isfinite(x[k])
            r[k] = np.average(x[k][ok], weights=x.chr1_share[ok]) if ok.any() else np.nan
        out.append(r)
    t = pd.concat([t, pd.DataFrame(out)], ignore_index=True)
    print(f"miRNA GC tertile cut points (training positives): <= {lo:g} | <= {hi:g} | above")
    print("mixing miRNAs per tertile:", m.drop_duplicates("noncodingRNA").groupby(["mirna_species", "tert"]).size().to_dict())
    t.to_csv(f"{RES}/seedless_transport_{model.replace('/', '_')}.tsv", sep="\t", index=False, float_format="%.4f")
    print(t.round(3).to_string(index=False))
    return t


if __name__ == "__main__":
    cmd = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "base150"
    {"build": build, "occupancy": lambda: occupancy(model), "mixing": lambda: mixing(model),
     "transport": lambda: transport(model)}[cmd]()
