#!/usr/bin/env python3
"""Twin-aware training arms for the seed-less plan (Stage 3.1); replaces noseed_arms.drop().

miRBench negatives are twins: 99.5% reuse the exact 50-nt `gene` window of a positive (other-family miRNA)
and family frequencies match between classes. An arm sets a weight w in [0, 1] on each positive; then
  1. twin step   each negative is scaled by P1(W)/P0(W), the share of its window's positive mass that
                 survives (0 when every positive on the window is dropped -> no orphan negatives);
                 negatives on windows without any positive (0.5%) keep weight 1;
  2. IPF         alternately rescale negatives to A0's per-family and per-window-stratum (GC decile of
                 gene150 x region) negative:positive ratios, ending on the family step.
Arms are row-aligned .npy sampler weights for `train --sample-weights X.npy --balance` on the unmodified
data/mre150/manakov_train_nochr2.tsv. A0 = all ones (the raw table).

    python3 src/artifact/seedless_arms.py build A0 A1        -> data/seedless/arms/<arm>.npy (+ .diag.tsv)
    python3 src/artifact/seedless_arms.py build A2 A3        (after Stage 3.0 OOF + results/seedless_cells.tsv)
    python3 src/artifact/seedless_arms.py build A2X          (A2 + G1' v2 RNA-side classes, noise follow-up)
    python3 src/artifact/seedless_arms.py build A2X_hi025    DEFAULT removal: A2X with the ARTIFACT cell at 0.1-0.25
                                                        (user decision 2026-11-23; A2/A2X = pre-registered 0.1-0.3)
    python3 src/artifact/seedless_arms.py diag  A0 A1 old_keep_partial
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRATA = f"{ROOT}/data/seedless/strata/manakov_train_nochr2.tsv.gz"
TRAIN = f"{ROOT}/data/mre150/manakov_train_nochr2.tsv"
OUT = f"{ROOT}/data/seedless/arms"
CELLS = f"{ROOT}/results/seedless_cells.tsv"
OOF = f"{ROOT}/data/seedless/oof/oof_train150.tsv"
BASE_TEST = f"{ROOT}/data/seedless/strata/manakov_test.tsv.gz"
DEFAULT_HI = 0.25
BINS = [0, 0.1, 0.3, 0.5, 0.8, 1.0001]
BLAB = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.8", ">=0.8"]


def load():
    d = pd.read_csv(STRATA, sep="\t", low_memory=False)
    g = pd.read_csv(TRAIN, sep="\t", usecols=["gene", "gene150"], low_memory=False)
    assert len(g) == len(d) and (g.gene.values == d.gene.values).all()
    d["gene150"] = g.gene150.values
    s = d.gene150.str.upper()
    d["gc150"] = (s.str.count("G") + s.str.count("C")) / s.str.len()
    d["wstrat"] = (pd.qcut(d.gc150, 10, labels=False, duplicates="drop").astype(str) + "|" +
                   d.feature.fillna("na").astype(str))
    return d


def assign_cells(stratum, sub, comp3p, score, label, col="G1_final"):
    """G1 label and R_emp (chr1 rows of results/seedless_cells.tsv) for every noseed300 positive scored < 0.5.
    A comp3p row takes the comp3p cell; a (group, bin) cell with < 30 positives (absent from the table) falls back
    to (group, 'missed'), then to the subclass. Rows outside scope get label '' and R_emp NaN."""
    c = pd.read_csv(CELLS, sep="\t")
    c = c[c.set == "manakov_test"]
    look = {(g, b): (l, r) for g, b, l, r in zip(c.group, c.bin, c[col], c.R_emp)}
    score = np.asarray(score, float)
    stratum, sub = np.asarray(stratum, object), np.asarray(pd.Series(sub).fillna(""), object)
    comp = np.asarray(pd.Series(comp3p).fillna(False), bool)
    b = np.asarray(pd.cut(score, BINS, labels=BLAB, right=False).astype(str), object)
    grp = np.where(comp, "comp3p", sub)
    lab = np.full(len(score), "", dtype=object)
    r = np.full(len(score), np.nan)
    for i in np.where((np.asarray(label) == 1) & (stratum == "noseed300") & (score < 0.5))[0]:
        for k in ((grp[i], b[i]), (grp[i], "missed"), (sub[i], b[i]), (sub[i], "missed")):
            if k in look:
                lab[i], r[i] = look[k]
                break
    return lab, r


def oof_scores(d):
    """Stage 3.0 out-of-fold probabilities, row-aligned to the training table, quantile-mapped onto the base150
    chr1 score distribution when the two differ (max |quantile gap| > 0.02 over the 5-95% range)."""
    o = pd.read_csv(OOF, sep="\t", usecols=["row", "label", "interaction_probability"])
    assert len(o) == len(d) and (o.row.values == d.row.values).all() and (o.label.values == d.label.values).all()
    p = o.interaction_probability.to_numpy(float)
    t = pd.read_csv(BASE_TEST, sep="\t", usecols=["p150"]).p150.to_numpy(float)
    qs = np.linspace(0.05, 0.95, 19)
    gap = np.abs(np.quantile(p, qs) - np.quantile(t, qs)).max()
    print(f"OOF vs base150-chr1 score quantiles: max gap {gap:.3f}", flush=True)
    if gap > 0.02:
        grid = np.linspace(0, 1, 2001)
        p = np.interp(np.searchsorted(np.sort(p), p, side="right") / len(p), grid, np.quantile(t, grid))
        print("  -> quantile-mapped onto base150 chr1", flush=True)
    return p


def artifact_cell(d, p, hi):
    """ARTIFACT cell with upper score edge hi: seed-less (noseed300) 'none' positives, not comp3p, 0.1 <= p < hi.
    hi = 0.3 is the pre-registered cell (identical rows to assign_cells' ARTIFACT label, checked in cell_extra)."""
    comp = np.asarray(pd.Series(d.comp3p).fillna(False), bool)
    return ((d.stratum.values == "noseed300") & (d["sub"].fillna("").values == "none") & ~comp
            & (d.label.values == 1) & (p >= 0.1) & (p < hi))


def cell_extra(d, arm, hi=None):
    """A2: 0 for missed positives in ARTIFACT cells; A3: R_emp (clipped to [0, 1]) in ARTIFACT/MIXED cells.
    hi: A2 only, replace the cell table's 0.1-0.3 ARTIFACT bin by 0.1-hi (DEFAULT_HI = 0.25, user decision)."""
    p = oof_scores(d)
    if arm == "A2" and hi is not None:
        w = np.ones(len(d))
        art = artifact_cell(d, p, hi)
        if os.path.exists(CELLS):                       # cross-check the rule against the cell table at 0.3
            lab, _ = assign_cells(d.stratum.values, d["sub"].values, d.comp3p.values, p, d.label.values)
            assert ((lab == "ARTIFACT") == artifact_cell(d, p, 0.3)).all(), "0.3 rule != cell-table ARTIFACT"
        w[art] = 0.0
        print(f"A2 (edge {hi}): ARTIFACT positives {art.sum():,}; positive mass kept "
              f"{w[d.label.values == 1].mean():.3f}", flush=True)
        return w
    lab, r = assign_cells(d.stratum.values, d["sub"].values, d.comp3p.values, p, d.label.values)
    w = np.ones(len(d))
    if arm == "A2":
        w[lab == "ARTIFACT"] = 0.0
    else:
        m = np.isin(lab, ["ARTIFACT", "MIXED"])
        w[m] = np.clip(r[m], 0.0, 1.0)
    pos = d.label.values == 1
    print(f"{arm}: positives in scope {np.isin(lab, ['ARTIFACT', 'MIXED', 'REAL-MISSED']).sum():,} "
          f"(ARTIFACT {(lab == 'ARTIFACT').sum():,}, MIXED {(lab == 'MIXED').sum():,}, "
          f"REAL-MISSED {(lab == 'REAL-MISSED').sum():,}); positive mass kept {w[pos].sum() / pos.sum():.3f}", flush=True)
    return w


def g1_extra(d, base):
    """A2X = A2 plus the G1' v2 RNA-side classes (results/seedless_decision_report.md, deviations F1-F2): seed-less
    positives of the 16 self-enriched and the 14 low-complexity small RNAs (results/lowcomplexity_rnas.tsv), and
    seed-less positives whose 150-nt window holds >= 12 nt of their own small RNA on the same strand"""
    u = pd.read_csv(f"{ROOT}/results/lowcomplexity_rnas.tsv", sep="\t", usecols=["seq", "signature", "low_complexity"])
    rnas = set(u.seq[u.signature == "self"]) | set(u.seq[u.low_complexity.astype(str) == "True"])
    assert len(rnas) == 27, len(rnas)
    seq = d.noncodingRNA.astype(str).str.upper().str.replace("U", "T").values
    self_len = np.load(f"{ROOT}/data/lowcomplexity/manakov_train_nochr2.npz")["self_len"]
    assert len(self_len) == len(d)
    m = (d.label.values == 1) & (d.stratum.values == "noseed300") & (np.isin(seq, list(rnas)) | (self_len >= 12))
    w = base.copy()
    w[m] = 0.0
    print(f"A2X: class positives {m.sum():,}, newly dropped beyond A2 {(m & (base > 0)).sum():,}", flush=True)
    return w


def positive_weights(d, arm, extra=None):
    pos = d.label.values == 1
    w = np.ones(len(d))
    if arm == "A0":
        pass
    elif arm == "A1":                       # keep_partial, twin-aware: drop every 'none' positive
        w[pos & (d.stratum.values == "noseed300") & (d["sub"].fillna("").values == "none")] = 0.0
    elif arm == "S1":                       # Stage 2.3 specialist: seed-less positives only (twins follow)
        w[pos & (d.stratum.values != "noseed300")] = 0.0
        w[~pos] = np.nan
        return w
    elif extra is not None:                 # A2/A3: caller supplies the positive weights
        w[pos] = extra[pos]
    else:
        raise SystemExit(f"unknown arm {arm}")
    w[d.stratum.values == "flank300"] = 1.0  # seed 51-125 nt out: invisible at 150 nt, never touched
    w[~pos] = np.nan
    return w


def arm_weights(d, wpos, iters=20):
    pos = d.label.values == 1
    neg = ~pos
    win = pd.factorize(d.gene.values)[0]
    P0 = np.bincount(win[pos], minlength=win.max() + 1).astype(float)
    P1 = np.bincount(win[pos], weights=wpos[pos], minlength=win.max() + 1)
    w = wpos.copy()
    r = np.ones(len(d))
    has = P0[win] > 0
    r[has] = P1[win[has]] / P0[win[has]]
    w[neg] = r[neg]
    margins = {}
    for col in ["noncodingRNA_fam", "wstrat"]:
        c = pd.factorize(d[col].astype(str).values)[0]
        k = c.max() + 1
        p0 = np.bincount(c[pos], minlength=k).astype(float)
        n0 = np.bincount(c[neg], minlength=k).astype(float)
        margins[col] = (c, np.divide(n0, p0, out=np.full(k, np.nan), where=p0 > 0))
    for it in range(iters):
        for col in ["wstrat", "noncodingRNA_fam"]:          # end on the family step
            c, ratio = margins[col]
            k = len(ratio)
            pm = np.bincount(c[pos], weights=w[pos], minlength=k)
            nm = np.bincount(c[neg], weights=w[neg], minlength=k)
            f = np.ones(k)
            ok = (nm > 0) & ~np.isnan(ratio)
            f[ok] = ratio[ok] * pm[ok] / nm[ok]
            w[neg] *= f[c[neg]]
    w *= len(w) / w.sum()
    return w


def margin_dev(d, w, col):
    pos = d.label.values == 1
    c = pd.factorize(d[col].astype(str).values)[0]
    k = c.max() + 1
    p0 = np.bincount(c[pos], minlength=k).astype(float); n0 = np.bincount(c[~pos], minlength=k).astype(float)
    pm = np.bincount(c[pos], weights=w[pos], minlength=k); nm = np.bincount(c[~pos], weights=w[~pos], minlength=k)
    ok = (p0 > 0) & (n0 > 0) & (pm > 0.5)
    want = n0[ok] / p0[ok] * pm[ok]
    return float(np.average(np.abs(nm[ok] - want) / want, weights=pm[ok])), float(np.max(np.abs(nm[ok] - want) / want))


def kmer3(seqs):
    lut = np.full(256, 255, np.uint8)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i
    a = np.frombuffer("".join(seqs).encode(), np.uint8).reshape(len(seqs), -1)
    x = lut[a].astype(np.int64)
    bad = x == 255
    code = x[:, :-2] * 16 + x[:, 1:-1] * 4 + x[:, 2:]
    code[bad[:, :-2] | bad[:, 1:-1] | bad[:, 2:]] = 64
    out = np.zeros((len(seqs), 65))
    np.add.at(out, (np.repeat(np.arange(len(seqs)), code.shape[1]), code.ravel()), 1.0)
    return out[:, :64] / code.shape[1]


def shortcut_auc(d, w, n=300_000, seed=0):
    """Weighted window-only (3-mers of gene150 + GC + region) and family-only classifier AUCs."""
    rng = np.random.default_rng(seed)
    p = w / w.sum()
    idx = rng.choice(len(d), size=2 * n, replace=True, p=p)
    a, b = idx[:n], idx[n:]
    y = d.label.values
    reg = pd.get_dummies(d.feature.fillna("na").astype(str)).values.astype(float)
    def X(i):
        return np.hstack([kmer3(d.gene150.values[i].tolist()), d.gc150.values[i, None], reg[i]])
    m = LogisticRegression(max_iter=300, C=1.0).fit(X(a), y[a])
    win_auc = roc_auc_score(y[b], m.predict_proba(X(b))[:, 1])
    fam = pd.factorize(d.noncodingRNA_fam.astype(str))[0]
    # family-only: score = training-half positive rate of the family (smoothed)
    k = fam.max() + 1
    pr = (np.bincount(fam[a], weights=y[a], minlength=k) + 1) / (np.bincount(fam[a], minlength=k) + 2)
    fam_auc = roc_auc_score(y[b], pr[fam[b]])
    return win_auc, fam_auc


def old_keep_partial(d, seed=0):
    """Row mask of the old noseed_arms.drop() keep_partial train set (random same-family negatives)."""
    rm = ((d.label.values == 1) & (d.stratum.values == "noseed300") & (d["sub"].fillna("").values == "none"))
    rng = np.random.default_rng(seed)
    keep = ~rm
    fam = d.noncodingRNA_fam.astype(str).values
    negm = d.label.values == 0
    for f, n in pd.Series(fam[rm]).value_counts().items():
        cand = np.where(negm & (fam == f) & keep)[0]
        keep[rng.choice(cand, min(n, len(cand)), replace=False)] = False
    return keep.astype(float)


def diag(d, w, name):
    pos = d.label.values == 1
    win = pd.factorize(d.gene.values)[0]
    pm_win = np.bincount(win[pos], weights=w[pos], minlength=win.max() + 1)
    p0_win = np.bincount(win[pos], minlength=win.max() + 1)
    orphan = (~pos) & (w > 0) & (p0_win[win] > 0) & (pm_win[win] == 0)
    wa, fa = shortcut_auc(d, w)
    fam_mean, fam_max = margin_dev(d, w, "noncodingRNA_fam")
    ws_mean, ws_max = margin_dev(d, w, "wstrat")
    r = {"arm": name, "pos_zero_share": float((w[pos] == 0).mean()), "neg_zero_share": float((w[~pos] == 0).mean()),
         "pos_mass": float(w[pos].sum()), "neg_mass": float(w[~pos].sum()),
         "ess": float(w.sum() ** 2 / (w ** 2).sum()),
         "orphan_neg_rows": int(orphan.sum()),
         "orphan_neg_share": float(((~pos) & (w > 0) & (pm_win[win] == 0)).sum() / max(((~pos) & (w > 0)).sum(), 1)),
         "fam_dev_mean": fam_mean, "fam_dev_max": fam_max, "wstrat_dev_mean": ws_mean, "wstrat_dev_max": ws_max,
         "window_only_auc": wa, "family_only_auc": fa}
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    return r


def main():
    cmd, arms = sys.argv[1], sys.argv[2:]
    d = load()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for arm in arms:
        if cmd == "build":
            base, _, edge = arm.partition("_hi")          # A2_hi025 / A2X_hi025 -> edge 0.25
            hi = int(edge) / 100 if edge else None
            extra = (cell_extra(d, base, hi) if base in ("A2", "A3")
                     else g1_extra(d, cell_extra(d, "A2", hi)) if base == "A2X" else None)
            w = arm_weights(d, positive_weights(d, base, extra=extra))
            np.save(f"{OUT}/{arm}.npy", w)
        elif arm == "old_keep_partial":
            w = old_keep_partial(d)
        else:
            w = np.load(f"{OUT}/{arm}.npy")
        rows.append(diag(d, w, arm))
    pd.DataFrame(rows).to_csv(f"{OUT}/diag_{'_'.join(arms)}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
