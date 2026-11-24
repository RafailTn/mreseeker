#!/usr/bin/env python3
"""Low-complexity small-RNA audit of the miRBench v7-format chimera tables (follow-up to deviation 10 of
results/seedless_decision_report.md, where hsa-miR-8485, a CA repeat, carried most arm differences).

Step 1, sequence only (rule fixed before any outcome was looked at): every unique small RNA gets a DUST triplet score
(sum over triplets of c(c-1)/2, divided by l-1, l = number of triplets) and the fraction of its length inside perfect
tandem repeats (period 1/2/3/4, total length >= 6/8/9/12). Low complexity = DUST >= 1.0 or tandem cover >= 0.5
(14 of the 1,912 unique small RNAs).

Step 2, window signature (every small RNA): the longest exact match between the small RNA and its 150-nt window on the
same strand ('self') and as a perfect complement ('anti'). A repeat-derived read split into a fake small-RNA arm and a
target arm puts the RNA's own sequence in the window (self); pairing puts its complement there (anti). Positives are
compared with the same RNA's negatives, which in the twin design are other positives' windows (the background).
  self-enriched = share of positives with a self match >= 8 nt exceeds the negatives' share by >= 0.20 and >= 5-fold
                  (pooled over sets, >= 20 positives); anti-enriched likewise.

  Also required: Fisher exact test of the self-match counts (positives vs negatives), BH q < 0.01 over the RNAs with
  >= 20 positives. 'palindromic' = the RNA shares an 8-mer with its own reverse complement (e.g. an AU repeat), so
  self and anti matches are the same rows and the window signature cannot separate them ('ambiguous').
  Alu hallmark in the window (both arms, both orientations) is reported per RNA as supporting evidence.

Step 3, impact: share of positives per set, base150 auPRC with and without the rows (both labels) of (a) the
low-complexity self-enriched RNAs and (b) every self-enriched RNA, A1/A2/A3 - A0 auPRC with and without (3 seeds),
and the mixing readout (rows, cross-species share, arm mixing gains without them). The arm columns need the trained
A0-A3 arms (train_arms.sh) and are left out when their predictions are missing. Row level: share of positives vs
negatives whose window holds >= 12 nt of their small RNA on the same strand.

    python3 src/artifact/lowcomplexity_audit.py
      -> data/lowcomplexity/<set>.npz (per-row match lengths, cached)
         results/lowcomplexity_rnas.tsv, results/lowcomplexity_by_set.tsv, results/lowcomplexity_impact.tsv
"""
import math
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_common import ROOT  # noqa: E402
OUT = f"{ROOT}/data/lowcomplexity"
EW = f"{ROOT}/data/relabel/noise/eval_win_v2/150"
SETS = ["manakov_train_nochr2", "manakov_val_chr2", "manakov_test", "manakov_leftout", "gse304955", "gse263552",
        "gse180129_HSPC", "gse180129_Kasumi1", "hejret_train", "hejret_test", "klimentova_test"]
EVAL = ["manakov_test", "manakov_leftout", "gse304955", "gse263552", "gse180129_HSPC", "gse180129_Kasumi1",
        "hejret_test", "klimentova_test", "manakov_val_chr2"]          # sets with arm predictions
SEEDS = ["s42", "s43", "s44"]
DUST_MIN, REP_MIN = 1.0, 0.5
K, MIN_POS, D_MIN, FOLD, Q_MAX = 8, 20, 0.20, 5.0, 0.01
ALU = "GTAATCCCAGCA|TGCTGGGATTAC|GTAGTCCCAGCT|AGCTGGGACTAC"   # Alu left/right-arm hallmark, both orientations
COMP = str.maketrans("ACGT", "TGCA")


def norm(s):
    return s.astype(str).str.upper().str.replace("U", "T")


def rc(s):
    return s.translate(COMP)[::-1]


def dust(s):
    l = len(s) - 2
    if l < 2:
        return 0.0
    c = Counter(s[i:i + 3] for i in range(l))
    return sum(v * (v - 1) / 2 for v in c.values()) / (l - 1)


def tandem_cover(s):
    cov = np.zeros(len(s), bool)
    for p, minlen in [(1, 6), (2, 8), (3, 9), (4, 12)]:
        i = 0
        while i < len(s) - p:
            j = i
            while j + p < len(s) and s[j] == s[j + p]:
                j += 1
            if j > i and j - i + p >= minlen:
                cov[i:j + p] = True
            i = max(j, i + 1)
    return float(cov.mean())


def entropy(s):
    c = Counter(s)
    return -sum(v / len(s) * math.log2(v / len(s)) for v in c.values())


def longest(r, w):
    """longest k such that some k-mer of r occurs in w (binary search; containment is monotone in k)"""
    lo, hi = 0, len(r)
    while lo < hi:
        k = (lo + hi + 1) // 2
        if any(r[i:i + k] in w for i in range(len(r) - k + 1)):
            lo = k
        else:
            hi = k - 1
    return lo


def _match(args):
    rs, ws = args
    return [(longest(r, w), longest(rc(r), w)) for r, w in zip(rs, ws)]


def match_lengths(rna, win, workers=24, chunk=20000):
    jobs = [(rna[i:i + chunk], win[i:i + chunk]) for i in range(0, len(rna), chunk)]
    with Pool(workers) as pool:
        res = [x for part in pool.map(_match, jobs) for x in part]
    a = np.array(res, dtype=np.uint8).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def scores(name, n):
    """held-out base150 probability per row: OOF for the training table, base150 elsewhere"""
    if name == "manakov_train_nochr2":
        from seedless_arms import OOF
        o = pd.read_csv(OOF, sep="\t", usecols=["row", "interaction_probability"])
        assert len(o) == n and (o.row.values == np.arange(n)).all()
        return o.interaction_probability.to_numpy(float)
    f = f"{ROOT}/data/seedless/strata/{name}.tsv.gz"
    if os.path.exists(f):
        p = pd.read_csv(f, sep="\t", usecols=lambda c: c == "p150").p150.to_numpy(float)
    else:
        p = pd.read_csv(f"{ROOT}/data/mre150/pred_mre150_{name}.tsv", sep="\t",
                        usecols=["interaction_probability"]).interaction_probability.to_numpy(float)
    assert len(p) == n, name
    return p


def load(name):
    t = pd.read_csv(f"{ROOT}/data/mre150/{name}.tsv", sep="\t", low_memory=False,
                    usecols=["gene150", "noncodingRNA", "noncodingRNA_name", "noncodingRNA_fam", "label"])
    t["seq"] = norm(t.noncodingRNA).values
    t["alu"] = norm(t.gene150).str.contains(ALU).values
    os.makedirs(OUT, exist_ok=True)
    f = f"{OUT}/{name}.npz"
    if os.path.exists(f) and len(np.load(f)["self_len"]) == len(t):
        z = np.load(f)
        t["self_len"], t["anti_len"] = z["self_len"], z["anti_len"]
    else:
        s, a = match_lengths(t.seq.tolist(), norm(t.gene150).tolist())
        np.savez_compressed(f, self_len=s, anti_len=a)
        t["self_len"], t["anti_len"] = s, a
    t["p"] = scores(name, len(t))
    st = f"{ROOT}/data/seedless/strata/{name}.tsv.gz"
    if os.path.exists(st):
        x = pd.read_csv(st, sep="\t", usecols=["stratum", "sub"], low_memory=False)
        t["none"] = ((x.stratum == "noseed300") & (x["sub"] == "none")).values
    else:
        t["none"] = np.nan
    t["set"] = name
    print(f"{name}: {len(t):,} rows, {t.seq.nunique()} small RNAs", flush=True)
    return t.drop(columns=["gene150", "noncodingRNA"])


def per_rna(t):
    t = t.assign(self8=t.self_len >= K, self12=t.self_len >= 12, anti8=t.anti_len >= K, anti12=t.anti_len >= 12)
    g = t.groupby(["seq", "label"]).agg(n=("label", "size"), self8=("self8", "mean"), self12=("self12", "mean"),
                                        anti8=("anti8", "mean"), anti12=("anti12", "mean"), alu=("alu", "mean"),
                                        self_len_med=("self_len", "median")).unstack("label")
    g.columns = [f"{a}_{'pos' if b == 1 else 'neg'}" for a, b in g.columns]
    g = g.fillna({"n_pos": 0, "n_neg": 0})
    return g


def fisher_q(u):
    """BH q of a two-sided Fisher test on self-match (>= K nt) counts, positives vs negatives, RNAs with >= MIN_POS"""
    from scipy.stats import fisher_exact
    ok = u.n_pos >= MIN_POS
    p = pd.Series(np.nan, index=u.index)
    for q in u.index[ok]:
        a, b = round(u.at[q, "self8_pos"] * u.at[q, "n_pos"]), round(u.at[q, "self8_neg"] * u.at[q, "n_neg"])
        p[q] = fisher_exact([[a, u.at[q, "n_pos"] - a], [b, u.at[q, "n_neg"] - b]])[1]
    x = p[ok].sort_values()
    q = (x * len(x) / np.arange(1, len(x) + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(u.index)


def auc_within(x):
    y = x.label.values
    if (y == 1).sum() >= 10 and (y == 0).sum() >= 10:
        return roc_auc_score(y, x.p.values)
    return np.nan


def classify(r):
    if r.n_pos < MIN_POS:
        return "too few"
    s = r.self8_pos - r.self8_neg >= D_MIN and r.self8_pos >= FOLD * max(r.self8_neg, 1e-3) and r.q_self < Q_MAX
    a = r.anti8_pos - r.anti8_neg >= D_MIN and r.anti8_pos >= FOLD * max(r.anti8_neg, 1e-3)
    if s and r.palindromic:
        return "ambiguous"                          # self and anti matches are the same rows
    if s:
        return "self"                               # own sequence on the same strand (self dominates or alone)
    return "anti" if a else "none"


def have_arm_preds():
    """True when every A0-A3 seed has its test-set and mixing predictions (step 3's arm columns)."""
    return all(os.path.exists(f"{ROOT}/data/seedless/arms/{a}/{k}/pred_{s}.npy")
               for a in ["A0", "A1", "A2", "A3"] for k in SEEDS for s in EVAL + ["mixing"])


def main():
    tabs = [load(s) for s in SETS]
    t = pd.concat(tabs, ignore_index=True)
    names = t.groupby("seq").agg(name=("noncodingRNA_name", lambda x: "|".join(sorted(set(map(str, x))))[:60]),
                                 fam=("noncodingRNA_fam", "first"))
    u = names.join(per_rna(t))
    u["len"] = u.index.str.len()
    u["dust"] = [dust(s) for s in u.index]
    u["tandem_cover"] = [tandem_cover(s) for s in u.index]
    u["entropy"] = [entropy(s) for s in u.index]
    u["low_complexity"] = (u.dust >= DUST_MIN) | (u.tandem_cover >= REP_MIN)
    u["palindromic"] = [longest(s, rc(s)) >= K for s in u.index]
    u["q_self"] = fisher_q(u)
    u["signature"] = u.apply(classify, axis=1)
    pos = t[t.label == 1]
    u["none_share_pos"] = pos[pos.none.notna()].groupby("seq").none.mean()
    held = t[t.set != "manakov_train_nochr2"]
    u["auc_within_test"] = held.groupby("seq").apply(auc_within)
    u["auc_within_train_oof"] = t[t.set == "manakov_train_nochr2"].groupby("seq").apply(auc_within)
    m = pd.read_csv(f"{EW}/mixing.tsv", sep="\t", low_memory=False)
    m["seq"] = norm(m.noncodingRNA).values
    base = m.groupby("mirna_species").real.mean()
    mg = m.groupby("seq").agg(mix_n=("real", "size"), mix_same_share=("real", "mean"),
                              mix_dir=("mirna_species", "first"))
    u = u.join(mg)
    u["mix_dir_same_share"] = u.mix_dir.map(base)
    u = u.sort_values(["low_complexity", "n_pos"], ascending=False)
    u.index.name = "seq"
    u.to_csv(f"{ROOT}/results/lowcomplexity_rnas.tsv", sep="\t", float_format="%.4f")

    flag = u[u.low_complexity].index
    art = u[u.low_complexity & (u.signature == "self")].index
    other_self = u[~u.low_complexity & (u.signature == "self")].index
    allself = u[u.signature == "self"].index
    print(f"\nlow complexity: {len(flag)} small RNAs; artifact-like (low complexity + self-enriched): {len(art)}; "
          f"ambiguous (palindromic): {int((u.low_complexity & (u.signature == 'ambiguous')).sum())}; "
          f"self-enriched but not low complexity: {len(other_self)}", flush=True)

    amb = u[u.signature == "ambiguous"].index
    listed = flag.union(allself).union(amb)
    rows = []
    for s, x in t[t.seq.isin(listed)].groupby("set", sort=False):
        npos = (t.set.values == s) & (t.label.values == 1)
        for q, y in x.groupby("seq"):
            yp = y[y.label == 1]
            rows.append({"set": s, "seq": q, "name": u.at[q, "name"], "low_complexity": u.at[q, "low_complexity"],
                         "signature": u.at[q, "signature"], "n_pos": len(yp), "n_neg": int((y.label == 0).sum()),
                         "share_of_set_pos": len(yp) / npos.sum(), "self8_pos": yp.self_len.ge(K).mean(),
                         "self8_neg": y[y.label == 0].self_len.ge(K).mean(), "anti8_pos": yp.anti_len.ge(K).mean(),
                         "anti8_neg": y[y.label == 0].anti_len.ge(K).mean(), "none_share_pos": yp.none.mean(),
                         "mean_p_pos": yp.p.mean(), "mean_p_neg": y[y.label == 0].p.mean(), "auc_within": auc_within(y)})
    bs = pd.DataFrame(rows)
    bs.to_csv(f"{ROOT}/results/lowcomplexity_by_set.tsv", sep="\t", index=False, float_format="%.4f")

    arms = have_arm_preds()
    if not arms:
        print("\nA0-A3 arm predictions (data/seedless/arms/A*/s*/pred_*.npy) not found: impact table without the "
              "arm columns (train them with src/artifact/train_arms.sh to add them)", flush=True)
    imp = []
    for s in SETS:
        x = t[t.set == s]
        y, p = x.label.values, x.p.values
        excl = {"art": x.seq.isin(art).values, "allself": x.seq.isin(allself).values,
                "self12pos": (x.self_len.values >= 12) & (y == 1)}            # row level: positives only
        r = {"set": s, "pos": int((y == 1).sum()),
             "lowcomp_pos_share": (x.seq.isin(flag).values & (y == 1)).sum() / (y == 1).sum(),
             "self12_pos": (x.self_len.values[y == 1] >= 12).mean(), "self12_neg": (x.self_len.values[y == 0] >= 12).mean(),
             "base150_ap_all": ap(y, p)}
        for e, drop in excl.items():
            r[f"{e}_pos_share"] = (drop & (y == 1)).sum() / (y == 1).sum()
            r[f"base150_ap_no_{e}"] = ap(y[~drop], p[~drop])
        if arms and s in EVAL:
            for arm in ["A1", "A2", "A3"]:
                for v, keep in {"all": np.ones(len(y), bool), "no_art": ~excl["art"], "no_allself": ~excl["allself"],
                                "no_self12pos": ~excl["self12pos"]}.items():
                    d = []
                    for k in SEEDS:
                        p0 = np.load(f"{ROOT}/data/seedless/arms/A0/{k}/pred_{s}.npy")
                        px = np.load(f"{ROOT}/data/seedless/arms/{arm}/{k}/pred_{s}.npy")
                        d.append(ap(y[keep], px[keep]) - ap(y[keep], p0[keep]))
                    r[f"{arm}_d_{v}"] = np.mean(d)
        imp.append(r)
    r = {"set": "mixing (readout)", "pos": len(m), "lowcomp_pos_share": m.seq.isin(flag).mean()}
    mexcl = {"art": m.seq.isin(art).values, "allself": m.seq.isin(allself).values}
    for e, drop in mexcl.items():
        r[f"{e}_pos_share"] = drop.mean()
    for arm in (["A1", "A2", "A3"] if arms else []):
        for v, keep in {"all": np.ones(len(m), bool), "no_art": ~mexcl["art"], "no_allself": ~mexcl["allself"]}.items():
            d = []
            for k in SEEDS:
                aucs = []
                for a in ["A0", arm]:
                    pp = np.load(f"{ROOT}/data/seedless/arms/{a}/{k}/pred_mixing.npy")
                    z = m[keep].assign(p=pp[keep])
                    aucs.append(np.mean([roc_auc_score(g.real, g.p) for _, g in z.groupby("mirna_species")]))
                d.append(aucs[1] - aucs[0])
            r[f"{arm}_d_{v}"] = np.mean(d)
    imp.append(r)
    ip = pd.DataFrame(imp)
    ip.to_csv(f"{ROOT}/results/lowcomplexity_impact.tsv", sep="\t", index=False, float_format="%.5f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 30)
    cols = ["name", "dust", "tandem_cover", "n_pos", "self8_pos", "self8_neg", "anti8_pos", "anti8_neg", "q_self",
            "signature", "self_len_med_pos", "alu_pos", "alu_neg", "none_share_pos", "auc_within_test", "mix_n",
            "mix_same_share", "mix_dir_same_share"]
    print("\nlow-complexity small RNAs:")
    print(u.loc[flag, cols].round(3).to_string())
    print("\nself-enriched small RNAs that are not low complexity:")
    print(u.loc[other_self, cols].round(3).to_string() if len(other_self) else "  none")
    print("\nsignature counts (>= 20 positives), by low-complexity flag:")
    print(pd.crosstab(u.signature, u.low_complexity).to_string())
    print("\nimpact:")
    print(ip.round(4).T.to_string())
    print("\nlargest contributors per set (positives; self-enriched, ambiguous or low complexity):")
    for s, x in bs.groupby("set", sort=False):
        x = x[x.n_pos > 0].sort_values("n_pos", ascending=False).head(5)
        print(f"  {s}: " + "; ".join(f"{n} {k} ({100 * f:.2f}%, {g})" for n, k, f, g in
                                    zip(x.name, x.n_pos, x.share_of_set_pos, x.signature)))


if __name__ == "__main__":
    main()
