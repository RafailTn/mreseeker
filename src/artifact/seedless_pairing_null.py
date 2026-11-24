#!/usr/bin/env python3
"""Stage 1.1 of the seed-less plan: does the ligated miRNA pair with the window better than chance?

IntaRNA (MREseeker's flags, as in Hybridetector audit_noncanonical.py) on the 300-nt window with context
accessibility (--tAccW=150 --tAccL=100) and interactions restricted to the central 150 nt the model sees
(--tRegion). Queries per row: own miRNA + 10 expressed decoys (HEK293T small RNA, CPM >= 10, other nt2-8 seed,
GC count within +-1, widened until >= 20 candidates; CPM-weighted draw = exchangeable with the own miRNA
under random ligation) + 10 dinucleotide shuffles of the own miRNA.

    python3 src/artifact/seedless_pairing_null.py select            -> data/seedless/pairing/rows.tsv.gz
    python3 src/artifact/seedless_pairing_null.py bench 200
    python3 src/artifact/seedless_pairing_null.py run [--workers 40] -> data/seedless/pairing/chunks/*.npz (resumable)
    python3 src/artifact/seedless_pairing_null.py collect           -> data/seedless/pairing/pairing.tsv.gz
"""
import glob
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artifact_common import HYB_DIR, INTARNA, ROOT, dinuc_shuffle, has_seed, rand_p  # noqa: E402

OUT = f"{ROOT}/data/seedless/pairing"
STRATA = f"{ROOT}/data/seedless/strata"
EW = f"{ROOT}/data/relabel/noise/eval_win_v2"
SMALLRNA = f"{HYB_DIR}/data/smallrna/gse196460_mirna_cpm.tsv"
FLAGS = ["--mode=M", "--tAccW=150", "--tAccL=100", "--qAcc=N", "--outMode=C",
         "--outCsvCols=id2,start1,end1,start2,end2,hybridDP,E,E_hybrid,ED1", "--outDeltaE=100",
         "--outOverlap=B", "--noSeed", "--outMaxE=100", "--outNoLP", "--model=S", "-n", "1"]
N_EXPR, N_SHUF = 10, 10
NQ = 1 + N_EXPR + N_SHUF
CHUNK = 5000


def norm(s):
    return s.astype(str).str.upper().str.replace("U", "T")


# ── row selection ───────────────────────────────────────────────────────────
def _strata(name):
    d = pd.read_csv(f"{STRATA}/{name}.tsv.gz", sep="\t", low_memory=False)
    g = pd.read_csv(f"{ROOT}/data/mre300/{name}.tsv", sep="\t", usecols=["gene", "gene300"], low_memory=False)
    assert (g.gene.values == d.gene.values).all()
    d["target"] = g.gene300.values
    d["set"] = name
    return d


def _matched_neg(d, rng, n_per=None):
    """noseed300 negatives sampled per subclass to the positives' subclass counts."""
    ns = d[d.stratum == "noseed300"]
    out = []
    for s, x in ns.groupby("sub"):
        n = int((x.label == 1).sum()) if n_per is None else n_per.get(s, 0)
        neg = x[x.label == 0]
        out.append(neg.sample(min(n, len(neg)), random_state=int(rng.integers(1 << 31))))
    return pd.concat(out)


def select():
    rng = np.random.default_rng(0)
    parts = []
    t = _strata("manakov_test")
    ns = t[t.stratum == "noseed300"]
    parts += [ns[ns.label == 1].assign(role="noseed_pos"), _matched_neg(t, rng).assign(role="noseed_neg")]
    for s in ["core", "flank150"]:
        x = t[(t.stratum == s)]
        xp, xn = x[x.label == 1], x[x.label == 0]
        parts.append(xp.sample(min(10000, len(xp)), random_state=1).assign(role=f"{s}_pos"))
        parts.append(xn.sample(min(10000, len(xn)), random_state=2).assign(role=f"{s}_neg"))
    lo = _strata("manakov_leftout")
    parts.append(lo[lo.stratum == "noseed300"].assign(role="noseed_all"))
    sa = _strata("gse304955")
    ns = sa[(sa.stratum == "noseed300") & (sa.label == 1)].sample(30000, random_state=3)
    n_per = ns["sub"].value_counts().to_dict()
    parts += [ns.assign(role="noseed_pos"), _matched_neg(sa, rng, n_per).assign(role="noseed_neg")]
    keep = ["set", "role", "row", "gene", "noncodingRNA", "noncodingRNA_fam", "label", "stratum", "sub", "comp3p",
            "p150", "chr", "start", "strand", "feature", "target"]
    rows = pd.concat(parts)[keep]
    # label-free sites: Expt8 occupancy and human/rat mixing, seed-less by the 300-nt window
    for s in ["occupancy", "mixing"]:
        a = pd.read_csv(f"{EW}/150/{s}.tsv", sep="\t", low_memory=False)
        b = pd.read_csv(f"{EW}/300/{s}.tsv", sep="\t", usecols=["gene"], low_memory=False)
        p = pd.read_csv(f"{EW}/150/pred_w150_{s}.tsv", sep="\t", low_memory=False)
        assert len(a) == len(b) == len(p)
        m = norm(a.noncodingRNA)
        g300 = norm(b.gene)
        ns = np.array([not has_seed(x, y) for x, y in zip(g300, m)])
        x = pd.DataFrame({"set": s, "role": "site", "row": np.arange(len(a)), "gene": a.gene.values,
                          "noncodingRNA": a.noncodingRNA.values, "label": np.nan, "stratum": "noseed300",
                          "p150": p["interaction_probability"].values, "target": b.gene.values})[ns]
        rows = pd.concat([rows, x])
    rows["noncodingRNA"] = norm(rows.noncodingRNA)
    rows["target"] = norm(rows.target)
    rows = rows.reset_index(drop=True)
    rows.insert(0, "uid", np.arange(len(rows)))
    os.makedirs(OUT, exist_ok=True)
    rows.to_csv(f"{OUT}/rows.tsv.gz", sep="\t", index=False)
    print(rows.groupby(["set", "role"]).size().to_string(), f"\ntotal {len(rows):,}")


# ── IntaRNA ─────────────────────────────────────────────────────────────────
def decoy_pool():
    sr = pd.read_csv(SMALLRNA, sep="\t")
    sr = sr[sr.cpm_mean >= 10.0]
    seq = norm(sr.noncodingRNA).to_numpy()
    return seq, sr.cpm_mean.to_numpy(), np.array([s[1:8] for s in seq]), np.array([s.count("G") + s.count("C") for s in seq])


def queries(m, pool, nrng, prng):
    seq, cpm, seed, gc = pool
    g = m.count("G") + m.count("C")
    for tol in range(1, 10):
        ok = (seed != m[1:8]) & (np.abs(gc - g) <= tol)
        if ok.sum() >= 20:
            break
    dec = list(nrng.choice(seq[ok], size=N_EXPR, replace=False, p=cpm[ok] / cpm[ok].sum()))
    return [m] + dec + [dinuc_shuffle(m, prng) for _ in range(N_SHUF)]


def run_one(job):
    i, target, qs, tmpdir = job
    L = len(target)
    c0 = max(1, (L - 150) // 2 + 1)
    fa = os.path.join(tmpdir, f"q{i}.fa")
    with open(fa, "w") as fh:
        for j, q in enumerate(qs):
            fh.write(f">q{j}\n{q.replace('T', 'U')}\n")
    r = subprocess.run([INTARNA, "--target", target.replace("T", "U"), "--query", fa, "--tId=t",
                        f"--tRegion={c0}-{min(L, c0 + 149)}", *FLAGS], capture_output=True, text=True)
    os.remove(fa)
    res = np.full((7, NQ), np.nan)            # E, E_hybrid, ED1, start1, end1, start2, end2
    hyb = ""
    for line in r.stdout.splitlines()[1:]:
        f = line.split(";")
        if len(f) < 9:
            continue
        j = int(f[0][1:])
        res[:, j] = [float(f[6]), float(f[7]), float(f[8]), int(f[1]), int(f[2]), int(f[3]), int(f[4])]
        if j == 0:
            hyb = f[5]
    return i, res, hyb, r.returncode


def run(workers=40, limit=None):
    rows = pd.read_csv(f"{OUT}/rows.tsv.gz", sep="\t", low_memory=False)
    if limit:
        rows = rows.sample(limit, random_state=0).reset_index(drop=True)
    pool = decoy_pool()
    os.makedirs(f"{OUT}/chunks", exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="intarna.", dir=OUT)
    t0 = time.time()
    for c0 in range(0, len(rows), CHUNK):
        f = f"{OUT}/chunks/{c0:07d}.npz"
        if limit is None and os.path.exists(f):
            continue
        x = rows.iloc[c0:c0 + CHUNK]
        nrng, prng = np.random.default_rng(c0), random.Random(c0)
        jobs, Q = [], []
        for k, (t, m) in enumerate(zip(x.target, x.noncodingRNA)):
            qs = queries(m, pool, nrng, prng)
            Q.append(qs)
            jobs.append((k, t, qs, tmpdir))
        R = np.full((len(x), 7, NQ), np.nan)
        H = [""] * len(x)
        bad = 0
        with ThreadPoolExecutor(workers) as ex:
            for k, res, hyb, rc_ in ex.map(run_one, jobs, chunksize=16):
                R[k], H[k] = res, hyb
                bad += rc_ != 0
        if limit:
            dt = time.time() - t0
            print(f"bench {len(x)} rows in {dt:.1f}s with {workers} workers -> {dt / len(x) * 1e3:.1f} ms/row; "
                  f"failed calls {bad}; own E median {np.nanmedian(R[:, 0, 0]):.1f}", flush=True)
            return R, H, Q, x
        np.savez_compressed(f, uid=x.uid.values, R=R, H=np.array(H, dtype=object), Q=np.array(Q, dtype=object))
        el = time.time() - t0
        print(f"  chunk {c0:,}-{c0 + len(x):,} done ({bad} failed calls); {el / 3600:.2f} h elapsed", flush=True)
    os.rmdir(tmpdir)
    print("PAIRING_DONE", flush=True)


def collect():
    rows = pd.read_csv(f"{OUT}/rows.tsv.gz", sep="\t", low_memory=False).set_index("uid")
    parts = []
    nrng = np.random.default_rng(7)
    for f in sorted(glob.glob(f"{OUT}/chunks/*.npz")):
        z = np.load(f, allow_pickle=True)
        R, uid = z["R"], z["uid"]
        E = R[:, 0]
        o = pd.DataFrame({"uid": uid, "E_real": E[:, 0], "ED_real": R[:, 2, 0], "Eh_real": R[:, 1, 0],
                          "s1_real": R[:, 3, 0], "e1_real": R[:, 4, 0], "s2_real": R[:, 5, 0], "e2_real": R[:, 6, 0],
                          "hyb_real": z["H"]})
        o["E_expr_mean"] = np.nanmean(E[:, 1:1 + N_EXPR], axis=1)
        o["E_shuf_mean"] = np.nanmean(E[:, 1 + N_EXPR:], axis=1)
        o["p_expr"] = [rand_p(e[0], e[1:1 + N_EXPR], nrng) for e in E]
        o["p_shuf"] = [rand_p(e[0], e[1 + N_EXPR:], nrng) for e in E]
        o["p_all"] = [rand_p(e[0], e[1:], nrng) for e in E]
        parts.append(o)
    p = pd.concat(parts).set_index("uid")
    out = rows.join(p, how="inner")
    out.drop(columns=["target"]).to_csv(f"{OUT}/pairing.tsv.gz", sep="\t")
    print(f"collected {len(out):,} / {len(rows):,} rows")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        select()
    elif cmd == "bench":
        run(workers=int(sys.argv[3]) if len(sys.argv) > 3 else 40, limit=int(sys.argv[2]))
    elif cmd == "run":
        run(workers=int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 40)
    elif cmd == "collect":
        collect()
