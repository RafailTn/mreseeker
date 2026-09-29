#!/usr/bin/env python3
"""
Is there a cooperativity signature in AGO2 binding data?

Cooperative miRNA sites sit roughly 8-40 nt apart (Grimson 2007; Saetrom 2007);
closer than ~8 nt two AGO2 complexes cannot both fit.  If binding (not just
repression) is cooperative, a seed match near an *observed* site should itself
be observed more often at 8-40 nt than at 200-400 nt, and less often below 8.

The test, on the per-UTR labels from build_utr_labels.py:

  candidates  every 7mer-m8 / 7mer-A1 / 8mer match, in the chosen 3'UTRs, of a
              seed that has at least --min-sites observed chimeras (the
              expressed repertoire).  Positions are the 6mer core (the target
              stretch paired with miRNA nt 2-7), so spacing is seed to seed.
  observed    a chimeric v7 window of the same seed contains the match.
  pairs       every two candidates in one UTR within --max-dist nt.  For each
              anchor, the partner's observation rate by distance bin.

Everything is compared inside UTRs, so transcript expression cancels.  What
does not cancel is local accessibility: an open, AU-rich stretch gets several
sites observed at once.  That gives enrichment falling smoothly with
distance; cooperativity plus steric exclusion gives a dip below ~8 nt, a bump
at 8-40, then flat.  The anchor-unobserved curve is the second control: a
hotspot lifts nearby partners whether or not the anchor was caught.

Pairs are split into same seed (homotypic, the classic two-site case) and
different seed.  Distances are measured between cores, so overlapping 50-nt
windows from one read pile-up do not create pairs; only separate matches do.
Two same-seed matches can share one chimeric window, which would mark both
observed from a single chimera; a partner only counts as observed through a
window that does not also contain the anchor.

Expression within a UTR is not flat (a long 3'UTR is often only partly used),
so nearby sites share expression more than distant ones, which alone makes
co-observation decay with distance.  With input_reads.npz from the builder,
every pair is also scored in an expression-matched set: both sites at
>= --min-input input reads, within --max-log2-diff of each other.

Output: a TSV of partner observation rates per bin and the ratio to the
200-400 nt baseline, with 95% Wilson intervals.

Example
-------
python3 src/training/transformer/site_spacing.py --labels-dir results/utr_labels --out results/site_spacing.tsv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [0, 8, 16, 26, 41, 61, 101, 201, 401]
LABELS = ["0-7", "8-15", "16-25", "26-40", "41-60", "61-100", "101-200", "201-400"]
BASELINE = "201-400"
_CODE = np.full(256, -1, dtype=np.int64)
for _i, _c in enumerate("ACGT"):
    _CODE[ord(_c)] = _i
_COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def kmer_code(s: str) -> int:
    v = 0
    for c in s:
        v = v * 4 + "ACGT".index(c)
    return v


def kmer_codes(seq: str, k: int = 7) -> np.ndarray:
    """Base-4 code of every k-mer; -1 where the k-mer holds a non-ACGT base."""
    a = _CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]
    n = len(a) - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    codes = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    for j in range(k):
        col = a[j:j + n]
        codes = codes * 4 + np.maximum(col, 0)
        bad |= col < 0
    codes[bad] = -1
    return codes


def read_fasta(path: Path) -> dict:
    seqs, tid, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if tid:
                    seqs[tid] = "".join(buf)
                tid, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip().upper())
    if tid:
        seqs[tid] = "".join(buf)
    return seqs


def wilson(k, n, z=1.96):
    n = np.maximum(n, 1)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", required=True, help="build_utr_labels.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="UTR splits to use; test stays untouched by default")
    ap.add_argument("--min-sites", type=int, default=50,
                    help="observed chimeras a seed needs to count as expressed")
    ap.add_argument("--max-dist", type=int, default=400)
    ap.add_argument("--window", type=int, default=50, help="chimeric window width")
    ap.add_argument("--min-input", type=int, default=10,
                    help="expression-matched set: input reads both sites need")
    ap.add_argument("--max-log2-diff", type=float, default=1.0,
                    help="expression-matched set: largest |log2| input ratio of a pair")
    args = ap.parse_args()
    d = Path(args.labels_dir)

    sites = pd.read_csv(d / "sites.tsv", sep="\t",
                        usecols=["transcript_id", "split", "role", "label", "tx_start",
                                 "tx_end", "mirna_seq"])
    pos = sites[(sites["label"] == 1) & (sites["role"] != "external")
                & sites["split"].isin(args.splits)].copy()
    pos["seed"] = pos["mirna_seq"].str.upper().str.replace("U", "T").str[1:8]
    counts = pos["seed"].value_counts()
    seeds = sorted(counts[counts >= args.min_sites].index)
    seed_id = {s: i for i, s in enumerate(seeds)}
    pos = pos[pos["seed"].isin(seed_id)]
    print(f"[seeds] {len(seeds)} expressed seeds (>= {args.min_sites} chimeras), "
          f"{len(pos):,} observed windows")

    # 7mer-m8 site = revcomp(nt 2-8), core at +1; 7mer-A1 = revcomp(nt 2-7) + A,
    # core at +0.  An 8mer hits both with the same core, and is counted once.
    targets = defaultdict(list)
    for s, i in seed_id.items():
        targets[kmer_code(revcomp(s))].append((i, 1))
        targets[kmer_code(revcomp(s[:6]) + "A")].append((i, 0))
    target_codes = np.array(sorted(targets), dtype=np.int64)

    windows = defaultdict(list)
    for t, s, a, b in zip(pos["transcript_id"], pos["seed"], pos["tx_start"], pos["tx_end"]):
        windows[(t, seed_id[s])].append((a, b))

    utrs = pd.read_csv(d / "utrs.tsv", sep="\t", usecols=["transcript_id", "split"])
    keep = set(utrs.loc[utrs["split"].isin(args.splits), "transcript_id"])
    seqs = {t: s for t, s in read_fasta(d / "utr3.fa").items() if t in keep}
    track_path = d / "input_reads.npz"
    tracks = np.load(track_path) if track_path.exists() else None
    if tracks is None:
        print("[input] no input_reads.npz: rerun build_utr_labels.py for the matched set")

    nb = len(LABELS)
    # [all/matched][anchor observed][same seed] -> per-bin pairs and observed partners
    n_pairs = np.zeros((2, 2, 2, nb), dtype=np.int64)
    n_obs = np.zeros((2, 2, 2, nb), dtype=np.int64)
    half = args.window // 2
    n_cand = n_cand_obs = 0
    edges = np.asarray(BINS)
    for t, seq in seqs.items():
        codes = kmer_codes(seq)
        hit = np.nonzero(np.isin(codes, target_codes))[0]
        if len(hit) < 2:
            continue
        cand = {}
        for p in hit:
            for sid, off in targets[int(codes[p])]:
                cand[(p + off, sid)] = True
        core = np.array([c for c, _ in cand], dtype=np.int64)
        sid = np.array([s for _, s in cand], dtype=np.int64)
        obs = np.zeros(len(core), dtype=bool)
        wins = [()] * len(core)
        for k, (c, s) in enumerate(zip(core, sid)):
            w = tuple((a, b) for a, b in windows.get((t, int(s)), ()) if a <= c and c + 6 <= b)
            obs[k], wins[k] = bool(w), w
        if tracks is not None and t in tracks.files:
            tr = tracks[t].astype(np.int64)
            expr = tr[np.clip(core - half + 3, 0, len(tr) - 1)]  # window centred on the core
        else:
            expr = np.zeros(len(core), dtype=np.int64)
        n_cand += len(core)
        n_cand_obs += int(obs.sum())
        order = np.argsort(core, kind="stable")
        core, sid, obs, expr = core[order], sid[order], obs[order], expr[order]
        wins = [wins[k] for k in order]
        # Ordered pairs (anchor i, partner j), both directions.
        for off in range(1, len(core)):
            dist = core[off:] - core[:-off]
            live = dist <= args.max_dist
            if not live.any():
                break
            i = np.nonzero(live)[0]
            j = i + off
            b = np.searchsorted(edges, dist[i], side="right") - 1
            same = (sid[i] == sid[j]).astype(np.int64)
            lo_e = np.minimum(expr[i], expr[j])
            matched = ((lo_e >= args.min_input)
                       & (np.abs(np.log2((expr[i] + 1) / (expr[j] + 1))) <= args.max_log2_diff))
            shared = np.nonzero(same.astype(bool) & (dist[i] < args.window))[0]
            for anc, par in ((i, j), (j, i)):
                a_obs = obs[anc].astype(np.int64)
                p_obs = obs[par].astype(np.int64)
                for q in shared:  # only a window without the anchor counts
                    ca = core[anc[q]]
                    p_obs[q] = int(any(not (a <= ca and ca + 6 <= bb) for a, bb in wins[par[q]]))
                for m, sel in ((0, slice(None)), (1, matched)):
                    np.add.at(n_pairs[m], (a_obs[sel], same[sel], b[sel]), 1)
                    np.add.at(n_obs[m], (a_obs[sel], same[sel], b[sel]), p_obs[sel])

    print(f"[candidates] {n_cand:,} seed matches, {n_cand_obs:,} observed "
          f"({n_cand_obs / max(n_cand, 1):.2%})")
    rows = []
    for m in (0, 1):
        for a_obs in (1, 0):
            for same in (1, 0):
                k, n = n_obs[m, a_obs, same], n_pairs[m, a_obs, same]
                rate = k / np.maximum(n, 1)
                lo, hi = wilson(k, n)
                base = rate[LABELS.index(BASELINE)]
                for bi, lab in enumerate(LABELS):
                    rows.append({
                        "set": "matched" if m else "all",
                        "anchor": "observed" if a_obs else "unobserved",
                        "pair": "same_seed" if same else "different_seed",
                        "distance": lab, "pairs": int(n[bi]), "partner_observed": int(k[bi]),
                        "rate": rate[bi], "rate_lo": lo[bi], "rate_hi": hi[bi],
                        "ratio_to_baseline": rate[bi] / base if base > 0 else np.nan,
                        "ratio_lo": lo[bi] / base if base > 0 else np.nan,
                        "ratio_hi": hi[bi] / base if base > 0 else np.nan})
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    with pd.option_context("display.width", 160):
        for name, g in out.groupby("set", sort=False):
            print(f"--- {name} pairs: partner observation rate / {BASELINE} nt baseline")
            print(g.pivot_table(index="distance", columns=["anchor", "pair"],
                                values="ratio_to_baseline", sort=False).round(2).to_string())
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
