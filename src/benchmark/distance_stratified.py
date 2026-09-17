"""
Performance of each model as a function of how close a test miRNA is to training.

A family-disjoint split only promises that no test miRNA shares a miRBase family
label with a training miRNA. Family labels are curated, not a distance: two
families can share a seed shifted by one position, differ by a single seed
mismatch, or share most of the 3' end. A model can then score well on "unseen"
families by recognising near-copies of training miRNAs. This script measures
that directly, without re-splitting or retraining.

For every unique test miRNA it records its nearest training miRNA by:

  seed_cat     first match in order: identical sequence / same seed (nt 2-8) /
               offset seed (nt 2-7 of one == nt 3-8 of the other) / one mismatch
               in nt 2-8 / novel seed
  nn_identity  max normalised Levenshtein similarity to any training miRNA,
               1 - distance / max(len_a, len_b)
  seed_support number of training *pairs* whose miRNA has the same nt 2-8 seed

and then scores each model separately for the pairs in each bin.

Only the miRNA side is considered; target similarity is deliberately ignored.

Reading the output
------------------
Prevalence differs between bins, so raw APS is not comparable across them:
compare AUROC, or `aps_lift` (APS / prevalence). Intervals are percentile
bootstraps that resample *miRNAs*, not rows - pairs sharing a miRNA are
correlated, and a row bootstrap would give intervals that are far too narrow.
A drop in the far bins is only evidence of memorisation if it is larger than
the drop other models show in the same bins; a bin that is hard for every
model is simply a hard bin.

Inputs mirror head_to_head.py: prediction files carry a probability column,
the source table carries the sequences and labels, and the two are joined by
row order with the order verified.

Example
-------
pixi run --manifest-path dependencies/gluon/pixi.toml python3 src/benchmark/distance_stratified.py \\
    --train  ../msc-thesis/data/AGO2_eCLIP_Manakov2022_train_v7.tsv \\
    --source ../msc-thesis/data/AGO2_eCLIP_Manakov2022_leftout_v7.tsv \\
    --name   "Manakov leftout" \\
    --pred   "sequence CNN:../msc-thesis/data/manakov_leftout_errors_v7_restructure.tsv" \\
    --pred   "IntaRNA features + CatBoost:results/preds/gluon_manakov_leftout.tsv" \\
    --mirna-table results/distance_stratified_manakov_leftout_mirnas.csv \\
    -o results/distance_stratified_manakov_leftout.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from head_to_head import LABEL_COL, PROB_COLS, _read

MIRNA_COL = "noncodingRNA"

SEED_CATS = ["identical", "same seed", "offset seed", "1-mismatch seed", "novel seed"]
SUPPORT_EDGES = [0, 1, 100, 1000, 10000, np.inf]
SUPPORT_LABELS = ["0", "1-99", "100-999", "1000-9999", ">=10000"]


def _norm(seqs: pd.Series) -> pd.Series:
    return seqs.astype(str).str.upper().str.replace("U", "T", regex=False)


def seed_category(m: str, train_set: set, seeds: set,
                  seeds_2_7: set, seeds_3_8: set) -> str:
    if m in train_set:
        return "identical"
    s = m[1:8]
    if s in seeds:
        return "same seed"
    if m[1:7] in seeds_3_8 or m[2:8] in seeds_2_7:
        return "offset seed"
    for i in range(len(s)):
        for b in "ACGT":
            if b != s[i] and s[:i] + b + s[i + 1:] in seeds:
                return "1-mismatch seed"
    return "novel seed"


def nearest_identity(queries: list[str], refs: list[str]) -> tuple[np.ndarray, list[str]]:
    """Max normalised Levenshtein similarity of each query to any ref.

    The DP is vectorised across refs: one pass per query updates a
    (n_refs, max_ref_len + 1) row matrix, so ~1.2k training miRNAs cost a few
    milliseconds per query and no string-distance dependency is needed.
    """
    lut = np.zeros(256, dtype=np.uint8)
    for code, ch in enumerate("ACGT", start=1):
        lut[ord(ch)] = code
    lens = np.array([len(r) for r in refs])
    L = int(lens.max())
    R = np.zeros((len(refs), L), dtype=np.uint8)
    for k, r in enumerate(refs):
        R[k, :len(r)] = lut[np.frombuffer(r.encode(), dtype=np.uint8)]
    rows = np.arange(len(refs))

    best_sim = np.empty(len(queries))
    best_ref = []
    for qi, q in enumerate(queries):
        qc = lut[np.frombuffer(q.encode(), dtype=np.uint8)]
        prev = np.tile(np.arange(L + 1), (len(refs), 1))
        for i in range(1, len(q) + 1):
            cur = np.empty_like(prev)
            cur[:, 0] = i
            sub = (R != qc[i - 1]).astype(prev.dtype)
            for j in range(1, L + 1):
                cur[:, j] = np.minimum(np.minimum(prev[:, j] + 1, cur[:, j - 1] + 1),
                                       prev[:, j - 1] + sub[:, j - 1])
            prev = cur
        dist = prev[rows, lens]
        sim = 1.0 - dist / np.maximum(len(q), lens)
        k = int(sim.argmax())
        best_sim[qi] = sim[k]
        best_ref.append(refs[k])
    return best_sim, best_ref


def mirna_distance_table(train_mirnas: pd.Series, test_mirnas: pd.Series) -> pd.DataFrame:
    counts = train_mirnas.value_counts()
    refs = counts.index.tolist()
    seed_support = train_mirnas.str[1:8].value_counts()
    train_set = set(refs)
    seeds = {r[1:8] for r in refs}
    seeds_2_7 = {r[1:7] for r in refs}
    seeds_3_8 = {r[2:8] for r in refs}

    queries = sorted(test_mirnas.unique())
    sim, nn = nearest_identity(queries, refs)
    return pd.DataFrame({
        MIRNA_COL: queries,
        "seed_cat": [seed_category(q, train_set, seeds, seeds_2_7, seeds_3_8)
                     for q in queries],
        "nn_identity": sim,
        "nn_train_mirna": nn,
        "seed_support": [int(seed_support.get(q[1:8], 0)) for q in queries],
    })


def load_scores(spec: str, source: pd.DataFrame) -> tuple[str, np.ndarray]:
    """Split a MODEL:PATH spec and return (model, scores aligned to `source`)."""
    if ":" not in spec:
        raise SystemExit(f"ERROR: '{spec}' is not in NAME:PATH form.")
    name, path = spec.rsplit(":", 1)
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"ERROR: no such file: {p}")
    df = _read(p)
    prob = next((c for c in PROB_COLS if c in df.columns), None)
    if prob is None:
        raise SystemExit(f"ERROR: {p} has no probability column (looked for {PROB_COLS}).")
    if len(df) != len(source):
        raise SystemExit(f"ERROR: {p} has {len(df)} rows but the source has "
                         f"{len(source)}. These are not the same set of pairs.")
    for col in ("gene", MIRNA_COL, LABEL_COL):
        if col in df.columns and col in source.columns:
            a, b = df[col].astype(str).to_numpy(), source[col].astype(str).to_numpy()
            if col == MIRNA_COL:
                a, b = _norm(pd.Series(a)).to_numpy(), _norm(pd.Series(b)).to_numpy()
            bad = int((a != b).sum())
            if bad:
                raise SystemExit(f"ERROR: {col} disagrees on {bad} rows between {p} "
                                 f"and the source; the row order does not match.")
    return name, df[prob].to_numpy(dtype=float)


def _metrics(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    if y.min() == y.max():
        return np.nan, np.nan
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def bin_metrics(y: np.ndarray, s: np.ndarray, mirna: np.ndarray,
                n_boot: int, rng: np.random.Generator, min_mirnas: int) -> dict:
    ok = ~np.isnan(s)
    y, s, mirna = y[ok], s[ok], mirna[ok]
    prev = float(y.mean()) if len(y) else np.nan
    auroc, aps = _metrics(y, s)
    out = {"n": len(y), "n_mirnas": len(np.unique(mirna)), "n_pos": int(y.sum()),
           "prevalence": prev, "auroc": auroc, "auroc_lo": np.nan, "auroc_hi": np.nan,
           "aps": aps, "aps_lo": np.nan, "aps_hi": np.nan,
           "aps_lift": aps / prev if prev else np.nan}
    # With a handful of miRNAs the resampled set is nearly always one of a few
    # combinations and the percentile interval is noise, not uncertainty.
    if n_boot < 1 or np.isnan(auroc) or out["n_mirnas"] < min_mirnas:
        return out

    groups = pd.Series(np.arange(len(y))).groupby(mirna).indices
    idx_by_mirna = list(groups.values())
    au, ap = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_mirna), len(idx_by_mirna))
        idx = np.concatenate([idx_by_mirna[k] for k in pick])
        a, b = _metrics(y[idx], s[idx])
        # Single-class resamples have no defined score; skip rather than inject 0/1.
        if not np.isnan(a):
            au.append(a)
            ap.append(b)
    if au:
        out.update(auroc_lo=float(np.percentile(au, 2.5)), auroc_hi=float(np.percentile(au, 97.5)),
                   aps_lo=float(np.percentile(ap, 2.5)), aps_hi=float(np.percentile(ap, 97.5)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True,
                    help="Training table the models were fit on (needs noncodingRNA).")
    ap.add_argument("--source", type=Path, required=True,
                    help="Test table the predictions were made from "
                         "(needs noncodingRNA and label, in prediction row order).")
    ap.add_argument("--name", default=None, help="Dataset name for the output.")
    ap.add_argument("--pred", action="append", default=[], metavar="MODEL:PATH",
                    required=True, help="Predictions for the source set. Repeatable.")
    ap.add_argument("--identity-edges", default="0,0.6,0.7,0.8,0.9,1",
                    help="Bin edges for nn_identity, left-closed; 1.0 gets its own "
                         "bin (default 0,0.6,0.7,0.8,0.9,1).")
    ap.add_argument("--bootstrap", type=int, default=200, metavar="N",
                    help="miRNA-level bootstrap resamples per bin (default 200; 0 disables).")
    ap.add_argument("--min-mirnas", type=int, default=10,
                    help="Leave the interval blank for bins with fewer miRNAs (default 10).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mirna-table", type=Path, default=None,
                    help="Also write the per-miRNA distance table here.")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("results/distance_stratified.csv"))
    args = ap.parse_args()

    dataset = args.name or args.source.stem
    # Only the miRNA column: the training table is millions of rows.
    sep = "\t" if args.train.suffix in (".tsv", ".txt") else ","
    train_mirnas = _norm(pd.read_csv(args.train, sep=sep, usecols=[MIRNA_COL])[MIRNA_COL])
    source = _read(args.source)
    for col in (MIRNA_COL, LABEL_COL):
        if col not in source.columns:
            raise SystemExit(f"ERROR: {args.source} has no '{col}' column.")
    source[MIRNA_COL] = _norm(source[MIRNA_COL])

    print(f"train: {len(train_mirnas)} pairs, {train_mirnas.nunique()} miRNAs | "
          f"{dataset}: {len(source)} pairs, {source[MIRNA_COL].nunique()} miRNAs", flush=True)
    mt = mirna_distance_table(train_mirnas, source[MIRNA_COL])

    edges = [float(e) for e in args.identity_edges.split(",")]
    labels = [f"[{a:.2f}, {b:.2f})" for a, b in zip(edges[:-1], edges[1:])]
    mt["identity_bin"] = pd.cut(mt["nn_identity"], bins=edges + [np.inf], right=False,
                                labels=labels + ["1.00"])
    mt["seed_cat"] = pd.Categorical(mt["seed_cat"], categories=SEED_CATS, ordered=True)
    mt["support_bin"] = pd.cut(mt["seed_support"], bins=SUPPORT_EDGES, right=False,
                               labels=SUPPORT_LABELS)

    print("\nPer-miRNA seed category:")
    print(mt["seed_cat"].value_counts(sort=False).to_string())
    if args.mirna_table:
        args.mirna_table.parent.mkdir(parents=True, exist_ok=True)
        mt.to_csv(args.mirna_table, index=False)
        print(f"miRNA table written to: {args.mirna_table}")

    pairs = source[[MIRNA_COL, LABEL_COL]].merge(
        mt[[MIRNA_COL, "seed_cat", "identity_bin", "support_bin"]],
        on=MIRNA_COL, how="left", validate="many_to_one")
    y = pairs[LABEL_COL].astype(int).to_numpy()
    mirna = pairs[MIRNA_COL].to_numpy()

    rng = np.random.default_rng(args.seed)
    rows = []
    for spec in args.pred:
        model, s = load_scores(spec, source)
        rows.append({"dataset": dataset, "model": model, "stratifier": "all", "bin": "all",
                     **bin_metrics(y, s, mirna, args.bootstrap, rng, args.min_mirnas)})
        for strat in ("seed_cat", "identity_bin", "support_bin"):
            for b in pairs[strat].cat.categories:
                m = (pairs[strat] == b).to_numpy()
                if not m.any():
                    continue
                rows.append({"dataset": dataset, "model": model, "stratifier": strat,
                             "bin": str(b),
                             **bin_metrics(y[m], s[m], mirna[m],
                                           args.bootstrap, rng, args.min_mirnas)})

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    show = out[["model", "stratifier", "bin", "n", "n_mirnas", "prevalence",
                "auroc", "auroc_lo", "auroc_hi", "aps_lift"]]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
