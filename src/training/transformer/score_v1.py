#!/usr/bin/env python3
"""
Score the v1 recipes (A: old negatives, B: old + NONE negatives) side by side.

Every model is scored on the same rows, so differences are the models'.

  benchmark files   the existing v7 test sets, as they are (Manakov test and
                    leftout, Hejret, Klimentova, SAEC).  Reported overall and
                    split by `feature`: A and B saw only 3'UTR positives in
                    training, so `three_prime_utr` is the fair comparison and
                    `other` shows what UTR-only training costs.
  test_old          chr1 3'UTR positives against old negatives
  test_none         chr1 3'UTR positives against NONE windows, overall and per
                    negative kind (none_seed: the miRNA has a 7mer+ match there)
  test_wrong_mirna  each chr1 positive window re-paired with a wrong miRNA;
                    reports how far the score falls when only the miRNA changes
                    (paired_auc: share of pairs where the true miRNA scores higher)

APS depends on the positive share, which differs between files (test_none is
~85% positive): compare models within a row, not rows with each other.
AUROC does not move with the ratio.

Example
-------
python3 src/training/transformer/score_v1.py --model A:cnn_checkpoints/v1_A.pt --model B:cnn_checkpoints/v1_B.pt --model current:cnn_checkpoints/cnn_branches_mirbind_embed16_restruct.pt --benchmark manakov_test:../msc-thesis/data/AGO2_eCLIP_Manakov2022_test_v7.tsv --pairwise-dir results/pairwise_v1 --out results/v1_scores.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

_CNN = Path(__file__).resolve().parents[2] / "cnn"
sys.path.insert(0, str(_CNN))
from predict_cnn import score_dataframe  # noqa: E402

MRE, MIR = "gene", "noncodingRNA"


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    both = len(np.unique(y)) == 2
    return {"n": len(y), "pos_rate": float(y.mean()) if len(y) else np.nan,
            "aps": average_precision_score(y, p) if both else np.nan,
            "auroc": roc_auc_score(y, p) if both else np.nan}


def parse(specs):
    out = []
    for s in specs:
        name, _, path = s.partition(":")
        if not path:
            raise SystemExit(f"expected NAME:PATH, got {s!r}")
        out.append((name, path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME:PATH")
    ap.add_argument("--benchmark", action="append", default=[], metavar="NAME:PATH")
    ap.add_argument("--pairwise-dir", default="results/pairwise_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores-dir", default=None, help="also write per-row scores here")
    ap.add_argument("--max-rows", type=int, default=None, help="subsample each file (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    models = parse(args.model)

    files = [(n, p, "benchmark") for n, p in parse(args.benchmark)]
    pw = Path(args.pairwise_dir)
    for n in ("test_old", "test_none", "test_wrong_mirna"):
        if (pw / f"{n}.tsv").exists():
            files.append((n, str(pw / f"{n}.tsv"), n))

    rows = []
    for fname, path, kind in files:
        df = pd.read_csv(path, sep="\t")
        if kind != "test_wrong_mirna":
            # One score per chimeric pair, as the benchmark does; pairs in the
            # wrong-miRNA file are distinct by construction.
            key = df[MIR].str.upper() + df[MRE].str.upper()
            df = df[~key.duplicated()].reset_index(drop=True)
        if args.max_rows and len(df) > args.max_rows:
            if kind == "test_wrong_mirna":
                ids = df["pair_id"].drop_duplicates().sample(args.max_rows // 2, random_state=0)
                df = df[df["pair_id"].isin(ids)].reset_index(drop=True)
            else:
                df = df.sample(args.max_rows, random_state=0).reset_index(drop=True)
        print(f"[score] {fname}: {len(df):,} rows")
        for mname, ckpt in models:
            scored, _ = score_dataframe(ckpt, df, device=args.device, mre_col=MRE, mirna_col=MIR,
                                        batch_size=args.batch_size, num_workers=args.num_workers)
            p, y = scored["interaction_probability"].to_numpy(), scored["label"].to_numpy()
            if args.scores_dir:
                Path(args.scores_dir).mkdir(parents=True, exist_ok=True)
                scored.to_csv(Path(args.scores_dir) / f"{fname}__{mname}.tsv", sep="\t", index=False)

            def add(subset, mask, **extra):
                rows.append({"model": mname, "test": fname, "subset": subset,
                             **metrics(y[mask], p[mask]), **extra})

            add("all", np.ones(len(y), bool))
            if kind == "benchmark" and "feature" in scored:
                utr = scored["feature"].astype(str).eq("three_prime_utr").to_numpy()
                add("three_prime_utr", utr)
                add("other", ~utr)
            elif kind == "test_none":
                for neg in ("none_seed", "none_random"):
                    add(f"pos_vs_{neg}", (y == 1) | scored["neg_type"].eq(neg).to_numpy())
            elif kind == "test_wrong_mirna":
                w = scored.pivot_table(index="pair_id", columns="label",
                                       values="interaction_probability")
                drop = (w[1] - w[0]).to_numpy()
                rows.append({"model": mname, "test": fname, "subset": "paired",
                             "n": len(drop), "paired_auc": float((drop > 0).mean()),
                             "mean_drop": float(drop.mean()), "median_drop": float(np.median(drop))})

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for metric in ("aps", "auroc"):
            print(f"\n--- {metric}")
            print(out.pivot_table(index=["test", "subset"], columns="model", values=metric,
                                  sort=False).round(4).to_string())
        paired = out[out["subset"] == "paired"]
        if len(paired):
            print("\n--- wrong miRNA (same window, miRNA swapped)")
            print(paired.set_index("model")[["n", "paired_auc", "mean_drop", "median_drop"]]
                  .round(4).to_string())
    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
