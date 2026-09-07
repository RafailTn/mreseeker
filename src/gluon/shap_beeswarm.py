#!/usr/bin/env python3
"""
TreeSHAP beeswarm for the LightGBM predictor on a stratified sample.

Unlike the -explain path in predict_target.py this explains BOTH classes (a beeswarm
built only from predicted positives shows what drives confident positives, not what
drives the decision) and keeps the full per-sample x per-feature matrix on disk so the
figure can be restyled without recomputing.

SHAP values are exact TreeSHAP in log-odds, averaged over the bagging folds.

Usage
-----
python3 shap_beeswarm.py \
    --model models_gluon_lgbm \
    --data  ~/Downloads/msc-thesis/data/manakov_test_v7_forgluon_56ft.csv \
    --out-prefix results/shap/manakov_test \
    [--n 25000] [--folds 8] [--jobs 20] [--top-n 20]
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Tokens that name nucleotides (AU, GU, UUU, GGG) or IUPAC purine/pyrimidine triplets
# (RYR, YYY) — these read as sequence content and belong in caps on a figure.
_NUC = re.compile(r"^(?:[acgu]{2,4}|[ry]{3})$")
# Molecule names that have a conventional casing of their own.
_TERMS = {"mirna": "miRNA", "mre": "MRE", "sd": "SD", "rna": "RNA"}
# Transcript-end shorthands, spelled the way the literature writes them.
_PRIME = {"3p": "3'", "5p": "5'", "3prime": "3'", "5prime": "5'"}
_PRIME_PAIRS = {("five", "prime"): "5'", ("three", "prime"): "3'"}


def prettify_feature(name: str) -> str:
    """Column name -> figure label: no underscores, nucleotides capitalised."""
    toks = name.split("_")
    out: list[str] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        # "five_prime_flank" is one concept spread over two tokens.
        if nxt is not None and (tok.lower(), nxt.lower()) in _PRIME_PAIRS:
            out.append(_PRIME_PAIRS[(tok.lower(), nxt.lower())])
            i += 2
            continue
        # "seed_2_8_pos" is a position range; plain spaces would read as "2 8".
        if tok.isdigit() and nxt is not None and nxt.isdigit():
            out.append(f"{tok}-{nxt}")
            i += 2
            continue
        if tok.lower() in _PRIME:
            out.append(_PRIME[tok.lower()])
        elif _NUC.fullmatch(tok):
            out.append(tok.upper())
        elif tok.lower() in _TERMS:
            out.append(_TERMS[tok.lower()])
        else:
            out.append(tok)
        i += 1
    return " ".join(out)


def make_beeswarm(shap_values, feature_values, features, baseline, png_path,
                  title: str, top_n: int = 20):
    """Beeswarm with left-aligned title so it clears the colour bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    expl = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(shap_values), baseline),
        data=feature_values,
        feature_names=[prettify_feature(f) for f in features],
    )
    plt.figure()
    shap.plots.beeswarm(expl, max_display=top_n, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, max(4, 0.34 * top_n))
    ax = fig.axes[0]
    ax.set_xlabel("SHAP value (log-odds)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # Anchored to the figure's left edge, not the axes': the plot axes start well to the
    # right (feature labels occupy that space), so an axes-anchored title still runs
    # into the colour bar. Figure coords put it clear of both.
    fig.suptitle(title, x=0.01, y=0.99, ha="left", va="top", fontsize=11)
    fig.savefig(png_path, dpi=300)
    return png_path


# Set by _init in each worker; on Linux (fork) this is inherited copy-on-write so the
# boosters are never pickled through the pipe.
_BOOSTERS: list = []


def _init(boosters):
    global _BOOSTERS
    _BOOSTERS = boosters


def _contrib_chunk(args):
    """SHAP contributions for one row-chunk, summed over folds."""
    fold_idx, X = args
    booster = _BOOSTERS[fold_idx]
    # pred_contrib returns (n, n_features + 1); the trailing column is the baseline.
    return fold_idx, booster.predict(X, pred_contrib=True, num_threads=1)


def _fair_allocation(available: pd.Series, budget: int) -> pd.Series:
    """
    Spread `budget` as evenly as possible over groups, capped by what each holds.

    Groups that cannot fill an equal share contribute everything they have and their
    shortfall is redistributed over the groups that still have capacity, repeating
    until the budget is met or the rows run out. This is what stops one abundant
    family from dominating: it is held to the same share as everyone else.
    """
    alloc = pd.Series(0, index=available.index, dtype=int)
    remaining = min(int(budget), int(available.sum()))
    while remaining > 0:
        headroom = available - alloc
        open_groups = headroom[headroom > 0]
        if open_groups.empty:
            break
        share = max(1, remaining // len(open_groups))
        take = open_groups.clip(upper=share)
        # Never overshoot the budget on the final partial pass.
        if take.sum() > remaining:
            take = take.sort_values()
            csum = take.cumsum()
            take = take[csum <= remaining]
            if take.empty:
                # Budget smaller than one row per group: hand it out one at a time.
                take = pd.Series(1, index=open_groups.index[:remaining])
        alloc.loc[take.index] += take
        remaining -= int(take.sum())
    return alloc


def stratified_sample(df: pd.DataFrame, label_col: str, n: int, seed: int,
                      balance_by: str | None = None) -> pd.DataFrame:
    """
    Equal-sized draw per class, capped by the rarest class.

    With `balance_by` set (e.g. mir_fam) the per-class budget is further spread evenly
    across that column's groups rather than drawn at random, so an abundant miRNA
    family cannot swamp the sample. Note this changes the estimand: mean |SHAP| then
    describes the average FAMILY, not the average interaction.
    """
    classes = sorted(df[label_col].unique())
    per_class = n // len(classes)
    parts = []
    for c in classes:
        sub = df[df[label_col] == c]
        take = min(per_class, len(sub))
        if take < per_class:
            print(f"  class {c}: only {len(sub)} rows available, taking all")
        if balance_by is None:
            parts.append(sub.sample(n=take, random_state=seed))
            continue
        groups = sub[balance_by].fillna("__NA__").astype(str)
        available = groups.value_counts()
        alloc = _fair_allocation(available, take)
        for g, k in alloc[alloc > 0].items():
            parts.append(sub[groups == g].sample(n=int(k), random_state=seed))
        used = int(alloc.sum())
        print(f"  class {c}: {used} rows over {int((alloc > 0).sum())} "
              f"{balance_by} groups (max {int(alloc.max())}/group)")
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)  # shuffle
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Saved AutoGluon predictor directory")
    p.add_argument("--data", required=True, help="Feature CSV containing the label column")
    p.add_argument("--out-prefix", required=True,
                   help="Path stem for the .npz matrix, .tsv summary and .png figure")
    p.add_argument("--n", type=int, default=25000, help="Total sampled rows (default 25000)")
    p.add_argument("--label", default="label", help="Label column (default: label)")
    p.add_argument("--balance-by", default=None,
                   help="Column (e.g. mir_fam) to spread the sample evenly across, so "
                        "abundant groups do not dominate. Default: plain random draw.")
    p.add_argument("--folds", type=int, default=None,
                   help="Use only the first N bagging folds (default: all)")
    p.add_argument("--jobs", type=int, default=os.cpu_count(),
                   help="Worker processes (default: all cores)")
    p.add_argument("--top-n", type=int, default=20, help="Features to plot (default: 20)")
    p.add_argument("--chunk", type=int, default=1000, help="Rows per work unit")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import lightgbm as lgb  # noqa: F401  (needed for the isinstance check downstream)
    from autogluon.tabular import TabularPredictor

    # -- Model -----------------------------------------------------------------
    predictor = TabularPredictor.load(args.model, require_version_match=False)
    bag = predictor._trainer.load_model(predictor.model_best)
    if not (hasattr(bag, "models") and hasattr(bag, "load_child")):
        print(f"{predictor.model_best} is not a bagged model.", file=sys.stderr)
        return 1
    boosters = [bag.load_child(c).model for c in bag.models]
    if args.folds:
        boosters = boosters[:args.folds]
    features = boosters[0].feature_name()
    print(f"Model {predictor.model_best}: {len(boosters)} fold(s), {len(features)} features")

    # -- Data ------------------------------------------------------------------
    print(f"Loading {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    if args.label not in df.columns:
        print(f"Label column '{args.label}' not in {args.data}.", file=sys.stderr)
        return 1

    print(f"Stratified sample: {args.n} rows from {len(df)}")
    if args.balance_by and args.balance_by not in df.columns:
        print(f"--balance-by column '{args.balance_by}' not in {args.data}.",
              file=sys.stderr)
        return 1
    sample = stratified_sample(df, args.label, args.n, args.seed, args.balance_by)
    y = sample[args.label].to_numpy()
    print("  class counts:", dict(zip(*np.unique(y, return_counts=True))))

    # transform_features applies the same preprocessing the model saw at fit time;
    # feeding raw columns straight to the booster would silently misalign them.
    X_model = predictor.transform_features(sample.drop(columns=[args.label]))
    missing = [f for f in features if f not in X_model.columns]
    if missing:
        print(f"Features missing after transform: {missing}", file=sys.stderr)
        return 1
    X_model = X_model[features].astype(np.float64)

    # -- TreeSHAP, parallel over (fold, row-chunk) -----------------------------
    from multiprocessing import Pool

    n_rows = len(X_model)
    starts = [i for _ in range(len(boosters)) for i in range(0, n_rows, args.chunk)]
    chunks = [(f, X_model.iloc[i:i + args.chunk])
              for f in range(len(boosters))
              for i in range(0, n_rows, args.chunk)]
    print(f"Computing TreeSHAP: {n_rows} rows x {len(boosters)} folds "
          f"= {len(chunks)} work units on {args.jobs} process(es)")

    t0 = time.time()
    acc = np.zeros((n_rows, len(features) + 1), dtype=np.float64)
    with Pool(args.jobs, initializer=_init, initargs=(boosters,)) as pool:
        done = 0
        # imap preserves input order, so result idx lines up with starts[idx].
        for idx, (fold_idx, out) in enumerate(
                pool.imap(_contrib_chunk, chunks, chunksize=1)):
            start = starts[idx]
            acc[start:start + len(out)] += out
            done += 1
            if done % 50 == 0 or done == len(chunks):
                el = time.time() - t0
                print(f"  {done}/{len(chunks)} units  {el:.0f}s elapsed  "
                      f"~{el / done * (len(chunks) - done):.0f}s left", flush=True)

    acc /= len(boosters)
    shap_values = acc[:, :-1]
    baseline = float(acc[:, -1].mean())
    print(f"Done in {time.time() - t0:.0f}s. Baseline E[f(x)] = {baseline:.4f} (log-odds)")

    # -- Persist ---------------------------------------------------------------
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.balance_by:
        comp = (sample[args.balance_by].fillna("__NA__").astype(str)
                .value_counts().rename_axis(args.balance_by).reset_index(name="n_sampled"))
        comp_path = out_prefix.with_suffix(".sample_composition.tsv")
        comp.to_csv(comp_path, sep="\t", index=False)
        print(f"Sample composition -> {comp_path}")

    npz_path = out_prefix.with_suffix(".shap.npz")
    np.savez_compressed(npz_path, shap_values=shap_values,
                        feature_values=X_model.to_numpy(),
                        features=np.array(features), label=y, baseline=baseline)
    print(f"SHAP matrix -> {npz_path}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    summary = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs,
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    summary["rank"] = np.arange(1, len(summary) + 1)
    tsv_path = out_prefix.with_suffix(".shap_summary.tsv")
    summary.to_csv(tsv_path, sep="\t", index=False, float_format="%.6f")
    print(f"Summary -> {tsv_path}")
    print(summary.head(10).to_string(index=False))

    # -- Beeswarm --------------------------------------------------------------
    png_path = out_prefix.with_suffix(".beeswarm.png")
    make_beeswarm(shap_values, X_model.to_numpy(), features, baseline, png_path,
                  title=f"TreeSHAP - {predictor.model_best.replace('_', ' ')}  "
                        f"(n={n_rows:,}, both classes)",
                  top_n=args.top_n)
    print(f"Beeswarm -> {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
