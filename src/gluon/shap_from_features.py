"""
TreeSHAP for one AutoGluon model, straight from a pre-extracted feature table.

`predict_target.py -explain` recomputes the whole pipeline - IntaRNA, the
suboptimal ensemble, conservation - before it can explain anything. When the
features already exist (the *_forgluon_56ft.csv tables), all of that is wasted:
SHAP needs the feature matrix and the model, nothing else. On the Manakov test
set that is the difference between hours and a couple of minutes.

Family-balanced sampling
------------------------
The test set is dominated by a handful of abundant miRNA families, so an
unweighted sample lets those families decide the feature ranking - which turns a
claim about mechanism into a claim about composition. `--family-balanced` draws
an equal quota per family instead, and writes what it drew alongside the values.

Writes the same .npz layout the figure scripts already read:
shap_values, feature_values, features, label, baseline.

Example
-------
python3 src/gluon/shap_from_features.py \
    --model models/gluon_total_try --model-name CatBoost_BAG_L1 \
    --features ../msc-thesis/data/manakov_test_v7_forgluon_56ft.csv \
    --family-balanced -n 25000 \
    -o results/shap/manakov_test_fambal_catboost
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "training" / "gluon"))

from predict_target import _tree_boosters, compute_tree_shap  # noqa: E402
from feature_extraction import SELECTED_26  # noqa: E402
from gluon_train_total import SEQUENCE_COLS, preprocess_dataframe  # noqa: E402

FAM_COL = "mir_fam"


def sample(df: pd.DataFrame, n: int, balanced: bool, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    if not balanced or FAM_COL not in df.columns:
        return df.iloc[rng.choice(len(df), n, replace=False)]
    # Equal quota per family, capped by what each family actually has; the
    # shortfall from small families is redistributed by simply taking more from
    # the rest, so the total still lands at n.
    fams = df[FAM_COL].fillna("unknown")
    groups = {k: idx.to_numpy() for k, idx in fams.groupby(fams).groups.items()}
    quota = max(1, n // len(groups))
    picked: list[np.ndarray] = []
    for g in groups.values():
        take = min(quota, len(g))
        picked.append(rng.choice(g, take, replace=False))
    got = np.concatenate(picked)
    if len(got) < n:                       # top up from whatever is left
        rest = np.setdiff1d(df.index.to_numpy(), got)
        if len(rest):
            got = np.concatenate([got, rng.choice(
                rest, min(n - len(got), len(rest)), replace=False)])
    return df.loc[got[:n]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="TabularPredictor directory")
    ap.add_argument("--model-name", default=None,
                    help="Model inside the predictor (default: its best).")
    ap.add_argument("--features", required=True,
                    help="Pre-extracted *_forgluon_*.csv feature table.")
    ap.add_argument("-n", type=int, default=25000, help="Rows to explain.")
    ap.add_argument("--family-balanced", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", required=True,
                    help="Output basename; writes <base>.shap.npz and "
                         "<base>.shap_summary.tsv.")
    args = ap.parse_args()

    predictor = TabularPredictor.load(args.model)
    name = args.model_name or predictor.model_best
    if name not in predictor.model_names():
        raise SystemExit(f"ERROR: '{name}' not in this predictor.\n"
                         f"Available: {predictor.model_names()}")

    found = _tree_boosters(predictor, name)
    if not found:
        raise SystemExit(
            f"ERROR: '{name}' is not a bagged tree model TreeSHAP can walk "
            f"(LightGBM or CatBoost). Use predict_target.py -explain, which "
            f"falls back to a sampled KernelExplainer.")
    boosters, feat_names = found

    raw = pd.read_csv(args.features)
    keep = raw[[FAM_COL]].copy() if FAM_COL in raw.columns else pd.DataFrame(index=raw.index)
    df, _ = preprocess_dataframe(raw, SEQUENCE_COLS, list(SELECTED_26))
    df = df.drop(columns=[FAM_COL], errors="ignore")
    df = df.join(keep, how="left")
    print(f"{len(df)} rows, {int(df.label.sum())} positives")

    picked = sample(df, args.n, args.family_balanced, args.seed)
    labels = picked["label"].to_numpy()
    X = picked.drop(columns=["label", FAM_COL], errors="ignore")
    print(f"explaining {len(picked)} rows with {name} "
          f"({len(boosters)} bagged folds)")

    vals, baseline, cols = compute_tree_shap(predictor, boosters, X, feat_names)
    feat_vals = predictor.transform_features(X)[cols].to_numpy(dtype=float)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".shap.npz"),
                        shap_values=vals, feature_values=feat_vals,
                        features=np.array(cols), label=labels,
                        baseline=np.float64(baseline))
    order = np.argsort(np.abs(vals).mean(axis=0))[::-1]
    pd.DataFrame({"feature": [cols[i] for i in order],
                  "mean_abs_shap": np.abs(vals).mean(axis=0)[order],
                  "mean_shap": vals.mean(axis=0)[order],
                  "rank": np.arange(1, len(cols) + 1)}).to_csv(
        out.with_suffix(".shap_summary.tsv"), sep="\t", index=False)
    if args.family_balanced and FAM_COL in picked.columns:
        picked[FAM_COL].value_counts().rename_axis(FAM_COL).to_csv(
            str(out) + ".sample_composition.tsv", sep="\t", header=["n"])
    print(f"wrote {out}.shap.npz and {out}.shap_summary.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
