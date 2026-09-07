#!/usr/bin/env python3
"""
Native LightGBM feature importance (gain + split) from a saved AutoGluon predictor.

Unlike permutation_importance.py this needs no labeled data - it reads the split
gains straight out of the trained boosters, averaged over bagging folds.

Usage
-----
python3 lgbm_importance.py --model models_gluon_lgbm \
    [--lgbm-model LightGBMLarge_BAG_L1]   # default: first LightGBM bag found
    [--output results/lgbm_importance.tsv]
    [--plot   results/figures/lgbm_importance.png]
    [--top-n  25]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


def _boosters(predictor: TabularPredictor, model_name: str | None):
    """Raw lgb.Booster objects behind `model_name` (one per bagging fold)."""
    import lightgbm as lgb

    names = [model_name] if model_name else list(predictor.model_names())
    for name in names:
        try:
            model = predictor._trainer.load_model(name)
        except Exception:
            continue
        if hasattr(model, "models") and hasattr(model, "load_child"):
            children = [getattr(model.load_child(c), "model", None) for c in model.models]
        else:
            children = [getattr(model, "model", None)]
        if children and all(isinstance(c, lgb.Booster) for c in children):
            return name, children
    return None, None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Saved AutoGluon predictor directory")
    p.add_argument("--lgbm-model", default=None,
                   help="Which model to read (default: first LightGBM bag found)")
    p.add_argument("--output", default="lgbm_importance.tsv", help="Output TSV path")
    p.add_argument("--plot", default=None, help="Optional PNG path for a bar plot")
    p.add_argument("--top-n", type=int, default=25, help="Features to plot (default: 25)")
    args = p.parse_args()

    predictor = TabularPredictor.load(args.model, require_version_match=False)
    name, boosters = _boosters(predictor, args.lgbm_model)
    if boosters is None:
        print(f"No LightGBM booster found in {args.model}.", file=sys.stderr)
        return 1

    features = boosters[0].feature_name()
    if any(b.feature_name() != features for b in boosters):
        print("Fold boosters disagree on column order; cannot average.", file=sys.stderr)
        return 1

    print(f"Reading importance from {name} ({len(boosters)} fold(s), "
          f"{len(features)} features)")

    gain = np.mean([b.feature_importance("gain") for b in boosters], axis=0)
    split = np.mean([b.feature_importance("split") for b in boosters], axis=0)

    tbl = pd.DataFrame({
        "feature": features,
        "gain": gain,
        "gain_pct": 100 * gain / gain.sum() if gain.sum() else 0.0,
        "split": split,
    }).sort_values("gain", ascending=False)
    tbl["gain_rank"] = np.arange(1, len(tbl) + 1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out, sep="\t", index=False, float_format="%.6f")
    print(f"\nImportance table -> {out}")
    print(tbl.head(10).to_string(index=False))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = tbl.head(args.top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.28 * len(top))))
        ax.barh(top["feature"], top["gain_pct"], color="#4C72B0")
        ax.set_xlabel("Importance (% of total split gain)")
        ax.set_title(f"LightGBM feature importance - {name}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300)
        print(f"Plot -> {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
