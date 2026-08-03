"""
Per-model APS comparison for a trained AutoGluon predictor.

The stacked ensemble that `gluon_train_total.py` produces is expensive at inference:
every L1 base model has to run before any L2 stacker or the weighted ensemble can
score a row. A single bagged LightGBM (LightGBM_BAG_L1) is self-contained and orders
of magnitude faster. This script quantifies what that costs in average precision, on
the same held-out sets the training script reports on.

APS is computed exactly like `evaluate_df` in gluon_train_total.py - unweighted
sklearn `average_precision_score` on the positive-class probability - so the numbers
line up with the ones already written to the results files.

Example
-------
python3 src/training/gluon/compare_models.py \
    --model models_gluon \
    --data test:data/test.csv leftout:data/leftout.csv \
    --models LightGBM_BAG_L1 LightGBMLarge_BAG_L1 WeightedEnsemble_L3
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import average_precision_score

from autogluon.tabular import TabularPredictor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'gluon'))
from feature_extraction import SELECTED_26  # noqa: E402

from gluon_train_total import SEQUENCE_COLS, preprocess_dataframe  # noqa: E402


def load_eval_set(path: str) -> pd.DataFrame:
    """Apply the training-time preprocessing so the columns match what the model saw."""
    raw = pd.read_csv(path)
    df, _ = preprocess_dataframe(raw, SEQUENCE_COLS, list(SELECTED_26))
    return df.drop(columns=['mir_fam'], errors='ignore')


def _aps(y_true: pd.Series, proba: pd.DataFrame) -> float:
    pos_col = 1 if 1 in proba.columns else True
    return float(average_precision_score(y_true, proba[pos_col]))


def aps_per_model(
    predictor: TabularPredictor,
    df: pd.DataFrame,
    models: List[str],
    fast: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    APS - and, unless `fast`, wall-clock inference time - per model on one dataset.

    Two modes, because the two things being measured want opposite call patterns:

    `fast=False` (default) gives each model its own isolated `predict_proba` call, so
    the elapsed time is the real deployment cost of serving with that model alone -
    a stacker or weighted ensemble pays for its full base layer, LightGBM_BAG_L1 pays
    only for itself. That is the number the speed argument rests on, and it only
    exists if the models do not share work.

    `fast=True` uses `predict_proba_multi`, which scores everything in one pass by
    reusing each L1 model's output across the L2 stackers. Much cheaper when scoring
    all 25 models, but it makes per-model timing meaningless, so none is reported.
    """
    y_true = df['label']
    scores: Dict[str, float] = {}
    times: Dict[str, float] = {}

    if fast:
        # Returns the base models it had to run as a side effect - keep what was asked.
        proba = predictor.predict_proba_multi(df, models=models)
        for name in models:
            scores[name] = _aps(y_true, proba[name])
        return scores, times

    for name in models:
        t0 = time.perf_counter()
        proba = predictor.predict_proba(df, model=name)
        times[name] = time.perf_counter() - t0
        scores[name] = _aps(y_true, proba)
        print(f"    {name:<26} APS={scores[name]:.4f}  {times[name]:8.1f}s")

    return scores, times


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare per-model average precision against the full stacked ensemble."
    )
    parser.add_argument('--model', required=True,
                        help='Path to the saved AutoGluon TabularPredictor directory')
    parser.add_argument('--data', nargs='+', required=True, metavar='NAME:PATH',
                        help='One or more evaluation sets as name:path '
                             '(e.g. test:data/test.csv leftout:data/leftout.csv)')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Model names to score. Default: every model in the '
                             'predictor. Names come from predictor.model_names().')
    parser.add_argument('-o', '--output',
                        help='Optional CSV to write the model x dataset APS table to')
    parser.add_argument('--fast', action='store_true',
                        help='Score all models in one shared pass. Much faster when '
                             'comparing many models, but reports no inference times.')

    args = parser.parse_args()

    predictor = TabularPredictor.load(args.model)
    available = predictor.model_names()

    models = args.models if args.models else available
    unknown = [m for m in models if m not in available]
    if unknown:
        raise SystemExit(
            f"ERROR: not in this predictor: {unknown}\nAvailable: {available}"
        )

    datasets: Dict[str, pd.DataFrame] = {}
    for spec in args.data:
        if ':' not in spec:
            raise SystemExit(f"ERROR: --data entry '{spec}' is not in name:path form.")
        name, path = spec.split(':', 1)
        datasets[name] = load_eval_set(path)
        print(f"{name}: {len(datasets[name])} rows, "
              f"{int(datasets[name]['label'].sum())} positives")

    cols: Dict[str, Dict[str, float]] = {}
    for name, df in datasets.items():
        print(f"\nScoring {name} ...")
        scores, times = aps_per_model(predictor, df, models, fast=args.fast)
        cols[f'APS_{name}'] = scores
        if times:
            cols[f'seconds_{name}'] = times

    table = pd.DataFrame(cols)

    # Rank by the first dataset given - conventionally the primary held-out test set.
    aps_cols = [c for c in table.columns if c.startswith('APS_')]
    table = table.sort_values(aps_cols[0], ascending=False)

    best = predictor.model_best
    if best in table.index:
        # The delta column is what the decision actually hinges on: how much APS a
        # single-model deployment gives up relative to the model in use today. Timing
        # columns get a speedup ratio instead, which is the other half of that trade.
        for col in aps_cols:
            table[f'{col}_delta'] = table[col] - table.loc[best, col]
        for col in [c for c in table.columns if c.startswith('seconds_')]:
            table[f'{col}_speedup'] = table.loc[best, col] / table[col]

    print(f"\nAPS per model (best model in predictor: {best})")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        table.to_csv(args.output, index_label='model')
        print(f"\nWritten to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
