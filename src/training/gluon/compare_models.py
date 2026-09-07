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

Timing repeats
--------------
One timed call per model is not enough to compare models whose costs are within a
small factor of each other. Two independent single-run timings of the same models on
the same leftout set (results/model_aps_manakov.csv against
results/model_aps_leftout.csv) disagree by 14-54% per model, and the
seconds_test/seconds_leftout ratio - which should be the constant n_test/n_leftout
for every model if timing were clean - instead spans 9.9x to 14.0x. That noise is
larger than several of the gaps the speed argument leans on.

`--repeats N` fixes that: N timed calls per model, reported as a median plus the
spread, with `--warmup` calls first to pay lazy model loading and page-cache costs
outside the measurement. Repeats are interleaved round-robin across models rather
than run model-by-model, so a noisy stretch of machine time is spread evenly instead
of landing entirely on whichever model happened to be running.

`--repeats 1 --warmup 0` (the default) reproduces the original single-shot behaviour
and writes exactly the original columns, so it stays comparable with the committed
results files. `--repeats 5 --warmup 1` is the recommended setting for any timing
that a conclusion depends on.

Example
-------
python3 src/training/gluon/compare_models.py \
    --model models_gluon \
    --data test:data/test.csv leftout:data/leftout.csv \
    --models LightGBM_BAG_L1 LightGBMLarge_BAG_L1 WeightedEnsemble_L3 \
    --repeats 5 --warmup 1
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
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
    repeats: int = 1,
    warmup: int = 0,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
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

    Returns the raw per-repeat timings rather than a summary, so the caller decides
    how to reduce them and nothing is thrown away before it is written out.
    """
    y_true = df['label']
    scores: Dict[str, float] = {}
    times: Dict[str, List[float]] = {name: [] for name in models}

    if fast:
        # Returns the base models it had to run as a side effect - keep what was asked.
        proba = predictor.predict_proba_multi(df, models=models)
        for name in models:
            scores[name] = _aps(y_true, proba[name])
        return scores, {}

    # Untimed first, so lazy model loading and cold page cache are paid outside the
    # measurement. APS comes from here when there is a warmup: it is deterministic,
    # so scoring it once keeps it out of the timed section entirely.
    for _ in range(warmup):
        for name in models:
            scores[name] = _aps(y_true, predictor.predict_proba(df, model=name))

    # Round-robin, not model-by-model: a slow stretch of machine time then hits every
    # model about equally instead of inflating whichever one it landed on.
    for rep in range(repeats):
        for name in models:
            t0 = time.perf_counter()
            proba = predictor.predict_proba(df, model=name)
            times[name].append(time.perf_counter() - t0)
            # Deterministic, so score once: an explicit check rather than
            # setdefault, whose argument would be evaluated on every repeat.
            if name not in scores:
                scores[name] = _aps(y_true, proba)
        if repeats > 1:
            print(f"    repeat {rep + 1}/{repeats} done")

    for name in models:
        print(f"    {name:<26} APS={scores[name]:.4f}  {_fmt_times(times[name])}")

    return scores, times


def _fmt_times(samples: List[float]) -> str:
    """One-line timing summary: the bare number for a single shot, else the spread."""
    if len(samples) == 1:
        return f"{samples[0]:8.1f}s"
    s = summarise_times(samples)
    return (f"median {s['']:7.2f}s  "
            f"[min {s['_min']:.2f}, IQR {s['_p25']:.2f}-{s['_p75']:.2f}, "
            f"n={int(s['_n'])}]")


def summarise_times(samples: List[float]) -> Dict[str, float]:
    """Reduce repeat timings to the columns written out.

    Median rather than mean: interference from other processes only ever adds time,
    so the distribution has a hard floor and a long right tail, and the mean chases
    the tail. `_min` is kept alongside it as the least-contaminated estimate, and the
    quartiles are what the figure draws as error bars.
    """
    arr = np.asarray(samples, dtype=float)
    return {
        '': float(np.median(arr)),
        '_min': float(arr.min()),
        '_p25': float(np.percentile(arr, 25)),
        '_p75': float(np.percentile(arr, 75)),
        '_n': float(len(arr)),
    }


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
    parser.add_argument('--repeats', type=int, default=1, metavar='N',
                        help='Timed predict_proba calls per model (default 1). With '
                             'N>1 the median is written to seconds_<set> and the '
                             'spread to seconds_<set>_{min,p25,p75,n}. Repeats are '
                             'interleaved across models. Use 5 or more for any '
                             'timing a conclusion depends on: single-shot timings of '
                             'these models vary by up to ~50%% run to run.')
    parser.add_argument('--warmup', type=int, default=0, metavar='N',
                        help='Untimed calls per model before timing starts (default '
                             '0), to pay lazy model loading and cold page cache '
                             'outside the measurement. Use 1 with --repeats.')

    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit(f"ERROR: --repeats must be at least 1, got {args.repeats}")
    if args.warmup < 0:
        raise SystemExit(f"ERROR: --warmup cannot be negative, got {args.warmup}")
    if args.fast and (args.repeats > 1 or args.warmup):
        raise SystemExit(
            "ERROR: --fast reports no per-model timings, so --repeats/--warmup have "
            "nothing to measure. Drop --fast to time models individually."
        )

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
    median_cols: List[str] = []   # the seconds_<set> columns, tracked rather than
                                  # pattern-matched so the spread columns added below
                                  # never get a speedup ratio of their own
    for name, df in datasets.items():
        print(f"\nScoring {name} "
              f"({args.repeats} timed call(s) per model, {args.warmup} warmup) ...")
        scores, times = aps_per_model(predictor, df, models, fast=args.fast,
                                      repeats=args.repeats, warmup=args.warmup)
        cols[f'APS_{name}'] = scores
        if times:
            summaries = {m: summarise_times(s) for m, s in times.items()}
            # One repeat writes just seconds_<set>, so the default run stays
            # column-identical to the files already in results/.
            suffixes = [''] if args.repeats == 1 else ['', '_min', '_p25', '_p75', '_n']
            for suffix in suffixes:
                cols[f'seconds_{name}{suffix}'] = {
                    m: s[suffix] for m, s in summaries.items()}
            median_cols.append(f'seconds_{name}')

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
        for col in median_cols:
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
