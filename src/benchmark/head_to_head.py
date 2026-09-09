"""
Both models, every evaluation set, one table.

The poster asks whether sequence alone matches engineered thermodynamics, but
until now no panel put the two models on the same axis: the Pareto panel is
AutoGluon-only and the generalisation panel is CNN-only. This builds the
comparison, with the interval needed to read it honestly - the evaluation sets
span 954 rows to 324,171, so a raw APS difference between two of them says
almost nothing without knowing how precisely each was measured.

Confidence intervals are bootstrap percentiles over rows. APS is not a mean, so
it has no closed-form standard error; resampling the rows is the standard answer
and it is the only one that reflects the class balance of each particular set.
Resamples that come back single-class are dropped rather than scored, which can
only happen for the small sets and would otherwise inject a meaningless 0 or 1.

Inputs are prediction TSV/CSVs that already carry a probability column, so this
script never loads a model - the CNN files come from predict_cnn.py and the
AutoGluon files from predict_target.py.

Example
-------
python3 src/benchmark/head_to_head.py \
    --cnn "Manakov test:../msc-thesis/data/manakov_test_errors_v7_restructure.tsv" \
    --cnn "SAEC:../msc-thesis/results/novel/gse304955_labelled_pred.tsv" \
    --gluon "Manakov test:results/gluon_pred_manakov_test.tsv" \
    -o results/head_to_head.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# predict_cnn.py and predict_target.py both call it interaction_probability;
# the older restructure files repeat it as `prob`. First hit wins.
PROB_COLS = ["interaction_probability", "prob", "proba", "score"]
LABEL_COL = "label"

# Series names written to the output and read back by the figure. Named after
# what the model is rather than which library it happens to use, so a change of
# shipped algorithm does not silently invalidate an existing results file.
CNN_LABEL = "sequence CNN"
GLUON_LABEL = "IntaRNA features + CatBoost"


def _read(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def load(spec: str, labels: dict[str, Path]) -> tuple[str, np.ndarray, np.ndarray]:
    """Split a NAME:PATH spec and return (name, y_true, y_score).

    predict_target.py drops the label column before scoring, so its output has
    probabilities and no truth. `labels` supplies the source table for those
    files. The whole pipeline is row-index aligned by construction - make_fastas
    establishes the order and feature_extraction joins back on it - so the join
    is positional, but it is checked: lengths must match, and where both tables
    carry the sequence columns every row must agree. A silent misalignment here
    would score real probabilities against the wrong truth and look plausible.
    """
    if ":" not in spec:
        raise SystemExit(f"ERROR: '{spec}' is not in NAME:PATH form.")
    name, path = spec.rsplit(":", 1)
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"ERROR: no such file: {p}")
    df = _read(p)
    prob = next((c for c in PROB_COLS if c in df.columns), None)
    if prob is None:
        raise SystemExit(
            f"ERROR: {p} has no probability column (looked for {PROB_COLS}).")

    if LABEL_COL not in df.columns:
        src = labels.get(name)
        if src is None:
            raise SystemExit(
                f"ERROR: {p} has no '{LABEL_COL}' column and no --labels entry "
                f"for '{name}'. Pass --labels \"{name}:<the v7 TSV it was "
                f"scored from>\".")
        truth = _read(src)
        if len(truth) != len(df):
            raise SystemExit(
                f"ERROR: {src} has {len(truth)} rows but {p} has {len(df)}. "
                f"These are not the same set of pairs; refusing to join.")
        for col in ("gene", "noncodingRNA"):
            if col in df.columns and col in truth.columns:
                bad = int((df[col].astype(str).to_numpy()
                           != truth[col].astype(str).to_numpy()).sum())
                if bad:
                    raise SystemExit(
                        f"ERROR: {col} disagrees on {bad} rows between {p} and "
                        f"{src}; the row order does not match.")
        df = df.assign(**{LABEL_COL: truth[LABEL_COL].to_numpy()})

    d = df[[LABEL_COL, prob]].dropna()
    return name, d[LABEL_COL].to_numpy(), d[prob].to_numpy(dtype=float)


def aps_ci(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int = 0):
    """Point APS plus a percentile bootstrap interval over rows."""
    point = float(average_precision_score(y, s))
    if n_boot < 1:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        # A single-class resample has no defined average precision. Only ever
        # reachable for the small sets; scoring it would inject a spurious value.
        if yb.min() == yb.max():
            continue
        vals.append(average_precision_score(yb, s[idx]))
    if not vals:
        return point, np.nan, np.nan
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnn", action="append", default=[], metavar="NAME:PATH",
                    help="Sequence-CNN predictions for one set. Repeatable.")
    ap.add_argument("--gluon", action="append", default=[], metavar="NAME:PATH",
                    help="Feature-model predictions for one set. Repeatable.")
    ap.add_argument("--labels", action="append", default=[], metavar="NAME:PATH",
                    help="Truth table for a set whose prediction file carries no "
                         "label column - predict_target.py drops it. Joined by "
                         "row order, with the order verified. Repeatable.")
    ap.add_argument("--bootstrap", type=int, default=200, metavar="N",
                    help="Bootstrap resamples per entry (default 200). Cost is "
                         "N scorings of the whole set, so the big sets dominate; "
                         "0 disables intervals.")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("results/head_to_head.csv"))
    args = ap.parse_args()

    if not args.cnn and not args.gluon:
        raise SystemExit("ERROR: give at least one --cnn or --gluon entry.")

    labels: dict[str, Path] = {}
    for spec in args.labels:
        if ":" not in spec:
            raise SystemExit(f"ERROR: '{spec}' is not in NAME:PATH form.")
        n, pth = spec.rsplit(":", 1)
        labels[n] = Path(pth)

    rows: List[dict] = []
    for model, specs in ((CNN_LABEL, args.cnn), (GLUON_LABEL, args.gluon)):
        for spec in specs:
            name, y, s = load(spec, labels)
            point, lo, hi = aps_ci(y, s, args.bootstrap)
            rows.append({"model": model, "dataset": name, "n": len(y),
                         "pos_rate": float(y.mean()), "aps": point,
                         "ci_lo": lo, "ci_hi": hi})
            print(f"{model:<28} {name:<18} n={len(y):>7} "
                  f"pos={y.mean():5.1%} APS={point:.4f} "
                  f"[{lo:.4f}, {hi:.4f}]", flush=True)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWritten to: {args.output}")

    # The comparison the poster is actually making, where both models ran.
    both = out.pivot_table(index="dataset", columns="model", values="aps")
    if both.shape[1] == 2:
        both["delta (CNN - features)"] = both[CNN_LABEL] - both[GLUON_LABEL]
        print("\n" + both.to_string(float_format=lambda v: f"{v:+.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
