#!/usr/bin/env python3
"""
Redraw beeswarm figures from saved .shap.npz matrices - no TreeSHAP recompute.

shap_beeswarm.py stores the full per-sample matrix alongside each figure precisely so
restyling costs seconds instead of the ~20 min the explanation itself takes.

Usage
-----
python3 replot_beeswarm.py results/shap/*.shap.npz [--top-n 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from shap_beeswarm import make_beeswarm


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", nargs="+", help="One or more .shap.npz files")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--model-name", default="LightGBMLarge_BAG_L1",
                   help="Model name shown in the title")
    args = p.parse_args()

    for path in args.npz:
        path = Path(path)
        d = np.load(path, allow_pickle=False)
        shap_values = d["shap_values"]
        features = [str(f) for f in d["features"]]
        n = len(shap_values)
        # ".shap.npz" is a double suffix; strip both to recover the run's stem.
        stem = path.with_suffix("").with_suffix("")
        png = stem.with_suffix(".beeswarm.png")
        make_beeswarm(shap_values, d["feature_values"], features,
                      float(d["baseline"]), png,
                      title=f"TreeSHAP - {args.model_name.replace('_', ' ')}  "
                            f"(n={n:,}, both classes)",
                      top_n=args.top_n)
        print(f"{path.name}  ->  {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
