#!/usr/bin/env python3
"""Published miRBench predictors on the re-centred, ARTIFACT-dropped test sets.

Inputs: data/mirbench_recentre_noart/<set>.tsv (gene = re-centred 50-nt window, noncodingRNA, label), built from
data/flank_recentre/test/<set>.tsv (gene_rc) and the edge-0.25 masks data/flank_recentre/test_noart_hi025/<set>.tsv
(rows with drop_art kept out). Runs in the miRBench env (Python 3.8, the miRBench package from
https://github.com/katarinagresova/miRBench installed with its predictor dependencies):

    /mnt/raid1/rafail/envs/mirbench/bin/python src/figures/mirbench_recentre_noart_eval.py [predictor ...]

Writes data/mirbench_recentre_noart/pred_<predictor>_<set>.npy and results/mirbench_recentre_noart_evaluation.csv
(columns as results/mirbench_predictor_evaluation.csv). Random = uniform scores, numpy seed 0 (the miRBench package
has no Random predictor). Rows whose window has a non-ACGT base are left out of every predictor (one SAEC row).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = f"{ROOT}/data/mirbench_recentre_noart"
OUT = f"{ROOT}/results/mirbench_recentre_noart_evaluation.csv"
SETS = ["klimentova_test", "hejret_test", "manakov_test", "manakov_leftout", "gse304955", "gse263552",
        "gse180129_HSPC", "gse180129_Kasumi1"]
DEFAULT = ["miRBenchCNN_Manakov", "TargetScanCnn_McGeary2019", "Random"]
CHUNK = 50000


def load(s):
    """The set without rows whose window has a non-ACGT base (TargetScanCnn cannot encode N; one SAEC row)."""
    df = pd.read_csv(f"{D}/{s}.tsv", sep="\t")
    return df[~df.gene.str.upper().str.contains("[^ACGTU]")].reset_index(drop=True)


def predict(name, df):
    if name == "Random":
        return np.random.default_rng(0).random(len(df))
    from miRBench.encoder import get_encoder
    from miRBench.predictor import get_predictor
    enc, pr = get_encoder(name), get_predictor(name)
    out = [np.asarray(pr(enc(df.iloc[i:i + CHUNK].reset_index(drop=True)))).reshape(-1)
           for i in range(0, len(df), CHUNK)]
    return np.concatenate(out).astype(float)


def main():
    names = sys.argv[1:] or DEFAULT
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    done = {(r["dataset"], r["predictor"]) for r in rows if not isinstance(r.get("error"), str)}
    for name in names:
        for s in SETS:
            if (s, name) in done:
                continue
            df = load(s)
            t0, err = time.time(), ""
            try:
                p = predict(name, df)
                assert len(p) == len(df), (len(p), len(df))
                np.save(f"{D}/pred_{name}_{s}.npy", p)
                ap, auc = average_precision_score(df.label, p), roc_auc_score(df.label, p)
            except Exception as e:  # keep going, record the failure
                ap = auc = np.nan
                err = f"{type(e).__name__}: {e}"
            dt = round(time.time() - t0, 1)
            rows = [r for r in rows if (r["dataset"], r["predictor"]) != (s, name)]
            rows.append(dict(dataset=s, predictor=name, n_samples=len(df), n_positive=int(df.label.sum()),
                             n_negative=int((df.label == 0).sum()), average_precision=ap, roc_auc=auc,
                             runtime_seconds=dt, error=err))
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"{name:28} {s:18} n={len(df):>7,} AP={ap:.4f} AUC={auc:.4f} {dt:>7.1f}s {err}", flush=True)


if __name__ == "__main__":
    main()
