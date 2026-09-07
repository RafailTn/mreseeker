#!/usr/bin/env python3
"""
End-to-end wall-clock cost of the two prediction pipelines, stage by stage.

Why this exists
---------------
`compare_models.py` times `predict_proba` on a CSV of features that already
exist. That is the right measurement for choosing between AutoGluon candidates,
which all consume the same features - but it is the wrong one for comparing the
AutoGluon path against the CNN, because it silently omits the step that makes
them different. The AutoGluon model needs IntaRNA, a conservation lookup and a
shuffle-background pass before it can score anything; the CNN needs the two
sequences. Putting a CNN point on an axis that excludes IntaRNA would compare
the CNN's whole pipeline against the LightGBM's last step and invert the
conclusion.

So this measures everything from "two FASTAs on disk" to "a probability per
pair", broken down by stage, for whichever pipeline is asked for.

One pipeline per invocation, because they live in different environments
--------------------------------------------------------------------------
dependencies/gluon and dependencies/cnn are deliberately disjoint (see the
README: featurewiz/autogluon pin incompatible xgboost ranges), so no single
interpreter can import both. Running each pipeline under `pixi run` from a
neutral parent would fold pixi startup and a multi-second `import torch` into
the measurement, which would swamp the CNN's actual forward pass. Instead the
same script runs twice, once in each environment, writing one CSV each:

    pixi run --manifest-path dependencies/gluon/pixi.toml python3 \\
        src/benchmark/pipeline_timing.py --pipeline gluon \\
        --mre-fasta mre.fa --mirna-fasta mirna.fa --conservation-tsv sites.tsv \\
        --gluon-model models_gluon_lgbm \\
        --mirna-background data/mirna_background.tsv \\
        --panel-fasta data/mirna_background.fa \\
        --threads 8 --repeats 3 -o results/timing_gluon.csv

    pixi run --manifest-path dependencies/cnn/pixi.toml python3 \\
        src/benchmark/pipeline_timing.py --pipeline cnn \\
        --mre-fasta mre.fa --mirna-fasta mirna.fa --conservation-tsv sites.tsv \\
        --cnn-checkpoint cnn_checkpoints/cnn_branches_mirbind_embed16_restruct.pt \\
        --cnn-device cpu --repeats 5 --warmup 1 -o results/timing_cnn.csv

Reading the numbers
-------------------
Repeats follow the protocol in compare_models.py: N timed passes, median
reported with the spread, `--warmup` untimed passes first. A gluon pass runs
IntaRNA twice over the whole input, so repeats there are expensive - 3 is
usually enough to see whether a stage is stable, and warmup is off by default
for that reason.

`--cnn-device` matters and is recorded in the sidecar: a CUDA forward pass
against a CPU-bound LightGBM is not a like-for-like comparison, so pass
`--cnn-device cpu` for the headline figure and treat a GPU run as a separate
claim.

Model load is timed as its own stage and is EXCLUDED from the reported total: a
served model is loaded once and scores many batches, so folding it into a
per-batch cost would flatter whichever model loads faster. It is in the CSV if
you want it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_GLUON_DIR = _REPO / "src" / "gluon"
_CNN_DIR = _REPO / "src" / "cnn"

# Stages excluded from the end-to-end total. Loading is amortised across every
# batch a served process handles, so charging it per run is misleading.
_EXCLUDE_FROM_TOTAL = {"model_load"}


# ---------------------------------------------------------------------------
# Timing helpers - the protocol is the one in compare_models.py
# ---------------------------------------------------------------------------

def summarise_times(samples: List[float]) -> Dict[str, float]:
    """Reduce repeat timings to the columns written out.

    Median rather than mean: interference from other processes only ever adds
    time, so the distribution has a hard floor and a long right tail, and the
    mean chases the tail. `min` is the least-contaminated estimate and the
    quartiles are what a figure draws as error bars.
    """
    arr = np.asarray(samples, dtype=float)
    return {
        "seconds": float(np.median(arr)),
        "seconds_min": float(arr.min()),
        "seconds_p25": float(np.percentile(arr, 25)),
        "seconds_p75": float(np.percentile(arr, 75)),
        "seconds_n": float(len(arr)),
    }


class Stopwatch:
    """Collects one stage->seconds mapping per pass."""

    def __init__(self) -> None:
        self.stages: Dict[str, float] = {}

    def run(self, cmd: List[str], stage: str) -> None:
        """Time a subprocess, failing loudly - a crashed stage must not be
        recorded as a fast one."""
        t0 = time.perf_counter()
        result = subprocess.run(cmd)
        self.stages[stage] = time.perf_counter() - t0
        if result.returncode != 0:
            raise SystemExit(
                f"ERROR: stage '{stage}' exited {result.returncode}\n"
                f"  {' '.join(cmd)}"
            )

    def time(self, stage: str):
        """Context manager for an in-process stage."""
        return _StageTimer(self, stage)


class _StageTimer:
    def __init__(self, watch: Stopwatch, stage: str) -> None:
        self.watch, self.stage = watch, stage

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.watch.stages[self.stage] = time.perf_counter() - self.t0
        return False


# ---------------------------------------------------------------------------
# The two pipelines
# ---------------------------------------------------------------------------

def run_gluon_pass(args, tmp: Path) -> tuple[Dict[str, float], np.ndarray]:
    """One full FASTA-to-probability pass of the AutoGluon pipeline.

    The stage list mirrors src/gluon/predict_target.py exactly, including its
    use of subprocesses - those interpreter startups are part of the pipeline as
    it actually ships, so they are timed rather than engineered away.
    """
    from autogluon.tabular import TabularPredictor  # noqa: PLC0415

    w = Stopwatch()
    intarna = tmp / "intarna.tsv"
    intarna_ens = tmp / "intarna_ensemble.tsv"
    merged = tmp / "merged.tsv"
    best = tmp / "best.tsv"
    feats = tmp / "features.csv"

    w.run(["python3", str(_GLUON_DIR / "intarna_parallel.py"),
           args.mre_fasta, args.mirna_fasta, "-o", str(intarna),
           "--threads", str(args.threads)], "intarna")

    w.run(["python3", str(_GLUON_DIR / "intarna_parallel.py"),
           args.mre_fasta, args.mirna_fasta, "-o", str(intarna_ens),
           "--threads", str(args.threads), "--ensemble"], "intarna_ensemble")

    w.run(["python3", str(_GLUON_DIR / "merge_intarna.py"),
           "-m", str(intarna), "-e", str(intarna_ens), "-o", str(merged)], "merge")

    w.run(["python3", str(_GLUON_DIR / "best_intarna.py"),
           "--intarna", str(merged), "--mre-fasta", args.mre_fasta,
           "--mirna-fasta", args.mirna_fasta, "--output", str(best)],
          "best_intarna")

    if args.mirna_background:
        cmd = ["python3", str(_GLUON_DIR / "check_inference_background.py"),
               "--input", args.mirna_fasta, "--background", args.mirna_background,
               "--threads", str(args.threads)]
        if args.panel_fasta:
            cmd += ["--panel-fasta", args.panel_fasta]
        # Never --auto-extend here: it mutates the tracked background table, and a
        # benchmark that changes its own inputs between repeats is not a benchmark.
        t0 = time.perf_counter()
        subprocess.run(cmd)      # non-zero just means some z-scores stay NaN
        w.stages["background_check"] = time.perf_counter() - t0

    feat_cmd = ["python3", str(_GLUON_DIR / "feature_extraction.py"),
                "--intarna", str(best), "--mre-fasta", args.mre_fasta,
                "--mirna-fasta", args.mirna_fasta, "--output", str(feats),
                "--threads", str(args.threads), "--v7", args.conservation_tsv]
    if args.mirna_background:
        feat_cmd += ["--mirna-background", args.mirna_background]
    if args.allow_missing_conservation:
        feat_cmd += ["--allow-missing-conservation"]
    w.run(feat_cmd, "feature_extraction")

    with w.time("model_load"):
        predictor = TabularPredictor.load(args.gluon_model)

    X = pd.read_csv(feats).drop(columns=["label"], errors="ignore")
    with w.time("predict"):
        proba = predictor.predict_proba(X, model=args.gluon_model_name)

    pos = 1 if 1 in proba.columns else True
    return w.stages, np.asarray(proba[pos], dtype=float)


def run_cnn_pass(args, df: pd.DataFrame) -> tuple[Dict[str, float], np.ndarray]:
    """One full sequence-to-probability pass of the CNN pipeline."""
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    sys.path.insert(0, str(_REPO / "src" / "training" / "cnn"))
    from cnn_branches_mirbind import (  # noqa: PLC0415
        MiRNAInteractionDataset, predict_logits, _load_ckpt_model)

    w = Stopwatch()
    dev = torch.device(args.cnn_device)

    with w.time("model_load"):
        model, _ = _load_ckpt_model(args.cnn_checkpoint, dev)

    # Tokenising the pair into the two index arrays - the CNN's entire
    # equivalent of the gluon path's IntaRNA + feature-extraction stages.
    with w.time("encode"):
        ds = MiRNAInteractionDataset.from_df(
            df, has_labels=False, mre_col="mre_seq", mirna_col="mirna_seq")

    with w.time("forward"):
        loader = DataLoader(ds, batch_size=args.cnn_batch_size, shuffle=False,
                            num_workers=args.cnn_workers, pin_memory=True)
        logits, _ = predict_logits(model, loader, dev)

    return w.stages, 1.0 / (1.0 + np.exp(-logits))


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage-by-stage end-to-end timing of one prediction pipeline.")
    p.add_argument("--pipeline", required=True, choices=["gluon", "cnn"])
    p.add_argument("--mre-fasta", required=True)
    p.add_argument("--mirna-fasta", required=True)
    p.add_argument("--conservation-tsv", required=True,
                   help="Row-aligned v7 sites TSV. Supplies conservation columns "
                        "for the gluon path, the sequences for the CNN path, and "
                        "the labels both need for APS.")
    p.add_argument("--label-col", default="label")
    p.add_argument("--mre-col", default="gene")
    p.add_argument("--mirna-col", default="noncodingRNA")

    p.add_argument("--gluon-model", help="Predictor directory (gluon pipeline)")
    p.add_argument("--gluon-model-name", default="LightGBMLarge_BAG_L1")
    p.add_argument("--mirna-background")
    p.add_argument("--panel-fasta")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--allow-missing-conservation", action="store_true",
                   help="Let the five conservation features fall back to 0.0 when "
                        "the sites TSV carries no phastCons column. Times the same "
                        "stages, but the APS it reports is not the shipped model's.")

    p.add_argument("--cnn-checkpoint", help="Checkpoint .pt (cnn pipeline)")
    p.add_argument("--cnn-device", default="cpu",
                   help="cpu (default) or cuda. Use cpu for a like-for-like "
                        "comparison against the CPU-bound LightGBM; a cuda run is "
                        "a separate claim and is recorded as such.")
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-workers", type=int, default=4)

    p.add_argument("--repeats", type=int, default=1, metavar="N",
                   help="Timed passes (default 1). A gluon pass runs IntaRNA "
                        "twice over the whole input, so keep N small there.")
    p.add_argument("--warmup", type=int, default=0, metavar="N",
                   help="Untimed passes first (default 0). Worth 1 for the CNN; "
                        "expensive for gluon.")
    p.add_argument("-o", "--output", required=True,
                   help="Summary CSV. Raw per-pass timings go to <output>.raw.csv "
                        "and run metadata to <output>.meta.json.")
    args = p.parse_args()

    if args.repeats < 1:
        raise SystemExit(f"ERROR: --repeats must be at least 1, got {args.repeats}")
    if args.warmup < 0:
        raise SystemExit(f"ERROR: --warmup cannot be negative, got {args.warmup}")
    if args.pipeline == "gluon" and not args.gluon_model:
        raise SystemExit("ERROR: --pipeline gluon requires --gluon-model")
    if args.pipeline == "cnn" and not args.cnn_checkpoint:
        raise SystemExit("ERROR: --pipeline cnn requires --cnn-checkpoint")

    sites = pd.read_csv(args.conservation_tsv, sep="\t")
    y_true = sites[args.label_col].to_numpy() if args.label_col in sites else None
    if y_true is None:
        print(f"Warning: no '{args.label_col}' column; APS will not be computed.",
              file=sys.stderr)
    cnn_df = None
    if args.pipeline == "cnn":
        cnn_df = pd.DataFrame({"mre_seq": sites[args.mre_col].astype(str),
                               "mirna_seq": sites[args.mirna_col].astype(str)})

    print(f"{args.pipeline}: {len(sites)} pairs, {args.warmup} warmup + "
          f"{args.repeats} timed pass(es)")

    tmp = Path(tempfile.mkdtemp(prefix=f"timing_{args.pipeline}_"))
    passes: List[Dict[str, float]] = []
    proba = None
    try:
        for i in range(args.warmup + args.repeats):
            timed = i >= args.warmup
            tag = f"pass {i - args.warmup + 1}/{args.repeats}" if timed else "warmup"
            print(f"  [{tag}] ...", flush=True)
            if args.pipeline == "gluon":
                stages, proba = run_gluon_pass(args, tmp)
            else:
                stages, proba = run_cnn_pass(args, cnn_df)
            if timed:
                passes.append(stages)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Every pass runs the same stages, so any pass gives the stage order.
    order = list(passes[0])
    rows, raw = [], []
    for stage in order:
        samples = [pss[stage] for pss in passes]
        raw += [{"pipeline": args.pipeline, "stage": stage, "pass": k,
                 "seconds": v} for k, v in enumerate(samples, 1)]
        rows.append({"pipeline": args.pipeline, "stage": stage,
                     **summarise_times(samples)})

    totals = [sum(v for k, v in pss.items() if k not in _EXCLUDE_FROM_TOTAL)
              for pss in passes]
    rows.append({"pipeline": args.pipeline, "stage": "TOTAL",
                 **summarise_times(totals)})

    summary = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    pd.DataFrame(raw).to_csv(out.with_suffix(".raw.csv"), index=False)

    aps = None
    if y_true is not None and proba is not None and len(proba) == len(y_true):
        from sklearn.metrics import average_precision_score  # noqa: PLC0415
        aps = float(average_precision_score(y_true, proba))
    elif y_true is not None:
        print(f"Warning: {len(proba)} predictions for {len(y_true)} labels; "
              f"skipping APS.", file=sys.stderr)

    meta = {
        "pipeline": args.pipeline,
        "n_pairs": int(len(sites)),
        "aps": aps,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "threads": args.threads if args.pipeline == "gluon" else None,
        "device": args.cnn_device if args.pipeline == "cnn" else "cpu",
        "model": (args.gluon_model_name if args.pipeline == "gluon"
                  else Path(args.cnn_checkpoint).name),
        "excluded_from_total": sorted(_EXCLUDE_FROM_TOTAL),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    width = max(len(s) for s in summary.stage)
    print(f"\n{args.pipeline} — {len(sites)} pairs"
          + (f", APS {aps:.4f}" if aps is not None else ""))
    for _, r in summary.iterrows():
        share = r.seconds / summary.iloc[-1].seconds * 100
        bar = "" if r.stage == "TOTAL" else f"  {share:5.1f}%"
        print(f"  {r.stage:<{width}}  {r.seconds:8.2f}s"
              f"  [{r.seconds_min:.2f}–{r.seconds_p75:.2f}, n={int(r.seconds_n)}]{bar}")
    print(f"\nWritten: {out}, {out.with_suffix('.raw.csv')}, "
          f"{out.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
