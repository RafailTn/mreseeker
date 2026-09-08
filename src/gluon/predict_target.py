import argparse
import subprocess
import os
import sys
import shutil
import tempfile
import warnings
import requests
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from autogluon.tabular import TabularPredictor

# The official UCSC URL for hg38 470-way conservation
_BW_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons470way/hg38.phastCons470way.bw"
_BW_LOCAL = "hg38.phastCons470way.bw"

# Script directory — all pipeline helpers are expected to sit next to this file
_HERE = Path(__file__).parent


# =============================================================================
# SUBPROCESS HELPER
# =============================================================================

def _count_records(fasta: str) -> int:
    """Records in a FASTA, for sizing the per-step timeout."""
    n = 0
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def _step_timeout(n_pairs: int, override: int | None) -> int | None:
    """Seconds to allow one pipeline step, or None for no limit.

    A fixed cap cannot serve both a 954-pair test set and a 200k-pair one: the
    old 600 s killed IntaRNA partway through the latter and threw away the whole
    run. The budget therefore scales with the work. 0.06 s/pair is roughly 4x
    the measured single-threaded IntaRNA cost, so it is a runaway guard rather
    than a schedule - a step that trips it is stuck, not merely large.
    """
    if override is not None:
        return None if override <= 0 else override
    return int(900 + 0.06 * n_pairs)


def _run(cmd: list[str], step: str, timeout: int | None = None) -> None:
    """
    Run a subprocess command (as a list — no shell=True needed).
    Raises RuntimeError with a clear message if the step fails.
    """
    print(f"[{step}] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Step '{step}' exceeded its {timeout}s budget.\n"
            f"Command: {' '.join(cmd)}\n"
            f"If the input is simply large, raise or remove the limit with "
            f"-timeout <seconds> (0 = no limit)."
        ) from None
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{step}' failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}"
        )


# =============================================================================
# BIGWIG DOWNLOAD
# =============================================================================

def _download_bigwig(dest: Path) -> None:
    """Stream-download the phastCons470way BigWig file."""
    print(f"Downloading BigWig from UCSC -> {dest} ...")
    r = requests.get(_BW_URL, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)
    print("Download complete.")


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def _drop_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns - SHAP needs floats."""
    return df.select_dtypes(include=[np.number])


def _positive_class_shap(vals) -> np.ndarray:
    """
    Flatten shap's per-version output into (n_samples, n_features).

    Binary LightGBM has been returned as a bare array, as a 2-element list, and as a
    trailing-axis-2 array across shap releases, so normalise rather than assume.
    """
    if isinstance(vals, list):
        vals = vals[1] if len(vals) == 2 else vals[0]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[..., 1] if vals.shape[-1] == 2 else vals[..., 0]
    return vals


def _tree_boosters(predictor: TabularPredictor,
                   model_name: str | None = None) -> tuple[list, list[str]] | None:
    """
    The raw tree models behind one predictor model, plus their feature order.

    Returns None for anything TreeSHAP cannot walk - a stacked ensemble mixing in
    neural nets has no tree structure, and must fall back to KernelExplainer. A
    bagged model contributes one booster per fold.

    Both LightGBM and CatBoost are accepted: the shipped default moved from
    LightGBMLarge to CatBoost, and returning None for CatBoost would have
    silently downgraded `-explain` from exact TreeSHAP to a sampled
    approximation for the one model that actually ships.
    """
    model = predictor._trainer.load_model(model_name or predictor.model_best)
    if hasattr(model, "models") and hasattr(model, "load_child"):
        children = [getattr(model.load_child(c), "model", None) for c in model.models]
    else:
        children = [getattr(model, "model", None)]
    if not children or any(c is None for c in children):
        return None

    def names_of(c):
        if hasattr(c, "feature_name"):          # lightgbm.Booster
            return list(c.feature_name())
        if getattr(c, "feature_names_", None):  # catboost.CatBoostClassifier
            return list(c.feature_names_)
        return None

    names = names_of(children[0])
    if names is None:
        return None
    # Averaging fold SHAP values is only valid if the folds agree on column order.
    if any(names_of(c) != names for c in children):
        return None
    return children, names


def compute_tree_shap(
    predictor: TabularPredictor,
    boosters: list,
    X_explain: pd.DataFrame,
    names: list[str],
) -> tuple[np.ndarray, float, list[str]]:
    """
    Exact SHAP values via TreeSHAP, in LOG-ODDS space.

    TreeSHAP walks the trees directly instead of sampling coalitions, so the values are
    exact rather than approximated - there is no nsamples knob and no background set to
    choose. Because SHAP is linear in the model output, averaging the per-fold values
    reproduces the bagged model exactly rather than approximating it.

    Note the units: LightGBM's binary objective is fit in log-odds, so contributions sum
    to the raw margin, not to the probability. The bag averages probabilities, so
    sigmoid(sum(shap) + baseline) is close to but not identical to the reported
    interaction_probability. Signs and rankings are unaffected.
    """
    import shap

    X_model = predictor.transform_features(X_explain)

    print(f"Computing TreeSHAP for {len(X_explain)} samples "
          f"across {len(boosters)} bagged fold(s) ...")
    fold_vals, fold_base = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for booster in boosters:
            explainer = shap.TreeExplainer(booster)
            fold_vals.append(_positive_class_shap(explainer.shap_values(X_model[names])))
            fold_base.append(float(np.mean(explainer.expected_value)))

    return np.mean(fold_vals, axis=0), float(np.mean(fold_base)), names


def compute_shap(
    predictor: TabularPredictor,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    n_background_clusters: int = 25,
    nsamples: int = 200,
) -> tuple[np.ndarray, float, list[str]]:
    """
    SHAP values via KernelExplainer - fully model-agnostic, approximate, PROBABILITY
    space. Used only when the model is not pure LightGBM; see compute_tree_shap.

    Returns
    -------
    shap_values : np.ndarray  shape (n_explain_samples, n_features)
    baseline : float expected model output over background (E[f(x)])
    feature_names : the columns the values are aligned to
    """
    import shap

    pos_col = 1 if 1 in predictor.predict_proba(X_background.head(2)).columns else True

    def _predict_fn(arr: np.ndarray) -> np.ndarray:
        """Wrapper: ndarray -> positive-class probability vector."""
        df = pd.DataFrame(arr, columns=X_background.columns)
        return predictor.predict_proba(df)[pos_col].values

    print(f"Building SHAP background ({n_background_clusters} k-means clusters) ...")
    background = shap.kmeans(X_background, n_background_clusters)

    print(f"Computing SHAP for {len(X_explain)} samples "
          f"(nsamples={nsamples} per call) ...")
    explainer = shap.KernelExplainer(_predict_fn, background)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)

    return (np.array(shap_values), float(explainer.expected_value),
            list(X_explain.columns))


def build_explanation_outputs(
    X_pred: pd.DataFrame,
    predictor: TabularPredictor,
    pos_proba: pd.Series,
    binary_labels: pd.Series,
    out_stem: Path,
    top_n: int = 3,
    max_shap_samples: int = 200,
    nsamples_per_shap: int = 200,
    n_background_clusters: int = 25,
) -> dict[str, list]:
    """
    Run the full explainability stack and write two companion files.

    Outputs
    -------
    1. <out_stem>_global_importance.tsv
       One row per feature, columns:
         feature | mean_abs_shap | shap_rank

    2. <out_stem>_shap_per_sample.tsv
       One row per explained sample (predicted positives), columns:
         row_index | interaction_probability | top1_feature | top1_shap |
         top2_feature | top2_shap | ... | topN_feature | topN_shap | shap_baseline

       The sign of each SHAP value tells you the direction of the feature's
       effect: positive pushes toward interaction, negative pushes away.

       Units depend on which explainer ran. For a LightGBM model this is TreeSHAP and
       the values are exact, in LOG-ODDS. For anything else it falls back to the
       approximate KernelExplainer, whose values are in probability. Both are written
       to the same columns, so note which model produced a given file before comparing
       magnitudes across runs.

    Returns
    -------
    per_sample_cols : dict mapping column names to lists, aligned to the full
                      input length, for appending to the main results TSV.
                      Non-positive rows have empty strings / None values.
    """
    feat_cols = list(_drop_non_numeric(X_pred).columns)
    X_numeric = X_pred[feat_cols]

    # -- SHAP: explain predicted positives (capped for runtime) ----------------
    pos_mask = binary_labels == 1
    X_pos = X_numeric[pos_mask.values].reset_index(drop=True)

    if len(X_pos) == 0:
        print("No predicted positives - skipping SHAP.")
        return {}

    if len(X_pos) > max_shap_samples:
        print(f"{len(X_pos)} positives found; explaining a random sample of "
              f"{max_shap_samples} (set -explain-samples to change).")
        X_explain = X_pos.sample(n=max_shap_samples, random_state=42)
    else:
        X_explain = X_pos

    # TreeSHAP where the model allows it - exact, ~20x faster per row, and it needs no
    # background set. Anything that is not pure LightGBM has no trees to walk and falls
    # back to the sampling explainer.
    found = _tree_boosters(predictor)
    if found:
        boosters, names = found
        shap_vals, baseline, feat_cols = compute_tree_shap(
            predictor, boosters, X_explain, names,
        )
        units = "log-odds"
    else:
        print(f"{predictor.model_best} is not a LightGBM model - "
              f"falling back to KernelExplainer.")
        shap_vals, baseline, feat_cols = compute_shap(
            predictor,
            X_background = X_numeric,
            X_explain = X_explain,
            n_background_clusters = n_background_clusters,
            nsamples = nsamples_per_shap,
        )
        units = "probability"

    shap_df = pd.DataFrame(shap_vals, columns=feat_cols)

    # -- Global SHAP summary ---------------------------------------------------
    mean_abs_shap = shap_df.abs().mean().rename("mean_abs_shap")
    global_tbl = mean_abs_shap.reset_index().rename(columns={"index": "feature"})
    global_tbl["shap_rank"] = global_tbl["mean_abs_shap"].rank(ascending=False).astype(int)
    global_tbl = global_tbl.sort_values("mean_abs_shap", ascending=False)

    global_path = Path(str(out_stem) + "_global_importance.tsv")
    global_tbl.to_csv(global_path, sep="\t", index=False, float_format="%.6f")
    print(f"\n  Global importance -> {global_path}")
    print(f"Top 5 features (by mean |SHAP|, {units}):")
    for _, row in global_tbl.head(5).iterrows():
        print(f"{row['feature']:45s}  "
              f"SHAP={row['mean_abs_shap']:.4f} (rank {row['shap_rank']:>3})")

    # -- Per-sample top-N driver columns ---------------------------------------
    per_sample_top: dict[str, list] = {}
    for k in range(1, top_n + 1):
        per_sample_top[f"top{k}_feature"] = []
        per_sample_top[f"top{k}_shap"]    = []

    for i in range(len(shap_df)):
        row_abs = shap_df.iloc[i].abs()
        top_feats = row_abs.nlargest(top_n).index.tolist()
        for k, feat in enumerate(top_feats, start=1):
            per_sample_top[f"top{k}_feature"].append(feat)
            per_sample_top[f"top{k}_shap"].append(
                round(float(shap_df.iloc[i][feat]), 6)
            )
        for k in range(len(top_feats) + 1, top_n + 1):
            per_sample_top[f"top{k}_feature"].append("")
            per_sample_top[f"top{k}_shap"].append(0.0)

    # Index by X_explain's own labels rather than re-drawing a sample: an independent
    # .sample() call only lines up as long as the seed, length and pandas version all
    # agree, and a silent mismatch would pair each row's probability with another row's
    # drivers.
    proba_pos = pos_proba[pos_mask.values].reset_index(drop=True).loc[X_explain.index]

    per_sample_df = pd.DataFrame(
        {"interaction_probability": proba_pos.values,
         **per_sample_top,
         "shap_baseline": [round(baseline, 6)] * len(shap_df)}
    )
    per_sample_path = Path(str(out_stem) + "_shap_per_sample.tsv")
    per_sample_df.to_csv(per_sample_path, sep="\t", index_label="row_index",
                         float_format="%.6f")
    print(f"Per-sample SHAP  -> {per_sample_path}")
    print(f"Baseline E[f(x)] = {baseline:.4f} ({units})  "
          f"(model output with no feature contributions)")

    # -- Align to full output length (non-positives get empty / None) ----------
    n_total = len(binary_labels)
    full_top: dict[str, list] = {
        f"top{k}_feature": [""] * n_total for k in range(1, top_n + 1)
    }
    full_top.update(
        {f"top{k}_shap": [None] * n_total for k in range(1, top_n + 1)}
    )

    # X_explain is a RANDOM sample of the positives, so the explained rows are not the
    # first max_shap_samples of them. X_explain.index holds each explained row's position
    # within X_pos, which pos_indices maps back to its row in the full input; taking a
    # leading slice instead would attach every row's drivers to the wrong row.
    pos_indices = [i for i, v in enumerate(pos_mask.values) if v]
    sampled_pos = [pos_indices[j] for j in X_explain.index]

    for shap_i, orig_i in enumerate(sampled_pos):
        if shap_i >= len(shap_df):
            break
        for k in range(1, top_n + 1):
            full_top[f"top{k}_feature"][orig_i] = per_sample_top[f"top{k}_feature"][shap_i]
            full_top[f"top{k}_shap"][orig_i] = per_sample_top[f"top{k}_shap"][shap_i]

    return full_top


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the miRNA target prediction pipeline end-to-end."
    )
    # -- Input / output --------------------------------------------------------
    parser.add_argument("-target_fasta", required=True,
                        help="FASTA file containing target/mRNA sequences")
    parser.add_argument("-query_fasta", required=True,
                        help="FASTA file containing query/miRNA sequences")
    parser.add_argument("-conservation_tsv",
                        help="TSV with conservation vectors (--conservation). "
                             "Mutually exclusive with -bigwig.")
    parser.add_argument("-bigwig",
                        help="phastCons BigWig file. If neither -conservation_tsv "
                             "nor -bigwig is given, hg38 470-way BigWig is "
                             "downloaded automatically.")
    parser.add_argument("-model", required=True,
                        help="Path to the saved AutoGluon TabularPredictor directory")
    parser.add_argument("-o", default="./results.tsv",
                        help="Output TSV file (default: ./results.tsv)")
    # -- Shuffle background (z-score features) ---------------------------------
    parser.add_argument("-mirna_background",
                        help="mirna_background.tsv so the four shuffle z-score features "
                             "(E_z_mirna, E_hybrid_z_mirna, E_bg_mean_mirna, "
                             "E_bg_sd_mirna) get filled. WITHOUT it they are NaN, which "
                             "matches the model ONLY if it was trained the same way. "
                             "Query miRNAs missing from the table are scored against its "
                             "frozen panel by default (disable with -no_extend_background).")
    parser.add_argument("-panel_fasta",
                        help="Frozen panel the background table was built on "
                             "(default: <mirna_background>_panel.fa). Any query miRNA "
                             "missing from the table is scored on THIS panel, so its "
                             "z-scores stay comparable to the trained ones.")
    parser.add_argument("-no_extend_background", action="store_true",
                        help="Do not score query miRNAs missing from the background; "
                             "leave their z-scores NaN and only warn.")
    # -- Prediction ------------------------------------------------------------
    parser.add_argument("-threshold", type=float, default=0.5,
                        help="Probability threshold for positive interactions "
                             "(default: 0.5)")
    parser.add_argument("-timeout", type=int, default=None,
                        help="Seconds allowed per pipeline step. Default scales "
                             "with the number of pairs (900s + 0.06s/pair); "
                             "0 removes the limit entirely. Raise it if a large "
                             "run is killed mid-IntaRNA.")
    parser.add_argument("-threads", type=int, default=4,
                        help="Threads for IntaRNA (default: 4)")
    parser.add_argument("-keep_files", action="store_true",
                        help="Keep intermediate files after completion")
    # -- Explainability --------------------------------------------------------
    parser.add_argument("-explain", action="store_true",
                        help="Compute feature importance + SHAP explanations. "
                             "Requires: pip install shap. "
                             "Produces two companion files and adds top-N SHAP "
                             "driver columns to the main results TSV. LightGBM models "
                             "use exact TreeSHAP (log-odds); other models fall back to "
                             "approximate KernelExplainer (probability).")
    parser.add_argument("-explain-samples", type=int, default=200,
                        dest="explain_samples",
                        help="Max positive predictions to explain with SHAP. "
                             "Higher = more accurate but slower. (default: 200)")
    parser.add_argument("-explain-top-n", type=int, default=3,
                        dest="explain_top_n",
                        help="Top-N SHAP driver columns per sample in main output "
                             "(default: 3)")
    parser.add_argument("-shap-nsamples", type=int, default=200,
                        dest="shap_nsamples",
                        help="SHAP nsamples per explained row (default: 200). "
                             "KernelExplainer fallback only - TreeSHAP is exact and "
                             "does not sample.")
    parser.add_argument("-shap-clusters", type=int, default=25,
                        dest="shap_clusters",
                        help="k-means clusters for SHAP background (default: 25). "
                             "KernelExplainer fallback only - TreeSHAP needs no "
                             "background set.")

    args = parser.parse_args()

    # -- Validate --------------------------------------------------------------
    if args.threads < 1:
        print("Error: -threads must be >= 1.", file=sys.stderr)
        return 1

    cpu_count = os.cpu_count() or 4
    if args.threads > cpu_count * 2:
        print(
            f"Warning: {args.threads} threads on a {cpu_count}-CPU system. "
            f"Consider {cpu_count}-{cpu_count * 2}.",
            file=sys.stderr,
        )

    if args.explain:
        try:
            import shap  # noqa: F401
        except ImportError:
            print("Error: -explain requires the `shap` package.\n"
                  "Install with: pip install shap", file=sys.stderr)
            return 1

    # -- Resolve BigWig --------------------------------------------------------
    bigwig_path: Path | None = None
    if args.bigwig:
        bigwig_path = Path(args.bigwig)
    elif not args.conservation_tsv:
        local_bw = Path(_BW_LOCAL)
        if not local_bw.exists():
            _download_bigwig(local_bw)
        else:
            print(f"Using existing BigWig: {local_bw}")
        bigwig_path = local_bw

    # -- Per-step timeout ------------------------------------------------------
    n_pairs = _count_records(args.query_fasta)
    step_timeout = _step_timeout(n_pairs, args.timeout)
    print(f"{n_pairs} pairs; per-step timeout: "
          + ("none" if step_timeout is None else f"{step_timeout}s"))

    # -- Temp directory --------------------------------------------------------
    tmp_dir = Path(tempfile.mkdtemp(prefix="mirna_pipeline_"))
    print(f"Temp directory: {tmp_dir}\n")

    try:
        # -- Step 1: IntaRNA standard ------------------------------------------
        intarna_out = tmp_dir / "intarna_results.tsv"
        intarna_ens_out = tmp_dir / "intarna_results_ensemble.tsv"
        _run(
            ["python3", str(_HERE / "intarna_parallel.py"),
             args.target_fasta, args.query_fasta,
             "-o", str(intarna_out),
             "--threads", str(args.threads)],
            step="intarna", timeout=step_timeout,
        )

        # -- Step 2: IntaRNA ensemble ------------------------------------------
        _run(
            ["python3", str(_HERE / "intarna_parallel.py"),
             args.target_fasta, args.query_fasta,
             "-o", str(intarna_ens_out),
             "--threads", str(args.threads),
             "--ensemble"],
            step="intarna-ensemble", timeout=step_timeout,
        )

        # -- Step 3: Merge -----------------------------------------------------
        merged_out = tmp_dir / "intarna_merged.tsv"
        _run(
            ["python3", str(_HERE / "merge_intarna.py"),
             "-m", str(intarna_out),
             "-e", str(intarna_ens_out),
             "-o", str(merged_out)],
            step="merge", timeout=step_timeout,
        )

        # -- Step 4: Best structure per pair -----------------------------------
        best_out = tmp_dir / "intarna_best.tsv"
        _run(
            ["python3", str(_HERE / "best_intarna.py"),
             "--intarna", str(merged_out),
             "--mre-fasta", args.target_fasta,
             "--mirna-fasta", args.query_fasta,
             "--output", str(best_out)],
            step="best-intarna", timeout=step_timeout,
        )

        # -- Step 4b: Shuffle-background coverage ------------------------------
        # The query miRNAs are the "new miRNAs" case: any not in the background table
        # would make feature_extraction emit NaN for the four z-score features, which is
        # off-distribution for a model trained with them filled. Score the missing ones
        # against the table's FROZEN panel so they stay comparable. On by default;
        # -no_extend_background downgrades it to a report-only warning.
        if args.mirna_background:
            check_cmd = [
                "python3", str(_HERE / "check_inference_background.py"),
                "--input", args.query_fasta,
                "--background", args.mirna_background,
                "--threads", str(args.threads),
            ]
            if args.panel_fasta:
                check_cmd += ["--panel-fasta", args.panel_fasta]
            if not args.no_extend_background:
                check_cmd += ["--auto-extend"]
            print(f"[background-check] {' '.join(check_cmd)}")
            # Tolerate a non-zero exit: it means some query miRNA stays NaN (missing in
            # report-only mode, or n_bg<2 after extension). That degrades those rows but
            # must not abort the run - and extending can be slow, so no timeout here.
            if subprocess.run(check_cmd).returncode != 0:
                print("Warning: some query miRNAs have no usable background; their "
                      "z-score features will be NaN for this run.", file=sys.stderr)
        else:
            print("Warning: -mirna_background not given; the four shuffle z-score "
                  "features (E_z_mirna, E_hybrid_z_mirna, E_bg_mean_mirna, "
                  "E_bg_sd_mirna) will be NaN. This matches the model ONLY if it was "
                  "trained without them.", file=sys.stderr)

        # -- Step 5: Feature extraction ----------------------------------------
        features_out = tmp_dir / "samples_4_pred.csv"
        feat_cmd = [
            "python3", str(_HERE / "feature_extraction.py"),
            "--intarna", str(best_out),
            "--mre-fasta", args.target_fasta,
            "--mirna-fasta", args.query_fasta,
            "--output", str(features_out),
            # Same budget as the IntaRNA steps. Extraction falls back to a serial path
            # under its own row threshold, so a small query set pays no pool startup.
            "--threads", str(args.threads),
        ]
        if args.conservation_tsv:
            feat_cmd += ["--v7", args.conservation_tsv]
        else:
            # No conservation track: point --v7 at the IntaRNA table purely so the row
            # count lines up, and let the conservation features fall back to 0.0.
            feat_cmd += ["--v7", str(best_out), "--allow-missing-conservation"]
        if bigwig_path:
            feat_cmd += ["--bigwig", str(bigwig_path)]
        if args.mirna_background:
            feat_cmd += ["--mirna-background", args.mirna_background]
        _run(feat_cmd, step="feature-extraction", timeout=step_timeout)

        # -- Step 6: Load model + predict --------------------------------------
        print(f"\nLoading predictor from: {args.model}")
        predictor  = TabularPredictor.load(args.model)
        input4pred = pl.read_csv(features_out).to_pandas()
        input4pred = input4pred.drop(columns=["label"], errors="ignore")

        proba = predictor.predict_proba(input4pred)
        pos_col = 1 if 1 in proba.columns else True
        pos_proba = proba[pos_col]
        binary = (pos_proba >= args.threshold).astype(int)

        n_pos = int(binary.sum())
        n_total = len(binary)
        print(f"{n_pos} interactions above threshold {args.threshold} "
              f"out of {n_total} pairs.")

        # -- Step 7: Explainability (optional) ---------------------------------
        shap_cols: dict[str, list] = {}
        if args.explain:
            print("\n--- Explainability ---")
            out_stem  = Path(args.o).with_suffix("")
            shap_cols = build_explanation_outputs(
                X_pred = input4pred,
                predictor = predictor,
                pos_proba = pos_proba,
                binary_labels = binary,
                out_stem = out_stem,
                top_n = args.explain_top_n,
                max_shap_samples = args.explain_samples,
                nsamples_per_shap = args.shap_nsamples,
                n_background_clusters = args.shap_clusters,
            )

        # -- Step 8: Write main output -----------------------------------------
        seq_cols = [c for c in ("mre_seq", "mirna_seq") if c in input4pred.columns]

        extra_pl_cols = [
            pl.Series("interaction_probability",
                      [round(float(v), 5) for v in pos_proba.tolist()]),
            pl.Series("prediction", binary.tolist()),
        ]
        for col_name, col_vals in shap_cols.items():
            extra_pl_cols.append(pl.Series(col_name, col_vals))

        output = pl.from_pandas(input4pred[seq_cols]).with_columns(extra_pl_cols)
        output.write_csv(args.o, separator="\t")
        print(f"\nResults written to: {args.o}  ({n_total} rows, {output.width} columns)")

    finally:
        if args.keep_files:
            print(f"\nIntermediate files kept at: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\nTemp directory removed: {tmp_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
