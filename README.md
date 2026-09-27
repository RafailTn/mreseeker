# MREseeker

Predict whether a microRNA binds a candidate target site (MRE) on an mRNA.

Two models are trained on the same data:

| Model | Input | Test APS |
|---|---|---|
| **Sequence CNN** | the two raw sequences | 0.87 |
| **CatBoost** (AutoGluon, `CatBoost_BAG_L1`) | 26 IntaRNA / conservation features | 0.84 |

The CNN needs only the sequences. CatBoost needs a feature-extraction pass
(IntaRNA + phyloP) but gives exact per-prediction SHAP explanations.

Data: miRBench (Gresova et al., 2025), built on the AGO2 chimeric eCLIP of
Manakov et al. (2022), with false negatives corrected against TarBase.

---

## Installation

Environments are managed with [Pixi](https://pixi.prefix.dev/latest/installation/).
Install only the one you need:

```bash
pixi install --manifest-path dependencies/cnn/pixi.toml    # PyTorch + Optuna (CUDA)
pixi install --manifest-path dependencies/gluon/pixi.toml  # AutoGluon + IntaRNA + ViennaRNA (~5.4 GB)
```

Each example first enters its environment with `pixi shell`, then runs the script.
Use the shell rather than the env's Python binary directly: the CatBoost pipeline
calls `IntaRNA` via `PATH`, and without it fails with `ERROR: IntaRNA not found.`

---

## Sequence CNN

### Predict

```bash
pixi shell --manifest-path dependencies/cnn/pixi.toml
python3 src/cnn/predict_cnn.py --checkpoint cnn_checkpoints/cnn_branches_mirbind_embed16_restruct.pt --input pairs.tsv -o predictions.tsv --mre-col mre_sequence --mirna-col mirna_sequence
```

- Input is a TSV with one MRE and one miRNA sequence per row; no `label` needed.
- Rows are de-duplicated on the miRNA+MRE sequence; `--no-dedup` scores every row.
- `--threshold` (default 0.5) sets the binary call.
- Checkpoints are self-contained (they store their own `model_args`), so any
  single checkpoint or k-fold member can be used on its own.

### Train

```bash
pixi shell --manifest-path dependencies/cnn/pixi.toml
python3 src/training/cnn/cnn_branches_mirbind.py train --train train.tsv --val val.tsv --test test.tsv leftout.tsv --mre-col gene --mirna-col noncodingRNA --family-col noncodingRNA_fam --epochs 40 --batch-size 256 --lr 1e-3 --out cnn_checkpoints/cnn_mirbind.pt
```

- `--folds K` — stratified group k-fold by miRNA family, writes `*_fold{1..K}.pt`.
- `--oof-out FILE` — with `--folds`, also writes out-of-fold predictions.
- `--no-val` — fit on the full training set without a validation split.
- `--test` — scores one or more labelled sets with the final checkpoint.

### Evaluate

Scoring a labelled file (a `label` column) prints metrics:

```bash
pixi shell --manifest-path dependencies/cnn/pixi.toml
# single checkpoint
python3 src/training/cnn/cnn_branches_mirbind.py predict --checkpoint cnn_checkpoints/cnn_mirbind.pt --input test.tsv --output test_pred.tsv [--error-dump errors.tsv]

# average of k-fold checkpoints over several test sets
python3 src/training/cnn/cnn_branches_mirbind.py predict-ensemble --checkpoints cnn_checkpoints/cnn_mirbind_fold*.pt --inputs test.tsv leftout.tsv --output-dir predictions
```

`explain` (integrated gradients or occlusion) is also available as a subcommand.

### Hyper-parameter search

```bash
pixi shell --manifest-path dependencies/cnn/pixi.toml
python3 src/training/cnn/optuna_cnn_mirbind.py --train train.tsv --val val.tsv --trials 50 --epochs 30 --ckpt-dir optuna_ckpts --storage optuna_mirbind.db
```

Maximises validation AUPRC (TPE sampler, median pruner). Every checkpoint it
writes can be scored directly with `predict_cnn.py`.

---

## CatBoost (AutoGluon)

### Predict

`predict_target.py` runs the full chain: IntaRNA → best structure per pair →
shuffle-background z-scores → feature extraction → prediction.

```bash
pixi shell --manifest-path dependencies/gluon/pixi.toml
python3 src/gluon/predict_target.py -target_fasta mre.fa -query_fasta mirna.fa -bigwig data/hg38.phyloP100way.bw -model models_gluon_catboost -o results.tsv -threads 8 [-explain]
```

- **The FASTAs are paired by position**, not all-vs-all: record *i* of
  `-target_fasta` is scored against record *i* of `-query_fasta`. Build them from
  a table with
  `python3 src/gluon/make_fastas.py --v7 sites.tsv --mre-fasta mre.fa --mirna-fasta mirna.fa`.
- **Conservation** — give exactly one of `-bigwig` (hg38 phyloP **100-way**; other
  tracks give different scores) or `-conservation_tsv` (MRE coordinates, optionally
  with a precomputed `gene_phyloP` column). With neither, the ~9 GB BigWig is
  downloaded into the working directory.
- **Shuffle background** — miRNAs missing from `background/mirna_background.tsv`
  are scored and **appended to that file in place**. Pass a copy via
  `-mirna_background`, or `-no_extend_background` to leave those features `NaN`.
- **`-explain`** adds SHAP output: global and per-sample tables plus top-N driver
  columns. Values are exact TreeSHAP in **log-odds**.
- Output: a TSV with `interaction_probability` and a thresholded `prediction`
  (`-threshold`, default 0.5).

`models_gluon_catboost` (27 MB) is a deployment clone of the default model. The
full AutoGluon stack (`models_gluon`, 6.8 GB, not in git) scores only marginally
higher (0.8465 vs 0.8430 APS on test) at far greater inference cost.

### Train and evaluate

Features are extracted once per split, then both trainers fit on the fixed 26
columns (`SELECTED_26` in `src/gluon/feature_extraction.py`) and report metrics
on the test and leftout sets.

```bash
pixi shell --manifest-path dependencies/gluon/pixi.toml
# features from an IntaRNA best-structure table
python3 src/gluon/feature_extraction.py --intarna best.tsv --mre-fasta mre.fa --mirna-fasta mirna.fa --v7 sites.tsv --output train.csv

# single fit on the full training set
python3 src/training/gluon/gluon_train_total.py --input train.csv --test test.csv --leftout leftout.csv --modelpath models/gluon_total --time 3600

# stratified group k-fold (grouped by miRNA family)
python3 src/training/gluon/gluon_training_kfold.py --input train.csv --test test.csv --leftout leftout.csv --folds 5 --modelpath models/gluon_fold --time 3600
```

Results go to `--results_path` and misclassified rows to `--misclassified_dir`
(defaults under `results/`). Permutation importance on a trained model:

```bash
pixi shell --manifest-path dependencies/gluon/pixi.toml
python3 src/gluon/permutation_importance.py --model models_gluon_catboost --data test.csv --output permutation_importance.tsv --shuffles 10
```

---

## Repository layout

```
src/cnn/              CNN inference (predict_cnn.py)
src/training/cnn/     CNN model + training CLI, Optuna search
src/gluon/            CatBoost inference pipeline (predict_target.py), feature extraction
src/training/gluon/   AutoGluon training, feature selection
src/benchmark/        model comparison and timing
src/eqtl_analysis/    GTEx eQTL scoring with the CNN
cnn_checkpoints/      shipped CNN checkpoint
models_gluon_catboost/ shipped CatBoost model
background/           shuffle-background panel
dependencies/         Pixi manifests (cnn, gluon, featurewiz)
```

---

## Citations

- Gresova, K., et al. (2025). miRBench datasets (v6). Zenodo. https://doi.org/10.5281/zenodo.14734014
- Manakov, S. A., et al. (2022). Scalable and deep profiling of mRNA targets for individual microRNAs with chimeric eCLIP. bioRxiv. https://doi.org/10.1101/2022.02.13.480296
- Erickson, N., et al. (2020). AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. arXiv:2003.06505.
- Mann, M., Wright, P. R., & Backofen, R. (2017). IntaRNA 2.0. *Nucleic Acids Research*, 45(W1), W435–W439. https://doi.org/10.1093/nar/gkx279
- Lorenz, R., et al. (2011). ViennaRNA Package 2.0. *Algorithms for Molecular Biology*, 6, 26.
- Sethupathy, P., Corda, B., & Hatzigeorgiou, A. G. (2006). TarBase. *RNA*, 12(2), 192–197. https://doi.org/10.1261/rna.2239606
- Seshadri, R. (2020). featurewiz. https://github.com/AutoViML/featurewiz
