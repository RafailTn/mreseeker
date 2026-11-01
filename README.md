# MREseeker (CNN)

Predict whether a microRNA binds a candidate target site (MRE) on an mRNA, with a
**sequence CNN** that reads the raw miRNA and MRE sequences (~0.87 average
precision; chance = 0.5). No feature-extraction pass is needed, and `explain
--duplex` reports which bases pair and what kind of site the model relied on.

`src/eqtl_analysis/` applies the CNN to GTEx eQTLs that fall in miRNA target sites.

Data: miRBench (Gresova et al., 2025), built on the AGO2 chimeric eCLIP of
Manakov et al. (2022).

---

## Installation

The environment is managed with [Pixi](https://pixi.prefix.dev/latest/installation/):

```bash
pixi install --manifest-path dependencies/cnn/pixi.toml    # PyTorch + Optuna (CUDA)
```

Each example first enters it with `pixi shell`, then runs the script.

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

### Explain: which bases pair, and what kind of site

`explain` runs Integrated Gradients on the 2D pairing matrix. With `--duplex` it also
decodes that attribution map into the duplex the model relied on: an antiparallel,
bulge-tolerant alignment through the matrix (`src/training/cnn/duplex_decode.py`),
steered by the attribution (`--duplex-lambda`, 0 = sequence-only duplex).

```bash
python3 src/training/cnn/cnn_branches_mirbind.py explain \
    --checkpoint cnn_checkpoints/cnn_branches_mirbind_embed16_restruct.pt \
    --input pairs.tsv --output explain.tsv --mre-col gene --mirna-col noncodingRNA \
    --duplex --duplex-report duplex.txt --report-top 50 --dump-matrix attr.npz
python3 src/cnn/plot_duplex.py --explain explain.tsv --npz attr.npz --out-dir duplex_png --top 6
```

Added columns (all values in logits):

| column | meaning |
|---|---|
| `duplex_type` | binding type of the decoded duplex (same names as `binding_types.py`, plus `.bulge` for a seed bulge) |
| `seq_type` | strict-register, sequence-only type, for comparison |
| `duplex_pairs` | `pos:j:kind:attr;…` per miRNA position (1-based pos, 0-based MRE j; kind WC/GU/MM/bulge) |
| `dlogit_seed/central/supp/tail` | logit drop when the duplex cells of miRNA positions 1–8 / 9–12 / 13–17 / 18+ are zeroed |
| `dlogit_duplex`, `dlogit_random` | whole duplex zeroed vs. the same number of random cells (control) |
| `attr_*`, `duplex_frac` | IG mass on those cells; share of positive IG mass that sits on the duplex |

`duplex.txt` shows an IntaRNA-style diagram per row (`|` WC, `:` G·U, lower-case = unpaired,
`s` = seed 2–8, `3` = supplementary 13–16). On a 3,000-row GSE304955 sample, zeroing the
decoded duplex costs +0.41 logit on average, against 0.00 for random cells, and correlates
0.95 with logit − baseline. So the duplex accounts for the prediction, rather than just
being drawn over it. The type is most trustworthy where `prob` and `duplex_frac` are high.
On low-scoring pairs the attribution is diffuse and the decoded duplex is mostly chemistry.

### Hyper-parameter search

```bash
pixi shell --manifest-path dependencies/cnn/pixi.toml
python3 src/training/cnn/optuna_cnn_mirbind.py --train train.tsv --val val.tsv --trials 50 --epochs 30 --ckpt-dir optuna_ckpts --storage optuna_mirbind.db
```

Maximises validation AUPRC (TPE sampler, median pruner). Every checkpoint it
writes can be scored directly with `predict_cnn.py`.

---

## Repository layout

```
src/cnn/              CNN inference (predict_cnn.py), duplex plots (plot_duplex.py)
src/training/cnn/     CNN model + training/explain CLI, duplex decoder, Optuna search
src/eqtl_analysis/    GTEx eQTL scoring with the CNN
cnn_checkpoints/      shipped CNN checkpoint
data/                 eQTL fragment tables and the CNN-TP pair sets
dependencies/cnn/     Pixi manifest
```

---

## Citations

- Gresova, K., et al. (2025). miRBench datasets (v6). Zenodo. https://doi.org/10.5281/zenodo.14734014
- Manakov, S. A., et al. (2022). Scalable and deep profiling of mRNA targets for individual microRNAs with chimeric eCLIP. bioRxiv. https://doi.org/10.1101/2022.02.13.480296
