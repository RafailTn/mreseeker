# Artifact check and removal

Finds chimeric "positives" that look like chance ligations (the ARTIFACT cell) and removes them from CNN training with
twin-aware sampler weights. Full study: `results/seedless_decision_report.md` (not tracked).

**ARTIFACT cell** = positives with no canonical seed for their miRNA within 300 nt (`noseed300`), no partial seed pairing
in the 50-nt core (`none`, not `comp3p`), scored 0.1 to 0.25 by an out-of-fold 150-nt CNN. The study pre-registered
0.1 to 0.3 (arms `A2`, `A2X`). 0.25 is the default (decision 2026-11-23) because, band by band, 0.25 to 0.30 fails the
geometry bar.

**Default removal** = `A2X_hi025`. It sets the weight of the ARTIFACT positives to 0. It also drops the seed-less positives
of 27 flagged small RNAs (self-chimeras and low-complexity RNAs) and seed-less positives whose window holds at least 12 nt
of their own RNA. Twin negatives follow their positives: each negative is weighted by the share of its window's
positives that survive. The weights are then rebalanced to A0's per-family and per-window-stratum ratios.

## Scripts

| script | role |
|---|---|
| `artifact_common.py` | shared code: seed classes, partial-pairing subclasses, decoy shuffle, empirical p, mixing estimate, chimera reads |
| `seedless_strata.py` | per row: stratum (core / flank150 / flank300 / noseed300), subclass, comp3p, p150 |
| `seedless_predict.py` | 150-nt CNN probabilities (`.npy`, row-aligned) |
| `seedless_oof.py` | chromosome-fold table + `run.sh` for the out-of-fold 150-nt scores |
| `seedless_pairing_null.py` | check: IntaRNA pairing of the own miRNA vs 10 expressed decoys + 10 shuffles (R_emp) |
| `seedless_geometry.py` | check: ligation geometry of the reads (R_geom, swap control) |
| `seedless_readouts.py` | check: Expt8 overexpression response and human/rat mixing |
| `seedless_cells.py` | check: combines the three into the cell labels (`results/seedless_cells.tsv`, `_fine.tsv`) |
| `lowcomplexity_audit.py` | flags self-chimera and low-complexity small RNAs (`results/lowcomplexity_rnas.tsv`, `data/lowcomplexity/*.npz`) |
| `seedless_arms.py` | builds the removal weights (`data/seedless/arms/<arm>.npy`) |

## Run order

```bash
PY=/path/to/python   # pandas, numpy, scipy, sklearn
PX="pixi run --frozen --manifest-path dependencies/cnn/pixi.toml"

# 0. window tables (tracked builder): data/mre150 (gene150) and data/mre300 (gene300) for every set
# 1. 150-nt scores of the test sets, then strata
MRESEEKER_MRE_LEN=150 $PX python3 src/artifact/seedless_predict.py --checkpoint cnn_checkpoints/cnn_mre150.pt \
    --out-dir data/seedless/base150 <name>=<table>:gene150:noncodingRNA ...
$PY src/artifact/seedless_strata.py
# 2. out-of-fold scores of the training rows (GPU)
$PY src/artifact/seedless_oof.py build && nohup bash data/seedless/oof/run.sh > data/seedless/oof/run.log 2>&1
# 3. artifact check (optional for removal; needed to re-derive the cell labels)
$PY src/artifact/seedless_pairing_null.py select && $PY src/artifact/seedless_pairing_null.py run && \
    $PY src/artifact/seedless_pairing_null.py collect
$PY src/artifact/seedless_geometry.py
$PY src/artifact/seedless_readouts.py build && $PY src/artifact/seedless_readouts.py occupancy && \
    $PY src/artifact/seedless_readouts.py mixing
$PY src/artifact/seedless_cells.py && $PY src/artifact/seedless_cells.py fine
# 4. RNA-side flags, then removal weights
$PY src/artifact/lowcomplexity_audit.py
$PY src/artifact/seedless_arms.py build A2X_hi025
# 5. train with the weights
$PX python3 src/training/cnn/cnn_branches_mirbind.py train --train data/mre150/manakov_train_nochr2.tsv \
    --sample-weights data/seedless/arms/A2X_hi025.npy --balance ...
```

## What stays outside this folder

- Code (tracked): `src/cnn/build_mre150.py` (window tables, `extend`) and `src/training/cnn/cnn_branches_mirbind.py`.
  `--sample-weights` in the training script is still an **unstaged** change; it must be committed for step 5.
- Data (not code), read from `$HYBRIDETECTOR_DIR` (default `/mnt/raid1/rafail/Hybridetector_pipeline`): HybriDetector
  chimera reads, small-RNA CPM, Expt8 (`validate_overexpression`) and mixing tables.
- `data/relabel/noise/eval_win_v2` (Expt8 and mixing windows for the readouts), `cnn_checkpoints/cnn_mre150.pt`
  (gitignored), and IntaRNA (on PATH, or the repo's gluon Pixi env).

`src/cnn/<name>.py` stubs load these scripts so older analyses and `data/*.sh` queues keep working; only this folder
needs to be pushed.
