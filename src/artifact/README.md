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
| `eval_windows.py` | label-free readout windows: human/rat mixing chimeras and Expt8 occupancy sites (50/150/300 nt) + base150 scores |
| `seedless_strata.py` | per row: stratum (core / flank150 / flank300 / noseed300), subclass, comp3p, p150 |
| `seedless_predict.py` | 150-nt CNN probabilities (`.npy`, row-aligned) |
| `seedless_oof.py` | chromosome-fold table + `run.sh` for the out-of-fold 150-nt scores |
| `seedless_pairing_null.py` | check: IntaRNA pairing of the own miRNA vs 10 expressed decoys + 10 shuffles (R_emp) |
| `seedless_geometry.py` | check: ligation geometry of the reads (R_geom, swap control) |
| `seedless_readouts.py` | check: Expt8 overexpression response and human/rat mixing |
| `seedless_cells.py` | check: combines the three into the cell labels (`results/seedless_cells.tsv`, `_fine.tsv`) |
| `lowcomplexity_audit.py` | flags self-chimera and low-complexity small RNAs (`results/lowcomplexity_rnas.tsv`, `data/lowcomplexity/*.npz`) |
| `seedless_arms.py` | builds the removal weights (`data/seedless/arms/<arm>.npy`) |
| `train_arms.sh` | trains 150-nt arms with those weights and scores them (GPU, ~4.5 h per arm and seed) |
| `check_expected.py` | compares a rerun with `expected/` |

## Without the raw data

`fetch_data.sh` puts every input in place from public sources and one Zenodo bundle (10.5281/zenodo.24503913; `BUNDLE_README.md` lists it):
the miRBench v7 tables (Zenodo 10.5281/zenodo.20540907), the Ensembl GRCh38 genome, and the processed HybriDetector
outputs, GSE test sets and readout windows. Every file is md5-checked. `cnn_mre150.pt` is in git.

```bash
bash src/artifact/fetch_data.sh                         # needs curl, samtools; ~1.5 GB download
export HYBRIDETECTOR_DIR=$PWD/data/external/hybridetector
G=data/external/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa
python3 src/cnn/build_mre150.py --src-dir data/mirbench_v7 --genome $G
python3 src/cnn/build_mre150.py --src-dir data/mirbench_v7 --genome $G --up 125 --down 125 --col gene300 --out data/mre300
```

Then run the order below from step 1. The readout windows and their scores come with the bundle, so skip
`eval_windows.py`; it needs the hg38+rn7 genome. `make_bundle.sh` is the maintainer side that builds the bundle; how to
rebuild its contents from GEO is in `REPRODUCE_MRESEEKER.md` of https://github.com/RafailTn/Hybridetector_pipeline.

## Run order

Run from the repo root.

```bash
PY=/path/to/python   # pandas, numpy, scipy, sklearn
PX="pixi run --frozen --manifest-path dependencies/cnn/pixi.toml"
export HYBRIDETECTOR_DIR=/path/to/Hybridetector_pipeline
M=data/mre150; EW=data/relabel/noise/eval_win_v2/150

# 0. inputs
#    window tables (tracked builder, main README "Wider MRE windows"): data/mre150 (gene150) and data/mre300 (gene300)
#    from the miRBench v7 tables; point --genome / --src-dir / --novel-dir at local copies if the defaults do not exist
python3 src/cnn/build_mre150.py
python3 src/cnn/build_mre150.py --up 125 --down 125 --col gene300 --out data/mre300
#    base 150-nt model: cnn_checkpoints/cnn_mre150.pt (tracked; recipe in the main README)
#    readout windows + their base150 scores
$PY src/artifact/eval_windows.py build
MRESEEKER_MRE_LEN=150 $PX python3 src/artifact/eval_windows.py predict
$PY src/artifact/seedless_readouts.py build               # Expt8 sites -> data/seedless/readouts/expt8_sites150.tsv

# 1. base150 scores of the test sets and readouts, then strata
SPECS=""
for s in manakov_test manakov_leftout gse304955 gse263552 gse180129_HSPC gse180129_Kasumi1 \
         hejret_test klimentova_test manakov_val_chr2; do SPECS="$SPECS $s=$M/$s.tsv:gene150:noncodingRNA"; done
SPECS="$SPECS occupancy=$EW/occupancy.tsv:gene:noncodingRNA mixing=$EW/mixing.tsv:gene:noncodingRNA"
SPECS="$SPECS expt8_all=data/seedless/readouts/expt8_sites150.tsv:gene150:mirna_seq"
MRESEEKER_MRE_LEN=150 $PX python3 src/artifact/seedless_predict.py --checkpoint cnn_checkpoints/cnn_mre150.pt \
    --out-dir data/seedless/base150 $SPECS
$PY src/artifact/seedless_strata.py

# 2. out-of-fold scores of the training rows (GPU)
$PY src/artifact/seedless_oof.py build && nohup bash data/seedless/oof/run.sh > data/seedless/oof/run.log 2>&1

# 3. artifact check (needed to re-derive the cell labels; the default removal does not read them)
$PY src/artifact/seedless_pairing_null.py select && $PY src/artifact/seedless_pairing_null.py run && \
    $PY src/artifact/seedless_pairing_null.py collect
$PY src/artifact/seedless_geometry.py
$PY src/artifact/seedless_readouts.py occupancy && $PY src/artifact/seedless_readouts.py mixing
$PY src/artifact/seedless_cells.py && $PY src/artifact/seedless_cells.py fine

# 4. RNA-side flags, then removal weights
$PY src/artifact/lowcomplexity_audit.py
$PY src/artifact/seedless_arms.py build A2X_hi025

# 5. train with the weights (and score the test sets)
ARMS="A2X_hi025" SEEDS="42" nohup bash src/artifact/train_arms.sh > data/seedless/arms/train.log 2>&1 &

# 6. compare with the reference outputs
$PY src/artifact/check_expected.py
```

`lowcomplexity_audit.py` also reports how the study's arms A1 to A3 compare with A0 when those arms are trained
(`seedless_arms.py build A0 A1 A2 A3`, then `ARMS="A0 A1 A2 A3" SEEDS="42 43 44" train_arms.sh`, ~54 GPU hours).
Without them it writes the flag tables and an impact table without the arm columns; the removal needs only the flags.

## What a rerun should reproduce

`expected/` holds the study's outputs; `check_expected.py` compares whatever parts of a rerun exist:

| file | from | rerun should match |
|---|---|---|
| `strata_counts.tsv` | step 1 | exactly (rows per set x label x stratum x subclass) |
| `lowcomplexity_flagged.tsv` | step 4 | exactly (the 28 flagged small RNAs and their signature) |
| `seedless_cells.tsv`, `seedless_cells_fine.tsv` | step 3 | cell labels; counts and R values to the last digits |
| `seedless_readouts_*_base150.tsv` | step 3 | to the last digits |
| `arm_diag.tsv` | step 4 | within ~0.01 (share of rows set to weight 0) |

Training rows are binned on out-of-fold scores (step 2), and GPU training is not bit-reproducible, so the training
rows near the 0.1 and 0.25 edges can switch cells. The arm weights therefore match closely, not exactly. Everything
computed from the test sets is deterministic given `cnn_mre150.pt`: base150 scores from `seedless_predict.py` and from
the older `data/mre150/pred_mre150_<set>.tsv` files differ by at most 3e-8.

## What stays outside this folder

- Code (tracked): `src/cnn/build_mre150.py` (window tables, `extend`), `src/cnn/predict_cnn.py` and
  `src/training/cnn/cnn_branches_mirbind.py` (model, `--sample-weights`).
- Data from the HybriDetector pipeline folder (see below).
- `cnn_checkpoints/cnn_mre150.pt` (tracked), IntaRNA (on PATH, or the repo's gluon Pixi env) and samtools.
- `src/cnn/build_mre150.py` (step 0) reads the miRBench v7 tables (default `../microCLIP2/data`, `--src-dir`) and a
  GRCh38 primary assembly (default the HybriDetector one below, `--genome`).
  `seedless_readouts.py build` uses the same genome: `$GRCH38_FA` if set, else the `fetch_data.sh` copy in
  `data/external/genome/`, else the HybriDetector one.
- `results/` is gitignored; the scripts create it.

### Data from HybriDetector

Read from `$HYBRIDETECTOR_DIR` (default `/mnt/raid1/rafail/Hybridetector_pipeline`); paths below are relative to it.
Data only, no code is imported.

| path | size | read by | used for |
|---|---|---|---|
| `external/HybriDetector/hyb_pairs/<sample>.hyb_pairs_deduplicated.tsv` and `<sample>.all_hyb_pairs.tsv`, 10 samples: Expt1/4/9 293T ChimeCLIP, with and without gel (`SAMPLES` in `artifact_common.py`) | 3.6 GB | `seedless_geometry.py` | ligation-geometry check |
| `data/validate_overexpression/sites.tsv`, `occupancy.tsv` | 34 MB | `eval_windows.py`, `seedless_readouts.py` | Expt8 miR-1/miR-124 occupancy readout |
| `data/mixing_artifacts/chimeras.tsv` | 160 MB | `eval_windows.py`, `seedless_readouts.py` | human/rat mixing readout |
| `data/mixing_artifacts/artifact_rate.tsv` | 72 KB | `seedless_readouts.py` | per-block rates for the mixing estimate |
| `data/smallrna/gse196460_mirna_cpm.tsv` | 68 KB | `seedless_pairing_null.py` | expressed decoy miRNAs for the pairing null |
| `external/HybriDetector/DBs/Homo_sapiens.GRCh38.dna.primary_assembly.fa` (+ `.fai`) | 3.2 GB | `eval_windows.py`, `build_mre150.py` | occupancy windows, window tables |
| `external/HybriDetector_hsrn/DBs/genome_hsrn.fa` (+ `.fai`) | 5.8 GB | `eval_windows.py` | mixing windows (human + rat genome) |

The default removal (`seedless_arms.py build A2X_hi025`) needs none of these. They are needed to re-derive the cell
labels (steps 0 and 3) and for the Expt8 / mixing readouts.

Outside this machine: the hyb_pairs files are standard HybriDetector output and the genomes are public. The two
`validate_overexpression` tables, the two `mixing_artifacts` tables and the small-RNA CPM table (about 200 MB) are
made by the HybriDetector pipeline (`validate_overexpression.py`, `mixing_artifacts.py`). They are not in this repo:
share them, or the scripts that make them, alongside it.

`src/cnn/<name>.py` stubs load these scripts so older analyses and `data/*.sh` queues keep working; only this folder
needs to be pushed.
