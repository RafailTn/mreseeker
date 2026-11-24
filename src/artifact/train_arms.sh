#!/usr/bin/env bash
# Train 150-nt CNN arms with removal weights from seedless_arms.py, then score them (row-aligned .npy).
# One protocol for every arm (A0 = no removal, included): base150 recipe, no validation split, 30 epochs,
# --sample-weights data/seedless/arms/<arm>.npy --balance, one run per seed.
# Predictions: the 8 test sets and chr2 val (gene150), the occupancy and mixing windows (eval_windows.py) and,
# when seedless_readouts.py build has run, every Expt8 site (expt8_all).
#
#   ARMS="A2X_hi025" SEEDS="42" nohup bash src/artifact/train_arms.sh > data/seedless/arms/train.log 2>&1 &
#   ARMS="A0 A1 A2 A3" SEEDS="42 43 44" ...     # the arms behind lowcomplexity_audit.py's impact columns
# About 4.5 h of GPU time per arm and seed. Existing checkpoints are kept (only predictions are redone).
# Needs the --sample-weights option of src/training/cnn/cnn_branches_mirbind.py.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONUNBUFFERED=1 MRESEEKER_MRE_LEN=150
PX="pixi run --frozen --manifest-path dependencies/cnn/pixi.toml"
ARCH="--seq-pairing embed --pair-embed-dim 16 --seq-pool gem --seq-filters 64 --seq-dim 128 \
      --n-conv-blocks 6 --n-pool-blocks 4 --block-pool max --ema"
D=data/seedless/arms
M=data/mre150
EW=data/relabel/noise/eval_win_v2/150
ARMS=${ARMS:-"A2X_hi025"}
SEEDS=${SEEDS:-"42"}
EPOCHS=${EPOCHS:-30}

for SEED in $SEEDS; do
  for ARM in $ARMS; do
    [ -s $D/$ARM.npy ] || { echo "missing $D/$ARM.npy: run src/artifact/seedless_arms.py build $ARM"; exit 1; }
    O=$D/$ARM/s$SEED
    mkdir -p $O
    if [ ! -s $O/cnn150.pt ]; then
        echo "######## train $ARM seed $SEED $(date)"
        $PX python3 src/training/cnn/cnn_branches_mirbind.py train \
            --train $M/manakov_train_nochr2.tsv --no-val \
            --mre-col gene150 --mirna-col noncodingRNA --family-col noncodingRNA_fam \
            --epochs $EPOCHS --batch-size 256 --lr 1e-3 $ARCH \
            --sample-weights $D/$ARM.npy --balance --seed $SEED --out $O/cnn150.pt
    fi
    echo "######## predict $ARM seed $SEED $(date)"
    SPECS=""
    for s in manakov_test manakov_leftout gse304955 gse263552 gse180129_HSPC gse180129_Kasumi1 \
             hejret_test klimentova_test manakov_val_chr2; do
        SPECS="$SPECS $s=$M/$s.tsv:gene150:noncodingRNA"
    done
    SPECS="$SPECS occupancy=$EW/occupancy.tsv:gene:noncodingRNA mixing=$EW/mixing.tsv:gene:noncodingRNA"
    [ -s data/seedless/readouts/expt8_sites150.tsv ] && \
        SPECS="$SPECS expt8_all=data/seedless/readouts/expt8_sites150.tsv:gene150:mirna_seq"
    $PX python3 src/artifact/seedless_predict.py --checkpoint $O/cnn150.pt --out-dir $O $SPECS
  done
done
echo "ARMS_DONE $ARMS $(date)"
