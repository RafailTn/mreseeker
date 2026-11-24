#!/usr/bin/env python3
"""Stage 3.0 of the seed-less study: out-of-fold 150-nt CNN scores for every Manakov training row.

The OOF scores (data/seedless/oof/oof_train150.tsv) decide which training positives fall in the ARTIFACT cell
(seedless_arms.oof_scores -> flank_recentre_noart.py). Folds are chromosome-grouped: chromosomes are packed into
4 folds by LPT (largest row count first, each into the currently lightest fold; ties -> lowest fold), so no
chromosome straddles a fold. Training = base150 recipe, 15 epochs, 4-fold, --oof-out.

    python3 src/artifact/seedless_oof.py build     # -> data/seedless/oof/train150_chrfold.tsv + run.sh
    nohup bash data/seedless/oof/run.sh > data/seedless/oof/run.log 2>&1     # GPU (~4 x 15 epochs at 150 nt)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = f"{ROOT}/data/mre150/manakov_train_nochr2.tsv"
OUT = f"{ROOT}/data/seedless/oof"
COLS = ["gene150", "noncodingRNA", "noncodingRNA_fam", "label", "chr"]
K = 4


def lpt_folds(sizes, k=K):
    """chromosome -> fold name, LPT packing of row counts."""
    load = np.zeros(k)
    out = {}
    for c, n in sorted(sizes.items(), key=lambda x: (-x[1], str(x[0]))):
        f = int(np.argmin(load))
        out[c] = f"f{f}"
        load[f] += n
    return out


def build(out=OUT):
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(SRC, sep="\t", usecols=COLS, low_memory=False)[COLS]
    d.insert(0, "row", np.arange(len(d)))
    d["chr_fold"] = d.chr.map(lpt_folds(d.chr.value_counts().to_dict()))
    d.to_csv(f"{out}/train150_chrfold.tsv", sep="\t", index=False)
    print(d.groupby("chr_fold").size().to_string())
    with open(f"{out}/run.sh", "w") as fh:
        fh.write(f"""#!/usr/bin/env bash
# Stage 3.0 of the seed-less plan: out-of-fold 150-nt CNN scores for every training row.
# Chromosome-grouped 4-fold (chr_fold = LPT-balanced chromosome packs), base150 recipe, 15 epochs.
#   nohup bash data/seedless/oof/run.sh > data/seedless/oof/run.log 2>&1
set -euo pipefail
cd {ROOT}
export PYTHONUNBUFFERED=1 MRESEEKER_MRE_LEN=150
PX="pixi run --frozen --manifest-path dependencies/cnn/pixi.toml"
ARCH="--seq-pairing embed --pair-embed-dim 16 --seq-pool gem --seq-filters 64 --seq-dim 128 \\
      --n-conv-blocks 6 --n-pool-blocks 4 --block-pool max --ema"
D={os.path.relpath(out, ROOT)}
echo "######## OOF start $(date)"
$PX python3 src/training/cnn/cnn_branches_mirbind.py train \\
    --train $D/train150_chrfold.tsv --folds 4 --family-col chr_fold \\
    --mre-col gene150 --mirna-col noncodingRNA \\
    --epochs 15 --patience 6 --batch-size 256 --lr 1e-3 $ARCH \\
    --out $D/cnn150_oof.pt --oof-out $D/oof_train150.tsv
echo "OOF_DONE $(date)"
""")


if __name__ == "__main__":
    {"build": build}[sys.argv[1]](*sys.argv[2:])
