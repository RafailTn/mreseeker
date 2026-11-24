#!/usr/bin/env python3
"""Score several row-aligned tables with one CNN checkpoint, writing only probabilities (float32 .npy).

Same path as src/cnn/predict_cnn.py --no-dedup (from_df, shuffle=False, sigmoid of predict_logits), but the
model is loaded once and only the probability vector is kept. MRESEEKER_MRE_LEN must match the checkpoint.

    MRESEEKER_MRE_LEN=150 python3 src/artifact/seedless_predict.py --checkpoint CK --out-dir DIR \
        NAME=PATH[:MRE_COL[:MIRNA_COL]] ...           -> DIR/pred_NAME.npy
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "cnn"))
from cnn_branches_mirbind import MiRNAInteractionDataset, predict_logits, _load_ckpt_model, MRE_LEN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = _load_ckpt_model(a.checkpoint, dev)
    print(f"{a.checkpoint}: epoch {ck.get('epoch')}, weights {ck.get('weights')}, MRE_LEN {MRE_LEN}, "
          f"ckpt mre_len {ck.get('mre_len')}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    for spec in a.specs:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        path, mre_col, mirna_col = parts[0], (parts[1] if len(parts) > 1 else "gene150"), \
            (parts[2] if len(parts) > 2 else "noncodingRNA")
        out = f"{a.out_dir}/pred_{name}.npy"
        if os.path.exists(out) and not a.force:
            print(f"  {name}: exists, skipped", flush=True)
            continue
        df = pd.read_csv(path, sep="\t", usecols=[mre_col, mirna_col], low_memory=False)
        lens = df[mre_col].astype(str).str.len()
        if (lens != MRE_LEN).any():
            sys.exit(f"ERROR {name}: {int((lens != MRE_LEN).sum())} rows with MRE length != {MRE_LEN}")
        ds = MiRNAInteractionDataset.from_df(df, has_labels=False, mre_col=mre_col, mirna_col=mirna_col)
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True)
        logits, _ = predict_logits(model, loader, dev)
        p = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        np.save(out, p)
        print(f"  {name}: {len(p):,} rows -> {out} (mean p {p.mean():.4f})", flush=True)


if __name__ == "__main__":
    main()
