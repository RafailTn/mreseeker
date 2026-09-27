#!/usr/bin/env python3
"""
Query-conditioned miRNA binding-site localization on full 3' UTRs.

This is a companion model to cnn_branches_mirbind.py.  Instead of treating each
50-nt MRE as an independent binary-classification sample, this script collapses
observations to (transcript, miRNA) pairs and predicts a binding-site landscape
along the complete 3' UTR.

Core formulation
----------------
    transcript 3'UTR -> nucleotide-wise contextual representation
    miRNA query      -> contextual query-token representation
                    -> cross-attention (UTR positions query the miRNA)
                    -> per-position/site-start logits + pair-level ANY-SITE logit

The default model is deliberately hybrid rather than a full quadratic
Transformer over the UTR:

    UTR: 1D dilated residual CNN  -> nucleotide-wise features
    miRNA: small Transformer      -> query-token features (length <= 30)
    fusion: cross-attention       -> miRNA-conditioned UTR features
    head: 1D conv                 -> P(site-start at each UTR position)

This keeps attention complexity approximately O(L_UTR * L_miRNA), rather than
O(L_UTR^2), while still allowing Transformer-style conditioning.

Input format
------------
A CSV or TSV containing at least:

    transcript_id     transcript identifier
    utr3_sequence     full 3' UTR sequence (5' -> 3')
    mirna_sequence    miRNA sequence (18-30 nt)
    label              1 = observed interaction, 0 = negative pair

For positive rows, the preferred site coordinate column is:

    site_start        0-based start of the observed chimeric target fragment

If site_start is absent, the script can infer it from mre_sequence by locating
that sequence in utr3_sequence.  Inference succeeds only when the fragment
occurs exactly once.  For AGO2-chimeric data, an explicit mapped coordinate is
strongly preferred.

Multiple rows with the same (transcript_id, mirna_sequence) are collapsed into
one sample.  All distinct positive site_start coordinates are retained.

Important labeling assumption
-----------------------------
For a positive pair, positions not observed as chimeric sites are treated as
UNKNOWN by the localization loss; the main positive-pair loss asks the model to
rank annotated sites highly rather than declaring every unobserved position a
true negative.  For negative transcript-miRNA pairs (label=0), all valid
site-start positions are treated as negatives.  This is intentionally different
from naive per-base BCE over every unobserved position.

Example
-------
Train:
    python utr_mirna_locator.py train \
        --input data/chimera_with_utr.tsv \
        --out checkpoints/utr_locator.pt \
        --transcript-col transcript_id \
        --utr-col utr3_sequence \
        --mirna-col mirna_sequence \
        --label-col label \
        --site-start-col site_start \
        --group-col transcript_id \
        --epochs 40

Predict:
    python utr_mirna_locator.py predict \
        --checkpoint checkpoints/utr_locator.pt \
        --input data/test.tsv \
        --output predictions/utr_locator.tsv

Output columns include pair-level probability, top predicted site starts,
and the observed site starts when labels are available.  With --track-out,
full variable-length per-position probability tracks are also saved as NPZ.

Design notes
------------
1. site_start refers to the beginning of the 50-nt chimera/MRE window by
   default.  The model therefore learns "where would the observed 50-nt target
   fragment start?" rather than assuming that the whole 50-nt context is the
   molecular binding footprint.
2. The --site-window argument defines the valid start positions: positions
   j <= len(UTR)-site_window are scored.
3. For positive pairs with multiple annotated sites, the localization target is
   a normalized multi-positive distribution over those starts.
4. A separate ANY-SITE head learns P(any observed/true interaction for this
   miRNA-transcript pair), enabling an explicit NO-SITE outcome.

Dependencies: numpy, pandas, torch, scikit-learn (recommended for stratified
               group splitting and AUROC/AUPRC metrics).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Reproducibility / IO
# ---------------------------------------------------------------------------

VOCAB = {"A": 0, "C": 1, "G": 2, "U": 3, "N": 4}
PAD = 5
VOCAB_SIZE = 6


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_table(path: str | Path) -> pd.DataFrame:
    path = str(path)
    sep = "\t" if path.lower().endswith((".tsv", ".tab", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def norm_seq(s: str) -> str:
    return str(s).upper().replace("T", "U").replace(" ", "")


def encode_sequence(s: str) -> list[int]:
    return [VOCAB.get(ch, 4) for ch in norm_seq(s)]


# ---------------------------------------------------------------------------
# Collapsing raw chimera rows to transcript-miRNA samples
# ---------------------------------------------------------------------------

@dataclass
class CollapsedSample:
    transcript_id: str
    utr_sequence: str
    mirna_sequence: str
    label: int
    site_starts: list[int]


def _infer_site_start(utr: str, mre: str) -> Optional[int]:
    if not mre:
        return None
    u = norm_seq(utr)
    m = norm_seq(mre)
    hits: list[int] = []
    start = u.find(m)
    while start >= 0:
        hits.append(start)
        start = u.find(m, start + 1)
    if len(hits) == 1:
        return hits[0]
    return None


def collapse_samples(
    df: pd.DataFrame,
    transcript_col: str,
    utr_col: str,
    mirna_col: str,
    label_col: str,
    site_start_col: Optional[str],
    mre_col: Optional[str],
    site_window: int,
    require_explicit_negative: bool = True,
) -> list[CollapsedSample]:
    required = [transcript_col, utr_col, mirna_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    label_available = label_col in df.columns
    if not label_available and require_explicit_negative:
        raise ValueError(
            f"Training requires {label_col!r}. Use 0/1 pair labels so the model "
            "can learn an explicit NO-SITE outcome."
        )
    if site_start_col and site_start_col not in df.columns:
        raise ValueError(f"site-start column {site_start_col!r} is missing")
    if site_start_col is None and mre_col and mre_col not in df.columns:
        raise ValueError(f"mre column {mre_col!r} is missing")

    grouped: dict[tuple[str, str], CollapsedSample] = {}
    inferred = 0
    unresolved_positive = 0

    for row in df.itertuples(index=False):
        row_d = row._asdict()
        tid = str(row_d[transcript_col])
        utr = norm_seq(row_d[utr_col])
        mir = norm_seq(row_d[mirna_col])
        label = int(row_d[label_col]) if label_available else 0
        if label not in (0, 1):
            raise ValueError(f"label must be 0/1, got {label} for {tid}/{mir}")
        if not tid or not utr or not mir:
            continue

        key = (tid, mir)
        if key not in grouped:
            grouped[key] = CollapsedSample(tid, utr, mir, label, [])
        else:
            # A pair is positive if any observed row is positive.
            grouped[key].label = max(grouped[key].label, label)
            if grouped[key].utr_sequence != utr:
                raise ValueError(f"Multiple different UTR sequences for transcript {tid}")

        if label == 1 or not label_available:
            start: Optional[int] = None
            if site_start_col:
                raw = row_d[site_start_col]
                if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
                    try:
                        start = int(raw)
                    except (TypeError, ValueError):
                        start = None
            if start is None and mre_col:
                start = _infer_site_start(utr, str(row_d[mre_col]))
                if start is not None:
                    inferred += 1
            if start is None:
                unresolved_positive += 1
                continue
            if start < 0 or start + site_window > len(utr):
                continue
            grouped[key].site_starts.append(start)

    samples = list(grouped.values())
    for s in samples:
        s.site_starts = sorted(set(s.site_starts))
        if s.label == 1 and not s.site_starts:
            unresolved_positive += 1

    if unresolved_positive:
        print(
            f"[collapse] warning: {unresolved_positive} positive observations/pairs "
            "had no usable site coordinate and therefore contribute no localization target."
        )
    if inferred:
        print(f"[collapse] inferred {inferred} site_start values from unique MRE matches")

    if require_explicit_negative and not any(s.label == 0 for s in samples):
        raise ValueError(
            "No label=0 transcript-miRNA pairs remain after collapsing. An explicit "
            "negative-pair set is required for the NO-SITE head."
        )

    print(
        f"[collapse] raw rows={len(df):,} -> transcript-miRNA pairs={len(samples):,} "
        f"(positive={sum(s.label for s in samples):,}, negative={sum(s.label == 0 for s in samples):,})"
    )
    return samples


# ---------------------------------------------------------------------------
# Dataset / padding
# ---------------------------------------------------------------------------

class UTRMiRNADataset(Dataset):
    def __init__(self, samples: list[CollapsedSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CollapsedSample:
        return self.samples[idx]


def _pad_1d(seqs: list[list[int]], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) else 0
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, seq in enumerate(seqs):
        n = len(seq)
        if n:
            out[i, :n] = torch.tensor(seq, dtype=torch.long)
            mask[i, :n] = True
    return out, mask


def collate(samples: list[CollapsedSample]) -> dict[str, object]:
    utr, utr_mask = _pad_1d([encode_sequence(x.utr_sequence) for x in samples], PAD)
    mir, mir_mask = _pad_1d([encode_sequence(x.mirna_sequence) for x in samples], PAD)
    # Positive-site targets are kept in Python lists because UTRs are variable length.
    return {
        "utr": utr,
        "utr_mask": utr_mask,
        "mirna": mir,
        "mirna_mask": mir_mask,
        "labels": torch.tensor([x.label for x in samples], dtype=torch.float32),
        "site_starts": [x.site_starts for x in samples],
        "transcript_ids": [x.transcript_id for x in samples],
        "mirna_sequences": [x.mirna_sequence for x in samples],
        "utr_sequences": [x.utr_sequence for x in samples],
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(dim, hidden * 2)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return self.out(a * F.gelu(b))


class ResidualDilatedConv(nn.Module):
    def __init__(self, channels: int, kernel: int = 7, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.norm = nn.GroupNorm(8, channels)
        self.dw = nn.Conv1d(
            channels, channels, kernel_size=kernel, dilation=dilation,
            padding=padding, groups=channels, bias=False
        )
        self.pw1 = nn.Conv1d(channels, channels * 2, kernel_size=1, bias=False)
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.dw(h)
        a, b = self.pw1(h).chunk(2, dim=1)
        h = a * F.silu(b)
        h = self.pw2(h)
        h = self.dropout(h)
        return x + h


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: B,L,D; mask: B,L True for real tokens
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=-1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)


class UTRMiRNALocator(nn.Module):
    """Hybrid CNN + Transformer query-conditioned localization model."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        mirna_layers: int = 2,
        utr_blocks: int = 6,
        dropout: float = 0.15,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.utr_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8, 16, 1][:utr_blocks]
        self.utr_blocks = nn.ModuleList([
            ResidualDilatedConv(d_model, kernel=7, dilation=d, dropout=dropout)
            for d in dilations
        ])

        mir_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mirna_transformer = nn.TransformerEncoder(mir_layer, num_layers=mirna_layers)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_ffn_norm = nn.LayerNorm(d_model)
        self.cross_ffn = GEGLU(d_model, d_model * 4)

        self.site_head = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model // 2, 1, kernel_size=1),
        )

        self.utr_pool = AttentionPool(d_model)
        self.none_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode_utr(self, utr: torch.Tensor) -> torch.Tensor:
        # B,L -> B,L,D -> B,D,L
        x = self.embedding(utr).transpose(1, 2)
        x = self.utr_stem(x)
        for block in self.utr_blocks:
            x = block(x)
        return x.transpose(1, 2)

    def encode_mirna(self, mirna: torch.Tensor, mirna_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(mirna)
        x = self.mirna_transformer(x, src_key_padding_mask=~mirna_mask)
        return x

    def forward(
        self,
        utr: torch.Tensor,
        utr_mask: torch.Tensor,
        mirna: torch.Tensor,
        mirna_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        utr_h = self.encode_utr(utr)          # B,L,D
        mir_h = self.encode_mirna(mirna, mirna_mask)  # B,M,D

        # UTR positions query the miRNA. Complexity is O(L_UTR * L_miRNA).
        cross, _ = self.cross_attn(
            query=utr_h,
            key=mir_h,
            value=mir_h,
            key_padding_mask=~mirna_mask,
            need_weights=False,
        )
        h = self.cross_norm(utr_h + cross)
        h = self.cross_ffn_norm(h + self.cross_ffn(h))

        site_logits = self.site_head(h.transpose(1, 2)).squeeze(1)  # B,L

        pooled = self.utr_pool(h, utr_mask)
        mir_valid = mir_h.masked_fill(~mirna_mask.unsqueeze(-1), 0.0)
        mir_den = mirna_mask.sum(dim=1, keepdim=True).clamp_min(1).to(mir_h.dtype)
        mir_pool = mir_valid.sum(dim=1) / mir_den
        none_logit = self.none_head(torch.cat([pooled, mir_pool], dim=-1)).squeeze(-1)

        return {"site_logits": site_logits, "none_logits": none_logit}


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------


def valid_site_mask(utr_mask: torch.Tensor, site_window: int) -> torch.Tensor:
    # A site-start j is valid when the complete window [j, j+site_window) fits.
    lengths = utr_mask.sum(dim=1)
    pos = torch.arange(utr_mask.shape[1], device=utr_mask.device).unsqueeze(0)
    return pos + site_window <= lengths.unsqueeze(1)


def localization_distribution(
    site_starts: list[list[int]],
    valid_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build multi-positive normalized targets for positive pairs."""
    bsz, L = valid_mask.shape
    target = torch.zeros((bsz, L), device=valid_mask.device, dtype=dtype)
    for i, starts in enumerate(site_starts):
        usable = [p for p in starts if 0 <= p < L and bool(valid_mask[i, p])]
        if usable:
            v = 1.0 / len(usable)
            target[i, usable] = v
    return target


def compute_loss(
    out: dict[str, torch.Tensor],
    labels: torch.Tensor,
    site_starts: list[list[int]],
    valid_mask: torch.Tensor,
    site_weight: float = 1.0,
    negative_site_weight: float = 0.20,
) -> tuple[torch.Tensor, dict[str, float]]:
    site_logits = out["site_logits"]
    none_logits = out["none_logits"]

    any_loss = F.binary_cross_entropy_with_logits(none_logits, labels)

    pos = labels > 0.5
    neg = ~pos

    # Positive pairs: listwise localization loss over observed sites only.
    pos_loss = site_logits.new_zeros(())
    pos_count = int(pos.sum().item())
    if pos_count:
        logits_pos = site_logits[pos]
        mask_pos = valid_mask[pos]
        targets_pos = localization_distribution(
            [site_starts[i] for i, flag in enumerate(pos.tolist()) if flag],
            mask_pos,
            logits_pos.dtype,
        )
        lsm = logits_pos.masked_fill(~mask_pos, torch.finfo(logits_pos.dtype).min)
        # Cross-entropy with a multi-positive normalized target.
        logp = F.log_softmax(lsm, dim=-1)
        pos_loss = -(targets_pos * logp).sum(dim=-1).mean()

    # Negative pairs: every valid candidate site is a negative. This teaches the
    # localization track to remain quiet for transcript-miRNA pairs labelled 0.
    neg_loss = site_logits.new_zeros(())
    neg_count = int(neg.sum().item())
    if neg_count:
        logits_neg = site_logits[neg]
        mask_neg = valid_mask[neg]
        zeros = torch.zeros_like(logits_neg)
        per = F.binary_cross_entropy_with_logits(logits_neg, zeros, reduction="none")
        denom = mask_neg.sum().clamp_min(1)
        neg_loss = (per * mask_neg.to(per.dtype)).sum() / denom

    loc_loss = pos_loss + negative_site_weight * neg_loss
    total = any_loss + site_weight * loc_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "any_loss": float(any_loss.detach().cpu()),
        "pos_loc_loss": float(pos_loss.detach().cpu()),
        "neg_loc_loss": float(neg_loss.detach().cpu()),
    }


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def localization_metrics(
    site_logits: list[np.ndarray],
    labels: np.ndarray,
    site_starts: list[list[int]],
    valid_masks: list[np.ndarray],
    ks: tuple[int, ...] = (1, 5, 10),
    tolerance: int = 2,
) -> dict[str, float]:
    out: dict[str, float] = {}
    positive_rows = np.where(labels > 0.5)[0]
    if len(positive_rows) == 0:
        return {f"loc_hit@{k}": float("nan") for k in ks} | {"site_mrr": float("nan")}

    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    used = 0
    for i in positive_rows:
        probs = safe_sigmoid(site_logits[i])
        valid = valid_masks[i]
        true_sites = [x for x in site_starts[i] if 0 <= x < len(probs) and valid[x]]
        if not true_sites:
            continue
        order = np.argsort(np.where(valid, probs, -np.inf))[::-1]
        used += 1
        for k in ks:
            top = order[:k]
            ok = any(min(abs(int(p) - int(t)) for t in true_sites) <= tolerance for p in top)
            hits[k] += int(ok)
        rr = 0.0
        for rank, p in enumerate(order, start=1):
            if any(abs(int(p) - int(t)) <= tolerance for t in true_sites):
                rr = 1.0 / rank
                break
        rr_sum += rr

    for k in ks:
        out[f"loc_hit@{k}"] = hits[k] / max(used, 1)
    out["site_mrr"] = rr_sum / max(used, 1)
    return out


# ---------------------------------------------------------------------------
# Train/eval utilities
# ---------------------------------------------------------------------------


def split_samples(
    samples: list[CollapsedSample],
    group_ids: list[str],
    seed: int,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
) -> tuple[list[int], list[int], list[int]]:
    n = len(samples)
    labels = np.array([s.label for s in samples], dtype=np.int64)
    groups = np.asarray(group_ids)
    idx = np.arange(n)

    if HAS_SKLEARN:
        # Use stratified grouped K-fold approximately matching the requested fractions.
        n_splits = max(3, round(1.0 / max(test_frac, 1e-3)))
        n_splits = min(max(n_splits, 3), 10)
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = list(sgkf.split(np.zeros(n), labels, groups))
            test_idx = folds[0][1]
            rem = np.setdiff1d(idx, test_idx, assume_unique=False)
            rem_labels = labels[rem]
            rem_groups = groups[rem]
            n_val_splits = max(3, round(1.0 / max(val_frac / max(1 - test_frac, 1e-6), 1e-3)))
            n_val_splits = min(max(n_val_splits, 3), 10)
            sgkf2 = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=seed + 1)
            vf = list(sgkf2.split(np.zeros(len(rem)), rem_labels, rem_groups))[0][1]
            val_idx = rem[vf]
            train_idx = np.setdiff1d(rem, val_idx, assume_unique=False)
            return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
        except Exception as exc:
            print(f"[split] StratifiedGroupKFold failed ({exc}); falling back to GroupShuffleSplit")

        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        trainval, test = next(gss.split(np.zeros(n), labels, groups))
        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=val_frac / max(1.0 - test_frac, 1e-6),
            random_state=seed + 1,
        )
        train, val = next(gss2.split(np.zeros(len(trainval)), labels[trainval], groups[trainval]))
        return trainval[train].tolist(), trainval[val].tolist(), test.tolist()

    # Simple deterministic group split without sklearn.
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_test_groups = max(1, int(round(len(unique_groups) * test_frac)))
    test_groups = set(unique_groups[:n_test_groups])
    rem_groups = [g for g in unique_groups if g not in test_groups]
    n_val_groups = max(1, int(round(len(unique_groups) * val_frac)))
    val_groups = set(rem_groups[:n_val_groups])
    train = [i for i, g in enumerate(groups) if g not in test_groups and g not in val_groups]
    val = [i for i, g in enumerate(groups) if g in val_groups]
    test = [i for i, g in enumerate(groups) if g in test_groups]
    return train, val, test


def make_loader(samples: list[CollapsedSample], batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        UTRMiRNADataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
        collate_fn=collate,
    )


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "utr": batch["utr"].to(device, non_blocking=True),
        "utr_mask": batch["utr_mask"].to(device, non_blocking=True),
        "mirna": batch["mirna"].to(device, non_blocking=True),
        "mirna_mask": batch["mirna_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "site_starts": batch["site_starts"],
        "transcript_ids": batch["transcript_ids"],
        "mirna_sequences": batch["mirna_sequences"],
        "utr_sequences": batch["utr_sequences"],
    }


def evaluate(
    model: UTRMiRNALocator,
    loader: DataLoader,
    device: torch.device,
    site_window: int,
    site_tolerance: int,
) -> dict[str, float]:
    model.eval()
    all_none: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_site: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    all_starts: list[list[int]] = []
    loss_values: list[float] = []

    with torch.no_grad():
        for batch0 in loader:
            batch = move_batch(batch0, device)
            out = model(batch["utr"], batch["utr_mask"], batch["mirna"], batch["mirna_mask"])
            valid = valid_site_mask(batch["utr_mask"], site_window)
            loss, _ = compute_loss(out, batch["labels"], batch["site_starts"], valid)
            loss_values.append(float(loss.detach().cpu()))
            all_none.append(out["none_logits"].detach().cpu().numpy())
            all_labels.append(batch["labels"].detach().cpu().numpy())
            max_l = out["site_logits"].shape[1]
            all_site.extend(list(out["site_logits"].detach().cpu().numpy()))
            all_valid.extend(list(valid.detach().cpu().numpy()))
            all_starts.extend(batch["site_starts"])

    none_logits = np.concatenate(all_none)
    labels = np.concatenate(all_labels)
    m: dict[str, float] = {"loss": float(np.mean(loss_values))}
    none_probs = safe_sigmoid(none_logits)
    if HAS_SKLEARN and len(np.unique(labels)) == 2:
        m["pair_auroc"] = float(roc_auc_score(labels, none_probs))
        m["pair_auprc"] = float(average_precision_score(labels, none_probs))
    m.update(localization_metrics(all_site, labels, all_starts, all_valid, tolerance=site_tolerance))
    return m


def train_one_epoch(
    model: UTRMiRNALocator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    site_window: int,
    grad_clip: float,
    site_weight: float,
    negative_site_weight: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n = 0
    amp = device.type == "cuda"

    for batch0 in loader:
        batch = move_batch(batch0, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(batch["utr"], batch["utr_mask"], batch["mirna"], batch["mirna_mask"])
            valid = valid_site_mask(batch["utr_mask"], site_window)
            loss, stats = compute_loss(
                out,
                batch["labels"],
                batch["site_starts"],
                valid,
                site_weight=site_weight,
                negative_site_weight=negative_site_weight,
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        b = len(batch["labels"])
        n += b
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v * b

    return {k: v / max(n, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def top_sites(
    site_logits: np.ndarray,
    valid: np.ndarray,
    k: int,
    min_prob: float,
    nonmax_radius: int,
) -> list[tuple[int, float]]:
    probs = safe_sigmoid(site_logits.copy())
    probs[~valid] = 0.0
    picks: list[tuple[int, float]] = []
    work = probs.copy()
    for _ in range(k):
        p = int(np.argmax(work))
        score = float(work[p])
        if score < min_prob or not valid[p]:
            break
        picks.append((p, score))
        lo = max(0, p - nonmax_radius)
        hi = min(len(work), p + nonmax_radius + 1)
        work[lo:hi] = -1.0
    return picks


def cmd_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = UTRMiRNALocator(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        mirna_layers=cfg["mirna_layers"],
        utr_blocks=cfg["utr_blocks"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    df = read_table(args.input)
    site_start_col = args.site_start_col if args.site_start_col in df.columns else None
    mre_col = args.mre_col if args.mre_col and args.mre_col in df.columns else None
    samples = collapse_samples(
        df,
        args.transcript_col,
        args.utr_col,
        args.mirna_col,
        args.label_col,
        site_start_col,
        mre_col,
        args.site_window,
        require_explicit_negative=False,
    )
    loader = make_loader(samples, args.batch_size, False, args.num_workers)

    rows = []
    tracks: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch0 in loader:
            batch = move_batch(batch0, device)
            out = model(batch["utr"], batch["utr_mask"], batch["mirna"], batch["mirna_mask"])
            none_probs = torch.sigmoid(out["none_logits"]).cpu().numpy()
            site_logits = out["site_logits"].cpu().numpy()
            valid = valid_site_mask(batch["utr_mask"], args.site_window).cpu().numpy()

            for i in range(len(none_probs)):
                picks = top_sites(
                    site_logits[i], valid[i], args.top_k, args.min_site_prob, args.nms_radius
                )
                site_text = ";".join(f"{p}:{s:.6f}" for p, s in picks)
                observed = ";".join(str(x) for x in batch["site_starts"][i])
                key = f"{batch['transcript_ids'][i]}||{batch['mirna_sequences'][i]}"
                utr_len = int(batch["utr_mask"][i].sum().item())
                tracks[key] = safe_sigmoid(site_logits[i][:utr_len]).astype(np.float32)
                rows.append({
                    args.transcript_col: batch["transcript_ids"][i],
                    args.mirna_col: batch["mirna_sequences"][i],
                    "utr_length": utr_len,
                    "pred_any_site_prob": float(none_probs[i]),
                    "predicted_sites": site_text,
                    "observed_site_starts": observed,
                })

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, sep="\t", index=False)
    print(f"[predict] wrote {args.output} ({len(out_df):,} transcript-miRNA pairs)")

    if args.track_out:
        Path(args.track_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.track_out, **{k.replace("/", "_"): v for k, v in tracks.items()})
        print(f"[predict] wrote probability tracks: {args.track_out}")


# ---------------------------------------------------------------------------
# CLI / training command
# ---------------------------------------------------------------------------


def add_data_args(p: argparse.ArgumentParser, require_label: bool = True) -> None:
    p.add_argument("--input", required=True)
    p.add_argument("--transcript-col", default="transcript_id")
    p.add_argument("--utr-col", default="utr3_sequence")
    p.add_argument("--mirna-col", default="mirna_sequence")
    p.add_argument("--label-col", default="label", required=False)
    p.add_argument("--site-start-col", default="site_start")
    p.add_argument(
        "--mre-col", default=None,
        help="Optional 50-nt chimeric MRE column used to infer site_start when the coordinate is absent.",
    )
    p.add_argument("--site-window", type=int, default=50)
    if require_label:
        p.set_defaults(_require_label=True)


def cmd_train(args: argparse.Namespace) -> None:
    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)

    df = read_table(args.input)
    site_start_col = args.site_start_col if args.site_start_col in df.columns else None
    mre_col = args.mre_col if args.mre_col and args.mre_col in df.columns else None
    if site_start_col is None and mre_col is None:
        print("[train] no site coordinate source is available; positive pairs cannot contribute localization targets")

    samples = collapse_samples(
        df,
        args.transcript_col,
        args.utr_col,
        args.mirna_col,
        args.label_col,
        site_start_col,
        mre_col,
        args.site_window,
        require_explicit_negative=not args.allow_no_negatives,
    )
    if not any(s.label == 1 and s.site_starts for s in samples):
        raise ValueError("No positive transcript-miRNA pair has a usable site coordinate")

    group_col = args.group_col
    if group_col == args.transcript_col:
        group_ids = [s.transcript_id for s in samples]
    elif group_col in df.columns:
        # Build a transcript -> group lookup.  This is useful if your annotation
        # has a stable gene-level grouping column distinct from transcript_id.
        mapping = {
            str(r[args.transcript_col]): str(r[group_col])
            for _, r in df[[args.transcript_col, group_col]].drop_duplicates().iterrows()
        }
        group_ids = [mapping.get(s.transcript_id, s.transcript_id) for s in samples]
    else:
        raise ValueError(f"group column {group_col!r} not found")

    train_i, val_i, test_i = split_samples(
        samples, group_ids, args.seed, args.test_frac, args.val_frac
    )
    train_samples = [samples[i] for i in train_i]
    val_samples = [samples[i] for i in val_i]
    test_samples = [samples[i] for i in test_i]
    print(
        f"[split] train={len(train_samples):,} val={len(val_samples):,} test={len(test_samples):,} "
        f"groups={len(set(group_ids)):,}"
    )

    train_loader = make_loader(train_samples, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_samples, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_samples, args.batch_size, False, args.num_workers)

    model_cfg = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "mirna_layers": args.mirna_layers,
        "utr_blocks": args.utr_blocks,
        "dropout": args.dropout,
    }
    model = UTRMiRNALocator(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = math.inf if args.checkpoint_metric == "site_nll" else -math.inf
    patience_left = args.patience
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            args.site_window, args.grad_clip, args.site_loss_weight,
            args.negative_site_weight,
        )
        va = evaluate(model, val_loader, device, args.site_window, args.site_tolerance)
        scheduler.step()

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(
            f"epoch {epoch:03d} | train loss={tr['loss']:.4f} | "
            f"val loss={va['loss']:.4f} | pair AP={va.get('pair_auprc', float('nan')):.4f} | "
            f"loc@1={va.get('loc_hit@1', float('nan')):.4f} | "
            f"MRR={va.get('site_mrr', float('nan')):.4f}"
        )

        if args.checkpoint_metric == "site_nll":
            metric = va["loss"]
            improved = metric < best_metric
        elif args.checkpoint_metric == "pair_auprc":
            metric = va.get("pair_auprc", float("nan"))
            improved = np.isfinite(metric) and metric > best_metric
        else:
            metric = va.get("loc_hit@1", float("nan"))
            improved = np.isfinite(metric) and metric > best_metric

        if improved or best_state is None:
            best_metric = metric
            patience_left = args.patience
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": model_cfg,
                "cli": vars(args),
            }
            print(f"  [checkpoint] new best {args.checkpoint_metric}={metric:.6f}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("  [early-stop]")
                break

    assert best_state is not None
    model.load_state_dict(best_state["model_state"])
    test = evaluate(model, test_loader, device, args.site_window, args.site_tolerance)
    print("[test]", " ".join(f"{k}={v:.5f}" for k, v in test.items() if np.isfinite(v)))

    best_state["history"] = history
    best_state["test_metrics"] = test
    best_state["split_sizes"] = {
        "train": len(train_samples), "val": len(val_samples), "test": len(test_samples)
    }
    best_state["task"] = "query_conditioned_full_3UTR_site_localization"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out)
    print(f"[checkpoint] wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_data_args(tr)
    tr.add_argument("--group-col", default="transcript_id")
    tr.add_argument("--test-frac", type=float, default=0.20)
    tr.add_argument("--val-frac", type=float, default=0.10)
    tr.add_argument("--allow-no-negatives", action="store_true")
    tr.add_argument("--d-model", type=int, default=128)
    tr.add_argument("--n-heads", type=int, default=4)
    tr.add_argument("--mirna-layers", type=int, default=2)
    tr.add_argument("--utr-blocks", type=int, default=6)
    tr.add_argument("--dropout", type=float, default=0.15)
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight-decay", type=float, default=1e-4)
    tr.add_argument("--grad-clip", type=float, default=1.0)
    tr.add_argument("--site-loss-weight", type=float, default=1.0)
    tr.add_argument("--negative-site-weight", type=float, default=0.20)
    tr.add_argument("--site-tolerance", type=int, default=2)
    tr.add_argument("--patience", type=int, default=8)
    tr.add_argument("--checkpoint-metric", choices=["site_nll", "pair_auprc", "loc@1"], default="site_nll")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--deterministic", action="store_true")
    tr.add_argument("--num-workers", type=int, default=4)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--out", default="checkpoints/utr_mirna_locator.pt")

    pr = sub.add_parser("predict", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_data_args(pr, require_label=False)
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--track-out", default=None)
    pr.add_argument("--top-k", type=int, default=10)
    pr.add_argument("--min-site-prob", type=float, default=0.10)
    pr.add_argument("--nms-radius", type=int, default=10)
    pr.add_argument("--batch-size", type=int, default=8)
    pr.add_argument("--num-workers", type=int, default=4)
    pr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
