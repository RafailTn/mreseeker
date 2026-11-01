#!/usr/bin/env python3
"""
Shared miRNA–MRE binding-type classification (numpy-only, no scipy/torch).

Single source of truth for the deterministic binding-type scheme used by both
``error_analysis.py`` (error dumps) and ``cnn_branches_mirbind.py`` (negative
binding-type undersampling).  The classification works off the best antiparallel
register of the raw sequences — no IntaRNA structure needed — and mirrors the
scheme in ``feature_extraction.py`` (bulge tags omitted in the strict-register
model).

Two entry points share one core:
  - classify_binding_type(mirna_dna, mre_dna)  — nucleotide strings (T→U ok)
  - classify_binding_type_idx(mi, ti)          — pre-tokenised index arrays
                                                 (A=0,C=1,G=2,U=3, pad/unknown=4)
The string and index encodings use the *same* nucleotide indices, so token
arrays produced elsewhere (e.g. the dataset's mirna_idx/mre_idx) classify
identically to the equivalent strings.
"""
from __future__ import annotations

import numpy as np

# Bump when the classification logic changes so cached labels are invalidated.
# v2: the '1A' sub-types test the TARGET base opposite miRNA position 1 (t1A),
#     not the miRNA's own first base, and no longer require position 1 unpaired.
CLASSIFIER_VERSION = 2

# Non-canonical (decoy-like) categories with stable names — the ones worth
# targeting for undersampling.  Canonical "*mer*" subtypes carry variable
# suffixes (.GU/.3prime/1A) and are intentionally not listed.
UNDERSAMPLE_CATEGORIES = ("seedless", "3prime.compensatory", "3prime", "centered")

# ---------------------------------------------------------------------------
# Watson-Crick / G·U pairing tables (RNA, 5-index so pad index 4 → 0)
# ---------------------------------------------------------------------------

_NUC = {"A": 0, "C": 1, "G": 2, "U": 3}
_WC  = np.zeros((5, 5), dtype=np.float32)
_GU  = np.zeros((5, 5), dtype=np.float32)
for _a, _b in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]:
    _WC[_NUC[_a], _NUC[_b]] = 1.0
for _a, _b in [("G", "U"), ("U", "G")]:
    _GU[_NUC[_a], _NUC[_b]] = 1.0

_MAX_MI = 30
_MAX_TI = 50


def _tok(seq: str, length: int) -> np.ndarray:
    out = np.full(length, 4, dtype=np.intp)
    for i, c in enumerate(seq.upper().replace("T", "U")[:length]):
        out[i] = _NUC.get(c, 4)
    return out


# ---------------------------------------------------------------------------
# Antidiagonal feature extraction + classification (index-array core)
# ---------------------------------------------------------------------------

def _max_consec(arr: np.ndarray) -> int:
    best_run = curr = 0
    for v in arr:
        curr = curr + 1 if v else 0
        best_run = max(best_run, curr)
    return best_run


def _antidiag_features_idx(mi: np.ndarray, ti: np.ndarray) -> dict:
    """Best antiparallel register features from tokenised index arrays.

    ``mi`` / ``ti`` are integer index arrays of length _MAX_MI / _MAX_TI
    (A=0,C=1,G=2,U=3, pad/unknown=4).  Returns the richer per-position info the
    classification cascade needs.
    """
    mi = np.asarray(mi, dtype=np.intp)
    ti = np.asarray(ti, dtype=np.intp)

    wc   = _WC[mi[:, None], ti[None, :]]          # (30, 50)
    gu   = _GU[mi[:, None], ti[None, :]]
    pair = wc + gu

    P, Q   = np.indices((_MAX_MI, _MAX_TI))
    d_flat = (P + Q).ravel()
    pair_d = np.bincount(d_flat, weights=pair.ravel(),
                         minlength=_MAX_MI + _MAX_TI - 1)
    best = int(np.argmax(pair_d))

    mi_len = int((mi != 4).sum())
    ti_len = int((ti != 4).sum())

    p_wc = np.zeros(_MAX_MI, dtype=bool)
    p_gu = np.zeros(_MAX_MI, dtype=bool)
    for i in range(min(mi_len, _MAX_MI)):
        j = best - i
        if 0 <= j < ti_len and ti[j] != 4:
            p_wc[i] = bool(_WC[mi[i], ti[j]])
            p_gu[i] = bool(_GU[mi[i], ti[j]])

    # t1: the target base opposite miRNA position 1 — on this register, j = best.
    t1 = int(ti[best]) if 0 <= best < ti_len else 4
    return _features_from_pairs(p_wc, p_gu, t1, mi_len)


def _features_from_pairs(p_wc: np.ndarray, p_gu: np.ndarray,
                         t1: int, mi_len: int) -> dict:
    """Classification features from per-miRNA-position pairing flags.

    ``p_wc`` / ``p_gu`` (length _MAX_MI, bool) say whether miRNA position i is
    Watson-Crick / G·U paired in whatever duplex the caller chose — the strict
    best register above, or a bulge-tolerant alignment (duplex_decode.py).
    Keeping this separate is what lets both paths share one cascade.

    ``t1`` is the nucleotide index of the target base opposite miRNA position 1
    (4 if there is none) — the base AGO reads for the "A1" site sub-types.  It is
    recognised in its own pocket whatever the miRNA's position-1 base is, so this
    is deliberately the target base, not ``mi[0]``.
    """
    any_pair = p_wc | p_gu

    # seed = miRNA positions 2–8, 0-indexed 1–7
    seed     = any_pair[1:8]
    seed_gu  = p_gu[1:8]

    # start_match: first paired miRNA position (1-indexed)
    start_match = 99
    for i in range(mi_len):
        if any_pair[i]:
            start_match = i + 1
            break

    return {
        "consec_seed":     _max_consec(seed),
        "total_1_9":       int(any_pair[0:9].sum()),
        "start_match":     start_match,
        "gu_in_seed":      int(seed_gu.sum()),
        "eff_3prime":      int(p_wc[8:mi_len].sum()),    # WC-only outside seed
        "consec_centered": _max_consec(any_pair[4:16]),  # positions 5–16
        "t1_A":            t1 == _NUC["A"],
    }


def _classify_from_features(f: dict) -> str:
    consec = f["consec_seed"]
    start  = f["start_match"]
    # A1 sub-types need the seed to begin at position 2 (start <= 2: position 1
    # itself may pair, as U1·A does) and an A on the target opposite position 1.
    t1A    = f["t1_A"] and start <= 2

    if f["total_1_9"] == 9:
        btype = "9mer"
    elif consec >= 5 and start < 4:
        btype = f"{consec}mer"
    else:
        btype = "seedless"

    if btype == "seedless":
        if f["consec_centered"] >= 8:
            return "centered"
        if start >= 13:
            return "3prime"
        if f["eff_3prime"] >= 6:
            return "3prime.compensatory"
        return "seedless"

    # canonical sub-types (mirroring feature_extraction.py)
    if btype in ("8mer", "7mer") and t1A:
        btype = "8mer1A"
    elif btype == "6mer":
        if start == 3:
            btype = "offset6mer"
        elif t1A:
            btype = "7mer1A"

    if "mer" in btype:
        if f["gu_in_seed"] > 0:
            btype += ".GU"
        if f["eff_3prime"] >= 3:
            btype += ".3prime"

    return btype


def _antidiag_features(mirna_dna: str, mre_dna: str) -> dict:
    """String-input wrapper around :func:`_antidiag_features_idx`."""
    return _antidiag_features_idx(_tok(mirna_dna, _MAX_MI),
                                  _tok(mre_dna, _MAX_TI))


def classify_binding_type(mirna_dna: str, mre_dna: str) -> str:
    """Classify a miRNA–MRE pair from nucleotide strings (T→U handled)."""
    return _classify_from_features(_antidiag_features(mirna_dna, mre_dna))


def classify_binding_type_idx(mi: np.ndarray, ti: np.ndarray) -> str:
    """Classify a miRNA–MRE pair from tokenised index arrays."""
    return _classify_from_features(_antidiag_features_idx(mi, ti))


def classify_index_arrays(mirna_idx: np.ndarray,
                          mre_idx: np.ndarray) -> np.ndarray:
    """Per-row binding categories for batched token arrays → (N,) string array."""
    n   = len(mirna_idx)
    out = np.empty(n, dtype="<U24")
    for i in range(n):
        out[i] = classify_binding_type_idx(mirna_idx[i], mre_idx[i])
    return out
