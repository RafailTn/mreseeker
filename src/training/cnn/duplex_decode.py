#!/usr/bin/env python3
"""
Decode the miRNA–MRE duplex a CNN prediction relied on (numpy-only, no torch).

The CNN never outputs a structure: it scores a (MAX_MIRNA × MRE_LEN) pairing
matrix, and Integrated Gradients on that matrix (``cmd_explain``) yields a same-
shaped attribution map.  That map says which *cells* mattered but not which
miRNA base pairs with which target base.  This module turns it into an explicit
duplex:

  decode_duplex   — antiparallel local alignment (Smith–Waterman, affine gaps)
                    over the matrix.  Both sequences are stored 5'→3', so a helix
                    runs along an anti-diagonal: the pair after (i, j) is
                    (i+1, j-1).  A miRNA bulge steps (i+1, j), a target bulge
                    (i, j-1).  Each cell scores a chemistry term (G·C > A·U >
                    G·U > mismatch) plus ``lam`` × the cell's attribution,
                    normalised by the pair's max |attribution| so ``lam`` is
                    scale-free.  Positive attribution pulls the path through the
                    cells the model used, negative pushes it away; ``lam=0`` is a
                    pure-sequence duplex for reference.
  classify_duplex — binding type of a decoded duplex, through the same cascade as
                    ``binding_types`` (so the names match everywhere), plus a
                    ``.bulge`` tag for a bulge inside the seed.
  render_duplex   — IntaRNA-style ASCII diagram.

Per-row outputs, all on miRNA positions i (0-based; biology numbering is i+1):
  mre_j  (MAX_MIRNA,) int16  — MRE index opposite miRNA i, -1 if none
  kind   (MAX_MIRNA,) int8   — KIND_* code below
  tbulge (MRE_LEN,)   bool   — target bases skipped inside the duplex
These three arrays fully describe the duplex; nothing else is needed to redraw it.
"""
from __future__ import annotations

import numpy as np

try:
    import binding_types as _bt
except ImportError:
    from cnn import binding_types as _bt

MAX_MIRNA = 30
MRE_LEN   = 50

KIND_NONE, KIND_WC, KIND_GU, KIND_MM, KIND_BULGE = 0, 1, 2, 3, 4
KIND_NAMES = {KIND_NONE: "-", KIND_WC: "WC", KIND_GU: "GU",
              KIND_MM: "MM", KIND_BULGE: "bulge"}

# miRNA regions (0-based slices over i).  "seed" includes position 1 so that the
# t1A / 8mer evidence sits with the seed rather than on its own.
REGIONS = {
    "seed":    slice(0, 8),    # positions 1–8
    "central": slice(8, 12),   # 9–12
    "supp":    slice(12, 17),  # 13–17, 3' supplementary
    "tail":    slice(17, MAX_MIRNA),
}

# Chemistry term, indexed [mirna_nt, mre_nt] with A=0 C=1 G=2 U=3, pad=4.  Same
# ordering as the model's _PAIR_STRENGTH prior; mismatches cost rather than score
# 0 so the local alignment does not wander through unpaired stretches.
_NEG = -1e9
_CHEM = np.full((5, 5), -3.0, dtype=np.float64)
for _a, _b, _v in [("A", "U", 2.0), ("U", "A", 2.0), ("G", "C", 3.0),
                   ("C", "G", 3.0), ("G", "U", 1.0), ("U", "G", 1.0)]:
    _CHEM[_bt._NUC[_a], _bt._NUC[_b]] = _v
_CHEM[4, :] = _NEG
_CHEM[:, 4] = _NEG

GAP_OPEN   = 5.0
GAP_EXTEND = 1.5
DEFAULT_LAMBDA = 3.0

# Traceback codes.
_FROM_START, _FROM_H, _FROM_E, _FROM_F = 0, 1, 2, 3


def decode_duplex(mi: np.ndarray, ti: np.ndarray, attr: np.ndarray,
                  lam: float = DEFAULT_LAMBDA
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched attribution-guided duplex decoding.

    ``mi`` (B, MAX_MIRNA) / ``ti`` (B, MRE_LEN) are nucleotide indices, ``attr``
    (B, MAX_MIRNA, MRE_LEN) the channel-summed attribution.  Returns
    ``(mre_j, kind, tbulge)`` as described in the module docstring.

    The DP runs on the miRNA against the *reversed* MRE (r = MRE_LEN-1-j), which
    turns the anti-diagonal helix into an ordinary alignment diagonal; the fill
    is vectorised across the batch and loops over the 30×50 cells.
    """
    mi = np.asarray(mi, dtype=np.intp)
    ti = np.asarray(ti, dtype=np.intp)
    attr = np.asarray(attr, dtype=np.float64)
    B, M, N = attr.shape

    scale = np.abs(attr).reshape(B, -1).max(axis=1)
    scale[scale == 0] = 1.0
    S = _CHEM[mi[:, :, None], ti[:, None, :]] + lam * attr / scale[:, None, None]
    S = S[:, :, ::-1]                                  # column r ↔ MRE j = N-1-r
    valid = S > _NEG / 2

    H = np.zeros((B, M + 1, N + 1))
    E = np.full((B, M + 1, N + 1), _NEG)               # gap along r: target bulge
    F = np.full((B, M + 1, N + 1), _NEG)               # gap along i: miRNA bulge
    tH = np.zeros((B, M + 1, N + 1), dtype=np.int8)
    tE = np.zeros((B, M + 1, N + 1), dtype=np.int8)
    tF = np.zeros((B, M + 1, N + 1), dtype=np.int8)

    for i in range(1, M + 1):
        for r in range(1, N + 1):
            ok = valid[:, i - 1, r - 1]
            # target bulge: target base r skipped, miRNA stays at i
            e_open, e_ext = H[:, i, r - 1] - GAP_OPEN, E[:, i, r - 1] - GAP_EXTEND
            E[:, i, r]  = np.where(ok, np.maximum(e_open, e_ext), _NEG)
            tE[:, i, r] = np.where(e_open >= e_ext, _FROM_H, _FROM_E)
            # miRNA bulge: miRNA base i skipped, target stays at r
            f_open, f_ext = H[:, i - 1, r] - GAP_OPEN, F[:, i - 1, r] - GAP_EXTEND
            F[:, i, r]  = np.where(ok, np.maximum(f_open, f_ext), _NEG)
            tF[:, i, r] = np.where(f_open >= f_ext, _FROM_H, _FROM_F)
            # pair / mismatch (diagonal); local alignment may restart at 0
            prev = np.stack([np.zeros(B), H[:, i - 1, r - 1],
                             E[:, i - 1, r - 1], F[:, i - 1, r - 1]])
            arg = prev.argmax(axis=0)
            h = prev[arg, np.arange(B)] + S[:, i - 1, r - 1]
            H[:, i, r]  = np.where(ok, h, _NEG)
            tH[:, i, r] = arg

    mre_j  = np.full((B, M), -1, dtype=np.int16)
    kind   = np.zeros((B, M), dtype=np.int8)
    tbulge = np.zeros((B, N), dtype=bool)
    for b in range(B):
        flat = int(np.argmax(H[b]))
        i, r = divmod(flat, N + 1)
        if H[b, i, r] <= 0:
            continue                                   # no positive-scoring duplex
        state = _FROM_H
        while i > 0 and r > 0:
            if state == _FROM_H:
                a, c = mi[b, i - 1], ti[b, N - r]
                mre_j[b, i - 1] = N - r
                kind[b, i - 1] = (KIND_WC if _bt._WC[a, c] else
                                  KIND_GU if _bt._GU[a, c] else KIND_MM)
                state = tH[b, i, r]
                i, r = i - 1, r - 1
                if state == _FROM_START:
                    break
            elif state == _FROM_E:
                tbulge[b, N - r] = True
                state = tE[b, i, r]
                r -= 1
            else:
                kind[b, i - 1] = KIND_BULGE
                state = tF[b, i, r]
                i -= 1
    return mre_j, kind, tbulge


def _seed_bulge(mre_j: np.ndarray, kind: np.ndarray, tbulge: np.ndarray) -> bool:
    """True if a miRNA or target bulge lies between seed positions 2 and 8."""
    if (kind[1:8] == KIND_BULGE).any():
        return True
    js = mre_j[1:8][mre_j[1:8] >= 0]
    if js.size < 2:
        return False
    return bool(tbulge[js.min():js.max() + 1].any())


def _t1(ti: np.ndarray, mre_j: np.ndarray, kind: np.ndarray) -> int:
    """Target base opposite miRNA position 1: the one just 3' of position 2's
    partner (4 if position 2 is unpaired or that base is off the end)."""
    if kind[1] not in (KIND_WC, KIND_GU):
        return 4
    j = int(mre_j[1]) + 1
    return int(ti[j]) if j < len(ti) else 4


def classify_duplex(mi: np.ndarray, ti: np.ndarray, mre_j: np.ndarray,
                    kind: np.ndarray, tbulge: np.ndarray) -> str:
    """Binding type of one decoded duplex (``binding_types`` names + ``.bulge``)."""
    mi = np.asarray(mi, dtype=np.intp)
    mi_len = int((mi != 4).sum())
    f = _bt._features_from_pairs(kind == KIND_WC, kind == KIND_GU,
                                 _t1(ti, mre_j, kind), mi_len)
    btype = _bt._classify_from_features(f)
    if _seed_bulge(mre_j, kind, tbulge):
        btype += ".bulge"
    return btype


def path_cells(mre_j: np.ndarray, kind: np.ndarray,
               region: slice = slice(None)) -> tuple[np.ndarray, np.ndarray]:
    """(i, j) index arrays of the paired/mismatched path cells in ``region``."""
    i = np.arange(len(kind))[region]
    keep = (kind[region] >= KIND_WC) & (kind[region] <= KIND_MM)
    return i[keep], mre_j[region][keep].astype(np.intp)


def format_pairs(mi: np.ndarray, mre_j: np.ndarray, kind: np.ndarray,
                 attr: np.ndarray) -> str:
    """Compact ``pos:j:kind:attr;…`` string (1-based miRNA pos, 0-based MRE j)."""
    out = []
    for i in range(int((np.asarray(mi) != 4).sum())):
        k = int(kind[i])
        j = int(mre_j[i])
        a = f"{attr[i, j]:.3f}" if j >= 0 else ""
        out.append(f"{i + 1}:{j if j >= 0 else ''}:{KIND_NAMES[k]}:{a}")
    return ";".join(out)


_NT = "ACGU"


def render_duplex(mi: np.ndarray, ti: np.ndarray, mre_j: np.ndarray,
                  kind: np.ndarray, tbulge: np.ndarray) -> str:
    """IntaRNA-style diagram: target 5'→3' over miRNA 3'→5'.

    ``|`` Watson–Crick, ``:`` G·U, blank = mismatch; bulged and flanking bases are
    lower-case, ``-`` fills the opposite strand of a bulge.  A fourth line marks
    seed positions 2–8 (``s``) and the 3' supplementary region 13–16 (``3``).
    """
    mi = np.asarray(mi); ti = np.asarray(ti)
    m = int((mi != 4).sum()); n = int((ti != 4).sum())
    mseq = "".join(_NT[c] for c in mi[:m]); tseq = "".join(_NT[c] for c in ti[:n])
    paired = np.flatnonzero((kind[:m] >= KIND_WC) & (kind[:m] <= KIND_MM))
    if paired.size == 0:
        return f"target 5'-{tseq.lower()}-3'\n(no duplex)\nmiRNA  3'-{mseq[::-1].lower()}-5'"

    i_lo, i_hi = int(paired.min()), int(paired.max())
    j_lo, j_hi = int(mre_j[i_hi]), int(mre_j[i_lo])
    top, mid, bot, reg = [], [], [], []

    def col(t, mk, q, i=None):
        top.append(t); mid.append(mk); bot.append(q)
        if i is None:
            reg.append(" ")
        else:
            reg.append("s" if 1 <= i <= 7 else "3" if 12 <= i <= 15 else " ")

    # left flanks: target 5' end opposite the miRNA 3' overhang
    lt = tseq[:j_lo].lower()
    lm = mseq[i_hi + 1:][::-1].lower()
    w = max(len(lt), len(lm))
    for k in range(w):
        t = lt[k - (w - len(lt))] if k >= w - len(lt) else " "
        q_off = k - (w - len(lm))
        q = lm[q_off] if q_off >= 0 else " "
        col(t, " ", q, (m - 1 - q_off) if q_off >= 0 else None)

    last_j = j_lo - 1
    for i in range(i_hi, i_lo - 1, -1):
        if kind[i] == KIND_BULGE:
            col("-", " ", mseq[i].lower(), i)
            continue
        j = int(mre_j[i])
        for jb in range(last_j + 1, j):               # target bulges before this pair
            col(tseq[jb].lower(), " ", "-")
        mk = {KIND_WC: "|", KIND_GU: ":"}.get(int(kind[i]), " ")
        col(tseq[j], mk, mseq[i], i)
        last_j = j

    # right flanks: target 3' end opposite the miRNA 5' overhang
    rt = tseq[j_hi + 1:].lower()
    rm = mseq[:i_lo][::-1].lower()
    for k in range(max(len(rt), len(rm))):
        q = rm[k] if k < len(rm) else " "
        col(rt[k] if k < len(rt) else " ", " ", q,
            (i_lo - 1 - k) if k < len(rm) else None)

    return "\n".join([
        "target 5'-" + "".join(top) + "-3'",
        "          " + "".join(mid),
        "miRNA  3'-" + "".join(bot) + "-5'",
        "          " + "".join(reg).rstrip(),
    ])
