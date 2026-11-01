#!/usr/bin/env python3
"""
plot_duplex.py — draw the duplex a CNN prediction relied on.

Reads the outputs of ``cnn_branches_mirbind.py explain --duplex --dump-matrix``
and writes one PNG per selected row:

  top     the (miRNA × MRE) Integrated-Gradients map, diverging around 0, with
          the decoded duplex drawn over it and the seed (2–8) and 3'
          supplementary (13–16) rows shaded;
  bottom  the duplex itself: target 5'→3' over miRNA 3'→5', one box per pair
          coloured by type, with a bar per pair giving its attribution.

Units are logits throughout (IG on the binding logit, zero-matrix baseline).

Usage
-----
  python src/cnn/plot_duplex.py --explain explain.tsv --npz attr.npz \\
      --out-dir duplex_png --top 6            # or --rows 12 40 77
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_MODEL_DIR = Path(__file__).resolve().parent.parent / "training" / "cnn"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

import duplex_decode as dd  # noqa: E402

_NT = "ACGU"
_KIND_COLOUR = {dd.KIND_WC: "#2a7ab9", dd.KIND_GU: "#7fb6dd",
                dd.KIND_MM: "#d9d9d9", dd.KIND_BULGE: "#f2b880"}
_REGION_ROWS = [((1, 7), "#fff2cc", "seed 2–8"), ((12, 15), "#e2f0d9", "3′ supp 13–16")]


def _duplex_columns(mi, ti, mj, kd, tb):
    """Ordered duplex columns (target j | None, miRNA i | None, kind) along the target."""
    paired = np.flatnonzero((kd >= dd.KIND_WC) & (kd <= dd.KIND_MM))
    if paired.size == 0:
        return []
    cols, last_j = [], int(mj[paired.max()]) - 1
    for i in range(int(paired.max()), int(paired.min()) - 1, -1):
        if kd[i] == dd.KIND_BULGE:
            cols.append((None, i, dd.KIND_BULGE))
            continue
        j = int(mj[i])
        cols.extend((jb, None, dd.KIND_BULGE) for jb in range(last_j + 1, j))
        cols.append((j, i, int(kd[i])))
        last_j = j
    return cols


def plot_row(n: int, row: pd.Series, z, out: Path) -> None:
    mi, ti = z["mirna_idx"][n], z["mre_idx"][n]
    attr = z["attribution"][n]
    mj, kd, tb = z["duplex_j"][n], z["duplex_kind"][n], z["duplex_tbulge"][n]
    m, L = int((mi != 4).sum()), int((ti != 4).sum())

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.4),
                                  gridspec_kw={"height_ratios": [3, 1.3]})
    lim = float(np.abs(attr).max()) or 1.0
    im = ax.imshow(attr[:m, :L], cmap="RdBu_r", vmin=-lim, vmax=lim,
                   aspect="auto", interpolation="nearest")
    for (a, b), c, lab in _REGION_ROWS:
        if a < m:
            ax.axhspan(a - 0.5, min(b, m - 1) + 0.5, color=c, alpha=0.35,
                       zorder=0, lw=0)
            ax.text(L - 0.3, (a + min(b, m - 1)) / 2, lab, ha="right",
                    va="center", fontsize=7, color="#555")
    ii, jj = dd.path_cells(mj, kd)
    ax.plot(jj, ii, "-", color="black", lw=1.0)
    ax.scatter(jj, ii, s=14, c=[_KIND_COLOUR[int(k)] for k in kd[ii]],
               edgecolors="black", linewidths=0.6, zorder=3)
    ax.set_xticks(range(L)); ax.set_xticklabels([_NT[c] for c in ti[:L]], fontsize=6)
    ax.set_yticks(range(m))
    ax.set_yticklabels([f"{_NT[c]}{i + 1}" for i, c in enumerate(mi[:m])], fontsize=6)
    ax.set_xlabel("MRE 5′→3′"); ax.set_ylabel("miRNA 5′→3′")
    fig.colorbar(im, ax=ax, pad=0.01, label="IG attribution (logit)")

    cols = _duplex_columns(mi, ti, mj, kd, tb)
    vals = [attr[i, j] if (i is not None and j is not None) else 0.0
            for j, i, _ in cols]
    vmax = max(1e-9, max((abs(v) for v in vals), default=0.0))
    for x, ((j, i, k), v) in enumerate(zip(cols, vals)):
        ax2.add_patch(Rectangle((x - 0.45, -0.5), 0.9, 2.0, color=_KIND_COLOUR[k],
                                alpha=0.55, lw=0))
        ax2.text(x, 1.0, _NT[ti[j]] if j is not None else "-", ha="center",
                 va="center", family="monospace", fontsize=9)
        ax2.text(x, 0.0, _NT[mi[i]] if i is not None else "-", ha="center",
                 va="center", family="monospace", fontsize=9)
        if i is not None:
            ax2.text(x, -0.85, str(i + 1), ha="center", va="center", fontsize=6,
                     color="#555")
        ax2.bar(x, 0.9 * v / vmax, bottom=1.6, width=0.6,
                color="#b2182b" if v > 0 else "#2166ac")
    ax2.set_xlim(-1, max(len(cols), 1)); ax2.set_ylim(-1.2, 2.6)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["miRNA 3′→5′", "target 5′→3′"],
                                                fontsize=8)
    ax2.set_xticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.55)
               for c in _KIND_COLOUR.values()]
    ax2.legend(handles, ["WC", "G·U", "mismatch", "bulge"], ncol=4, fontsize=7,
               loc="lower right", frameon=False, bbox_to_anchor=(1, -0.35))

    regions = "  ".join(f"{r} {row[f'dlogit_{r}']:+.2f}" for r in dd.REGIONS)
    fig.suptitle(f"row {n}  ·  p={row['prob']:.3f}  ·  {row['duplex_type']}"
                 f"  (sequence-only: {row['seq_type']})\n"
                 f"Δlogit when zeroed — {regions}  ·  duplex "
                 f"{row['dlogit_duplex']:+.2f} vs random {row['dlogit_random']:+.2f}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--explain", required=True, help="TSV from explain --duplex.")
    p.add_argument("--npz", required=True, help="npz from explain --dump-matrix.")
    p.add_argument("--out-dir", required=True, dest="out_dir")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rows", type=int, nargs="+", help="Row indices to plot.")
    g.add_argument("--top", type=int, default=6, help="Plot the N highest-prob rows.")
    args = p.parse_args()

    df = pd.read_csv(args.explain, sep="\t")
    z = np.load(args.npz)
    if "duplex_j" not in z:
        sys.exit("ERROR: the npz has no decoded duplex; rerun explain with "
                 "--duplex together with --dump-matrix.")
    rows = args.rows if args.rows else np.argsort(-df["prob"].values)[:args.top]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in rows:
        out = out_dir / f"duplex_row{int(n)}.png"
        plot_row(int(n), df.iloc[int(n)], z, out)
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
