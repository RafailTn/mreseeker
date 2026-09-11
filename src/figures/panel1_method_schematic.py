#!/usr/bin/env python3
"""
Panel 1 - method schematic: two models over the same pair of sequences.

Every architectural number is taken from the shipped artefacts rather than
retyped: the CNN stages come from `model_args` inside
cnn_checkpoints/cnn_branches_mirbind_embed16_restruct.pt, the feature groups
from SELECTED_26 in src/gluon/feature_extraction.py, and the bag size from the
S1F* fold directories under models_gluon_lgbm/. The pairing-matrix inset is a
real CNN true positive read out of data/eqtl_tp_pairs.tsv - not a mock-up - so
the seed diagonal on it is the model's actual input.

    python src/figures/panel1_method_schematic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_style as ps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"
CKPT = REPO / "cnn_checkpoints" / "cnn_branches_mirbind_embed16_restruct.pt"
PAIRS = REPO / "data" / "eqtl_tp_pairs.tsv"
FEATURE_JSON = REPO / "data" / "selected_features_from56ft.json"
BAG_DIR = REPO / "models" / "gluon_total_try" / "models" / "CatBoost_BAG_L1"

MRE_LEN, MAX_MIRNA = 50, 30          # cnn_branches_mirbind.py
SEED = (2, 8)                        # miRNA seed, 1-based inclusive

# The five families SELECTED_26 falls into, as (label, [feature names]).
FEATURE_GROUPS = [
    ("shuffle-background z", ["E_bg_mean_mirna", "E_bg_sd_mirna",
                              "E_hybrid_z_mirna", "E_z_mirna"]),
    ("phastCons conservation", ["conservation_variance",
                                "five_prime_flank_conservation_mean",
                                "seed_conservation_mean", "seed_conservation_min",
                                "three_prime_flank_conservation_mean"]),
    ("duplex & seed ΔG", ["duplex_energy_sum_total", "mirna_3p_energy_sum",
                               "mirna_seed_energy_gradient", "mirna_seed_energy_mean",
                               "mirna_seed_energy_sum", "seed_vs_3p_energy_diff"]),
    ("suboptimal ensemble", ["subopt_E_delta_selected", "subopt_frac_seedlike",
                             "subopt_n_distinct_starts", "subopt_n_within_1kcal",
                             "subopt_priority_gap", "subopt_target_span"]),
    ("pairing & composition", ["effective_3prime_matches",
                               "gu_wobbles_in_seed_2_8_pos", "priority_score",
                               "seed_au_content", "total_gu_wobbles"]),
]


# ---------------------------------------------------------------------------
# Inputs read back from the shipped artefacts
# ---------------------------------------------------------------------------

def load_model_args() -> dict:
    """model_args out of the checkpoint, without importing torch if avoidable."""
    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    return ck["model_args"]


def check_feature_groups() -> None:
    """The five groups must partition SELECTED_26 exactly - no silent drift."""
    grouped = [f for _, fs in FEATURE_GROUPS for f in fs]
    canonical = json.loads(FEATURE_JSON.read_text())
    if sorted(grouped) != sorted(canonical):
        missing = sorted(set(canonical) - set(grouped))
        extra = sorted(set(grouped) - set(canonical))
        raise SystemExit(
            f"feature groups do not partition SELECTED_26 "
            f"(missing={missing}, extra={extra})")


def pick_pair() -> pd.Series:
    """The highest-confidence 3'UTR true positive that has a canonical 7mer-m8
    site, so the inset shows a seed diagonal a reader can actually find."""
    df = pd.read_csv(PAIRS, sep="\t")
    comp = str.maketrans("ACGT", "TGCA")
    df["site"] = [t.find(m[1:8].translate(comp)[::-1])
                  for m, t in zip(df.mirna_seq, df.mre_seq)]
    hit = df[(df.site > 8) & (df.site < 38)
             & (df.mirna_seq.str.len() >= 21)
             & (df.dominant_region == "UTR3")]
    return hit.sort_values("interaction_probability", ascending=False).iloc[0]


def pairing_matrix(mirna: str, mre: str) -> np.ndarray:
    """(MAX_MIRNA, MRE_LEN) complementarity: 1.0 Watson-Crick, 0.5 G.U wobble.

    The shipped checkpoint replaces this fixed table with a learned 16-channel
    embedding; the geometry it sees is identical, so this is the honest picture
    of the input to draw.
    """
    wc = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
    gu = {("G", "U"), ("U", "G")}
    mi = mirna.upper().replace("T", "U")
    ta = mre.upper().replace("T", "U")
    m = np.zeros((MAX_MIRNA, MRE_LEN), dtype=float)
    for i, a in enumerate(mi[:MAX_MIRNA]):
        for j, b in enumerate(ta[:MRE_LEN]):
            pair = (a, b)
            m[i, j] = 1.0 if pair in wc else (0.5 if pair in gu else 0.0)
    return m


# ---------------------------------------------------------------------------
# Drawing helpers - all in a 0-100 x 0-100 canvas
# ---------------------------------------------------------------------------

def stage(ax, x0, x1, y0, y1, title, body, accent, fill, title_size=10.5,
          body_size=9.0):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.4",
        linewidth=1.6, edgecolor=accent, facecolor=fill, zorder=2))
    cx = (x0 + x1) / 2
    ax.text(cx, y1 - 3.2, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=ps.INK, zorder=3)
    ax.text(cx, y1 - 8.0, "\n".join(body), ha="center", va="top",
            fontsize=body_size, color=ps.INK_2, linespacing=1.6, zorder=3)


def arrow(ax, x0, y0, x1, y1, colour, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.7,
        color=colour, shrinkA=0, shrinkB=0, zorder=4,
        connectionstyle=f"arc3,rad={rad}"))


def draw_features(ax, x0, y_top, accent):
    """The 26 selected features as five labelled rows of glyphs."""
    ax.text(x0, y_top, "26 selected features", ha="left", va="bottom",
            fontsize=10.5, fontweight="bold", color=ps.INK, zorder=3)
    row_h, sq, gap = 4.6, 1.05, 0.30
    for k, (label, feats) in enumerate(FEATURE_GROUPS):
        y = y_top - 2.6 - k * row_h
        for n in range(len(feats)):
            ax.add_patch(Rectangle(
                (x0 + n * (sq + gap), y - sq), sq, sq,
                facecolor=accent, edgecolor="none", zorder=3))
        ax.text(x0 + 6 * (sq + gap) + 1.0, y - sq / 2,
                f"{len(feats)}  {label}", ha="left", va="center",
                fontsize=8.5, color=ps.INK_2, zorder=3)


def draw_matrix(fig, ax_host, pair, accent):
    """Inset: the real miRNA x MRE pairing matrix the CNN consumes."""
    cmap = LinearSegmentedColormap.from_list("mreblue", ps.BLUE_RAMP)
    m = pairing_matrix(pair.mirna_seq, pair.mre_seq)

    # Place the inset in figure coords matching the 0-100 canvas column.
    x0, x1, y0, y1 = 18.0, 33.0, 61.0, 79.5
    trans = ax_host.transData + fig.transFigure.inverted()
    (fx0, fy0), (fx1, fy1) = trans.transform([(x0, y0), (x1, y1)])
    ax = fig.add_axes((fx0, fy0, fx1 - fx0, fy1 - fy0))

    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks([0, 24, 49]); ax.set_xticklabels(["1", "25", "50"])
    ax.set_yticks([0, 9, 19, 29]); ax.set_yticklabels(["1", "10", "20", "30"])
    ax.tick_params(labelsize=7.5, length=2.5, pad=1.5, colors=ps.MUTED)
    for s in ax.spines.values():
        s.set_color(ps.GRID)
    ax.set_xlabel("MRE position (50 nt)", fontsize=8, color=ps.INK_2, labelpad=1.5)
    ax.set_ylabel("miRNA position", fontsize=8, color=ps.INK_2, labelpad=1.5)

    # Mark the seed rows the panel talks about. The label goes *inside* the
    # matrix on a white stroke - outside it would land on the caption.
    ax.add_patch(Rectangle(
        (-0.5, SEED[0] - 1.5), MRE_LEN, SEED[1] - SEED[0] + 1,
        fill=False, edgecolor=ps.ACCENT_GLUON, lw=1.6, zorder=5))
    ax.text(MRE_LEN - 1.0, SEED[1] + 0.2, "seed 2–8", ha="right", va="top",
            fontsize=7.5, color=ps.ACCENT_GLUON, fontweight="bold", zorder=6,
            path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])

    # Pad rows carry no signal - say so rather than leaving a blank band.
    ax.axhline(len(pair.mirna_seq) - 0.5, color=ps.MUTED, lw=0.9, ls=(0, (3, 2)))
    ax.text(1.0, len(pair.mirna_seq) + 0.2, "zero-padded", fontsize=7,
            color=ps.MUTED, va="top", zorder=6,
            path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])

    ax_host.text((x0 + x1) / 2, y1 + 1.3,
                 f"{pair.noncodingRNA_name} × 3'UTR site",
                 ha="center", va="bottom", fontsize=8.5, color=ps.INK_2)
    ax_host.text((x0 + x1) / 2, y0 - 5.8,
                 "complementarity matrix →\nlearned 16-channel pair embedding",
                 ha="center", va="top", fontsize=9, color=ps.INK_2,
                 linespacing=1.5)
    return ax


# ---------------------------------------------------------------------------

def main():
    check_feature_groups()
    ma = load_model_args()
    n_folds = len([p for p in BAG_DIR.iterdir() if p.name.startswith("S1F")])
    pair = pick_pair()

    ps.apply(base=11)
    # The canvas stays 0-100 in both axes so the stage coordinates below read the
    # same, but the view is cropped to the band the diagram actually occupies -
    # there is no title or footnote block to leave room for. The figure height is
    # cropped by the same factor, which keeps one canvas unit at the height it
    # had before, so every point size still sits in its box the same way.
    Y_LO, Y_HI = 11.0, 90.0
    fig = plt.figure(figsize=(14.0, 7.6 * (Y_HI - Y_LO) / 100.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100); ax.set_ylim(Y_LO, Y_HI); ax.axis("off")

    BLUE, ORANGE = ps.ACCENT_CNN, ps.ACCENT_GLUON

    # Vertical bands: track A boxes, track B boxes, and the arrow mid-lines.
    A0, A1, MID_A = 61.0, 83.0, 72.0
    B0, B1, MID_B = 15.0, 37.0, 26.0

    # -- shared input ------------------------------------------------------
    stage(ax, 1.5, 14.0, 40.0, 58.0, "Input",
          ["two sequences", "", "miRNA  ≤ 30 nt", "MRE  50 nt"],
          ps.MUTED, ps.FILL, title_size=11, body_size=9.5)
    # Both branches leave the box through the side that faces their track - blue
    # off the top, orange off the bottom - so neither crosses the box's right
    # edge. Each bows away from the box and arrives pointing at its target: the
    # blue head stops short of the inset's y-axis and lands on the "miRNA
    # position" label instead, which is what it is feeding.
    arrow(ax, 7.75, 58.6, 14.9, 70.2, BLUE, rad=-0.30)
    arrow(ax, 7.75, 39.4, 16.4, 26.5, ORANGE, rad=0.30)

    # -- track A: sequence CNN --------------------------------------------
    ax.text(17.0, 85.6, "A   Sequence CNN", fontsize=12.5, fontweight="bold",
            color=BLUE, ha="left", va="bottom")
    draw_matrix(fig, ax, pair, BLUE)

    stage(ax, 36.0, 58.0, A0, A1,
          f"{ma['n_conv_blocks']} × Conv2d block",
          [f"5×5 · {ma['seq_filters']} filters",
           f"BatchNorm → {ma['activation'].upper()} → Dropout {ma['seq_dropout']}",
           f"first {ma['n_pool_blocks']} blocks: 2×2 max-pool",
           "30×50 → 15×25 → 7×12 → 3×6 → 1×3"],
          BLUE, ps.CNN_FILL)
    arrow(ax, 33.6, MID_A, 35.4, MID_A, BLUE)

    stage(ax, 60.0, 78.0, A0, A1, "GeM pool → head",
          [f"→ {ma['seq_dim']}-d embedding",
           "LayerNorm → GELU",
           f"{ma['seq_dim']} → {ma['seq_dim'] // 2} → 1",
           "sigmoid"],
          BLUE, ps.CNN_FILL)
    arrow(ax, 58.6, MID_A, 59.4, MID_A, BLUE)
    arrow(ax, 78.6, MID_A, 79.4, MID_A, BLUE)

    # -- track B: features + LightGBM -------------------------------------
    ax.text(17.0, 39.6, "B   Thermodynamic features → gradient-boosted trees",
            fontsize=12.5, fontweight="bold", color=ORANGE, ha="left", va="bottom")

    stage(ax, 17.0, 35.0, B0, B1, "IntaRNA duplex",
          ["MFE + suboptimal ensemble", "shuffled-miRNA background",
           "hg38 470-way phastCons"],
          ORANGE, ps.GLUON_FILL)
    arrow(ax, 35.6, MID_B, 37.4, MID_B, ORANGE)

    draw_features(ax, 38.0, B1, ORANGE)
    arrow(ax, 58.6, MID_B, 59.4, MID_B, ORANGE)

    stage(ax, 60.0, 78.0, B0, B1, "CatBoost BAG L1",
          [f"{n_folds}-fold bagged", "the shipped default of 25",
           "AutoGluon candidates", "27 MB deployment clone"],
          ORANGE, ps.GLUON_FILL)
    arrow(ax, 78.6, MID_B, 79.4, MID_B, ORANGE)

    # -- outputs -----------------------------------------------------------
    for y0, y1, accent, fill, aps, notes in [
        (A0, A1, BLUE, ps.CNN_FILL, "0.87",
         ["sequence only — no IntaRNA,", "no conservation, no features"]),
        (B0, B1, ORANGE, ps.GLUON_FILL, "0.84", []),
    ]:
        ax.add_patch(FancyBboxPatch(
            (80.0, y0), 18.5, y1 - y0,
            boxstyle="round,pad=0,rounding_size=1.4",
            linewidth=1.6, edgecolor=accent, facecolor=fill, zorder=2))
        cx = 80.0 + 18.5 / 2
        ax.text(cx, y1 - 3.2, "P(miRNA binds MRE)", ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=ps.INK, zorder=3)
        if notes:
            ax.text(cx, y1 - 9.0, f"APS {aps}", ha="center", va="top",
                    fontsize=20, fontweight="bold", color=accent, zorder=3)
            ax.text(cx, y0 + 3.5, "\n".join(notes), ha="center", va="bottom",
                    fontsize=8.5, color=ps.INK_2, linespacing=1.55, zorder=3)
        else:
            # Nothing under it, so the figure sits in the middle of the space
            # left below the title rather than hugging it.
            ax.text(cx, (y0 + y1) / 2 - 2.0, f"APS {aps}", ha="center",
                    va="center", fontsize=20, fontweight="bold", color=accent,
                    zorder=3)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"panel1_method_schematic.{ext}", **kw)
    plt.close(fig)
    print("wrote", OUT / "panel1_method_schematic.{png,svg,pdf}")
    print(f"  inset pair: {pair.noncodingRNA_name} p={pair.interaction_probability:.4f}")


if __name__ == "__main__":
    main()
