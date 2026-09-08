"""
Panel 5 - what the feature model actually uses.

This is the panel the deployment argument rests on: the CNN matches the feature
model closely on accuracy and costs far less, so the reason to ship the feature
model is that its attributions land on physically meaningful quantities rather
than on nucleotide positions. That claim is only worth making if the quantities
at the top of this list are ones a biologist recognises - so the figure has to
show direction, not just magnitude. A bar chart of mean |SHAP| would rank the
features and throw away the part that carries the argument.

Family-balanced sample
----------------------
Reads the `_fambal` SHAP run rather than the plain one. The test set is
dominated by a handful of abundant miRNA families, and an unweighted sample
would let those families decide the ranking - which turns a claim about
mechanism into a claim about composition. results/shap/*.sample_composition.tsv
records what the balanced sample drew.

Colour
------
Not SHAP's default blue-to-red. The poster reads blue as the sequence-CNN track,
and this panel is entirely about the feature model, so a blue ramp here invites
exactly the wrong reading on the one panel that has to land. Feature value is a
magnitude rather than a divergence, so it gets a single-hue sequential ramp in
the feature model's own orange.

Usage
-----
python3 src/figures/panel5_shap_beeswarm.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

import poster_style as ps
from panel1_method_schematic import FEATURE_GROUPS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"

# Family-tag column, in axes-width units left of the y axis.
GUTTER = -0.46

# Low-to-high feature value, kept inside the feature model's hue.
VALUE_RAMP = LinearSegmentedColormap.from_list(
    "gluon_seq", ["#fde8db", "#f6a878", "#eb6834", "#a83c12", "#5c1e05"])

# Names that do not survive a mechanical underscore-to-space pass.
PRETTY = {
    "E_z_mirna": "energy z-score vs background",
    "E_hybrid_z_mirna": "hybrid energy z-score",
    "E_bg_mean_mirna": "background energy mean",
    "E_bg_sd_mirna": "background energy SD",
    "gu_wobbles_in_seed_2_8_pos": "G·U wobbles in seed (2–8)",
    "seed_au_content": "seed AU content",
    "mirna_seed_energy_sum": "seed energy sum",
    "mirna_seed_energy_mean": "seed energy mean",
    "mirna_seed_energy_gradient": "seed energy gradient",
    "mirna_3p_energy_sum": "3' energy sum",
    "seed_vs_3p_energy_diff": "seed vs 3' energy difference",
    "duplex_energy_sum_total": "duplex energy sum",
    "subopt_E_delta_selected": "suboptimal ΔE (selected)",
    "subopt_frac_seedlike": "suboptimal fraction seed-like",
    "subopt_n_distinct_starts": "suboptimal distinct starts",
    "subopt_n_within_1kcal": "suboptimal within 1 kcal",
    "subopt_priority_gap": "suboptimal priority gap",
    "subopt_target_span": "suboptimal target span",
    "five_prime_flank_conservation_mean": "5' flank conservation",
    "three_prime_flank_conservation_mean": "3' flank conservation",
    "seed_conservation_mean": "seed conservation mean",
    "seed_conservation_min": "seed conservation min",
    "conservation_variance": "conservation variance",
    "effective_3prime_matches": "effective 3' matches",
    "total_gu_wobbles": "G·U wobbles, total",
    "priority_score": "IntaRNA priority score",
}

# Short tags for the five families Fig 1 introduces, so the last panel closes
# the loop the first one opened.
GROUP_TAG = {
    "shuffle-background z": "background",
    "phastCons conservation": "conservation",
    "duplex & seed ΔG": "duplex ΔG",
    "suboptimal ensemble": "suboptimal",
    "pairing & composition": "pairing",
}
FEATURE_GROUP = {f: GROUP_TAG[name] for name, feats in FEATURE_GROUPS for f in feats}


def swarm_offsets(x: np.ndarray, nbins: int = 90, spread: float = 0.40) -> np.ndarray:
    """Vertical offsets that turn a 1-D scatter into a beeswarm row.

    Points are binned along x and spread symmetrically within their bin, scaled
    so the densest bin just fills the row. That keeps every row the same height
    whatever its density, which is what makes rows comparable by eye - a row
    that ballooned with its point count would read as important when it is only
    peaked.
    """
    if len(x) == 0:
        return np.zeros(0)
    edges = np.linspace(np.min(x), np.max(x) + 1e-12, nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    out = np.zeros(len(x), dtype=float)
    counts = np.bincount(idx, minlength=nbins)
    peak = max(counts.max(), 1)
    for b in np.flatnonzero(counts):
        members = np.flatnonzero(idx == b)
        n = len(members)
        # Alternate outward from the centre line so the row stays symmetric.
        rank = np.arange(n) - (n - 1) / 2.0
        out[members] = rank / peak * 2.0 * spread
    # A touch of jitter: evenly spaced ranks leave visible vertical striping in
    # dense bins, which reads as structure that is not in the data.
    step = 2.0 * spread / peak
    out += np.random.default_rng(1).uniform(-0.35, 0.35, len(x)) * step
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path,
                    default=RESULTS / "shap" / "manakov_test_fambal.shap.npz")
    ap.add_argument("--max-points", type=int, default=4000,
                    help="Rows drawn per feature. The full sample is 25,000; "
                         "beyond a few thousand the swarm is solid ink and the "
                         "vector outputs get large for no readable gain.")
    ap.add_argument("--top", type=int, default=20,
                    help="Features shown, ranked by mean |SHAP|.")
    ap.add_argument("--bare", action="store_true",
                    help="Drop the figure legend. Use for the poster, where the "
                         "caption under the panel is the explanatory text.")
    ap.add_argument("--name", default="panel5_shap_beeswarm")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    shap, values = z["shap_values"], z["feature_values"]
    names = [str(f) for f in z["features"]]

    order = np.argsort(np.abs(shap).mean(axis=0))[::-1][:args.top]
    rng = np.random.default_rng(0)
    keep = (rng.choice(len(shap), args.max_points, replace=False)
            if len(shap) > args.max_points else np.arange(len(shap)))

    ps.apply(9.5)
    # Height tracks the row count so a shorter --top keeps the same row pitch
    # instead of stretching a few rows over a full-height figure. The constant
    # is the fixed chrome: axis, colour key and legend.
    chrome = 1.35 if args.bare else 2.60
    fig, ax = plt.subplots(figsize=(12.2, chrome + 0.36 * len(order)))
    # Wide left margin: the row labels carry two columns, the family tag in the
    # outer gutter and the feature name against the axis.
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.285, right=0.885, top=1 - 0.245 / h,
                        bottom=(0.90 if args.bare else 1.52) / h)

    ax.axvline(0.0, color=ps.MUTED, lw=1.0, zorder=2)

    for row, fi in enumerate(order):
        y0 = len(order) - 1 - row
        sv, fv = shap[keep, fi], values[keep, fi]
        # Clip the colour scale to the 5-95 range: a single extreme feature
        # value would otherwise flatten every other point to one shade.
        lo, hi = np.percentile(fv, [5, 95])
        norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
        ax.scatter(sv, y0 + swarm_offsets(sv), c=fv, cmap=VALUE_RAMP, norm=norm,
                   s=3.0, linewidths=0, alpha=0.85, zorder=3, rasterized=True)

    labels = [PRETTY.get(names[i], names[i].replace("_", " ")) for i in order]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_ylim(-0.75, len(order) - 0.25)

    mean_abs = np.abs(shap).mean(axis=0)
    for row, fi in enumerate(order):
        y0 = len(order) - 1 - row
        # Family tag in the left gutter, in axes-independent figure space so it
        # never collides with the longest feature name.
        ax.text(GUTTER, y0, FEATURE_GROUP.get(names[fi], ""),
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.5, color=ps.MUTED)
        ax.text(1.012, y0, f"{mean_abs[fi]:.3f}",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.5, color=ps.INK_2)
    ax.text(1.012, len(order) - 0.35, "mean |SHAP|",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=ps.INK)
    ax.text(GUTTER, len(order) - 0.35, "family",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=ps.INK)

    ax.set_xlabel("SHAP value  (log-odds contribution to P(bind))")
    ax.tick_params(axis="y", length=0)
    ps.despine(ax, keep=("bottom",))
    ax.grid(False)

    # Colour key: a gradient strip, since a categorical legend cannot show a
    # continuous encoding honestly.
    cax = fig.add_axes([0.285, (0.30 if args.bare else 0.80) / h,
                        0.115, 0.10 / h])
    cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto",
               cmap=VALUE_RAMP)
    cax.set_xticks([]); cax.set_yticks([])
    for side in cax.spines.values():
        side.set_visible(False)
    cax.text(-0.03, 0.5, "low", transform=cax.transAxes, ha="right", va="center",
             fontsize=8.5, color=ps.INK_2)
    cax.text(1.03, 0.5, "high", transform=cax.transAxes, ha="left", va="center",
             fontsize=8.5, color=ps.INK_2)

    # Bare mode has no room above the strip before the x label, so the key's
    # own label goes inline after "high" instead of over it.
    if args.bare:
        cax.text(1.16, 0.5, "feature value", transform=cax.transAxes,
                 ha="left", va="center", fontsize=8.5, color=ps.INK_2)
    else:
        cax.text(0.5, 2.6, "feature value", transform=cax.transAxes,
                 ha="center", va="bottom", fontsize=8.5, color=ps.INK_2)

    if not args.bare:
        # Set flush to the figure edge and wrapped to its full width: the legend
        # describes the whole panel, so indenting it under the plotting area would
        # read as a note about the axes.
        fig.text(0.030, 0.20 / h,
                 "TreeSHAP attributions for LightGBMLarge_BAG_L1 on a "
                 "family-balanced sample of the Manakov v7 test set (25,000 pairs, "
                 "both classes; 4,000 drawn per row). Each point is one "
                 "(miRNA, MRE) pair;\nhorizontal position is that feature's "
                 "log-odds contribution for that pair, colour is the feature's "
                 "value. Features are ranked by mean |SHAP| and tagged with the "
                 "family they belong to in Fig 1.",
                 fontsize=8.5, color=ps.INK_2, va="bottom", linespacing=1.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=400)), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{args.name}.{ext}", **kw)
    plt.close(fig)

    print(f"wrote {OUT}/{args.name}.{{png,svg,pdf}}")
    print("\nTop features by mean |SHAP| (family-balanced):")
    for fi in order[:8]:
        d = np.corrcoef(values[:, fi], shap[:, fi])[0, 1]
        print(f"  {names[fi]:<36} {mean_abs[fi]:.4f}  "
              f"corr(value, SHAP) = {d:+.2f}")


if __name__ == "__main__":
    main()
