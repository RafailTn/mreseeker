#!/usr/bin/env python3
"""
Would scoring each target site against its tissue's *expressed* miRNA repertoire
recover high-PIP eQTLs that the single miRBench-assigned miRNA throws away?

The merged GTEx x miRBench files pair every fragment with the one miRNA the
AGO2 eCLIP happened to capture, in one cell type. Only ~40% of eQTL rows have
that miRNA among the top 100 expressed in the tissue where the effect was
actually measured, and requiring it to be a CNN true positive collapses the
high-PIP set from 201 variants to 45 - which is what limits the separation
test. This asks whether the tissue's real repertoire fills that gap, *before*
committing to the full 49-tissue re-scoring.

Two numbers decide it:
  1. RESCUE   - high-PIP variants with >=1 tissue-expressed miRNA predicted to
                bind their site, versus the 45 available now.
  2. CALIBRATION - the same scores against composition-matched shuffled sites.
                The CNN was trained on eCLIP-supported pairs, so scoring
                arbitrary miRNA x site combinations is out of distribution. If
                shuffled sites score like real ones, a "rescue" is just the
                model saying yes to everything and means nothing.

Reporting (1) without (2) would be actively misleading, so both always run.

    pixi run -m dependencies/cnn python src/eqtl_analysis/mirna_repertoire_diagnostic.py \
        --eqtl data/GTEx_..._pip_gt_0_9_..._merged_with_miRBench_fragments.tsv \
        --top-mirnas results/gtex_mirna_expression/top100_by_subtissue.tsv \
        --mirna-seqs data/mirbase_mature_hg38.tsv \
        --checkpoint cnn_checkpoints/cnn_branches_mirbind_embed16_restruct.pt \
        --annotated results/eqtl_intarna/eqtl_annotated_pip_gt_0_9.tsv \
        -o results/eqtl_intarna/repertoire_diagnostic_pip_gt_0_9
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_CNN_DIR = _HERE.parent / "cnn"
if str(_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_DIR))

from predict_cnn import score_dataframe  # noqa: E402

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

# Merged-file columns, as in eqtl_analysis_cnn.py.
COL_SNP_POS, COL_REF, COL_ALT = "start", "allele1", "allele2"
COL_KEY, COL_MRE_SEQ, COL_MIRNA = "unique_key", "gene_b", "noncodingRNA"
COL_MRE_START, COL_MRE_END, COL_STRAND = "start_b", "end_b", "strand"


def norm_tissue(x: str) -> str:
    """GTEx v8 eQTL and v11 miRNA tables differ only in punctuation runs."""
    return re.sub(r"_+", "_", str(x)).strip("_")


def build_sites(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique (variant, fragment) with REF and ALT site sequences.

    Same offset/strand convention as eqtl_analysis_cnn.py; indels and rows whose
    REF base disagrees with the stored fragment are dropped, because an ALT
    sequence cannot be built for them.
    """
    cols = [COL_SNP_POS, COL_REF, COL_ALT, COL_KEY, COL_MRE_SEQ,
            COL_MRE_START, COL_MRE_END, COL_STRAND, "variant"]
    uniq = df[cols].drop_duplicates()

    out, n_indel, n_bad = [], 0, 0
    for r in uniq.itertuples(index=False):
        ref, alt = str(getattr(r, COL_REF)).upper(), str(getattr(r, COL_ALT)).upper()
        if len(ref) != 1 or len(alt) != 1:
            n_indel += 1
            continue
        seq = str(getattr(r, COL_MRE_SEQ)).upper()
        pos = int(float(getattr(r, COL_SNP_POS)))
        lo, hi = int(float(getattr(r, COL_MRE_START))), int(float(getattr(r, COL_MRE_END)))
        strand = str(getattr(r, COL_STRAND))

        off = (pos - lo) if strand == "+" else (hi - pos)
        exp = ref if strand == "+" else ref.translate(_COMPLEMENT)
        alt_in = alt if strand == "+" else alt.translate(_COMPLEMENT)
        if not (0 <= off < len(seq)) or seq[off] != exp:
            n_bad += 1
            continue

        out.append({
            "variant": r.variant, "unique_key": getattr(r, COL_KEY),
            "ref_site": seq, "alt_site": seq[:off] + alt_in + seq[off + 1:],
        })
    print(f"  unique (variant, fragment) sites : {len(out):,}  "
          f"(skipped {n_indel:,} indel, {n_bad:,} REF-mismatch)")
    return pd.DataFrame(out)


def shuffled(seq: str, rng: np.random.Generator) -> str:
    """Mononucleotide shuffle — same base composition, destroyed order."""
    a = np.frombuffer(seq.encode(), dtype="S1").copy()
    rng.shuffle(a)
    return b"".join(a.tolist()).decode()


def score_grid(sites: pd.Series, mirnas: pd.DataFrame, args, label: str
               ) -> pd.DataFrame:
    """Score every (site, miRNA) combination; returns a long frame."""
    site_list = list(dict.fromkeys(sites))
    grid = pd.DataFrame({
        "mre_sequence"  : np.repeat(site_list, len(mirnas)),
        "mirna_sequence": np.tile(mirnas["sequence"].to_numpy(), len(site_list)),
        "miRNA"         : np.tile(mirnas["miRNA"].to_numpy(), len(site_list)),
    })
    print(f"    [{label}] scoring {len(grid):,} pairs "
          f"({len(site_list):,} sites x {len(mirnas):,} miRNAs)")
    scored, _ = score_dataframe(
        args.checkpoint, grid, device=args.device,
        mre_col="mre_sequence", mirna_col="mirna_sequence",
        batch_size=args.batch_size, num_workers=args.num_workers)
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eqtl", required=True)
    ap.add_argument("--top-mirnas", required=True)
    ap.add_argument("--mirna-seqs", required=True,
                    help="From mirna_sequences_from_gff.py.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--annotated", required=True,
                    help="eqtl_annotated_*.tsv, for the current in_interaction baseline.")
    ap.add_argument("-o", required=True, help="Output directory.")
    ap.add_argument("--bind-threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- inputs ----------------------------------------------------------
    print("Loading inputs...")
    eqtl = pd.read_csv(args.eqtl, sep="\t", low_memory=False)
    eqtl["_t"] = eqtl["tissue"].map(norm_tissue)
    print(f"  eQTL rows {len(eqtl):,}, variants {eqtl['variant'].nunique():,}, "
          f"tissues {eqtl['_t'].nunique():,}")

    top = pd.read_csv(args.top_mirnas, sep="\t")
    top["_t"] = top["tissue"].map(norm_tissue)
    top = top[top["_t"].isin(set(eqtl["_t"]))]

    seqs = pd.read_csv(args.mirna_seqs, sep="\t")
    # Prefer the dataset's own sequence where it exists: that is what the CNN
    # was trained on, and it differs from the current miRBase build for ~6% of
    # names (including a miR-18a/18b label swap).
    bench: dict[str, str] = {}
    for nm, sq in zip(eqtl[COL_MIRNA + "_name"].fillna(""), eqtl[COL_MIRNA]):
        for alias in str(nm).split("|"):
            if alias.strip():
                bench[alias.strip()] = str(sq).upper()
    gff = dict(zip(seqs["name"], seqs["sequence"]))

    names = sorted(set(top["miRNA"]))
    mirnas = pd.DataFrame({
        "miRNA": names,
        "sequence": [bench.get(n, gff.get(n)) for n in names],
    }).dropna(subset=["sequence"])
    n_bench = sum(1 for n in mirnas["miRNA"] if n in bench)
    print(f"  repertoire: {len(mirnas):,} miRNAs "
          f"({n_bench:,} from dataset, {len(mirnas) - n_bench:,} from GFF3)")

    sites = build_sites(eqtl)
    if sites.empty:
        print("No scoreable sites.", file=sys.stderr)
        return 1

    # ---- scoring ---------------------------------------------------------
    print("\nScoring repertoire against REF sites...")
    real = score_grid(sites["ref_site"], mirnas, args, "real")

    print("Scoring repertoire against composition-shuffled sites (control)...")
    shuf_map = {s: shuffled(s, rng) for s in dict.fromkeys(sites["ref_site"])}
    ctrl = score_grid(pd.Series(list(shuf_map.values())), mirnas, args, "shuffled")

    thr = args.bind_threshold
    p_real = real["interaction_probability"].to_numpy()
    p_ctrl = ctrl["interaction_probability"].to_numpy()

    # Positive control: the assigned eCLIP miRNA against its own site, and
    # against a shuffled site. Without it a weak real-vs-shuffled enrichment is
    # ambiguous between "the repertoire idea fails" and "the checkpoint is
    # broken" - this separates the two.
    print("Positive control: assigned miRNA vs own site...")
    pairs = eqtl[[COL_MRE_SEQ, COL_MIRNA]].drop_duplicates()
    pc = pd.DataFrame({
        "mre_sequence": ([s.upper() for s in pairs[COL_MRE_SEQ]]
                         + [shuffled(s.upper(), rng) for s in pairs[COL_MRE_SEQ]]),
        "mirna_sequence": [s.upper() for s in pairs[COL_MIRNA]] * 2,
        "grp": ["real"] * len(pairs) + ["shuffled"] * len(pairs),
    })
    pc_scored, _ = score_dataframe(
        args.checkpoint, pc, device=args.device, mre_col="mre_sequence",
        mirna_col="mirna_sequence", batch_size=args.batch_size,
        num_workers=args.num_workers)
    pc_real = pc_scored.loc[pc_scored["grp"] == "real", "interaction_probability"].to_numpy()
    pc_shuf = pc_scored.loc[pc_scored["grp"] == "shuffled", "interaction_probability"].to_numpy()

    # ---- rescue ----------------------------------------------------------
    # A variant is "rescued" if any miRNA expressed in a tissue where that
    # variant was actually measured is predicted to bind its site.
    binders = real[real["interaction_probability"] >= thr]
    site_binders = (binders.groupby("mre_sequence")["miRNA"]
                    .agg(set).to_dict())
    tis_by_variant = eqtl.groupby("variant")["_t"].agg(set).to_dict()
    mir_by_tissue = top.groupby("_t")["miRNA"].agg(set).to_dict()

    rescued, per_variant = set(), []
    for r in sites.itertuples(index=False):
        cand = site_binders.get(r.ref_site, set())
        expressed: set[str] = set()
        for t in tis_by_variant.get(r.variant, set()):
            expressed |= mir_by_tissue.get(t, set())
        hit = cand & expressed
        if hit:
            rescued.add(r.variant)
        per_variant.append({"variant": r.variant, "unique_key": r.unique_key,
                            "n_binders_any": len(cand),
                            "n_binders_expressed": len(hit),
                            "top_binders": ",".join(sorted(hit)[:5])})
    pv = pd.DataFrame(per_variant)

    ann = pd.read_csv(args.annotated, sep="\t", low_memory=False)
    baseline = ann.loc[ann["in_interaction"] == True, "variant"].nunique()  # noqa: E712
    total = eqtl["variant"].nunique()

    # ---- report ----------------------------------------------------------
    def q(a):
        return (f"median={np.median(a):.3f} "
                f"p90={np.quantile(a, .9):.3f} frac>={thr}: {(a >= thr).mean():.1%}")

    print(f"\n{'='*66}\n  CALIBRATION (is the CNN discriminating off-distribution?)\n{'='*66}")
    print(f"  real sites     : {q(p_real)}")
    print(f"  shuffled sites : {q(p_ctrl)}")
    enr = ((p_real >= thr).mean() / max((p_ctrl >= thr).mean(), 1e-12))
    print(f"  enrichment real/shuffled at p>={thr} : {enr:.2f}x")

    # Enrichment across thresholds: a weak result at 0.5 could just be a badly
    # placed cut, so show whether it recovers anywhere.
    sweep = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        fr, fc = float((p_real >= t).mean()), float((p_ctrl >= t).mean())
        sweep.append({"threshold": t, "frac_real": fr, "frac_shuffled": fc,
                      "enrichment": fr / fc if fc else np.nan})
    print("\n  threshold sweep (repertoire vs shuffled sites):")
    print("    thr    real%    shuf%   enrichment")
    for s in sweep:
        e = "n/a" if not np.isfinite(s["enrichment"]) else f"{s['enrichment']:.2f}x"
        print(f"    {s['threshold']:<6} {s['frac_real']:7.3%} {s['frac_shuffled']:7.3%}   {e}")

    pc_enr = ((pc_real >= thr).mean() / max((pc_shuf >= thr).mean(), 1e-12))
    print(f"\n  POSITIVE CONTROL - assigned miRNA on its own site:")
    print(f"    real site     : n={len(pc_real):,} median={np.median(pc_real):.3f} "
          f"frac>={thr}: {(pc_real >= thr).mean():.1%}")
    print(f"    shuffled site : n={len(pc_shuf):,} median={np.median(pc_shuf):.3f} "
          f"frac>={thr}: {(pc_shuf >= thr).mean():.1%}")
    print(f"    enrichment    : {pc_enr:.2f}x   "
          f"(vs {enr:.2f}x for arbitrary expressed miRNAs)")

    print(f"\n{'='*66}\n  RESCUE (high-PIP variants with a usable interaction)\n{'='*66}")
    print(f"  variants in file                        : {total:,}")
    print(f"  currently analysable (in_interaction)    : {baseline:,}")
    print(f"  with >=1 expressed predicted binder      : {len(rescued):,}")
    if baseline:
        print(f"  fold change                             : {len(rescued)/baseline:.2f}x")
    print(f"  median expressed binders per site        : "
          f"{pv['n_binders_expressed'].median():.0f}")
    print("=" * 66)

    pv.to_csv(out_dir / "per_variant_binders.tsv", sep="\t", index=False)
    real.to_csv(out_dir / "repertoire_scores_real.tsv.gz", sep="\t", index=False)
    pd.DataFrame({"real": pd.Series(p_real).describe(),
                  "shuffled": pd.Series(p_ctrl).describe()}).to_csv(
        out_dir / "calibration_summary.tsv", sep="\t")
    pd.DataFrame(sweep).to_csv(out_dir / "calibration_threshold_sweep.tsv",
                               sep="\t", index=False, float_format="%.6g")

    # The verdict, in one machine-readable place. The rescue count is only
    # interpretable next to the calibration, so they are stored together.
    json.dump({
        "inputs": {"eqtl": str(Path(args.eqtl).resolve()),
                   "checkpoint": str(Path(args.checkpoint).resolve()),
                   "top_mirnas": str(Path(args.top_mirnas).resolve())},
        "bind_threshold": thr,
        "repertoire": {"n_mirnas": int(len(mirnas)), "n_sites": int(len(sites)),
                       "n_pairs_scored": int(len(real))},
        "calibration": {
            "median_real": float(np.median(p_real)),
            "median_shuffled": float(np.median(p_ctrl)),
            "frac_real_ge_thr": float((p_real >= thr).mean()),
            "frac_shuffled_ge_thr": float((p_ctrl >= thr).mean()),
            "enrichment": float(enr), "threshold_sweep": sweep},
        "positive_control": {
            "n": int(len(pc_real)),
            "median_real": float(np.median(pc_real)),
            "median_shuffled": float(np.median(pc_shuf)),
            "frac_real_ge_thr": float((pc_real >= thr).mean()),
            "frac_shuffled_ge_thr": float((pc_shuf >= thr).mean()),
            "enrichment": float(pc_enr)},
        "rescue": {"n_variants_total": int(total),
                   "n_currently_analysable": int(baseline),
                   "n_with_expressed_binder": int(len(rescued)),
                   "fold_change": float(len(rescued) / baseline) if baseline else None,
                   "median_expressed_binders_per_site":
                       float(pv["n_binders_expressed"].median())},
    }, open(out_dir / "diagnostic_stats.json", "w"), indent=2)
    print(f"\nWritten to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
