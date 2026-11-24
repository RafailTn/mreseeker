#!/usr/bin/env bash
# Maintainer side: pack the processed inputs of the artifact analysis that are not public elsewhere into one tar for
# Zenodo. A reproducer unpacks it with fetch_data.sh. Nothing here is needed by someone who has the raw data.
#
#   bash src/artifact/make_bundle.sh [OUT_DIR]      # default data/zenodo_bundle -> <OUT_DIR>/mreseeker_artifact_inputs.tar
#
# Contents (paths inside the tar; every table gzipped, MD5SUMS over the uncompressed files):
#   hybridetector/...   mirrors $HYBRIDETECTOR_DIR, so it can be used as HYBRIDETECTOR_DIR after unpacking
#     external/HybriDetector/hyb_pairs/<sample>.{hyb_pairs_deduplicated,all_hyb_pairs}.tsv  10 samples (SAMPLES)
#     data/validate_overexpression/{sites,occupancy}.tsv     Expt8 occupancy (validate_overexpression.py)
#     data/mixing_artifacts/{chimeras,artifact_rate}.tsv     human/rat mixing (mixing_artifacts.py)
#     data/smallrna/gse196460_mirna_cpm.tsv                  HEK293T small-RNA CPM (quantify_smallrna.sh)
#   mreseeker/...       mirrors this repo
#     data/gse{304955,263552,180129_HSPC,180129_Kasumi1}_novel_labelled_v7.tsv   test sets not on the miRBench Zenodo
#     data/relabel/noise/eval_win_v2/{50,150,300}/{mixing,occupancy}.tsv, 150/pred_w150_*.tsv   (eval_windows.py)
# Not included: the miRBench v7 tables and the GRCh38 genome (fetch_data.sh downloads them) and cnn_mre150.pt (in git).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HYB=${HYBRIDETECTOR_DIR:-/mnt/raid1/rafail/Hybridetector_pipeline}
OUT=${1:-$ROOT/data/zenodo_bundle}
ST=$OUT/stage
rm -rf "$ST"; mkdir -p "$ST"

SAMPLES=$(cd "$ROOT/src/artifact" && python3 -c "
import re; s=open('artifact_common.py').read()
print(' '.join(re.findall(r'\"(Expt\d+_293T_\w+?_rep\d)\"', s)))")
HFILES=()
for s in $SAMPLES; do
    HFILES+=("external/HybriDetector/hyb_pairs/$s.hyb_pairs_deduplicated.tsv" "external/HybriDetector/hyb_pairs/$s.all_hyb_pairs.tsv")
done
HFILES+=(data/validate_overexpression/sites.tsv data/validate_overexpression/occupancy.tsv
         data/mixing_artifacts/chimeras.tsv data/mixing_artifacts/artifact_rate.tsv
         data/smallrna/gse196460_mirna_cpm.tsv)
MFILES=()
for s in gse304955 gse263552 gse180129_HSPC gse180129_Kasumi1; do MFILES+=("data/${s}_novel_labelled_v7.tsv"); done
for w in 50 150 300; do for t in mixing occupancy; do MFILES+=("data/relabel/noise/eval_win_v2/$w/$t.tsv"); done; done
MFILES+=(data/relabel/noise/eval_win_v2/150/pred_w150_mixing.tsv data/relabel/noise/eval_win_v2/150/pred_w150_occupancy.tsv)

pack() {   # pack <src root> <prefix> <files...>
    local src=$1 pre=$2; shift 2
    for f in "$@"; do
        [ -s "$src/$f" ] || { echo "missing $src/$f"; exit 1; }
        mkdir -p "$ST/$pre/$(dirname "$f")"
        (cd "$src" && md5sum "$f" | sed "s#  #  $pre/#") >> "$ST/MD5SUMS"
        gzip -6 -c "$src/$f" > "$ST/$pre/$f.gz"
        echo "  $pre/$f"
    done
}
: > "$ST/MD5SUMS"
pack "$HYB" hybridetector "${HFILES[@]}"
pack "$ROOT" mreseeker "${MFILES[@]}"
cp "$ROOT/src/artifact/BUNDLE_README.md" "$ST/README.md"
tar -C "$ST" -cf "$OUT/mreseeker_artifact_inputs.tar" .
(cd "$OUT" && md5sum mreseeker_artifact_inputs.tar > mreseeker_artifact_inputs.tar.md5)
rm -rf "$ST"
du -h "$OUT/mreseeker_artifact_inputs.tar"; cat "$OUT/mreseeker_artifact_inputs.tar.md5"
