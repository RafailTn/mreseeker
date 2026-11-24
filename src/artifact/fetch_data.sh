#!/usr/bin/env bash
# Reproducer side: get every input of the artifact analysis without the raw sequencing data.
#
#   bash src/artifact/fetch_data.sh                     # downloads the bundle from Zenodo (BUNDLE_URL below)
#   bash src/artifact/fetch_data.sh /path/to/mreseeker_artifact_inputs.tar   # or use a local copy
#
# 1. miRBench v7 tables (Zenodo 10.5281/zenodo.20540907) -> data/mirbench_v7/<name>_v7.tsv (md5-checked)
# 2. Ensembl GRCh38 primary assembly (release 105)     -> data/external/genome/ (+ samtools faidx)
#    release 105 = the HybriDetector copy byte for byte (later releases unmask the chrY PAR; no row is near it)
# 3. processed-inputs bundle (make_bundle.sh; BUNDLE_README.md lists it), md5-checked against its MD5SUMS:
#      hybridetector/...  -> data/external/hybridetector/   (use as HYBRIDETECTOR_DIR)
#      mreseeker/...      -> this repo (GSE test sets in data/, readout windows)
# Existing files are kept. About 1.5 GB to download, 10 GB unpacked (most of it the genome and hyb_pairs).
# Then follow the README run order with
#   export HYBRIDETECTOR_DIR=$PWD/data/external/hybridetector
#   step 0: build_mre150.py --src-dir data/mirbench_v7 --genome data/external/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
BUNDLE_URL=${BUNDLE_URL:-"https://zenodo.org/records/24503913/files/mreseeker_artifact_inputs.tar?download=1"}   # Zenodo 10.5281/zenodo.24503913
BUNDLE_MD5=${BUNDLE_MD5:-"6274cf202204191684995f511219b7e5"}
MIRBENCH=https://zenodo.org/records/20540907/files
GENOME_URL=${GENOME_URL:-https://ftp.ensembl.org/pub/release-105/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz}
EXT=data/external
need() { command -v "$1" >/dev/null || { echo "ERROR: $1 not found on PATH"; exit 1; }; }
need curl; need md5sum; need samtools; need tar; need gzip
md5ok() { [ "$(md5sum < "$1" | cut -c1-32)" = "$2" ]; }

echo "== 1. miRBench v7"
mkdir -p data/mirbench_v7
while read -r name md5; do
    out=data/mirbench_v7/${name}_v7.tsv
    [ -s "$out" ] && { echo "  have $out"; continue; }
    curl -fsSL -o "$out.gz" "$MIRBENCH/$name.tsv.gz?download=1"
    md5ok "$out.gz" "$md5" || { echo "ERROR: md5 mismatch for $name.tsv.gz"; exit 1; }
    gzip -dc "$out.gz" > "$out" && rm "$out.gz"
    echo "  $out"
done <<'EOF'
AGO2_eCLIP_Manakov2022_train 42a7e091f66cf12539d6ae47bc2e0c85
AGO2_eCLIP_Manakov2022_test bb6140b14f025899b38469d3c78d93f7
AGO2_eCLIP_Manakov2022_leftout 0bd263447cd83f5dbdf669a1d421bd56
AGO2_CLASH_Hejret2023_train 55f1fbc97234cb3747ca7480b2ef70c6
AGO2_CLASH_Hejret2023_test 224fe08e101b6997ebae54090664c9f7
AGO2_eCLIP_Klimentova2022_test 49b3ffb7752401a52b70f18a15fec4b6
EOF

echo "== 2. GRCh38 genome"
mkdir -p $EXT/genome
FA=$EXT/genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa
if [ ! -s "$FA.fai" ]; then
    [ -s "$FA" ] || { curl -fsSL "$GENOME_URL" | gzip -dc > "$FA.tmp" && mv "$FA.tmp" "$FA"; }
    samtools faidx "$FA"
fi
echo "  $FA"

echo "== 3. processed-inputs bundle"
TAR=${1:-$EXT/mreseeker_artifact_inputs.tar}
if [ ! -s "$TAR" ]; then
    case "$BUNDLE_URL" in *RECORD_ID*) echo "ERROR: BUNDLE_URL not set yet; pass a local tar or export BUNDLE_URL"; exit 1;; esac
    curl -fsSL -o "$TAR" "$BUNDLE_URL"
fi
[ -z "$BUNDLE_MD5" ] || md5ok "$TAR" "$BUNDLE_MD5" || { echo "ERROR: bundle md5 mismatch"; exit 1; }
ST=$(mktemp -d "$EXT/unpack.XXXX")
tar -C "$ST" -xf "$TAR"
while read -r md5 path; do
    case "$path" in
        hybridetector/*) dst=$EXT/$path ;;
        mreseeker/*)     dst=${path#mreseeker/} ;;
        *) echo "ERROR: unexpected path $path"; exit 1 ;;
    esac
    if [ -s "$dst" ]; then
        md5ok "$dst" "$md5" && { echo "  have $dst"; continue; }
        echo "ERROR: $dst exists but differs from the bundle; move it away first"; exit 1
    fi
    mkdir -p "$(dirname "$dst")"
    if [ -s "$ST/$path.gz" ]; then gzip -dc "$ST/$path.gz" > "$dst"; else cp "$ST/$path" "$dst"; fi
    md5ok "$dst" "$md5" || { echo "ERROR: md5 mismatch for $dst"; exit 1; }
    echo "  $dst"
done < "$ST/MD5SUMS"
rm -rf "$ST"

echo
echo "Done. Next:"
echo "  export HYBRIDETECTOR_DIR=$ROOT/$EXT/hybridetector"
echo "  then src/artifact/README.md, run order, step 0 with:"
echo "  python3 src/cnn/build_mre150.py --src-dir data/mirbench_v7 --genome $FA"
