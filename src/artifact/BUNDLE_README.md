# MREseeker artifact analysis: processed inputs

Processed inputs for the artifact check and removal in MREseeker (`src/artifact/`,
https://github.com/RafailTn/mreseeker). With them the analysis runs without the raw sequencing data.
Unpack and place them with `src/artifact/fetch_data.sh`, which also downloads the public inputs
(miRBench v7 tables, GRCh38 genome) and checks every file against `MD5SUMS`.

All tables are gzipped TSV; `MD5SUMS` lists the checksums of the uncompressed files.

## hybridetector/

Mirrors the folder layout of the HybriDetector pipeline (https://github.com/RafailTn/Hybridetector_pipeline), so the
unpacked folder can be used directly as `HYBRIDETECTOR_DIR`.

| file | content | made by (Hybridetector_pipeline) | source data |
|---|---|---|---|
| `external/HybriDetector/hyb_pairs/<sample>.hyb_pairs_deduplicated.tsv`, `<sample>.all_hyb_pairs.tsv` | HybriDetector chimeras of 10 HEK293T AGO2 chimeric eCLIP libraries (Expt1, Expt4, Expt9; with and without gel) | stages 1-3 (`01_download_geo.sh`, `02_preprocess_fastq.sh`, `03_run_hybridetector.sh`) | GSE198250 (Manakov et al. 2022) |
| `data/validate_overexpression/sites.tsv`, `occupancy.tsv` | miR-1 / miR-124 chimera sites and AGO2 occupancy change after over-expression (Expt8) | `run_expt8.sh`, `validate_overexpression.py` | GSE198250 Expt8 IP + input; GSE197483 RNA-seq |
| `data/mixing_artifacts/chimeras.tsv`, `artifact_rate.tsv` | chimeras of the human/rat mixed-lysate libraries with species calls; post-lysis artifact rates | `setup_hsrn.sh`, `run_phaseB.sh`, `mixing_artifacts.py` | GSE198250 Expt9/10 (293T, C9 and mixed lysates) |
| `data/smallrna/gse196460_mirna_cpm.tsv` | HEK293T mature-miRNA CPM | `quantify_smallrna.sh`, `count_mirna_reads.py` | GSE196460 |

## mreseeker/

Mirrors the MREseeker repository.

| file | content | made by |
|---|---|---|
| `data/gse304955_novel_labelled_v7.tsv`, `gse263552_…`, `gse180129_HSPC_…`, `gse180129_Kasumi1_…` | external chimeric eCLIP test sets in the miRBench v7 schema | Hybridetector_pipeline stages 1-4 on GSE304955, GSE263552, GSE180129 |
| `data/relabel/noise/eval_win_v2/{50,150,300}/{mixing,occupancy}.tsv` | 50/150/300-nt windows of the mixing and occupancy sites | `src/artifact/eval_windows.py build` (needs the hg38 and hg38+rn7 genomes) |
| `data/relabel/noise/eval_win_v2/150/pred_w150_{mixing,occupancy}.tsv` | base 150-nt CNN scores of those windows | `src/artifact/eval_windows.py predict` |

Sample sheets for every GEO source are in `Hybridetector_pipeline/samplesheets/`.

## Not included (public; downloaded by fetch_data.sh)

- miRBench v7 datasets: https://doi.org/10.5281/zenodo.20540907
- Ensembl GRCh38 primary assembly (release 105)
- the base 150-nt CNN `cnn_checkpoints/cnn_mre150.pt` is in the MREseeker repository
