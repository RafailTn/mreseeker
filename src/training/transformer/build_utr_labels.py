#!/usr/bin/env python3
"""
Per-3'UTR label tracks for the transcript-level miRNA model.

Collapses the pairwise evidence (v7 chimeric windows) and the AGO2 eCLIP
libraries onto one 3'UTR per gene.  Every UTR position gets one class:

    3  KNOWN   inside a chimeric v7 window: bound, and the miRNA is known
    2  BOUND   AGO2 IP enriched over input, no v7 chimera: bound, miRNA unknown
    1  NONE    expressed, with no IP read, chimera or v7 positive within
               --buffer nt: nothing bound in these cells
    0  MASKED  everything else (unexpressed, weak IP, non-v7 chimeras, ...)

These are training labels only.  Nothing here becomes a model input: at
prediction time the model still sees just the UTR and the miRNA sequences.

Why count reads here rather than reuse the genome-wide window tables
--------------------------------------------------------------------
The negatives pipeline tiles the genome into unstranded 50-nt windows.  A 3'UTR
track needs three things that tiling does not give: strand (an antisense
gene's reads would otherwise count as expression of this UTR), transcript
coordinates (a spliced 3'UTR is not one genomic interval), and a clearance zone
around every piece of AGO2 evidence.  Counting only over the chosen 3'UTRs also
reads a small fraction of each BAM instead of the whole genome.

Transcript choice
-----------------
Ensembl 90 predates MANE and has no canonical-transcript tag, so each gene
contributes its protein-coding, `basic`-tagged transcript with the longest
3'UTR.  The distal end of a long isoform that the cells do not express costs
nothing: it has no input coverage, so it is MASKED rather than NONE.  UTRs that
overlap on the same strand (read-through genes) are grouped and always land in
the same split.

Splits
------
miRBench held out chromosome 1 for the Manakov and Hejret test sets (neither
training table has a chr1 row), so chromosome 1 is the test split here too
(--test-chroms).  Scoring those benchmark fragments therefore stays leak-free
for any model trained on these tracks.  A random --val-frac of the remaining
gene groups is validation.

Leftout families are held out by miRNA family, not by chromosome, so their
windows can sit in training genes.  Outside the test split they are MASKED
(with --buffer) and flagged `heldout_family` in sites.tsv, which hides both
the identity and the occupancy of those exact sites.

Tables tagged `external` (other cell types, e.g. SAEC) go to sites.tsv and veto
NONE around their positives, but never become KNOWN: the tracks describe the
cells the BAMs came from.

Inputs
------
--gtf           Ensembl GTF (Homo_sapiens.GRCh38.90.gtf[.gz])
--table ROLE:PATH
                v7 tables; ROLE is train, test, leftout or external
--genome        matching FASTA; without it, no sequences and no sequence check
--bam-manifest  TSV with columns experiment, role (ip|input), bam,
                library_size [, strand (sense|antisense|auto)]; without it,
                no labels
--chimeras      TSV with columns chrom, start (0-based), end, strand: the target
                arm of every chimera, of every small-RNA type.  Vetoes NONE.

Outputs (in --out-dir)
----------------------
utrs.tsv         one row per gene: transcript, segments, split, class sizes
sites.tsv        every v7 row inside a chosen UTR, in transcript coordinates,
                 with its canonical seed site and the read counts around it
utr3.fa          UTR sequences, 5'->3'                      (with --genome)
labels.npz       int8 class track per transcript_id         (with --bam-manifest)
calibration.tsv  how often a real site (v7 positive) would pass the NONE rule,
                 by input depth; choose --min-input-reads from this
summary.json

Example
-------
python3 src/training/transformer/build_utr_labels.py \\
    --gtf Homo_sapiens.GRCh38.90.gtf.gz \\
    --genome Homo_sapiens.GRCh38.dna.primary_assembly.fa \\
    --bam-manifest bams.tsv --chimeras chimeras.tsv \\
    --table train:AGO2_eCLIP_Manakov2022_train_v7.tsv \\
    --table test:AGO2_eCLIP_Manakov2022_test_v7.tsv \\
    --table leftout:AGO2_eCLIP_Manakov2022_leftout_v7.tsv \\
    --out-dir data/utr_labels --threads 16

Dependencies: numpy, pandas, and pysam when --genome or --bam-manifest is given.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]

MASKED, NONE, BOUND, KNOWN = 0, 1, 2, 3
CLASS_NAMES = {MASKED: "masked", NONE: "none", BOUND: "bound", KNOWN: "known"}
ROLES = ("train", "test", "leftout", "external")
TRACK_ROLES = ("train", "test", "leftout")
DEFAULT_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
V7_COLS = ["gene", "noncodingRNA", "noncodingRNA_name", "noncodingRNA_fam",
           "feature", "label", "chr", "start", "end", "strand"]
INPUT_BINS = [0, 1, 2, 3, 5, 10, 20, 50, 100, np.inf]

_COMP = str.maketrans("ACGTN", "TGCAN")


def dna(seq) -> str:
    return str(seq).upper().replace("U", "T")


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def norm_chrom(c) -> str:
    c = str(c)
    if c.startswith("chr"):
        c = c[3:]
    return "MT" if c == "M" else c


def ref_name(references, chrom: str):
    """The name `chrom` goes by in a BAM/FASTA (Ensembl `1` or UCSC `chr1`)."""
    if chrom in references:
        return chrom
    alt = "chrM" if chrom == "MT" else "chr" + chrom
    return alt if alt in references else None


# ---------------------------------------------------------------------------
# 3'UTR selection and coordinate mapping
# ---------------------------------------------------------------------------

@dataclass
class UTR:
    gene_id: str
    gene_name: str
    transcript_id: str
    chrom: str
    strand: str
    segments: list  # [(start, end)] genomic, 0-based half-open, in 5'->3' order
    offsets: list   # transcript coordinate at which each segment begins

    @property
    def length(self) -> int:
        s, e = self.segments[-1]
        return self.offsets[-1] + e - s

    def span(self, blocks):
        """Transcript span [lo, hi) of aligned genomic blocks; None if none land in the UTR."""
        lo = hi = None
        for bs, be in blocks:
            for (s, e), off in zip(self.segments, self.offsets):
                a, b = max(bs, s), min(be, e)
                if a >= b:
                    continue
                if self.strand == "+":
                    ta, tb = off + a - s, off + b - s
                else:
                    ta, tb = off + e - b, off + e - a
                lo = ta if lo is None else min(lo, ta)
                hi = tb if hi is None else max(hi, tb)
        return None if lo is None else (lo, hi)


_TID = re.compile(r'transcript_id "([^"]+)"')
_GID = re.compile(r'gene_id "([^"]+)"')
_GNAME = re.compile(r'gene_name "([^"]+)"')
_TBIO = re.compile(r'transcript_biotype "([^"]+)"')


def select_utrs(gtf: str, chroms: set, min_length: int) -> list:
    """Longest 3'UTR per gene among protein-coding, `basic`-tagged transcripts."""
    opener = gzip.open if gtf.endswith(".gz") else open
    meta, segs = {}, defaultdict(list)
    with opener(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if len(f) < 9 or f[2] not in ("transcript", "three_prime_utr"):
                continue
            chrom = norm_chrom(f[0])
            if chrom not in chroms:
                continue
            tid = _TID.search(f[8]).group(1)
            if f[2] == "three_prime_utr":
                segs[tid].append((int(f[3]) - 1, int(f[4])))
                continue
            bio = _TBIO.search(f[8])
            if bio is None or bio.group(1) != "protein_coding" or 'tag "basic"' not in f[8]:
                continue
            name = _GNAME.search(f[8])
            meta[tid] = (_GID.search(f[8]).group(1), name.group(1) if name else "", chrom, f[6])

    best = {}
    for tid, (gid, gname, chrom, strand) in meta.items():
        parts = segs.get(tid)
        if not parts:
            continue
        parts = sorted(set(parts), reverse=(strand == "-"))
        offsets, off = [], 0
        for s, e in parts:
            offsets.append(off)
            off += e - s
        u = UTR(gid, gname, tid, chrom, strand, parts, offsets)
        cur = best.get(gid)
        # Longest wins; equal lengths fall to the smaller transcript id so reruns agree.
        if cur is None or (u.length, cur.transcript_id) > (cur.length, u.transcript_id):
            best[gid] = u
    utrs = [u for u in best.values() if u.length >= min_length]
    return sorted(utrs, key=lambda u: (u.chrom, min(s for s, _ in u.segments)))


class SegmentIndex:
    """Genomic (chrom, strand, interval) -> UTR transcript coordinates."""

    # Earlier-starting segments tested per query.  Same-strand segments only
    # nest where genes overlap (read-through transcripts), so a handful covers
    # every real case.
    MAX_BACK = 16

    def __init__(self, utrs: list):
        rows = defaultdict(list)
        for ui, u in enumerate(utrs):
            for (s, e), off in zip(u.segments, u.offsets):
                rows[(u.chrom, u.strand)].append((s, e, ui, off))
        self.groups = {}
        for key, r in rows.items():
            a = np.asarray(r, dtype=np.int64)
            self.groups[key] = a[np.argsort(a[:, 0], kind="stable")]

    def map(self, chrom, strand, a, b, contained: bool):
        """Map query intervals [a, b) to (query_idx, utr_idx, tx_start, tx_end).

        contained=True keeps only queries lying wholly inside one segment: a v7
        window that straddles a UTR intron is not a substring of the transcript.
        contained=False keeps every overlap, clipped to the segment.
        """
        empty = np.empty(0, dtype=np.int64)
        g = self.groups.get((chrom, strand))
        if g is None or len(a) == 0:
            return empty, empty, empty, empty
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        starts, ends = g[:, 0], g[:, 1]
        top = np.searchsorted(starts, b, side="left") - 1  # last segment starting before b
        qs, idxs = [], []
        for back in range(self.MAX_BACK):
            idx = top - back
            valid = idx >= 0
            if not valid.any():
                break
            i = np.where(valid, idx, 0)
            if contained:
                hit = valid & (starts[i] <= a) & (ends[i] >= b)
            else:
                hit = valid & (starts[i] < b) & (ends[i] > a)
            q = np.nonzero(hit)[0]
            qs.append(q)
            idxs.append(i[q])
        q = np.concatenate(qs) if qs else empty
        if len(q) == 0:
            return empty, empty, empty, empty
        i = np.concatenate(idxs)
        s, e, ui, off = g[i, 0], g[i, 1], g[i, 2], g[i, 3]
        qa, qb = np.maximum(a[q], s), np.minimum(b[q], e)
        if strand == "+":
            ta, tb = off + qa - s, off + qb - s
        else:
            ta, tb = off + e - qb, off + e - qa
        return q, ui, ta, tb


def assign_splits(utrs: list, test_chroms: set, val_frac: float, seed: int):
    """Group same-strand overlapping UTRs, then split whole groups."""
    parent = list(range(len(utrs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_key = defaultdict(list)
    for ui, u in enumerate(utrs):
        for s, e in u.segments:
            by_key[(u.chrom, u.strand)].append((s, e, ui))
    for rows in by_key.values():
        rows.sort()
        end, owner = -1, -1
        for s, e, ui in rows:
            if s < end:
                parent[find(ui)] = find(owner)
            if e > end:
                end, owner = e, ui

    members = defaultdict(list)
    for ui in range(len(utrs)):
        members[find(ui)].append(ui)
    rng = np.random.default_rng(seed)
    group = np.empty(len(utrs), dtype=np.int64)
    split = np.empty(len(utrs), dtype=object)
    for gi, root in enumerate(sorted(members)):
        ms = members[root]
        group[ms] = gi
        if utrs[ms[0]].chrom in test_chroms:
            split[ms] = "test"
        else:
            split[ms] = "val" if rng.random() < val_frac else "train"
    return group, split


# ---------------------------------------------------------------------------
# v7 tables -> sites in transcript coordinates
# ---------------------------------------------------------------------------

def load_tables(specs: list) -> pd.DataFrame:
    frames = []
    for spec in specs:
        role, _, path = spec.partition(":")
        if role not in ROLES or not path:
            raise SystemExit(f"--table wants ROLE:PATH with ROLE in {ROLES}, got {spec!r}")
        df = pd.read_csv(path, sep="\t", usecols=V7_COLS, dtype={"chr": str})
        df = df.rename(columns={"gene": "target_seq", "noncodingRNA": "mirna_seq",
                                "noncodingRNA_name": "mirna_names", "noncodingRNA_fam": "family"})
        df["chrom"] = df["chr"].map(norm_chrom)
        # v7 windows are 1-based inclusive (end - start == 49).
        df["start0"] = df["start"].astype(float).astype(np.int64) - 1
        df["end0"] = df["end"].astype(float).astype(np.int64)
        df["source"] = Path(path).name.split(".")[0]
        df["role"] = role
        frames.append(df.drop(columns=["chr", "start", "end"]))
        print(f"[tables] {role:8s} {Path(path).name}: {len(df):,} rows")
    out = pd.concat(frames, ignore_index=True)
    out["row_id"] = np.arange(len(out))
    return out


def map_intervals(df: pd.DataFrame, index: SegmentIndex, contained: bool) -> pd.DataFrame:
    """Rows of `df` (chrom, strand, start0, end0) repeated once per UTR they map into."""
    parts = []
    for (chrom, strand), g in df.groupby(["chrom", "strand"], sort=False):
        q, ui, ta, tb = index.map(chrom, strand, g["start0"].to_numpy(),
                                  g["end0"].to_numpy(), contained)
        if len(q):
            sub = g.iloc[q].copy()
            sub["utr_idx"], sub["tx_start"], sub["tx_end"] = ui, ta, tb
            parts.append(sub)
    if not parts:
        return df.iloc[0:0].assign(utr_idx=0, tx_start=0, tx_end=0)
    return pd.concat(parts, ignore_index=True)


def canonical_site(mirna: str, target: str):
    """Best canonical seed site of `mirna` in `target` (both 5'->3').

    Returns (type, offset of the 6mer core) or (None, -1).  The core is the
    target stretch paired with miRNA nt 2-7, which every canonical type
    contains, so offsets are comparable across types (site spacing uses them).
    """
    m = dna(mirna)
    if len(m) < 8:
        return None, -1
    m8, m7 = revcomp(m[1:8]), revcomp(m[1:7])  # pair miRNA nt 2-8 / 2-7
    t = dna(target)
    for name, pat, core in (("8mer", m8 + "A", 1), ("7mer-m8", m8, 1),
                            ("7mer-A1", m7 + "A", 0), ("6mer", m7, 0)):
        i = t.find(pat)
        if i >= 0:
            return name, i + core
    return None, -1


def annotate_sites(sites: pd.DataFrame, mirbase: dict) -> None:
    types, cores = [], []
    for mir, tgt, start in zip(sites["mirna_seq"], sites["target_seq"], sites["tx_start"]):
        kind, off = canonical_site(mir, tgt)
        types.append(kind or "none")
        cores.append(start + off if off >= 0 else -1)
    sites["site_type"] = types
    sites["site_core_tx"] = cores

    # A chimera whose miRNA arm fits several family members lists them all
    # (e.g. hsa-miR-20a-5p|hsa-miR-106a-5p); every one is a correct answer.
    cache = {}

    def compatible(names, own):
        key = (names, own)
        if key not in cache:
            seqs = {dna(own)}
            for n in str(names).split("|"):
                s = mirbase.get(n.strip())
                if s:
                    seqs.add(s)
            cache[key] = ";".join(sorted(seqs))
        return cache[key]

    sites["compatible_seqs"] = [compatible(n, s) for n, s in
                                zip(sites["mirna_names"], sites["mirna_seq"])]


# ---------------------------------------------------------------------------
# Read counting and labelling (worker processes)
# ---------------------------------------------------------------------------

def window_counts(starts, ends, L: int, W: int, pad: int = 0) -> np.ndarray:
    """Spans overlapping [j - pad, j + W + pad) for every window start j in [0, L - W].

    `starts` and `ends` are sorted separately.  Every span ending at or before
    the window's left edge also starts before its right edge, so the difference
    of the two counts is exactly the number of overlapping spans.
    """
    j = np.arange(L - W + 1)
    lo = np.clip(j - pad, 0, L)
    hi = np.clip(j + W + pad, 0, L)
    return np.searchsorted(starts, hi, "left") - np.searchsorted(ends, lo, "right")


def covered(win: np.ndarray, W: int, L: int) -> np.ndarray:
    """Positions inside at least one selected window."""
    d = np.zeros(L + 1, dtype=np.int32)
    j = np.nonzero(win)[0]
    np.add.at(d, j, 1)
    np.add.at(d, j + W, -1)
    return np.cumsum(d[:L]) > 0


def spans_array(spans) -> tuple:
    if not spans:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    a = np.asarray(spans, dtype=np.int64)
    return np.sort(a[:, 0]), np.sort(a[:, 1])


_W = {}  # per-process state, filled by _init_worker


def _init_worker(bams, genome, lib, params):
    import pysam
    _W["bams"] = [(b["experiment"], b["role"], b["flip"],
                   pysam.AlignmentFile(b["bam"], "rb")) for b in bams]
    _W["fasta"] = pysam.FastaFile(genome) if genome else None
    _W["lib"], _W["p"] = lib, params


def read_spans(bam, utr: UTR, flip: bool, min_mapq: int) -> list:
    """Transcript spans of the reads on this UTR's strand."""
    ref = ref_name(bam.references, utr.chrom)
    if ref is None:
        return []
    multi = len(utr.segments) > 1
    seen, spans = set(), []
    for s, e in utr.segments:
        for r in bam.fetch(ref, s, e):
            if (r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_qcfail
                    or r.is_duplicate or r.mapping_quality < min_mapq):
                continue
            if r.is_paired and not r.is_read1:
                continue
            if (r.is_reverse != flip) != (utr.strand == "-"):
                continue  # read comes from the opposite strand's RNA
            if multi:
                key = (r.query_name, r.reference_start, r.is_read1)
                if key in seen:
                    continue  # a spliced read, already fetched in another segment
                seen.add(key)
            sp = utr.span(r.get_blocks())
            if sp is not None:
                spans.append(sp)
    return spans


def utr_sequence(fasta, utr: UTR) -> str:
    ref = ref_name(fasta.references, utr.chrom)
    parts = [fasta.fetch(ref, s, e).upper() for s, e in utr.segments]
    if utr.strand == "-":
        parts = [revcomp(x) for x in parts]
    return "".join(parts)


def label_track(L, ip, inp, chim, veto, known, masked, lib, p):
    """Class per position, plus the window counts reused for site statistics."""
    W, buf = p["window"], p["buffer"]
    lab = np.zeros(L, dtype=np.int8)
    if L < W:
        return lab, None
    expts = sorted(set(ip) | set(inp))
    zero = (np.empty(0, np.int64), np.empty(0, np.int64))
    ip_w = {e: window_counts(*ip.get(e, zero), L, W) for e in expts}
    in_w = {e: window_counts(*inp.get(e, zero), L, W) for e in expts}
    ip_sum = sum(ip_w.values())
    in_sum = sum(in_w.values())
    ip_pad = sum(window_counts(*ip.get(e, zero), L, W, buf) for e in expts)

    clear = ((ip_pad == 0) & (window_counts(*chim, L, W, buf) == 0)
             & (window_counts(*veto, L, W, buf) == 0))
    none_win = clear & (in_sum >= p["min_input_reads"])

    # Enriched in an experiment: IP/ip_lib >= fold * (input + 1)/input_lib.  The
    # pseudo-read keeps a shallow input (Expt4: 8M against 36M IP) from turning
    # a couple of IP reads into a huge fold.
    enriched = sum(
        ((ip_w[e] >= 1) & (ip_w[e] * lib[(e, "input")]
                           >= p["bound_min_fold"] * (in_w[e] + 1) * lib[(e, "ip")])).astype(np.int8)
        for e in expts)
    bound_win = (ip_sum >= p["bound_min_ip"]) & (enriched >= p["bound_min_expts"])

    lab[covered(none_win, W, L)] = NONE
    lab[covered(bound_win, W, L)] = BOUND
    for s, e in known:
        lab[s:e] = KNOWN
    for s, e in masked:
        lab[max(s - buf, 0):e + buf] = MASKED
    return lab, (in_sum, ip_sum, ip_pad)


def _process(task):
    utr, p = task["utr"], _W["p"]
    L = utr.length
    out = {"utr_idx": task["utr_idx"]}
    seq = None
    if _W["fasta"] is not None:
        seq = utr_sequence(_W["fasta"], utr)
        out["sequence"] = seq
        out["seq_match"] = [seq[s:e] == t for _, s, e, t in task["sites"]]
    if _W["bams"]:
        ip, inp = defaultdict(list), defaultdict(list)
        for expt, role, flip, bam in _W["bams"]:
            (ip if role == "ip" else inp)[expt].extend(read_spans(bam, utr, flip, p["min_mapq"]))
        lab, counts = label_track(
            L, {e: spans_array(v) for e, v in ip.items()},
            {e: spans_array(v) for e, v in inp.items()},
            spans_array(task["chim"]), spans_array(task["veto"]),
            task["known"], task["masked"], _W["lib"], p)
        out["labels"] = lab
        stats = []
        for _, s, _e, _t in task["sites"]:
            j = min(s, L - p["window"])
            if counts is None or j < 0:
                stats.append((-1, -1, -1))
            else:
                stats.append((int(counts[0][j]), int(counts[1][j]), int(counts[2][j])))
        out["site_counts"] = stats
    return out


def detect_strand(bam_path: str, utrs: list, probe: list, min_mapq: int) -> bool:
    """True if read 1 lands on the strand opposite the RNA (libraries differ)."""
    import pysam
    same = opposite = 0
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for ui in probe:
            u = utrs[ui]
            ref = ref_name(bam.references, u.chrom)
            if ref is None:
                continue
            for s, e in u.segments:
                for r in bam.fetch(ref, s, e):
                    if (r.is_unmapped or r.is_secondary or r.is_supplementary
                            or r.mapping_quality < min_mapq or (r.is_paired and not r.is_read1)):
                        continue
                    if r.is_reverse == (u.strand == "-"):
                        same += 1
                    else:
                        opposite += 1
            if same + opposite >= 200_000:
                break
    frac = same / max(same + opposite, 1)
    print(f"[strand] {Path(bam_path).name}: {frac:.1%} of {same + opposite:,} reads "
          f"on the UTR strand")
    if same + opposite < 1000:
        raise SystemExit(f"too few reads over probe UTRs in {bam_path}; set its strand column")
    if 0.25 < frac < 0.75:
        raise SystemExit(f"{bam_path} looks unstranded ({frac:.1%} sense); a strand-specific "
                         "track cannot be built from it")
    return frac < 0.5


def load_manifest(path: str):
    m = pd.read_csv(path, sep="\t")
    missing = {"experiment", "role", "bam", "library_size"} - set(m.columns)
    if missing:
        raise SystemExit(f"--bam-manifest is missing columns {sorted(missing)}")
    m["experiment"] = m["experiment"].astype(str)
    m["role"] = m["role"].str.lower()
    if not set(m["role"]) <= {"ip", "input"}:
        raise SystemExit("--bam-manifest role must be ip or input")
    if "strand" not in m.columns:
        m["strand"] = "auto"
    lib = m.groupby(["experiment", "role"])["library_size"].sum()
    for e in m["experiment"].unique():
        for role in ("ip", "input"):
            if (e, role) not in lib.index:
                raise SystemExit(f"experiment {e} has no {role} BAM; enrichment needs both")
    return m, {k: float(v) for k, v in lib.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def per_utr(df: pd.DataFrame, cols: list) -> dict:
    return {ui: list(zip(*(g[c].tolist() for c in cols))) for ui, g in df.groupby("utr_idx")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--table", action="append", required=True, metavar="ROLE:PATH")
    ap.add_argument("--genome", default=None)
    ap.add_argument("--bam-manifest", default=None)
    ap.add_argument("--chimeras", default=None)
    ap.add_argument("--mirbase", default=str(REPO / "data" / "mirbase_mature_hg38.tsv"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chroms", nargs="+", default=DEFAULT_CHROMS)
    ap.add_argument("--test-chroms", nargs="+", default=["1"])
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--buffer", type=int, default=50,
                    help="clearance around IP reads, chimeras and v7 positives for NONE")
    ap.add_argument("--min-input-reads", type=int, default=5,
                    help="input reads (summed over experiments) a NONE window needs")
    ap.add_argument("--bound-min-ip", type=int, default=10)
    ap.add_argument("--bound-min-fold", type=float, default=8.0)
    ap.add_argument("--bound-min-expts", type=int, default=2)
    ap.add_argument("--min-mapq", type=int, default=10,
                    help="applied to IP and input alike: filtering only one of them makes "
                         "repeats look expressed but AGO2-free")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = {k: getattr(args, k) for k in ("window", "buffer", "min_input_reads", "bound_min_ip",
                                            "bound_min_fold", "bound_min_expts", "min_mapq")}

    utrs = select_utrs(args.gtf, set(args.chroms), args.window)
    print(f"[utr] {len(utrs):,} genes, {sum(u.length for u in utrs) / 1e6:.1f} Mb of 3'UTR")
    index = SegmentIndex(utrs)
    group, split = assign_splits(utrs, set(args.test_chroms), args.val_frac, args.seed)

    tables = load_tables(args.table)
    sites = map_intervals(tables, index, contained=True)
    sites["split"] = split[sites["utr_idx"].to_numpy()]
    leftout_fams = set(tables.loc[tables["role"] == "leftout", "family"])
    sites["heldout_family"] = sites["family"].isin(leftout_fams)
    mirbase = {}
    if Path(args.mirbase).exists():
        mb = pd.read_csv(args.mirbase, sep="\t", usecols=["name", "sequence"])
        mirbase = dict(zip(mb["name"], mb["sequence"].map(dna)))
    annotate_sites(sites, mirbase)

    mapping = {}
    for src, g in tables.groupby("source", sort=False):
        pos = g[g["label"] == 1]
        pos_utr3 = pos[pos["feature"] == "three_prime_utr"]
        mapped = set(sites.loc[sites["source"] == src, "row_id"])
        mapping[src] = {
            "rows": len(g), "positives": len(pos),
            "positives_mapped": int(pos["row_id"].isin(mapped).sum()),
            "utr3_positives": len(pos_utr3),
            "utr3_positives_mapped": int(pos_utr3["row_id"].isin(mapped).sum()),
        }
        m = mapping[src]
        print(f"[map] {src}: {m['positives_mapped']:,} positives inside a chosen UTR; "
              f"{m['utr3_positives_mapped']:,}/{m['utr3_positives']:,} of those annotated "
              f"three_prime_utr ({m['utr3_positives_mapped'] / max(m['utr3_positives'], 1):.1%})")

    pos = sites[sites["label"] == 1]
    veto = per_utr(pos, ["tx_start", "tx_end"])
    in_track = pos["role"].isin(TRACK_ROLES)
    hidden = pos["heldout_family"] & (pos["split"] != "test")
    # A test table's positive in a training gene would be trained on and then
    # scored; keep it out of the track (it still vetoes NONE).
    stray = (pos["role"] == "test") & (pos["split"] != "test")
    if stray.any():
        print(f"[split] {int(stray.sum()):,} test-table positives fall outside --test-chroms "
              f"({', '.join(sorted(pos.loc[stray, 'source'].unique()))}); kept out of the "
              f"track. Tag such tables `external`.")
    known = per_utr(pos[in_track & ~hidden & ~stray], ["tx_start", "tx_end"])
    masked = per_utr(pos[hidden], ["tx_start", "tx_end"])
    site_rows = per_utr(sites.assign(i=np.arange(len(sites))),
                        ["i", "tx_start", "tx_end", "target_seq"])

    chim = {}
    if args.chimeras:
        c = pd.read_csv(args.chimeras, sep="\t", dtype={"chrom": str})
        missing = {"chrom", "start", "end", "strand"} - set(c.columns)
        if missing:
            raise SystemExit(f"--chimeras is missing columns {sorted(missing)}")
        c["chrom"] = c["chrom"].map(norm_chrom)
        c = c.rename(columns={"start": "start0", "end": "end0"})
        cm = map_intervals(c, index, contained=False)
        chim = per_utr(cm, ["tx_start", "tx_end"])
        print(f"[chimeras] {len(c):,} target arms, {len(cm):,} overlaps with chosen UTRs")
    elif args.bam_manifest:
        print("[chimeras] none given: only v7 positives veto NONE; chimeras that miRBench "
              "filtered out (other small RNAs, failed filters) can still land in NONE")

    labels, seqs = {}, {}
    site_counts = np.full((len(sites), 3), -1, dtype=np.int64)
    seq_match = np.full(len(sites), np.nan)
    strand_calls = {}
    if args.genome or args.bam_manifest:
        try:
            import pysam  # noqa: F401
        except ImportError:
            raise SystemExit("--genome and --bam-manifest need pysam (pip install pysam)")
        bams, lib = [], {}
        if args.bam_manifest:
            m, lib = load_manifest(args.bam_manifest)
            # Probe strand on the UTRs with the most positives: they are expressed.
            probe = (pos["utr_idx"].value_counts().index[:500].tolist()
                     or list(range(min(500, len(utrs)))))
            for row in m.itertuples(index=False):
                if row.strand == "auto":
                    flip = detect_strand(row.bam, utrs, probe, args.min_mapq)
                else:
                    flip = row.strand == "antisense"
                strand_calls[row.bam] = "antisense" if flip else "sense"
                bams.append({"experiment": row.experiment, "role": row.role,
                             "bam": row.bam, "flip": flip})
        tasks = ({"utr_idx": ui, "utr": u, "chim": chim.get(ui, []), "veto": veto.get(ui, []),
                  "known": known.get(ui, []), "masked": masked.get(ui, []),
                  "sites": site_rows.get(ui, [])} for ui, u in enumerate(utrs))
        with ProcessPoolExecutor(args.threads, initializer=_init_worker,
                                 initargs=(bams, args.genome, lib, params)) as pool:
            for n, res in enumerate(pool.map(_process, tasks, chunksize=32), 1):
                ui = res["utr_idx"]
                rows = [r[0] for r in site_rows.get(ui, [])]
                if "sequence" in res:
                    seqs[ui] = res["sequence"]
                    seq_match[rows] = res["seq_match"]
                if "labels" in res:
                    labels[ui] = res["labels"]
                    if rows:
                        site_counts[rows] = res["site_counts"]
                if n % 2000 == 0:
                    print(f"[labels] {n:,}/{len(utrs):,} UTRs ({time.time() - t0:.0f}s)")

    # ---- outputs -----------------------------------------------------------
    utr_df = pd.DataFrame({
        "transcript_id": [u.transcript_id for u in utrs],
        "gene_id": [u.gene_id for u in utrs],
        "gene_name": [u.gene_name for u in utrs],
        "chrom": [u.chrom for u in utrs],
        "strand": [u.strand for u in utrs],
        "utr_length": [u.length for u in utrs],
        "segments": [",".join(f"{s}-{e}" for s, e in u.segments) for u in utrs],
        "group": group,
        "split": split,
        "n_positive_sites": np.bincount(pos["utr_idx"], minlength=len(utrs)),
    })
    if labels:
        for cls, name in CLASS_NAMES.items():
            utr_df[f"n_{name}_nt"] = [int((labels[ui] == cls).sum()) if ui in labels else 0
                                      for ui in range(len(utrs))]
        np.savez_compressed(out / "labels.npz",
                            **{utrs[ui].transcript_id: lab for ui, lab in labels.items()})
    utr_df.to_csv(out / "utrs.tsv", sep="\t", index=False)
    if seqs:
        with open(out / "utr3.fa", "w") as fh:
            for ui, s in seqs.items():
                u = utrs[ui]
                fh.write(f">{u.transcript_id} {u.gene_id} {u.gene_name} {split[ui]}\n{s}\n")

    sites["transcript_id"] = [utrs[i].transcript_id for i in sites["utr_idx"]]
    sites["gene_id"] = [utrs[i].gene_id for i in sites["utr_idx"]]
    sites["gene_name"] = [utrs[i].gene_name for i in sites["utr_idx"]]
    keep = ["transcript_id", "gene_id", "gene_name", "split", "source", "role", "label",
            "tx_start", "tx_end", "mirna_seq", "mirna_names", "family", "heldout_family",
            "compatible_seqs", "site_type", "site_core_tx", "feature", "target_seq"]
    if args.genome:
        sites["seq_match"] = seq_match
        keep.append("seq_match")
    if labels:
        sites["win_input_reads"], sites["win_ip_reads"], sites["pad_ip_reads"] = site_counts.T
        keep += ["win_input_reads", "win_ip_reads", "pad_ip_reads"]
    sites[keep].to_csv(out / "sites.tsv", sep="\t", index=False)

    summary = {"params": params, "genes": len(utrs), "mapping": mapping,
               "strand": strand_calls, "splits": {}}
    for sp in ("train", "val", "test"):
        ms = utr_df["split"] == sp
        entry = {"genes": int(ms.sum()), "utr_nt": int(utr_df.loc[ms, "utr_length"].sum()),
                 "positive_sites": int(((sites["split"] == sp) & (sites["label"] == 1)).sum())}
        if labels:
            for name in CLASS_NAMES.values():
                entry[f"{name}_nt"] = int(utr_df.loc[ms, f"n_{name}_nt"].sum())
        summary["splits"][sp] = entry
        print(f"[split] {sp:5s} {entry}")

    if args.genome:
        checked = sites["seq_match"].dropna()
        summary["seq_match_rate"] = float(checked.mean()) if len(checked) else None
        print(f"[check] v7 window sequence == UTR sequence for "
              f"{summary['seq_match_rate']:.2%} of {len(checked):,} sites")

    if labels:
        # How often a real site would pass NONE if its chimera had been missed:
        # enough input, and no IP read within the buffer.
        cal = sites[(sites["label"] == 1) & sites["role"].isin(TRACK_ROLES)
                    & (sites["win_input_reads"] >= 0)]
        cal = cal.assign(bin=pd.cut(cal["win_input_reads"], INPUT_BINS, right=False))
        table = cal.groupby("bin", observed=False).agg(
            sites=("pad_ip_reads", "size"),
            ip_free=("pad_ip_reads", lambda x: int((x == 0).sum())))
        table["frac_ip_free"] = table["ip_free"] / table["sites"].clip(lower=1)
        table.to_csv(out / "calibration.tsv", sep="\t")
        passing = ((cal["win_input_reads"] >= args.min_input_reads)
                   & (cal["pad_ip_reads"] == 0)).mean()
        summary["positive_would_pass_none"] = float(passing)
        print(table.to_string())
        print(f"[calibration] {passing:.3%} of positive sites would pass NONE at "
              f"--min-input-reads {args.min_input_reads}")

    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[done] {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
