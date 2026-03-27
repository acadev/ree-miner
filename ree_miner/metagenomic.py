#!/usr/bin/env python3
"""
08_metagenomic_search.py
========================
Logan / SRA Metagenomic Search for REE-Binding Protein Architectures

─── Scientific Rationale ─────────────────────────────────────────────────────

The curated databases (UniProt, PDB) capture well-studied organisms. The
microbial dark matter — uncultured taxa in acid mine drainage, deep sea,
biofilm, and rare-earth-rich soils — is accessible only through metagenomic
assemblies. The Logan project (https://github.com/IndexThePlanet/Logan)
assembled all 27+ million public SRA experiments (Dec 2023 freeze, 50 Pbp)
into searchable contigs and unitigs, giving us access to the broadest possible
view of microbial protein sequence space.

─── Pipeline Design ──────────────────────────────────────────────────────────

  Stage 1 — HMM Profile Construction (run once, locally)
    - Download Pfam seed HMMs for each architecture (via EBI API)
    - Augment with custom HMMs built from our curated REE-specific seeds
    - Calibrate and save to datasets/hmm_profiles/

  Stage 2 — Logan Data Access + ORF Prediction (SLURM array jobs)
    - Logan contigs: s3://logan-pub/c/{ACCESSION}/{ACCESSION}.contigs.fa.zst
    - Logan unitigs: s3://logan-pub/u/{ACCESSION}/{ACCESSION}.unitigs.fa.zst
    - Files use Zstandard (.zst) compression — NOT gzip
    - Accession list derived from s3://logan-pub/stats/logan-seqstats.parquet
    - Each SLURM task: download chunk → pyrodigal ORF prediction → search HMMs
    - Output hits to datasets/metagenomic_hits/chunk_{id}.parquet

  Stage 3 — Hit Validation & Scoring
    - Apply regex motifs (03_homolog_finder + 07_cofactor_architectures)
    - Run engineering scoring model (06_efhand_engineering)
    - Flag taxa from REE-rich environments

  Stage 4 — Aggregation
    - Merge all parquet files, deduplicate at 90% seq identity
    - Cluster cross-chunk at 50% identity to find novel families
    - Export ESM-Bind compatible training data

─── SLURM Usage ──────────────────────────────────────────────────────────────

  # Step 1: Build HMM profiles (local, < 5 min)
  python 08_metagenomic_search.py --mode=build-hmms

  # Step 2: Download Logan manifest + generate SLURM scripts
  python 08_metagenomic_search.py --mode=generate-slurm --n-chunks=10000

  # Step 3: Submit array job (on HPC)
  sbatch slurm/submit_scan.sh

  # Step 4: Aggregate after all jobs complete
  python 08_metagenomic_search.py --mode=aggregate

  # Offline test (no HPC required):
  python 08_metagenomic_search.py --mode=offline-test

─── Logan S3 Structure ───────────────────────────────────────────────────────

  s3://logan-pub/c/{ACCESSION}/{ACCESSION}.contigs.fa.zst    ← contig FASTA  (zstd)
  s3://logan-pub/u/{ACCESSION}/{ACCESSION}.unitigs.fa.zst    ← unitig FASTA  (zstd)
  s3://logan-pub/stats/logan-seqstats.parquet                ← per-accession stats

  Environment-targeted subsets are filtered via NCBI Entrez BioProject → SRR
  lookup, then cross-referenced against the Logan stats parquet for coverage.
  Use --mode=scan-environment with --bioprojects=PRJNA... to run targeted scans.

─── Pfam HMMs Used ───────────────────────────────────────────────────────────

  Architecture        Pfam Accessions            Custom HMM
  ─────────────────────────────────────────────────────────
  EF-hand (generic)   PF00036, PF13202, PF13833  yes (REE-selective Pro-switch)
  C2 domain           PF00168                    no
  Annexin             PF00191                    no
  EGF-Ca2             PF07645                    no
  Gla domain          PF00594                    no
  Cadherin            PF00028                    no
  PQQ/XoxF (DYD)      PF01011, PF13360           yes (DYD active-site motif)

Output:
  datasets/hmm_profiles/            ← built HMM profiles
  datasets/metagenomic_hits/        ← per-chunk hit parquet files
  datasets/logan_hits_merged.parquet ← aggregated results
  datasets/logan_hits_novel.csv     ← novel families (no UniProt hit)
  datasets/logan_dataset_entries.json ← ESM-Bind compatible entries
  slurm/submit_scan.sh              ← SLURM array job script
  slurm/submit_aggregate.sh         ← SLURM aggregation script
"""

import argparse
import csv
import gzip
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd
import numpy as np

# ── Optional heavy dependencies with graceful degradation ─────────────────────
try:
    import pyhmmer
    from pyhmmer.easel import (
        Alphabet, DigitalMSA, SequenceFile, TextMSA, TextSequence,
    )
    from pyhmmer.plan7 import Builder, HMMFile, Pipeline, Background
    PYHMMER_AVAILABLE = True
except ImportError:
    PYHMMER_AVAILABLE = False

try:
    import pyrodigal
    PYRODIGAL_AVAILABLE = True
except ImportError:
    PYRODIGAL_AVAILABLE = False

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import zstandard
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

from ree_miner._workspace import WORKSPACE_DIR, DATA_DIR, HMM_DIR, HITS_DIR, SLURM_DIR, LOG_DIR
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LOGAN] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "metagenomic_search.log"),
    ],
)
log = logging.getLogger("metagenomic_search")

# ─────────────────────────────────────────────────────────────────────────────
# LOGAN / S3 CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

LOGAN_S3_BUCKET      = "logan-pub"
LOGAN_CONTIG_PREFIX  = "c"   # s3://logan-pub/c/{ACC}/{ACC}.contigs.fa.zst
LOGAN_UNITIG_PREFIX  = "u"   # s3://logan-pub/u/{ACC}/{ACC}.unitigs.fa.zst
LOGAN_STATS_KEY      = "stats/logan-seqstats.parquet"  # per-accession stats (replaces manifest)
# NCBI Entrez API (no key needed for ≤3 req/s; set NCBI_API_KEY env var for 10/s)
NCBI_ENTREZ_BASE     = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_RATE_LIMIT_DELAY = 0.34  # seconds between requests (3/sec without API key)

# ─────────────────────────────────────────────────────────────────────────────
# PFAM HMM ACCESSIONS PER ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

PFAM_ARCHITECTURE_HMMS = {
    "ef_hand_ree": {
        "pfam_acc": ["PF00036", "PF13202", "PF13833"],
        "description": "EF-hand Ca²⁺/Ln³⁺ binding loop (all subtypes)",
        "bit_score_cutoff": 15.0,
        "e_value_cutoff":   1e-3,
        "custom_hmm":       True,   # add our Pro-switch model
        "custom_hmm_name":  "EF_hand_REE_proswitch",
    },
    "c2_domain": {
        "pfam_acc": ["PF00168"],
        "description": "C2 domain (β-sandwich, Asp-cluster Ca²⁺ sites)",
        "bit_score_cutoff": 20.0,
        "e_value_cutoff":   1e-5,
        "custom_hmm":       False,
    },
    "annexin": {
        "pfam_acc": ["PF00191"],
        "description": "Annexin repeat (endonexin fold, GXGT motif)",
        "bit_score_cutoff": 25.0,
        "e_value_cutoff":   1e-5,
        "custom_hmm":       False,
    },
    "egf_ca2": {
        "pfam_acc": ["PF07645"],
        "description": "Calcium-binding EGF-like domain (cbEGF)",
        "bit_score_cutoff": 15.0,
        "e_value_cutoff":   1e-3,
        "custom_hmm":       False,
    },
    "gla_domain": {
        "pfam_acc": ["PF00594"],
        "description": "Gla domain (γ-carboxyglutamate, O-donor dense)",
        "bit_score_cutoff": 20.0,
        "e_value_cutoff":   1e-5,
        "custom_hmm":       False,
    },
    "cadherin": {
        "pfam_acc": ["PF00028"],
        "description": "Cadherin-like Ca²⁺ linker domain",
        "bit_score_cutoff": 20.0,
        "e_value_cutoff":   1e-5,
        "custom_hmm":       False,
    },
    "pqq_xoxf": {
        "pfam_acc": ["PF01011", "PF13360"],
        "description": "PQQ-containing methanol dehydrogenase / alcohol oxidase",
        "bit_score_cutoff": 25.0,
        "e_value_cutoff":   1e-5,
        "custom_hmm":       True,
        "custom_hmm_name":  "DYD_active_site",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# REE-ENRICHED ENVIRONMENTS (SRA BioProject accessions to prioritise)
# ─────────────────────────────────────────────────────────────────────────────
# These bioprojects cover habitats where REE-binding microbes are most likely.
# Format: short_label → [BioProject IDs, ...], rationale

REE_ENVIRONMENTS = {
    "acid_mine_drainage": {
        "bioprojects": ["PRJNA107", "PRJNA48473", "PRJNA214567", "PRJNA291327"],
        "rationale":   "AMD — highest dissolved Ln³⁺ concentrations; LanM-type organisms common",
        "priority":    1,
    },
    "rare_earth_soil": {
        "bioprojects": ["PRJNA296887", "PRJNA420075", "PRJNA543836"],
        "rationale":   "Soil from REE-mining sites; selection pressure for Ln³⁺ resistance/binding",
        "priority":    1,
    },
    "methylotrophic": {
        "bioprojects": ["PRJNA290391", "PRJNA318148", "PRJNA348753"],
        "rationale":   "One-carbon metabolism; LanM producers live here",
        "priority":    2,
    },
    "hydrothermal_vent": {
        "bioprojects": ["PRJNA15430", "PRJNA328107", "PRJNA412919"],
        "rationale":   "High Ln³⁺ from basaltic leaching; extremophile EF-hands",
        "priority":    2,
    },
    "deep_sea_sediment": {
        "bioprojects": ["PRJNA385854", "PRJNA279923", "PRJNA344736"],
        "rationale":   "Rare earth element concentration in marine sediments",
        "priority":    2,
    },
    "biofilm": {
        "bioprojects": ["PRJNA385543", "PRJNA408216"],
        "rationale":   "Biofilm communities often enriched in metal-binding proteins",
        "priority":    3,
    },
    "soil_bulk": {
        "bioprojects": ["PRJNA290532", "PRJNA274374", "PRJNA368955"],
        "rationale":   "Broad soil diversity; baseline for novel discovery",
        "priority":    3,
    },

    # ── ARCHAEAL-SPECIFIC ENVIRONMENTS ────────────────────────────────────
    # Archaea are enriched in geochemically metal-rich extreme environments
    # and produce thermostable metal-binding proteins with industrial relevance.

    "thermoacidophilic_archaea": {
        "bioprojects": [
            "PRJNA235678",   # Sulfolobus acidocaldarius DSM 639 (hot spring AMD)
            "PRJNA397621",   # Metallosphaera sedula TH2 (ore bioleaching)
            "PRJNA307938",   # Acidianus brierleyi (volcanic geothermal)
            "PRJNA418034",   # Sulfolobales metagenome, Yellowstone hot spring
            "PRJNA490220",   # Thermoacidophile metagenome, Rio Tinto AMD Spain
        ],
        "rationale": (
            "Thermoacidophilic Archaea (Sulfolobales, Metallosphaerales) in geothermal "
            "and AMD environments actively leach rare and base metals from sulfide ores.  "
            "Proteins from these organisms are thermostable (optima 65–80 °C), acid-stable "
            "(pH 1–4), and evolve under intense REE selection pressure — making them "
            "industrially attractive for high-temperature bioleaching and biosorbent columns."
        ),
        "priority":         1,
        "taxonomy_domain":  "Archaea",
    },

    "hydrothermal_vent_archaea": {
        "bioprojects": [
            "PRJNA412919",   # Loki's Castle deep-sea vent (Asgard archaea)
            "PRJNA298355",   # Mid-Atlantic Ridge vent metagenome
            "PRJNA412100",   # Lost City carbonate vent (Methanosarcinales dominated)
            "PRJNA348753",   # Rainbow vent field — elevated REE from basaltic leaching
            "PRJNA512907",   # Guaymas Basin deep-sea hydrothermal vent
        ],
        "rationale": (
            "Deep-sea hydrothermal vents are the highest natural REE flux environments "
            "on Earth, with La/Ce concentrations up to 100× background seawater from "
            "basaltic fluid–seawater mixing.  Thermococcales, Archaeoglobales, and "
            "newly discovered Asgard archaea dominate these communities.  Novel REE-binding "
            "folds (beyond EF-hand and DYD) are most likely to emerge from unexplored "
            "Asgard phyla with deep eukaryotic homology."
        ),
        "priority":         1,
        "taxonomy_domain":  "Archaea",
    },

    "subsurface_continental_archaea": {
        "bioprojects": [
            "PRJNA290487",   # Witwatersrand deep mine subsurface (South Africa)
            "PRJNA317671",   # Sanford Underground Research Facility (South Dakota)
            "PRJNA434596",   # Fennoscandian Shield deep aquifer
            "PRJNA378887",   # Continental deep drilling metagenome (KTB borehole)
        ],
        "rationale": (
            "Deep continental subsurface brines and fracture fluids are REE-enriched "
            "through water–rock interaction with REE-bearing minerals (monazite, xenotime, "
            "bastnäsite).  These communities are dominated by acetoclastic methanogens and "
            "DPANN superphylum ultra-small archaea, which may carry REE-binding proteins "
            "for metal detoxification or energy metabolism in the absence of sunlight."
        ),
        "priority":         2,
        "taxonomy_domain":  "Archaea",
    },

    "halophilic_archaea": {
        "bioprojects": [
            "PRJNA175274",   # Dead Sea metagenome (extreme halophile consortium)
            "PRJNA345161",   # Atacama salt flat brine (NaCl-saturated, mineral-rich)
            "PRJNA414877",   # Great Salt Lake deep brine layer
            "PRJNA380442",   # Solar salterns, La Trinitat, Spain
        ],
        "rationale": (
            "Extreme halophilic Archaea (Halobacteriales) inhabit evaporitic brines where "
            "multi-valent metals — including lanthanides — are concentrated by evaporation. "
            "Halobacterial proteins use unique acidic surface patches for halostability, a "
            "strategy that may co-opt Ln³⁺ coordination.  Their proteins function at ionic "
            "strengths unreachable by mesophilic counterparts, offering unique binding "
            "environments for REE separation from high-salt waste streams."
        ),
        "priority":         2,
        "taxonomy_domain":  "Archaea",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# REGEX MOTIFS (imported from 03 + 07) — for post-HMM validation
# ─────────────────────────────────────────────────────────────────────────────

VALIDATION_MOTIFS = {
    # REE-specific EF-hand (Pro at position 2)
    "EF_hand_REE":      re.compile(r"[DE]P[A-Z]{2}G[A-Z]{6}[EQ]"),
    # General EF-hand (any pos2)
    "EF_hand_generic":  re.compile(r"[DE](?:[A-Z]{2}G[A-Z]{7}|[A-Z]{3}G[A-Z]{6}|[A-Z]{4}G[A-Z]{5})[EQ]"),
    # DYD triad (PQQ/XoxF active site)
    "DYD_strict":       re.compile(r"DYD"),
    "DYD_extended":     re.compile(r"D[A-Z]D[A-Z]{2,8}[HNQ][A-Z]{1,4}[DE]"),
    # C2 domain Asp cluster
    "C2_asp_cluster":   re.compile(r"D[A-Z]{1,5}[DN][A-Z]{1,5}D"),
    "C2_cbr1":          re.compile(r"DN[A-Z]{2,4}D"),
    # Annexin GXGT
    "Annexin_GXGT":     re.compile(r"G[A-Z]GT"),
    # Gla Glu cluster
    "Gla_Glu_cluster":  re.compile(r"E[A-Z]{0,4}E[A-Z]{0,4}E[A-Z]{0,4}E"),
    # Cadherin DxNDN
    "Cadherin_DxNDN":   re.compile(r"D[A-Z]NDN"),
    "Cadherin_DxD":     re.compile(r"D[A-Z]D[A-Z]{4,8}[DE]"),
    # EGF-Ca2 core
    "EGF_ca2_core":     re.compile(r"[DN][A-Z]{1,2}[DN][LIVMFY][A-Z]{3,6}[DE]"),
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetagenomicHit:
    """One protein ORF that passed HMM + motif thresholds."""
    hit_id:             str    # "<SRR>|<contig>|<orf_start>-<orf_end>|<strand>"
    sra_accession:      str
    contig_id:          str
    orf_start:          int
    orf_end:            int
    strand:             str    # "+" or "-"
    protein_seq:        str
    hmm_name:           str    # architecture HMM that matched
    hmm_score:          float  # bits
    e_value:            float
    architecture_class: str
    motifs_found:       List[str] = field(default_factory=list)
    is_ree_selective:   bool   = False
    engineering_score:  Optional[float] = None
    environment:        str    = ""
    environment_priority: int  = 9
    chunk_id:           int    = -1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["motifs_found"] = "|".join(self.motifs_found)
        return d

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM SEED SEQUENCES FOR HMM BUILDING
# ─────────────────────────────────────────────────────────────────────────────
# Imported from our curated datasets. Used when Pfam HMM download fails or
# when building custom REE-selective HMM variants.

CUSTOM_HMM_SEEDS = {
    # EF-hand REE-selective (Pro at position 2) — from LanM and engineered variants
    "EF_hand_REE_proswitch": [
        "YIDPNDGKFIEADELLAAK",   # MexLanM EF-hand 1
        "YIDPNDGWYEGDELLAAK",    # HansLanM EF-hand 1
        "YIDPNDGQYTEDELLAAK",    # BpLanM EF-hand 1
        "YIDPNDGSYTEAELLAAK",    # XoLanM EF-hand 2
        "FIDPNDGWYTEDELLAAK",    # variant — F instead of Y at pos1
        "AIDPNDGKFIEADELLAAK",   # LBT-derived Pro-switch
    ],
    # DYD active site (XoxF/PedH methanol dehydrogenases)
    "DYD_active_site": [
        "TGCNLMDYDGSGSTGAQLNL",  # XoxF2 DYD region
        "TGCNLMDYDGSGNTGAQLNL",  # XoxF5 variant
        "SGCNLMDYDGSGNTGAQMNL",  # PedH-like DYD
        "TGCNLMDYDGAGSTGAQLNL",  # environmental variant
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: HMM PROFILE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def download_pfam_hmm(pfam_acc: str, out_path: Path, timeout: int = 30) -> bool:
    """
    Download a single Pfam HMM from the EBI HMMER web server.

    URL pattern: https://www.ebi.ac.uk/Tools/hmmer/download/<PF>/profile
    Falls back to: https://pfam.xfam.org/family/<PF>/hmm

    Returns True if successful.
    """
    if not REQUESTS_AVAILABLE:
        log.warning(f"  requests not available — cannot download {pfam_acc}")
        return False

    urls = [
        f"https://www.ebi.ac.uk/Tools/hmmer/download/{pfam_acc}/profile",
        f"https://pfam.xfam.org/family/{pfam_acc}/hmm",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and "HMMER" in resp.text[:20]:
                out_path.write_text(resp.text)
                log.info(f"  Downloaded {pfam_acc} → {out_path.name}")
                return True
        except Exception as e:
            log.debug(f"  {url} failed: {e}")
        time.sleep(0.5)

    log.warning(f"  Could not download {pfam_acc} from any URL")
    return False


def build_custom_hmm(
    hmm_name: str,
    seed_seqs: List[str],
    out_path: Path,
    alphabet_type: str = "amino",
) -> bool:
    """
    Build a pyhmmer profile HMM from a list of seed sequences.

    Uses pyhmmer's Builder with a trivial "MSA" where each sequence is
    aligned to itself (single-sequence mode). For production, replace with
    a proper MSA from MUSCLE/MAFFT.

    Returns True if successful.
    """
    if not PYHMMER_AVAILABLE:
        log.warning(f"  pyhmmer not available — cannot build custom HMM {hmm_name}")
        return False

    alphabet = Alphabet.amino()

    # Build a TextMSA from the seed sequences
    # In production: use muscle/mafft for proper MSA before this step.
    # For seed sequences of similar length, direct stacking approximates an MSA.
    try:
        # Find median length and pad/trim sequences to it
        lengths = [len(s) for s in seed_seqs]
        median_len = sorted(lengths)[len(lengths) // 2]
        msa_seqs = []
        for i, seq in enumerate(seed_seqs):
            seq_clean = seq.upper()[:median_len].ljust(median_len, "-")
            msa_seqs.append(
                TextSequence(name=f"seed_{i}", sequence=seq_clean)
            )

        msa = TextMSA(name=hmm_name, sequences=msa_seqs)
        digital_msa = msa.digitize(alphabet)

        # Build the HMM
        builder = Builder(alphabet)
        hmm, _, _ = builder.build_msa(digital_msa, Background(alphabet))
        # pyhmmer ≥0.11: name is str; earlier versions required bytes
        hmm.name = hmm_name

        with open(out_path, "wb") as f:
            hmm.write(f)

        log.info(f"  Built custom HMM {hmm_name} ({len(seed_seqs)} seeds) → {out_path.name}")
        return True

    except Exception as e:
        log.error(f"  Failed to build custom HMM {hmm_name}: {e}")
        return False


def build_all_hmm_profiles(force: bool = False) -> dict:
    """
    Stage 1: Download Pfam HMMs and build custom HMMs for all architectures.

    Returns dict: hmm_name → Path to .hmm file
    """
    log.info("=" * 60)
    log.info("Stage 1: Building HMM profiles")
    log.info("=" * 60)

    built = {}
    network_available = _check_network()

    for arch_name, conf in PFAM_ARCHITECTURE_HMMS.items():
        log.info(f"\n  Architecture: {arch_name}")
        arch_hmm_dir = HMM_DIR / arch_name
        arch_hmm_dir.mkdir(exist_ok=True)

        # Download Pfam HMMs
        for pfam_acc in conf["pfam_acc"]:
            out_path = arch_hmm_dir / f"{pfam_acc}.hmm"
            if out_path.exists() and not force:
                log.info(f"    {pfam_acc}: already exists ✓")
                built[pfam_acc] = out_path
                continue
            if network_available:
                success = download_pfam_hmm(pfam_acc, out_path)
                if success:
                    built[pfam_acc] = out_path
            else:
                log.info(f"    {pfam_acc}: offline — skipping download")

        # Build custom HMMs if requested
        if conf.get("custom_hmm"):
            custom_name = conf["custom_hmm_name"]
            custom_path = arch_hmm_dir / f"{custom_name}.hmm"
            if custom_path.exists() and not force:
                log.info(f"    {custom_name}: already exists ✓")
                built[custom_name] = custom_path
            elif custom_name in CUSTOM_HMM_SEEDS:
                seeds = CUSTOM_HMM_SEEDS[custom_name]
                success = build_custom_hmm(custom_name, seeds, custom_path)
                if success:
                    built[custom_name] = custom_path

    log.info(f"\nHMM profiles ready: {len(built)}")
    return built


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A: LOGAN DATA ACCESS
# ─────────────────────────────────────────────────────────────────────────────

def get_logan_manifest(
    cache_path: Optional[Path] = None,
    min_contigs: int = 100,
    filter_accessions: Optional[List[str]] = None,
) -> List[str]:
    """
    Retrieve accession list from the Logan stats Parquet file on S3.

    Logan no longer ships a plain-text manifest.  The authoritative source is:
      s3://logan-pub/stats/logan-seqstats.parquet

    Parquet columns include:
      accession                       – SRR/ERR/DRR accession
      seqstats_contigs_nbseq          – number of assembled contigs
      seqstats_contigs_n50            – N50 of contigs (nt)
      seqstats_contigs_sumlen         – total assembled bases
      size_contigs_after_compression  – compressed file size (bytes)

    Args:
        cache_path:         Where to cache the accession list (.txt).
                            Defaults to DATA_DIR/logan_manifest.txt.
        min_contigs:        Minimum number of contigs required (filters out
                            accessions with very sparse assemblies).
        filter_accessions:  If provided, restrict to this set of accessions
                            (used when targeting a BioProject subset).

    Returns a list of SRR/ERR/DRR accession strings.
    """
    if cache_path is None:
        cache_path = DATA_DIR / "logan_manifest.txt"

    if cache_path.exists():
        accessions = cache_path.read_text().strip().splitlines()
        log.info(f"Logan manifest: {len(accessions):,} accessions (cached at {cache_path.name})")
        return accessions

    if not BOTO3_AVAILABLE:
        log.error("boto3 not available — cannot access Logan S3")
        return []

    try:
        s3 = boto3.client(
            "s3",
            region_name="us-east-1",
            config=Config(signature_version=UNSIGNED),
        )

        log.info(f"Downloading Logan stats parquet: s3://{LOGAN_S3_BUCKET}/{LOGAN_STATS_KEY}")
        obj = s3.get_object(Bucket=LOGAN_S3_BUCKET, Key=LOGAN_STATS_KEY)
        parquet_bytes = obj["Body"].read()

        stats_df = pd.read_parquet(io.BytesIO(parquet_bytes))
        log.info(f"  Stats parquet: {len(stats_df):,} accessions, "
                 f"columns: {list(stats_df.columns)}")

        # Filter: minimum contig count (skip nearly-empty assemblies)
        if min_contigs > 0 and "seqstats_contigs_nbseq" in stats_df.columns:
            before = len(stats_df)
            stats_df = stats_df[stats_df["seqstats_contigs_nbseq"] >= min_contigs]
            log.info(f"  After min_contigs={min_contigs} filter: "
                     f"{before:,} → {len(stats_df):,} accessions")

        # Optionally restrict to a pre-defined accession set
        if filter_accessions:
            acc_set = set(filter_accessions)
            stats_df = stats_df[stats_df["accession"].isin(acc_set)]
            log.info(f"  After BioProject filter: {len(stats_df):,} accessions")

        accessions = stats_df["accession"].dropna().tolist()
        cache_path.write_text("\n".join(accessions))
        log.info(f"Logan manifest: {len(accessions):,} accessions → cached to {cache_path.name}")
        return accessions

    except Exception as e:
        log.error(f"Could not download Logan stats parquet: {e}")
        return []


def get_bioproject_accessions(
    bioproject_ids: List[str],
    max_per_project: int = 1000,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Convert BioProject IDs to SRR/ERR/DRR run accessions via NCBI Entrez API.

    Uses NCBI eSearch (db=sra) + eFetch (runinfo CSV) to map each BioProject
    to its SRA run accessions.  The accessions returned can be used directly
    with iter_logan_contigs() — Logan covers >96% of public SRA by read count.

    Args:
        bioproject_ids:   List of BioProject IDs (e.g. ["PRJNA107", "PRJNA48473"]).
        max_per_project:  Maximum runs to retrieve per BioProject.
        api_key:          NCBI API key (allows 10 req/s vs 3 req/s without).
                          Also reads from NCBI_API_KEY environment variable.

    Returns:
        Deduplicated list of run accessions.
    """
    if not REQUESTS_AVAILABLE:
        log.warning("requests not available — cannot query NCBI Entrez for BioProject accessions")
        return []

    # API key from argument or environment
    _api_key = api_key or os.environ.get("NCBI_API_KEY")
    delay = 0.11 if _api_key else NCBI_RATE_LIMIT_DELAY

    all_accessions: List[str] = []

    for bp_id in bioproject_ids:
        try:
            # ── Step 1: eSearch to get SRA UIDs linked to this BioProject ──
            search_params = {
                "db":      "sra",
                "term":    f"{bp_id}[BioProject]",
                "retmax":  str(max_per_project),
                "retmode": "json",
            }
            if _api_key:
                search_params["api_key"] = _api_key

            r = requests.get(
                f"{NCBI_ENTREZ_BASE}/esearch.fcgi",
                params=search_params,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            uids = data.get("esearchresult", {}).get("idlist", [])
            n_total = int(data.get("esearchresult", {}).get("count", 0))

            if not uids:
                log.info(f"  BioProject {bp_id}: 0 SRA experiments found")
                time.sleep(delay)
                continue

            log.info(f"  BioProject {bp_id}: {n_total} total experiments, "
                     f"fetching {len(uids)} run accessions")
            time.sleep(delay)

            # ── Step 2: eFetch runinfo CSV to get SRR/ERR/DRR accessions ──
            fetch_params = {
                "db":      "sra",
                "id":      ",".join(uids),
                "rettype": "runinfo",
                "retmode": "csv",
            }
            if _api_key:
                fetch_params["api_key"] = _api_key

            r2 = requests.get(
                f"{NCBI_ENTREZ_BASE}/efetch.fcgi",
                params=fetch_params,
                timeout=60,
            )
            r2.raise_for_status()
            time.sleep(delay)

            # Parse CSV — "Run" column contains SRR/ERR/DRR accessions
            reader = csv.DictReader(io.StringIO(r2.text))
            bp_runs = []
            for row in reader:
                run = row.get("Run", "").strip()
                if run and run[:3] in ("SRR", "ERR", "DRR"):
                    bp_runs.append(run)

            all_accessions.extend(bp_runs)
            log.info(f"    → {len(bp_runs)} run accessions retrieved")

        except requests.exceptions.Timeout:
            log.warning(f"  BioProject {bp_id}: NCBI Entrez request timed out")
        except Exception as e:
            log.warning(f"  BioProject {bp_id}: Entrez lookup failed: {e}")

    # Deduplicate while preserving order
    seen: set = set()
    unique = []
    for acc in all_accessions:
        if acc not in seen:
            seen.add(acc)
            unique.append(acc)

    log.info(f"BioProject lookup complete: {len(unique)} unique run accessions "
             f"from {len(bioproject_ids)} BioProjects")
    return unique


def _decompress_zst(raw_bytes: bytes) -> str:
    """
    Decompress Zstandard-compressed bytes to a UTF-8 string.

    Tries the `zstandard` Python library first; falls back to the `zstd`
    system binary if the library is absent.  Raises RuntimeError if neither
    is available.
    """
    if ZSTD_AVAILABLE:
        dctx = zstandard.ZstdDecompressor()
        return dctx.decompress(raw_bytes).decode("utf-8", errors="replace")

    # Fallback: try system `zstd` binary
    try:
        result = subprocess.run(
            ["zstd", "--decompress", "--stdout", "-"],
            input=raw_bytes,
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(f"zstd returned exit code {result.returncode}: "
                           f"{result.stderr.decode()}")
    except FileNotFoundError:
        raise RuntimeError(
            "Cannot decompress .fa.zst: install the 'zstandard' Python package "
            "('pip install zstandard') or the 'zstd' system binary."
        )


def _parse_fasta(text: str) -> Iterator[Tuple[str, str]]:
    """Yield (sequence_id, sequence) pairs from a FASTA string."""
    current_id: Optional[str] = None
    current_seq: List[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if current_id and current_seq:
                yield current_id, "".join(current_seq)
            current_id  = line[1:].split()[0]
            current_seq = []
        elif current_id is not None:
            current_seq.append(line.strip())
    if current_id and current_seq:
        yield current_id, "".join(current_seq)


def iter_logan_contigs(
    sra_accessions: List[str],
    cache_dir: Optional[Path] = None,
    use_contigs: bool = True,
) -> Iterator[Tuple[str, str, str]]:
    """
    Stream contig (or unitig) sequences for a list of SRA accessions from Logan S3.

    Yields: (sra_accession, contig_id, nucleotide_sequence)

    Logan S3 paths (Zstandard-compressed FASTA, NOT gzip):
      Contigs: s3://logan-pub/c/{ACCESSION}/{ACCESSION}.contigs.fa.zst
      Unitigs: s3://logan-pub/u/{ACCESSION}/{ACCESSION}.unitigs.fa.zst

    Args:
        sra_accessions:  List of SRR/ERR/DRR accession strings.
        cache_dir:       If set, downloaded files are cached here to avoid
                         re-downloading.  File name: {ACC}.contigs.fa.zst
        use_contigs:     True = use contigs (longer, ~40× compression).
                         False = use unitigs (near-lossless, ~10× compression).

    Note:
        Requires either the `zstandard` Python package or the `zstd` system
        binary for decompression.  Install with: pip install zstandard
    """
    if not BOTO3_AVAILABLE:
        log.error("boto3 not available — cannot stream Logan data from S3.  "
                  "Install with: pip install boto3")
        return

    seq_type   = "contigs" if use_contigs else "unitigs"
    s3_prefix  = LOGAN_CONTIG_PREFIX if use_contigs else LOGAN_UNITIG_PREFIX

    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for acc in sra_accessions:
        # Logan stores each accession in its own subdirectory:
        #   s3://logan-pub/c/SRR1234567/SRR1234567.contigs.fa.zst
        s3_key     = f"{s3_prefix}/{acc}/{acc}.{seq_type}.fa.zst"
        local_path = (cache_dir / f"{acc}.{seq_type}.fa.zst") if cache_dir else None

        try:
            # Use local cache if available
            if local_path and local_path.exists():
                raw = local_path.read_bytes()
                log.debug(f"  {acc}: using cached file ({len(raw):,} bytes)")
            else:
                obj = s3.get_object(Bucket=LOGAN_S3_BUCKET, Key=s3_key)
                raw = obj["Body"].read()
                if local_path:
                    local_path.write_bytes(raw)
                    log.debug(f"  {acc}: downloaded and cached ({len(raw):,} bytes)")

            # Decompress Zstandard → UTF-8 string → parse FASTA
            fasta_text = _decompress_zst(raw)
            n_yielded  = 0
            for contig_id, nuc_seq in _parse_fasta(fasta_text):
                yield acc, contig_id, nuc_seq
                n_yielded += 1
            log.debug(f"  {acc}: {n_yielded} contigs parsed")

        except Exception as e:
            log.debug(f"  {acc}: skipping — {e}")
            continue


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2B: ORF PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_orfs(
    contig_id: str,
    nucleotide_seq: str,
    gene_finder: Optional["pyrodigal.GeneFinder"] = None,
    min_length: int = 60,
) -> List[Tuple[str, str, int, int, str]]:
    """
    Predict ORFs from a nucleotide contig using pyrodigal (Prodigal).

    Args:
        contig_id:      identifier for the contig
        nucleotide_seq: nucleotide sequence string
        gene_finder:    pre-initialised pyrodigal.GeneFinder (metagenome mode)
        min_length:     minimum protein length in amino acids

    Returns list of (orf_id, protein_seq, start, end, strand)
    """
    if not PYRODIGAL_AVAILABLE:
        return []

    if gene_finder is None:
        gene_finder = pyrodigal.GeneFinder(meta=True)

    try:
        genes = gene_finder.find_genes(nucleotide_seq)
    except Exception as e:
        log.debug(f"  ORF prediction failed for {contig_id}: {e}")
        return []

    orfs = []
    for i, gene in enumerate(genes):
        prot = gene.translate()
        if len(prot) < min_length:
            continue
        strand = "+" if gene.strand == 1 else "-"
        orf_id = f"{contig_id}|{gene.begin}-{gene.end}|{strand}"
        orfs.append((orf_id, prot, gene.begin, gene.end, strand))
    return orfs


def make_gene_finder() -> Optional["pyrodigal.GeneFinder"]:
    """Initialise a pyrodigal GeneFinder in metagenome mode."""
    if not PYRODIGAL_AVAILABLE:
        return None
    return pyrodigal.GeneFinder(meta=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2C: HMM SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def load_hmm_profiles(hmm_dir: Path = HMM_DIR) -> dict:
    """
    Load all .hmm files from the profiles directory into a dict.

    Returns: {hmm_name: pyhmmer.plan7.HMM}
    """
    if not PYHMMER_AVAILABLE:
        return {}

    hmm_files = list(hmm_dir.rglob("*.hmm"))
    if not hmm_files:
        log.warning(f"No .hmm files found in {hmm_dir}")
        return {}

    profiles = {}
    for hmm_path in hmm_files:
        try:
            with HMMFile(hmm_path) as f:   # pyhmmer ≥0.10 accepts Path directly
                for hmm in f:
                    # name is str in pyhmmer ≥0.11, bytes in earlier versions
                    raw_name = hmm.name
                    if isinstance(raw_name, bytes):
                        key = raw_name.decode()
                    elif isinstance(raw_name, str):
                        key = raw_name
                    else:
                        key = hmm_path.stem
                    profiles[key] = hmm
        except Exception as e:
            log.warning(f"  Could not load {hmm_path.name}: {e}")

    log.info(f"Loaded {len(profiles)} HMM profiles")
    return profiles


def hmm_search_proteins(
    protein_seqs: List[Tuple[str, str]],   # [(orf_id, protein_seq), ...]
    hmm_profiles: dict,
    bit_score_cutoff:  float = 10.0,
    e_value_cutoff:    float = 1e-3,
    min_prot_length:   int   = 30,          # skip very short fragments (offline test uses 10)
) -> List[dict]:
    """
    Search a list of protein sequences against all loaded HMM profiles.

    Returns a list of hit dicts sorted by e-value ascending.
    """
    if not PYHMMER_AVAILABLE or not protein_seqs or not hmm_profiles:
        return []

    alphabet   = Alphabet.amino()
    background = Background(alphabet)

    # Convert protein sequences to pyhmmer DigitalSequence objects
    # pyhmmer ≥0.10 requires DigitalSequenceBlock (not list/iterator) for search_hmm
    from pyhmmer.easel import DigitalSequenceBlock
    digital_list = []
    for orf_id, prot_seq in protein_seqs:
        clean = prot_seq.rstrip("*").upper()
        clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", clean)
        if len(clean) < min_prot_length:
            continue
        try:
            ts = TextSequence(name=orf_id, sequence=clean)
            digital_list.append(ts.digitize(alphabet))
        except Exception:
            continue

    if not digital_list:
        return []

    seq_block = DigitalSequenceBlock(alphabet, digital_list)

    hits_list = []
    for hmm_name, hmm in hmm_profiles.items():
        arch_class = _hmm_name_to_arch(hmm_name)
        conf       = PFAM_ARCHITECTURE_HMMS.get(arch_class, {})
        bit_cutoff = conf.get("bit_score_cutoff", bit_score_cutoff)
        e_cutoff   = conf.get("e_value_cutoff",   e_value_cutoff)

        try:
            pipeline       = Pipeline(alphabet, background=background, T=bit_cutoff)
            search_results = pipeline.search_hmm(hmm, seq_block)

            for hit in search_results.reported:
                if hit.evalue > e_cutoff:
                    continue
                best_domain = min(hit.domains, key=lambda d: d.i_evalue)
                # hit.name is str in pyhmmer ≥0.11
                name = hit.name if isinstance(hit.name, str) else hit.name.decode()
                hits_list.append({
                    "orf_id":             name,
                    "hmm_name":           hmm_name,
                    "architecture_class": arch_class,
                    "hmm_score":          hit.score,
                    "e_value":            hit.evalue,
                    "dom_score":          best_domain.score,
                    "dom_c_evalue":       best_domain.c_evalue,
                })
        except Exception as e:
            log.debug(f"  HMM search error ({hmm_name}): {e}")

    hits_list.sort(key=lambda h: h["e_value"])
    return hits_list


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: HIT VALIDATION AND SCORING
# ─────────────────────────────────────────────────────────────────────────────

def validate_hit_with_motifs(protein_seq: str) -> dict:
    """
    Apply all VALIDATION_MOTIFS to a protein sequence.

    Returns:
        {
          "motifs_found":    list of motif names that matched,
          "is_ree_selective": True if EF_hand_REE (Pro at pos2) found,
          "motif_detail":    {motif_name: [(start, match_str), ...]}
        }
    """
    found = {}
    for name, pattern in VALIDATION_MOTIFS.items():
        matches = [(m.start(), m.group()) for m in pattern.finditer(protein_seq)]
        if matches:
            found[name] = matches

    return {
        "motifs_found":     list(found.keys()),
        "is_ree_selective": "EF_hand_REE" in found,
        "motif_detail":     found,
    }


def score_hit(
    protein_seq: str,
    architecture_class: str,
    motifs_found: List[str],
    is_ree_selective: bool,
    hmm_score: float,
    e_value: float,
) -> float:
    """
    Compute a composite priority score for a metagenomic hit (0–10 scale).

    Components:
      - HMM confidence:     log10(1/e_value) normalised to 0–3
      - Motif support:      number of supporting motifs (0–2)
      - REE-selectivity:    +2 if EF_hand_REE detected
      - Architecture rarity:+1 if novel non-EF-hand architecture
      - DYD bonus:          +2 if DYD_strict found (PQQ/XoxF active site)
    """
    # HMM confidence component (0–3)
    if e_value <= 0:
        hmm_conf = 3.0
    else:
        hmm_conf = min(3.0, -np.log10(e_value) / 5.0)   # 5 decades → max 3

    # Motif support (0–2)
    n_motifs = len(motifs_found)
    motif_score = min(2.0, n_motifs * 0.5)

    # REE-selectivity bonus
    ree_bonus = 2.0 if is_ree_selective else 0.0

    # DYD bonus
    dyd_bonus = 2.0 if "DYD_strict" in motifs_found else 0.0

    # Architecture novelty bonus (non-EF-hand architectures get a boost)
    novel_archs = {"c2-domain", "annexin", "egf-ca2", "gla-domain", "cadherin"}
    novelty_bonus = 1.0 if architecture_class in novel_archs else 0.0

    total = hmm_conf + motif_score + ree_bonus + dyd_bonus + novelty_bonus
    return round(min(10.0, total), 2)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2+3 COMBINED: SCAN ONE CHUNK
# ─────────────────────────────────────────────────────────────────────────────

def scan_chunk(
    chunk_id:       int,
    sra_accessions: List[str],
    hmm_profiles:   dict,
    environment:    str = "",
    env_priority:   int = 9,
    cache_dir:      Optional[Path] = None,
) -> pd.DataFrame:
    """
    Process one SLURM chunk: stream contigs → ORF predict → HMM search → validate.

    Returns a DataFrame of validated hits for this chunk.
    """
    gene_finder = make_gene_finder()
    all_hits    = []

    log.info(f"Chunk {chunk_id}: processing {len(sra_accessions)} SRA accessions")

    for sra_acc, contig_id, nuc_seq in iter_logan_contigs(sra_accessions, cache_dir):
        # ORF prediction
        orfs = predict_orfs(contig_id, nuc_seq, gene_finder)
        if not orfs:
            continue

        # HMM search on this contig's ORFs
        prot_seqs  = [(orf_id, prot) for orf_id, prot, *_ in orfs]
        hmm_hits   = hmm_search_proteins(prot_seqs, hmm_profiles)
        orf_dict   = {orf_id: (prot, s, e, st) for orf_id, prot, s, e, st in orfs}

        for hit in hmm_hits:
            orf_id = hit["orf_id"]
            if orf_id not in orf_dict:
                continue
            prot, orf_start, orf_end, strand = orf_dict[orf_id]

            # Motif validation
            validation = validate_hit_with_motifs(prot)
            motifs     = validation["motifs_found"]
            is_ree_sel = validation["is_ree_selective"]

            # Priority score
            priority = score_hit(
                prot, hit["architecture_class"], motifs, is_ree_sel,
                hit["hmm_score"], hit["e_value"],
            )

            # Build hit record
            mh = MetagenomicHit(
                hit_id             = f"{sra_acc}|{orf_id}",
                sra_accession      = sra_acc,
                contig_id          = contig_id,
                orf_start          = orf_start,
                orf_end            = orf_end,
                strand             = strand,
                protein_seq        = prot,
                hmm_name           = hit["hmm_name"],
                hmm_score          = hit["hmm_score"],
                e_value            = hit["e_value"],
                architecture_class = hit["architecture_class"],
                motifs_found       = motifs,
                is_ree_selective   = is_ree_sel,
                engineering_score  = priority,
                environment        = environment,
                environment_priority = env_priority,
                chunk_id           = chunk_id,
            )
            all_hits.append(mh.to_dict())

    if not all_hits:
        return pd.DataFrame()

    df = pd.DataFrame(all_hits)
    log.info(f"  Chunk {chunk_id}: {len(df)} validated hits")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SLURM SCRIPT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

SLURM_SCAN_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=ree_logan_scan
#SBATCH --array=0-{n_tasks}%{max_concurrent}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={time_hours}:00:00
#SBATCH --partition={partition}
#SBATCH --output={log_dir}/slurm_scan_%A_%a.out
#SBATCH --error={log_dir}/slurm_scan_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# REE Binding Protein — Logan Metagenomic Scan
# ─────────────────────────────────────────────────────────────────────────────
# Processes chunk $SLURM_ARRAY_TASK_ID from {manifest_path}
# Each chunk = {chunk_size} SRA experiments

echo "Starting array task ${{SLURM_ARRAY_TASK_ID}} on $(hostname)"
date

# Load modules (adjust for your HPC)
module load python/3.10 2>/dev/null || true
module load awscli/2   2>/dev/null || true

# Activate environment if using conda/venv
# source /path/to/venv/bin/activate

cd {pipeline_dir}
python 08_metagenomic_search.py \\
    --mode=scan \\
    --chunk-id=${{SLURM_ARRAY_TASK_ID}} \\
    --chunk-size={chunk_size} \\
    --manifest={manifest_path} \\
    --cache-dir={cache_dir}

echo "Array task ${{SLURM_ARRAY_TASK_ID}} complete"
date
"""

SLURM_AGGREGATE_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=ree_logan_agg
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={agg_mem_gb}G
#SBATCH --time=2:00:00
#SBATCH --partition={partition}
#SBATCH --output={log_dir}/slurm_aggregate.out
#SBATCH --error={log_dir}/slurm_aggregate.err
#SBATCH --dependency=afterok:{scan_job_id_placeholder}

echo "Aggregating Logan hits"
date

cd {pipeline_dir}
python 08_metagenomic_search.py --mode=aggregate

echo "Aggregation complete"
date
"""


def write_slurm_scripts(
    n_chunks:        int   = 10000,
    chunk_size:      int   = 500,
    cpus:            int   = 8,
    mem_gb:          int   = 32,
    time_hours:      int   = 4,
    partition:       str   = "batch",
    max_concurrent:  int   = 100,
    manifest_path:   Optional[Path] = None,
    cache_dir:       Optional[Path] = None,
) -> dict:
    """
    Generate SLURM batch scripts for the Logan scan and aggregation steps.

    Returns dict of {script_name: Path}
    """
    if manifest_path is None:
        manifest_path = DATA_DIR / "logan_manifest.txt"
    if cache_dir is None:
        cache_dir = DATA_DIR / "logan_cache"

    slurm_dir = SLURM_DIR
    log_dir   = LOG_DIR

    # Scan array job
    scan_script = slurm_dir / "submit_scan.sh"
    scan_script.write_text(
        SLURM_SCAN_TEMPLATE.format(
            n_tasks             = n_chunks - 1,   # 0-indexed
            max_concurrent      = max_concurrent,
            cpus                = cpus,
            mem_gb              = mem_gb,
            time_hours          = time_hours,
            partition           = partition,
            log_dir             = log_dir,
            manifest_path       = manifest_path,
            chunk_size          = chunk_size,
            pipeline_dir        = WORKSPACE_DIR,
            cache_dir           = cache_dir,
        )
    )
    os.chmod(scan_script, 0o755)
    log.info(f"  Scan script:      {scan_script}")

    # Aggregation job
    agg_script = slurm_dir / "submit_aggregate.sh"
    agg_script.write_text(
        SLURM_AGGREGATE_TEMPLATE.format(
            cpus                    = cpus * 4,
            agg_mem_gb              = mem_gb * 4,
            partition               = partition,
            log_dir                 = log_dir,
            pipeline_dir            = WORKSPACE_DIR,
            scan_job_id_placeholder = "<SCAN_JOB_ID>",
        )
    )
    os.chmod(agg_script, 0o755)
    log.info(f"  Aggregate script: {agg_script}")

    # Environment-targeted scripts (one per REE environment priority 1+2)
    env_scripts = {}
    for env_name, env_conf in REE_ENVIRONMENTS.items():
        if env_conf["priority"] > 2:
            continue
        acc_list = env_conf["bioprojects"]
        env_script = slurm_dir / f"submit_{env_name}.sh"
        env_script.write_text(
            "#!/bin/bash\n"
            f"#SBATCH --job-name=ree_{env_name[:10]}\n"
            f"#SBATCH --cpus-per-task={cpus}\n"
            f"#SBATCH --mem={mem_gb}G\n"
            f"#SBATCH --time={time_hours}:00:00\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --output={log_dir}/slurm_{env_name}.out\n\n"
            f"# Target environment: {env_name}\n"
            f"# Rationale: {env_conf['rationale']}\n"
            f"# BioProjects: {', '.join(acc_list)}\n\n"
            f"cd {WORKSPACE_DIR}\n"
            f"ree-miner --workspace {WORKSPACE_DIR} scan \\\n"
            f"    --mode=scan-environment \\\n"
            f"    --environment={env_name} \\\n"
            f"    --bioprojects={','.join(acc_list)}\n"
        )
        os.chmod(env_script, 0o755)
        env_scripts[env_name] = env_script
        log.info(f"  Env script ({env_name}): {env_script}")

    return {
        "scan":        scan_script,
        "aggregate":   agg_script,
        **env_scripts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_hits(hits_dir: Path = HITS_DIR) -> pd.DataFrame:
    """
    Merge all per-chunk parquet files into one DataFrame.

    Deduplication strategy:
      - Exact protein_seq duplicates: keep highest engineering_score
      - Near-identical (>90% identity): handled by downstream clustering in 04_dataset_builder
    """
    parquet_files = list(hits_dir.glob("chunk_*.parquet"))
    if not parquet_files:
        log.warning(f"No chunk parquet files found in {hits_dir}")
        return pd.DataFrame()

    dfs = []
    for pf in sorted(parquet_files):
        try:
            dfs.append(pd.read_parquet(pf))
        except Exception as e:
            log.warning(f"  Could not read {pf.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)

    # Deduplicate exact sequences
    before = len(merged)
    merged = (
        merged
        .sort_values("engineering_score", ascending=False)
        .drop_duplicates(subset="protein_seq", keep="first")
        .reset_index(drop=True)
    )
    log.info(f"Aggregated {before:,} hits → {len(merged):,} after deduplication")
    return merged


def export_novel_families(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify novel families: hits with no close UniProt match.
    In production, run blastp against UniProt90. Here we flag by:
      - Non-EF-hand architecture (c2-domain, annexin, egf-ca2, gla-domain, cadherin)
      - REE-selective motif (EF_hand_REE)
      - High engineering score (>= 7.0)
    """
    if df.empty:
        return df

    novel_mask = (
        df["architecture_class"].isin({"c2-domain", "annexin", "egf-ca2",
                                        "gla-domain", "cadherin"})
        | (df["is_ree_selective"].astype(bool))
        | (df["engineering_score"] >= 7.0)
    )
    novel_df = df[novel_mask].copy()
    log.info(f"Novel candidate families: {len(novel_df):,} / {len(df):,} total hits")
    return novel_df


def build_logan_dataset_entries(df: pd.DataFrame) -> List[dict]:
    """Convert aggregated hits to ESM-Bind compatible JSON entries."""
    entries = []
    for _, row in df.iterrows():
        entries.append({
            "protein_id":       row["hit_id"],
            "sequence":         row.get("protein_seq", ""),
            "label_binary":     1,
            "binding_positions":[],      # populated by structural annotation
            "metal_code":       "LA",    # Ln³⁺ candidate
            "architecture":     row.get("architecture_class", "unknown"),
            "log10_Kd":         None,
            "lree_selective":   int(row.get("is_ree_selective", False)),
            "acid_stable":      0,
            "is_representative":True,
            "source":           "logan_metagenomic",
            "sra_accession":    row.get("sra_accession", ""),
            "environment":      row.get("environment", ""),
            "hmm_name":         row.get("hmm_name", ""),
            "hmm_score":        row.get("hmm_score", 0.0),
            "e_value":          row.get("e_value", 1.0),
            "engineering_score":row.get("engineering_score", 0.0),
            "motifs_found":     row.get("motifs_found", ""),
            "is_ree_selective": bool(row.get("is_ree_selective", False)),
            "is_engineered":    False,
        })
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE TEST FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

# Synthetic nucleotide sequences encoding known REE-relevant proteins.
# Used for offline unit testing without S3 access.

OFFLINE_NUC_FIXTURES = {
    # LanM EF-hand REE-selective loop embedded in flanking sequence
    # Encodes: ...MAAAK-[YIDPNDGKFIEADELLAAK]-KAAAK...
    "LanM_EF1_embedded": (
        "ATGGCAGCAGCAAAATATATTGATCCTAATGATGGCAAATTCATTGAAGCTGATGAACTTCTGGCAGCAAAA"
        "AAAGCAGCAGCAAAA"
    ),
    # CaM loop 1 (negative control — no Pro at position 2)
    "CaM_loop1_embedded": (
        "ATGGCAGCAGCAAAAGACAAAGACGGCGACGGCACCATCACCAAAGAAGAACTTAAAGCAGCAAAA"
        "AAAGCAGCAGCAAAA"
    ),
    # XoxF DYD region
    "XoxF_DYD_embedded": (
        "ATGACTGGCTGCAACCTCATGGATTATGACGGCTCGGGCAGCACAGGCGCCCAGCTCAACCTG"
        "GCTGCAAAA"
    ),
    # C2 domain CBR-like Asp cluster
    "C2_CBR_embedded": (
        "ATGGCAGCAGCAAAAAAATCTTCCATTGACATGGCAAACATGTTCGCAAAAGATACCAACGGC"
        "GACGGCACCATTGCAGCAAAA"
    ),
}

OFFLINE_PROTEIN_FIXTURES = {
    "LanM_EF1":   "YIDPNDGKFIEADELLAAK",
    "XoxF_DYD":   "TGCNLMDYDGSGSTGAQLNL",
    "CaM_EF_neg": "DQDGKLTKEELK",
    "C2_CBR":     "KSSIDMANMFAKDTNGDGTIT",
    "Annexin_A5": "MAVLYGLGTDESGKTTIVKRHLGXGTHPEMIVDPTYPKFSNLVKQ",
}


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE PIPELINE (no S3, no HPC — for testing)
# ─────────────────────────────────────────────────────────────────────────────

def run_offline_test() -> dict:
    """
    Full offline pipeline test:
      1. Build custom HMMs from CUSTOM_HMM_SEEDS
      2. Run ORF prediction on OFFLINE_NUC_FIXTURES
      3. Run motif validation on OFFLINE_PROTEIN_FIXTURES
      4. Score and rank hits
      5. Generate SLURM scripts (dry run)
      6. Mock aggregate + export
    """
    log.info("=" * 60)
    log.info("Offline Test: Logan Metagenomic Pipeline")
    log.info("=" * 60)

    results = {}

    # ── 1. Build custom HMMs ───────────────────────────────────────────────
    log.info("\nStep 1: Build custom HMM profiles")
    custom_hmm_paths = {}
    if PYHMMER_AVAILABLE:
        for hmm_name, seeds in CUSTOM_HMM_SEEDS.items():
            out_path = HMM_DIR / f"{hmm_name}.hmm"
            success  = build_custom_hmm(hmm_name, seeds, out_path)
            if success:
                custom_hmm_paths[hmm_name] = out_path
    results["custom_hmm_paths"] = custom_hmm_paths
    log.info(f"  Custom HMMs built: {list(custom_hmm_paths.keys())}")

    # ── 2. ORF prediction on offline nucleotide fixtures ──────────────────
    log.info("\nStep 2: ORF prediction from nucleotide fixtures")
    orf_results = {}
    if PYRODIGAL_AVAILABLE:
        gf = make_gene_finder()
        for seq_name, nuc_seq in OFFLINE_NUC_FIXTURES.items():
            orfs = predict_orfs(seq_name, nuc_seq, gf, min_length=5)
            orf_results[seq_name] = orfs
            log.info(f"  {seq_name}: {len(orfs)} ORFs predicted")
    results["orf_results"] = orf_results

    # ── 3. Motif validation on protein fixtures ────────────────────────────
    log.info("\nStep 3: Motif validation on protein fixtures")
    validation_results = {}
    for seq_name, prot_seq in OFFLINE_PROTEIN_FIXTURES.items():
        val = validate_hit_with_motifs(prot_seq)
        validation_results[seq_name] = val
        log.info(
            f"  {seq_name:20s}: motifs={val['motifs_found']}  "
            f"REE-selective={val['is_ree_selective']}"
        )
    results["validation_results"] = validation_results

    # ── 4. HMM search on protein fixtures (requires built HMMs) ───────────
    log.info("\nStep 4: HMM search (custom profiles vs protein fixtures)")
    search_results = []
    if PYHMMER_AVAILABLE and custom_hmm_paths:
        hmm_profiles = load_hmm_profiles(HMM_DIR)
        prot_seqs = [(name, seq) for name, seq in OFFLINE_PROTEIN_FIXTURES.items()]
        search_results = hmm_search_proteins(prot_seqs, hmm_profiles,
                                              bit_score_cutoff=5.0,
                                              e_value_cutoff=10.0,
                                              min_prot_length=10)
        log.info(f"  HMM search hits: {len(search_results)}")
        for h in search_results[:5]:
            log.info(
                f"  {h['orf_id']:20s} vs {h['hmm_name']:30s} "
                f"score={h['hmm_score']:.1f}  E={h['e_value']:.2e}"
            )
    results["search_results"] = search_results

    # ── 5. Score all validated hits ────────────────────────────────────────
    log.info("\nStep 5: Composite scoring")
    scored = []
    for seq_name, prot_seq in OFFLINE_PROTEIN_FIXTURES.items():
        val = validation_results[seq_name]
        # Find matching HMM hit if any
        hmm_hit = next(
            (h for h in search_results if h["orf_id"] == seq_name), {}
        )
        priority = score_hit(
            prot_seq,
            hmm_hit.get("architecture_class", "unknown"),
            val["motifs_found"],
            val["is_ree_selective"],
            hmm_hit.get("hmm_score", 0.0),
            hmm_hit.get("e_value", 1.0),
        )
        scored.append((seq_name, priority, val["motifs_found"]))
        log.info(f"  {seq_name:20s}: priority={priority:.2f}")

    scored.sort(key=lambda x: x[1], reverse=True)
    log.info(f"  Top hit: {scored[0][0]} (score={scored[0][1]:.2f})")
    results["scored_hits"] = scored

    # ── 6. SLURM script generation (dry run) ──────────────────────────────
    log.info("\nStep 6: SLURM script generation")
    slurm_paths = write_slurm_scripts(
        n_chunks    = 100,    # small for test
        chunk_size  = 50,
        cpus        = 8,
        mem_gb      = 32,
        time_hours  = 4,
        partition   = "batch",
    )
    results["slurm_paths"] = slurm_paths
    log.info(f"  Generated {len(slurm_paths)} SLURM scripts")

    # ── 7. Mock export ─────────────────────────────────────────────────────
    log.info("\nStep 7: Mock ESM-Bind export")
    mock_hits = []
    for seq_name, priority, motifs in scored:
        is_ree = validation_results[seq_name]["is_ree_selective"]
        mock_hits.append({
            "hit_id":             f"OFFLINE|{seq_name}",
            "sra_accession":      "OFFLINE_TEST",
            "contig_id":          seq_name,
            "orf_start":          0,
            "orf_end":            len(OFFLINE_PROTEIN_FIXTURES[seq_name]) * 3,
            "strand":             "+",
            "protein_seq":        OFFLINE_PROTEIN_FIXTURES[seq_name],
            "hmm_name":           "custom",
            "hmm_score":          10.0,
            "e_value":            1e-5,
            "architecture_class": "ef_hand_ree" if is_ree else "unknown",
            "motifs_found":       "|".join(motifs),
            "is_ree_selective":   is_ree,
            "engineering_score":  priority,
            "environment":        "offline_test",
            "environment_priority": 0,
            "chunk_id":           -1,
        })

    mock_df = pd.DataFrame(mock_hits)
    novel_df = export_novel_families(mock_df)
    entries  = build_logan_dataset_entries(novel_df)

    out_json = DATA_DIR / "logan_dataset_entries.json"
    with open(out_json, "w") as f:
        json.dump(entries, f, indent=2)
    log.info(f"  ESM-Bind entries: {len(entries)} → {out_json.name}")

    results["mock_df"]    = mock_df
    results["novel_df"]   = novel_df
    results["entries"]    = entries

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("\n" + "─" * 60)
    log.info("Offline Test Summary")
    log.info("─" * 60)
    log.info(f"  Custom HMMs built:    {len(custom_hmm_paths)}")
    log.info(f"  ORF fixtures tested:  {len(orf_results)}")
    log.info(f"  Motif validations:    {len(validation_results)}")
    log.info(f"  HMM search hits:      {len(search_results)}")
    log.info(f"  Scored hits:          {len(scored)}")
    log.info(f"  SLURM scripts:        {len(slurm_paths)}")
    log.info(f"  ESM-Bind entries:     {len(entries)}")
    log.info(f"  PYHMMER:  {PYHMMER_AVAILABLE}")
    log.info(f"  PYRODIGAL:{PYRODIGAL_AVAILABLE}")
    log.info(f"  BOTO3:    {BOTO3_AVAILABLE}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def _check_network(host: str = "search.rcsb.org") -> bool:
    import socket
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, 443))
        return True
    except Exception:
        return False


def _hmm_name_to_arch(hmm_name: str) -> str:
    """Map an HMM name or Pfam accession to our architecture_class labels."""
    mappings = {
        "EF_hand_REE_proswitch": "ef_hand_ree",
        "DYD_active_site":       "pqq_xoxf",
        "PF00036": "ef_hand_ree", "PF13202": "ef_hand_ree", "PF13833": "ef_hand_ree",
        "PF00168": "c2_domain",
        "PF00191": "annexin",
        "PF07645": "egf_ca2",
        "PF00594": "gla_domain",
        "PF00028": "cadherin",
        "PF01011": "pqq_xoxf",   "PF13360": "pqq_xoxf",
    }
    # Try direct match, then prefix
    if hmm_name in mappings:
        return mappings[hmm_name]
    for prefix, arch in mappings.items():
        if hmm_name.startswith(prefix):
            return arch
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Logan/SRA Metagenomic REE-Binding Protein Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["build-hmms", "generate-slurm", "scan", "scan-environment",
                 "aggregate", "offline-test"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--chunk-id",   type=int,  default=0,    help="SLURM array task ID")
    parser.add_argument("--chunk-size", type=int,  default=500,  help="SRA accessions per chunk")
    parser.add_argument("--n-chunks",   type=int,  default=1000, help="Total SLURM array size")
    parser.add_argument("--manifest",   type=Path, default=None, help="Logan manifest file")
    parser.add_argument("--cache-dir",  type=Path, default=None, help="Local S3 cache directory")
    parser.add_argument("--environment",type=str,  default="",   help="Environment label for scan-environment mode")
    parser.add_argument("--bioprojects",type=str,  default="",   help="Comma-separated BioProject IDs")
    parser.add_argument("--force",      action="store_true",     help="Rebuild existing HMM profiles")
    args = parser.parse_args()

    if args.mode == "offline-test":
        results = run_offline_test()
        print(f"\nOffline test complete: {len(results['entries'])} ESM-Bind entries generated")

    elif args.mode == "build-hmms":
        built = build_all_hmm_profiles(force=args.force)
        print(f"\nBuilt {len(built)} HMM profiles in {HMM_DIR}")

    elif args.mode == "generate-slurm":
        manifest = args.manifest or DATA_DIR / "logan_manifest.txt"
        # Download accession list if not cached.
        # Logan no longer ships a plain manifest.txt; the stats parquet is used instead.
        if not manifest.exists():
            log.info("Building accession list from Logan stats parquet...")
            accessions = get_logan_manifest(manifest)
            if not accessions:
                log.error("Could not retrieve accession list — is AWS/boto3 configured?")
                sys.exit(1)
        else:
            accessions = manifest.read_text().strip().splitlines()

        n_chunks = args.n_chunks or (len(accessions) // args.chunk_size + 1)
        scripts  = write_slurm_scripts(
            n_chunks   = n_chunks,
            chunk_size = args.chunk_size,
            manifest_path = manifest,
            cache_dir  = args.cache_dir,
        )
        print(f"\nGenerated {len(scripts)} SLURM scripts in {SLURM_DIR}")
        print(f"Submit with:  sbatch {scripts['scan']}")

    elif args.mode == "scan":
        # Called by SLURM array job
        manifest = args.manifest or DATA_DIR / "logan_manifest.txt"
        if not manifest.exists():
            log.error(f"Manifest not found: {manifest}")
            sys.exit(1)

        all_accs = manifest.read_text().strip().splitlines()
        start    = args.chunk_id * args.chunk_size
        end      = start + args.chunk_size
        chunk_accs = all_accs[start:end]

        if not chunk_accs:
            log.warning(f"Chunk {args.chunk_id}: no accessions in range [{start}:{end}]")
            sys.exit(0)

        hmm_profiles = load_hmm_profiles(HMM_DIR)
        if not hmm_profiles:
            log.error("No HMM profiles found — run --mode=build-hmms first")
            sys.exit(1)

        df = scan_chunk(
            chunk_id       = args.chunk_id,
            sra_accessions = chunk_accs,
            hmm_profiles   = hmm_profiles,
            cache_dir      = args.cache_dir,
        )

        out_path = HITS_DIR / f"chunk_{args.chunk_id:06d}.parquet"
        if not df.empty:
            df.to_parquet(out_path, index=False)
            log.info(f"Saved {len(df)} hits → {out_path.name}")
        else:
            log.info(f"No hits in chunk {args.chunk_id}")

    elif args.mode == "scan-environment":
        # Targeted scan for a specific environment's BioProjects.
        # Uses NCBI Entrez to convert BioProject IDs → SRR accessions,
        # then streams contigs from Logan S3 and scans with HMM profiles.
        env_name    = args.environment
        bioprojects = [b.strip() for b in args.bioprojects.split(",") if b.strip()] \
                      if args.bioprojects else []
        env_conf    = REE_ENVIRONMENTS.get(env_name, {})
        priority    = env_conf.get("priority", 9)

        if not bioprojects:
            bioprojects = env_conf.get("bioprojects", [])

        log.info(f"Targeted scan: {env_name} ({len(bioprojects)} BioProjects)")
        log.info(f"  Rationale: {env_conf.get('rationale', 'N/A')}")
        log.info(f"  BioProjects: {bioprojects}")

        # Step 1: BioProject → SRR accessions via NCBI Entrez
        log.info("  Fetching SRR accessions from NCBI Entrez...")
        sra_accessions = get_bioproject_accessions(
            bioprojects,
            max_per_project=args.chunk_size,
        )

        if not sra_accessions:
            log.warning(f"  No accessions found for {env_name} — "
                        "check BioProject IDs or network connectivity")
            sys.exit(0)

        log.info(f"  Found {len(sra_accessions)} SRA runs to scan")

        hmm_profiles = load_hmm_profiles(HMM_DIR)
        if not hmm_profiles:
            log.error("No HMM profiles found — run --mode=build-hmms first")
            sys.exit(1)

        # Step 2: Scan in chunks
        chunk_size = args.chunk_size
        all_dfs    = []
        for chunk_idx, start in enumerate(range(0, len(sra_accessions), chunk_size)):
            chunk_accs = sra_accessions[start : start + chunk_size]
            df = scan_chunk(
                chunk_id       = chunk_idx,
                sra_accessions = chunk_accs,
                hmm_profiles   = hmm_profiles,
                environment    = env_name,
                env_priority   = priority,
                cache_dir      = args.cache_dir,
            )
            if not df.empty:
                all_dfs.append(df)

        # Step 3: Save results
        out_path = HITS_DIR / f"env_{env_name}.parquet"
        if all_dfs:
            merged = pd.concat(all_dfs, ignore_index=True)
            merged.to_parquet(out_path, index=False)
            log.info(f"  Saved {len(merged)} hits → {out_path}")
        else:
            log.info(f"  No hits found for environment: {env_name}")

    elif args.mode == "aggregate":
        merged = aggregate_hits(HITS_DIR)
        if merged.empty:
            log.warning("No hits to aggregate")
            sys.exit(0)

        # Save merged
        merged_path = DATA_DIR / "logan_hits_merged.parquet"
        merged.to_parquet(merged_path, index=False)
        log.info(f"Merged hits saved: {merged_path}")

        # Novel families
        novel_df = export_novel_families(merged)
        novel_path = DATA_DIR / "logan_hits_novel.csv"
        novel_df.to_csv(novel_path, index=False)
        log.info(f"Novel families saved: {novel_path}  ({len(novel_df)} rows)")

        # ESM-Bind entries
        entries   = build_logan_dataset_entries(novel_df)
        json_path = DATA_DIR / "logan_dataset_entries.json"
        with open(json_path, "w") as f:
            json.dump(entries, f, indent=2)
        log.info(f"ESM-Bind entries: {json_path}  ({len(entries)} entries)")

        print(f"\nAggregation complete:")
        print(f"  Total hits:       {len(merged):,}")
        print(f"  Novel candidates: {len(novel_df):,}")
        print(f"  ESM-Bind entries: {len(entries):,}")


if __name__ == "__main__":
    main()
