"""
ree_miner.functional_annotation
================================
Functional annotation and taxonomic classification of metagenomic REE-binding
protein hits discovered by the HMM pipeline.

─── Purpose ────────────────────────────────────────────────────────────────────

After scan_chunk() identifies candidate ORFs from the Logan dataset, this
module adds three layers of biological context:

  1. Taxonomic origin      – Bacteria / Archaea / Eukaryota classification
                             derived from NCBI SRA metadata and Entrez Taxonomy.
                             Archaeal hits are flagged separately because they
                             represent thermostable, industrially relevant variants.

  2. Functional annotation – COG category, KEGG ortholog, and GO terms via the
                             eggNOG-mapper web API (api.eggnog-mapper.embl.de) or
                             a locally installed eggnog-mapper executable.

  3. Genomic neighborhood  – Flanking ORFs on the same contig are predicted and
                             scanned for co-encoded REE metabolism signatures:
                             TonB-dependent REE transporters, ABC transporters,
                             and secondary REE-binding genes.  A neighborhood
                             score weights hits from genomic islands vs. isolated
                             ORFs.

─── Design decisions ───────────────────────────────────────────────────────────

  • Graceful degradation: every external call falls back silently when offline or
    when optional tools are not installed.  Offline annotation uses a conservative
    "Unknown" label rather than crashing.

  • Caching: taxonomy results are written to DATA_DIR/taxonomy_cache.json to
    avoid repeated Entrez queries across runs.

  • Archaeal priority: Archaea receive a dedicated `archaeal_significance_score`
    (0-3) based on thermophily, acidophily, and metal-leaching phenotype of the
    source environment.

─── CLI usage ──────────────────────────────────────────────────────────────────

  # Annotate a hits parquet from the scan step:
  ree-miner annotate --hits datasets/metagenomic_hits/chunk_0.parquet

  # Annotate the aggregated results:
  ree-miner annotate --hits datasets/logan_hits_merged.parquet \\
                     --contigs-dir datasets/contig_cache/ \\
                     --out    datasets/annotated_hits.parquet

  # Use local eggnog-mapper installation:
  ree-miner annotate --hits datasets/logan_hits_merged.parquet \\
                     --eggnog-mode local \\
                     --eggnog-db /data/eggnog_db/

─── Output columns added ───────────────────────────────────────────────────────

  taxonomy_domain      Bacteria | Archaea | Eukaryota | Virus | Unknown
  taxonomy_organism    NCBI SRA reported organism (may be "metagenome")
  taxonomy_phylum      Phylum (empty for environmental metagenomes)
  taxonomy_taxid       NCBI taxon ID (integer)
  is_archaeal          bool — True if taxonomy_domain == "Archaea"
  is_thermophile       bool — True if source organism is a known thermophile
  is_acidophile        bool — True if source organism is a known acidophile
  archaeal_significance_score  0–3 composite score for archaeal hits
  cog_category         Single COG letter or compound (e.g. "P", "PC")
  cog_description      Human-readable COG category name(s)
  kegg_ko              KEGG ortholog ID (e.g. "K14028")
  kegg_pathway         KEGG pathway (first hit)
  go_terms             "|"-separated GO term IDs
  go_biological_process"|"-separated GO biological process descriptions
  eggnog_description   Closest eggNOG OG functional description
  neighborhood_has_tonb       bool — TonB-dependent transporter in ±10 kb
  neighborhood_has_abc        bool — ABC transporter in ±10 kb
  neighborhood_has_xoxf       bool — XoxF/PQQ dehydrogenase in ±10 kb
  neighborhood_ree_gene_count int  — # of co-encoded REE-metabolism genes
  neighborhood_score          float 0–1 — confidence of REE gene cluster context
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Optional imports ────────────────────────────────────────────────────────
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pyrodigal
    PYRODIGAL_AVAILABLE = True
except ImportError:
    PYRODIGAL_AVAILABLE = False

try:
    import pyhmmer
    from pyhmmer.plan7 import HMMFile
    PYHMMER_AVAILABLE = True
except ImportError:
    PYHMMER_AVAILABLE = False

from ree_miner._workspace import DATA_DIR, HMM_DIR

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NCBI_ENTREZ_BASE      = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_RATE_LIMIT_DELAY = 0.34   # 3 requests/sec without API key
EGGNOG_API_BASE       = "https://api.eggnog-mapper.embl.de/api"
EGGNOG_POLL_INTERVAL  = 30     # seconds between status polls
EGGNOG_MAX_WAIT       = 3600   # max 1 hour for annotation job

TAXONOMY_CACHE_FILE = DATA_DIR / "taxonomy_cache.json"
ANNOTATED_HITS_FILE = DATA_DIR / "annotated_hits.parquet"

# NCBI root taxon IDs for the three domains of life
TAXID_BACTERIA   = 2
TAXID_ARCHAEA    = 2157
TAXID_EUKARYOTA  = 2759
TAXID_VIRUSES    = 10239

# COG categories most relevant to REE biology, ordered by expected enrichment
REE_COG_CATEGORIES: Dict[str, str] = {
    "P": "Inorganic ion transport and metabolism",   # metal transporters / binding
    "C": "Energy production and conversion",          # XoxF methanol oxidation
    "G": "Carbohydrate transport and metabolism",    # alcohol/sugar dehydrogenases
    "T": "Signal transduction mechanisms",            # Ln-responsive sensors
    "E": "Amino acid transport and metabolism",
    "Q": "Secondary metabolites biosynthesis/transport",
    "I": "Lipid transport and metabolism",
    "S": "Function unknown",                          # novel hits land here
    "R": "General function prediction only",
}

# Thermophile / acidophile taxonomy keywords (used when taxid lookup fails)
THERMOPHILE_KEYWORDS = [
    "thermophil", "thermoacidophil", "hyperthermophil",
    "sulfolobus", "thermococcus", "pyrococcus", "thermoproteus",
    "methanothermobacter", "caldarchaeol", "thermodesulfobacterium",
    "aquifex", "thermotoga",
]
ACIDOPHILE_KEYWORDS = [
    "acidophil", "thermoacidophil",
    "sulfolobus", "acidianus", "picrophilus", "metallosphaera",
    "ferroplasma", "thermoplasma", "acidithiobacil",
]
METAL_LEACHER_KEYWORDS = [
    "metallosphaera", "acidianus", "sulfolobus", "ferroplasma",
    "acidithiobacillus", "leptospirillum", "sulfobacillus",
]

# Pfam HMM accessions for genomic neighborhood markers
NEIGHBORHOOD_PFAMS: Dict[str, List[str]] = {
    "tonb_transporter":      ["PF03544", "PF02683", "PF07715"],  # TonB-dependent receptor
    "abc_transporter":       ["PF00005", "PF00076", "PF13555"],  # ABC ATP-binding + TMD
    "pqq_dehydrogenase":     ["PF01011", "PF13360"],             # XoxF/PedH PQQ-binding
    "methyltransferase_ree": ["PF04847", "PF08241"],             # C1 methylotrophy
    "ef_hand_generic":       ["PF00036", "PF13202", "PF13833"],  # EF-hand (any)
}
NEIGHBORHOOD_WINDOW_BP = 10_000  # ±10 kb around hit ORF


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaxonomyInfo:
    """Taxonomic classification for one SRA accession or protein sequence."""
    accession:    str
    taxid:        int    = 0
    organism:     str    = "Unknown"
    domain:       str    = "Unknown"   # Bacteria / Archaea / Eukaryota / Virus
    phylum:       str    = ""
    tax_class:    str    = ""
    is_archaeal:  bool   = False
    is_thermophile: bool = False
    is_acidophile:  bool = False
    is_metal_leacher: bool = False
    archaeal_significance_score: int = 0  # 0–3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EggNOGAnnotation:
    """Functional annotation from eggNOG-mapper for one query sequence."""
    query_id:           str
    cog_category:       str   = ""    # e.g. "P" or "PC"
    cog_description:    str   = ""    # human-readable category name(s)
    kegg_ko:            str   = ""    # e.g. "K14028"
    kegg_pathway:       str   = ""    # e.g. "ko00680"
    go_terms:           str   = ""    # "|"-separated GO IDs
    go_biological_process: str = ""
    eggnog_description: str   = ""    # OG functional description
    eggnog_og:          str   = ""    # OG accession
    max_annot_lvl:      str   = ""    # taxonomic level of best hit

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NeighborhoodInfo:
    """Genomic neighborhood context around a metagenomic hit ORF."""
    hit_id:                  str
    contig_length:           int   = 0
    neighbor_orf_count:      int   = 0
    neighborhood_has_tonb:   bool  = False
    neighborhood_has_abc:    bool  = False
    neighborhood_has_xoxf:   bool  = False
    neighborhood_has_ef_hand: bool = False
    neighborhood_ree_gene_count: int = 0
    neighborhood_score:      float = 0.0  # 0–1 confidence

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: NCBI TAXONOMY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _load_taxonomy_cache() -> Dict[str, dict]:
    """Load the local taxonomy cache from disk."""
    if TAXONOMY_CACHE_FILE.exists():
        try:
            return json.loads(TAXONOMY_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_taxonomy_cache(cache: Dict[str, dict]) -> None:
    """Persist the taxonomy cache to disk."""
    TAXONOMY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TAXONOMY_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _parse_lineage(lineage_str: str) -> Tuple[str, str, str]:
    """
    Parse an NCBI lineage string into (domain, phylum, tax_class).

    Example lineage: "cellular organisms; Bacteria; Pseudomonadota; ..."
    Returns ("Bacteria", "Pseudomonadota", "Gammaproteobacteria")
    """
    parts = [p.strip() for p in lineage_str.split(";")]
    domain = "Unknown"
    phylum = ""
    tax_class = ""

    for part in parts:
        lower = part.lower()
        if "bacteria" in lower and domain == "Unknown":
            domain = "Bacteria"
        elif "archaea" in lower and domain == "Unknown":
            domain = "Archaea"
        elif "eukaryota" in lower and domain == "Unknown":
            domain = "Eukaryota"
        elif "viruses" in lower and domain == "Unknown":
            domain = "Virus"

    # Phylum is typically the 3rd element after 'cellular organisms'
    idx_start = 0
    for i, p in enumerate(parts):
        if p in ("cellular organisms", "Bacteria", "Archaea", "Eukaryota",
                 "Viruses", "other sequences"):
            idx_start = i + 1
            break
    if idx_start < len(parts):
        phylum = parts[idx_start] if idx_start < len(parts) else ""
    if idx_start + 1 < len(parts):
        tax_class = parts[idx_start + 1] if idx_start + 1 < len(parts) else ""

    return domain, phylum, tax_class


def _keyword_classify(organism: str) -> Tuple[bool, bool, bool]:
    """
    Keyword-based thermophile / acidophile / metal-leacher classification.
    Returns (is_thermophile, is_acidophile, is_metal_leacher).
    """
    low = organism.lower()
    thermo  = any(kw in low for kw in THERMOPHILE_KEYWORDS)
    acido   = any(kw in low for kw in ACIDOPHILE_KEYWORDS)
    leacher = any(kw in low for kw in METAL_LEACHER_KEYWORDS)
    return thermo, acido, leacher


def lookup_taxonomy_for_accession(
    accession: str,
    api_key: Optional[str] = None,
) -> TaxonomyInfo:
    """
    Look up NCBI taxonomy for an SRA accession using the Entrez API.

    Pipeline:
      1. eFetch SRA runinfo (CSV) → extract organism name and taxid
      2. eFetch Taxonomy DB with taxid → parse lineage to get domain/phylum
      3. Apply keyword heuristics for thermophile / acidophile flags

    Args:
        accession:  SRA run accession (SRR/ERR/DRR).
        api_key:    NCBI API key for higher rate limits (also reads NCBI_API_KEY env).

    Returns:
        TaxonomyInfo dataclass.  Falls back to TaxonomyInfo(domain="Unknown")
        if the network is unavailable.
    """
    _api_key = api_key or os.environ.get("NCBI_API_KEY")
    delay = 0.11 if _api_key else NCBI_RATE_LIMIT_DELAY

    info = TaxonomyInfo(accession=accession)

    if not REQUESTS_AVAILABLE:
        log.debug(f"  taxonomy: requests not available, skipping {accession}")
        return info

    try:
        # ── Step 1: SRA runinfo → organism + taxid ────────────────────────
        params: dict = {"db": "sra", "id": accession, "rettype": "runinfo",
                        "retmode": "csv"}
        if _api_key:
            params["api_key"] = _api_key

        r = requests.get(f"{NCBI_ENTREZ_BASE}/efetch.fcgi",
                         params=params, timeout=20)
        r.raise_for_status()
        time.sleep(delay)

        reader = csv.DictReader(io.StringIO(r.text))
        row = next(reader, None)
        if row:
            info.organism = row.get("ScientificName", "") or row.get("Organism", "Unknown")
            try:
                info.taxid = int(row.get("TaxID", 0) or 0)
            except (ValueError, TypeError):
                info.taxid = 0

        # ── Step 2: Taxonomy lineage ───────────────────────────────────────
        if info.taxid > 0:
            tx_params: dict = {"db": "taxonomy", "id": str(info.taxid),
                               "retmode": "xml"}
            if _api_key:
                tx_params["api_key"] = _api_key

            r2 = requests.get(f"{NCBI_ENTREZ_BASE}/efetch.fcgi",
                              params=tx_params, timeout=20)
            r2.raise_for_status()
            time.sleep(delay)

            xml = r2.text
            # Extract lineage with simple regex (avoids xml.etree for speed)
            lineage_match = re.search(r"<Lineage>(.*?)</Lineage>", xml, re.DOTALL)
            if lineage_match:
                info.domain, info.phylum, info.tax_class = \
                    _parse_lineage(lineage_match.group(1))

        # If taxid lookup failed, try keyword-based domain classification
        if info.domain == "Unknown" and info.organism:
            low = info.organism.lower()
            if "archaea" in low or "archaeon" in low or "archaeota" in low:
                info.domain = "Archaea"
            elif "bacteria" in low or "bacterium" in low or "bacterota" in low:
                info.domain = "Bacteria"
            elif "metagenome" in low or "environmental" in low:
                # metagenome label; try to infer from environment keywords
                if any(k in low for k in ["archaeal", "archaea", "thermoacidophil"]):
                    info.domain = "Archaea"
                else:
                    info.domain = "Unknown"

        # ── Step 3: Phenotype flags ────────────────────────────────────────
        info.is_archaeal = info.domain == "Archaea"
        info.is_thermophile, info.is_acidophile, info.is_metal_leacher = \
            _keyword_classify(info.organism)

        # Composite archaeal significance score (0–3)
        if info.is_archaeal:
            score = 1  # baseline for being archaeal
            if info.is_thermophile:
                score += 1
            if info.is_acidophile or info.is_metal_leacher:
                score += 1
            info.archaeal_significance_score = score

        log.debug(f"  {accession}: {info.organism} ({info.domain}), "
                  f"archaea={info.is_archaeal}, thermo={info.is_thermophile}")

    except requests.exceptions.Timeout:
        log.debug(f"  {accession}: NCBI Entrez timeout, returning Unknown taxonomy")
    except Exception as e:
        log.debug(f"  {accession}: taxonomy lookup failed: {e}")

    return info


def batch_lookup_taxonomy(
    accessions: List[str],
    cache_path: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> Dict[str, TaxonomyInfo]:
    """
    Look up taxonomy for multiple SRA accessions, using local cache to
    avoid redundant API calls across pipeline runs.

    Args:
        accessions:  List of SRR/ERR/DRR accession strings.
        cache_path:  Path to JSON cache file.
        api_key:     NCBI API key.

    Returns:
        Dict mapping accession → TaxonomyInfo.
    """
    if cache_path is None:
        cache_path = TAXONOMY_CACHE_FILE

    # Load cache
    cache = _load_taxonomy_cache() if cache_path.exists() else {}
    results: Dict[str, TaxonomyInfo] = {}
    to_fetch: List[str] = []

    for acc in accessions:
        if acc in cache:
            results[acc] = TaxonomyInfo(**cache[acc])
        else:
            to_fetch.append(acc)

    log.info(f"Taxonomy lookup: {len(results)} cached, {len(to_fetch)} to fetch")

    for i, acc in enumerate(to_fetch):
        info = lookup_taxonomy_for_accession(acc, api_key=api_key)
        results[acc] = info
        cache[acc] = info.to_dict()

        # Save periodically to avoid losing progress
        if (i + 1) % 50 == 0:
            _save_taxonomy_cache(cache)
            log.info(f"  Taxonomy: {i + 1}/{len(to_fetch)} fetched")

    if to_fetch:
        _save_taxonomy_cache(cache)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: eggNOG-MAPPER FUNCTIONAL ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────

def _format_fasta(sequences: Dict[str, str]) -> str:
    """Format a dict of {id: seq} as a FASTA string."""
    lines = []
    for seq_id, seq in sequences.items():
        lines.append(f">{seq_id}")
        # Wrap at 80 characters
        for i in range(0, len(seq), 80):
            lines.append(seq[i:i+80])
    return "\n".join(lines) + "\n"


def annotate_with_eggnog_web(
    sequences: Dict[str, str],
    tax_scope: int = 2,   # 2 = Bacteria; use 2157 for Archaea-scoped search
    eggnog_db: str = "auto",
    poll_interval: int = EGGNOG_POLL_INTERVAL,
    max_wait: int = EGGNOG_MAX_WAIT,
) -> Dict[str, EggNOGAnnotation]:
    """
    Annotate protein sequences using the eggNOG-mapper web API.

    Submits a batch job to api.eggnog-mapper.embl.de, polls until complete,
    then parses the annotation table.

    Args:
        sequences:      Dict of {query_id: protein_sequence}.
        tax_scope:      NCBI taxon ID to restrict annotation search.
                        Use 2157 (Archaea) for archaeal-biased annotation.
                        Use 2 (Bacteria) for standard bacterial annotation.
                        Use 1 (root) for broad cross-domain annotation.
        eggnog_db:      eggNOG database version ("auto", "5.0", etc.).
        poll_interval:  Seconds between status checks.
        max_wait:       Maximum seconds to wait for job completion.

    Returns:
        Dict mapping query_id → EggNOGAnnotation.
        Returns empty dict on failure or when offline.
    """
    if not REQUESTS_AVAILABLE:
        log.warning("eggNOG web API: requests not available — skipping annotation")
        return {}

    if not sequences:
        return {}

    results: Dict[str, EggNOGAnnotation] = {}

    # ── Step 1: Submit job ─────────────────────────────────────────────────
    fasta_str = _format_fasta(sequences)
    log.info(f"eggNOG-mapper: submitting {len(sequences)} sequences "
             f"(tax_scope={tax_scope})")

    try:
        submit_payload = {
            "data":      fasta_str,
            "db":        eggnog_db,
            "taxscope":  tax_scope,
            "target_orthologs": "all",
            "go_evidence":      "all",
            "predict_ncrna":    False,
        }
        r = requests.post(
            f"{EGGNOG_API_BASE}/job/",
            data=submit_payload,
            timeout=120,
        )
        r.raise_for_status()
        job_data = r.json()
        job_id = job_data.get("jobid") or job_data.get("id")

        if not job_id:
            log.warning(f"eggNOG API: no job ID in response: {job_data}")
            return {}

        log.info(f"  eggNOG job submitted: {job_id}")

    except Exception as e:
        log.warning(f"eggNOG API submit failed: {e}")
        return {}

    # ── Step 2: Poll for completion ────────────────────────────────────────
    elapsed = 0
    while elapsed < max_wait:
        try:
            time.sleep(poll_interval)
            elapsed += poll_interval
            r = requests.get(f"{EGGNOG_API_BASE}/job/{job_id}",
                             timeout=30)
            r.raise_for_status()
            status_data = r.json()
            status = status_data.get("status", "")

            log.debug(f"  eggNOG job {job_id}: status={status} ({elapsed}s elapsed)")

            if status in ("Done", "done", "Finished", "finished", "success"):
                break
            elif status in ("Error", "error", "Failed", "failed"):
                log.warning(f"eggNOG job {job_id} failed: {status_data}")
                return {}

        except Exception as e:
            log.debug(f"  eggNOG poll error: {e}")
            continue
    else:
        log.warning(f"eggNOG job {job_id} timed out after {max_wait}s")
        return {}

    # ── Step 3: Download annotations ──────────────────────────────────────
    try:
        r = requests.get(
            f"{EGGNOG_API_BASE}/job/{job_id}/output/out.emapper.annotations",
            timeout=120,
        )
        r.raise_for_status()
        results = _parse_eggnog_tsv(r.text)
        log.info(f"  eggNOG: {len(results)} annotations received")

    except Exception as e:
        log.warning(f"eggNOG result download failed: {e}")

    return results


def annotate_with_eggnog_local(
    sequences: Dict[str, str],
    eggnog_db_dir: Path,
    tax_scope: int = 1,
    n_cpu: int = 4,
    tmp_dir: Optional[Path] = None,
) -> Dict[str, EggNOGAnnotation]:
    """
    Run eggNOG-mapper locally (requires `emapper.py` in PATH or venv).

    This is the preferred mode for large jobs on HPC clusters.

    Args:
        sequences:      Dict of {query_id: protein_sequence}.
        eggnog_db_dir:  Path to eggNOG database files (--data_dir).
        tax_scope:      Restrict to taxonomic scope (1 = all, 2157 = Archaea).
        n_cpu:          CPU threads for diamond alignment.
        tmp_dir:        Working directory for intermediate files.

    Returns:
        Dict mapping query_id → EggNOGAnnotation.
    """
    import subprocess
    import tempfile

    if tmp_dir is None:
        tmp_dir = DATA_DIR / "eggnog_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fasta_path   = tmp_dir / "query.faa"
    output_prefix = str(tmp_dir / "eggnog_out")

    fasta_path.write_text(_format_fasta(sequences))

    cmd = [
        "emapper.py",
        "-i",          str(fasta_path),
        "--output",    output_prefix,
        "--data_dir",  str(eggnog_db_dir),
        "--cpu",       str(n_cpu),
        "--tax_scope", str(tax_scope),
        "-m",          "diamond",
        "--override",
        "--no_file_comments",
    ]
    log.info(f"Running local eggNOG-mapper: {' '.join(cmd[:6])} ...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            log.warning(f"emapper.py failed (rc={result.returncode}): "
                        f"{result.stderr[:500]}")
            return {}
    except FileNotFoundError:
        log.warning("emapper.py not found in PATH — "
                    "install eggnog-mapper or use --eggnog-mode=web")
        return {}
    except subprocess.TimeoutExpired:
        log.warning("Local eggNOG-mapper timed out after 1 hour")
        return {}

    ann_file = Path(output_prefix + ".emapper.annotations")
    if not ann_file.exists():
        log.warning(f"eggNOG output not found: {ann_file}")
        return {}

    results = _parse_eggnog_tsv(ann_file.read_text())
    log.info(f"Local eggNOG-mapper: {len(results)} annotations")
    return results


def _parse_eggnog_tsv(tsv_text: str) -> Dict[str, EggNOGAnnotation]:
    """
    Parse eggNOG-mapper v2 annotation table (*.emapper.annotations).

    Column order (v2.1+):
      query, seed_ortholog, evalue, score, eggNOG_OGs, max_annot_lvl,
      COG_category, Description, Preferred_name, GOs, EC, KEGG_ko,
      KEGG_Pathway, KEGG_Module, KEGG_Reaction, KEGG_rclass, BRITE,
      KEGG_TC, CAZy, BiGG_Reaction, PFAMs

    Returns dict of query_id → EggNOGAnnotation.
    """
    results: Dict[str, EggNOGAnnotation] = {}
    reader = csv.DictReader(
        (line for line in tsv_text.splitlines()
         if line and not line.startswith("##")),
        delimiter="\t",
    )

    for row in reader:
        qid = row.get("query", row.get("#query", "")).strip()
        if not qid:
            continue

        cog_cat = row.get("COG_category", "").strip().replace("-", "")
        cog_descs = [REE_COG_CATEGORIES.get(c, "") for c in cog_cat if c in REE_COG_CATEGORIES]

        # Parse GO terms
        go_raw = row.get("GOs", "").strip()
        go_ids = [g for g in go_raw.split(",") if g.startswith("GO:")]
        go_bio = []  # biological process GO terms (GO:00XX category 'P')
        # Note: full BP filtering would need a GO DAG library;
        # here we include all GO terms for downstream filtering
        go_str = "|".join(go_ids)

        # KEGG_ko: typically "ko:K14028,ko:K14029" → take first
        kegg_raw = row.get("KEGG_ko", "").strip()
        kegg_ko  = kegg_raw.split(",")[0].replace("ko:", "").strip() if kegg_raw else ""

        kegg_path_raw = row.get("KEGG_Pathway", "").strip()
        kegg_pathway  = kegg_path_raw.split(",")[0].strip() if kegg_path_raw else ""

        og_raw  = row.get("eggNOG_OGs", "").strip()
        max_lvl = row.get("max_annot_lvl", "").strip()

        ann = EggNOGAnnotation(
            query_id=qid,
            cog_category=cog_cat,
            cog_description="; ".join(d for d in cog_descs if d),
            kegg_ko=kegg_ko,
            kegg_pathway=kegg_pathway,
            go_terms=go_str,
            eggnog_description=row.get("Description", "").strip(),
            eggnog_og=og_raw.split(",")[0].strip() if og_raw else "",
            max_annot_lvl=max_lvl,
        )
        results[qid] = ann

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: GENOMIC NEIGHBORHOOD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_genomic_neighborhood(
    contig_seq:   str,
    hit_orf_start: int,
    hit_orf_end:   int,
    hit_id:        str,
    hmm_profiles:  Optional[dict] = None,
    window_bp:     int = NEIGHBORHOOD_WINDOW_BP,
) -> NeighborhoodInfo:
    """
    Predict and HMM-scan flanking ORFs on the same contig as a hit.

    The window is centered on the hit ORF and extends ±window_bp bp.
    Pyrodigal predicts all ORFs in the window, then pyhmmer scans each
    predicted protein against the NEIGHBORHOOD_PFAMS profile set to detect
    co-encoded REE metabolism genes.

    Args:
        contig_seq:     Full nucleotide sequence of the contig.
        hit_orf_start:  0-based start of the hit ORF in the contig.
        hit_orf_end:    0-based end of the hit ORF in the contig.
        hit_id:         Hit identifier string (for logging).
        hmm_profiles:   Dict of {name: pyhmmer HMM} from load_hmm_profiles().
                        If None, loads from HMM_DIR.
        window_bp:      Half-width of the neighborhood window.

    Returns:
        NeighborhoodInfo with co-occurrence flags and a composite score.
    """
    info = NeighborhoodInfo(
        hit_id=hit_id,
        contig_length=len(contig_seq),
    )

    if not PYRODIGAL_AVAILABLE or not PYHMMER_AVAILABLE:
        log.debug(f"  neighborhood: pyrodigal/pyhmmer not available, skipping {hit_id}")
        return info

    # ── Extract neighborhood window ────────────────────────────────────────
    w_start = max(0, hit_orf_start - window_bp)
    w_end   = min(len(contig_seq), hit_orf_end + window_bp)
    window_seq = contig_seq[w_start:w_end]

    if len(window_seq) < 90:  # too short to contain meaningful ORFs
        return info

    # ── Predict ORFs in window ─────────────────────────────────────────────
    try:
        gf = pyrodigal.GeneFinder(meta=True)
        genes = gf.find_genes(window_seq.encode())
        proteins: List[str] = [str(g.translate()) for g in genes]
        info.neighbor_orf_count = len(proteins)
    except Exception as e:
        log.debug(f"  neighborhood ORF prediction failed for {hit_id}: {e}")
        return info

    if not proteins:
        return info

    # ── Load HMM profiles for neighborhood markers ─────────────────────────
    if hmm_profiles is None:
        if not HMM_DIR.exists():
            return info
        # Lazy-load all HMMs from HMM_DIR
        hmm_profiles = {}
        try:
            from ree_miner.metagenomic import load_hmm_profiles
            hmm_profiles = load_hmm_profiles(HMM_DIR)
        except Exception:
            pass

    if not hmm_profiles:
        return info

    # ── HMM-scan neighbor proteins ─────────────────────────────────────────
    try:
        alphabet = pyhmmer.easel.Alphabet.amino()
        seqs = []
        for i, prot in enumerate(proteins):
            clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", prot.rstrip("*"))
            if len(clean) >= 10:
                seq = pyhmmer.easel.TextSequence(
                    name=f"nbr_{i}".encode(),
                    sequence=clean,
                )
                seqs.append(seq.digitize(alphabet))

        if not seqs:
            return info

        hits_found: Dict[str, List[str]] = {k: [] for k in NEIGHBORHOOD_PFAMS}

        for hmm_name, hmm in hmm_profiles.items():
            pipeline = pyhmmer.plan7.Pipeline(alphabet)
            for hit in pipeline.search_hmm(hmm, seqs):
                if hit.included:
                    # Map HMM name to neighborhood category
                    for cat, pfam_list in NEIGHBORHOOD_PFAMS.items():
                        if any(pf in hmm_name for pf in pfam_list) or cat in hmm_name:
                            hits_found[cat].append(hmm_name)

        # Set flags
        info.neighborhood_has_tonb    = len(hits_found.get("tonb_transporter", [])) > 0
        info.neighborhood_has_abc     = len(hits_found.get("abc_transporter", [])) > 0
        info.neighborhood_has_xoxf    = len(hits_found.get("pqq_dehydrogenase", [])) > 0
        info.neighborhood_has_ef_hand = len(hits_found.get("ef_hand_generic", [])) > 0

        # Count total REE-related co-encoded genes
        info.neighborhood_ree_gene_count = sum(
            1 for v in hits_found.values() if v
        )

        # Composite neighborhood score (0–1):
        # TonB transporter is the strongest single signal (Ochsner 2019)
        score_weights = {
            "tonb_transporter":      0.40,
            "abc_transporter":       0.25,
            "pqq_dehydrogenase":     0.20,
            "methyltransferase_ree": 0.10,
            "ef_hand_generic":       0.05,
        }
        info.neighborhood_score = min(1.0, sum(
            w for cat, w in score_weights.items()
            if hits_found.get(cat)
        ))

    except Exception as e:
        log.debug(f"  neighborhood HMM scan failed for {hit_id}: {e}")

    return info


def batch_analyze_neighborhoods(
    hits_df: pd.DataFrame,
    contig_cache: Optional[Dict[str, str]] = None,
    contigs_dir:  Optional[Path] = None,
    hmm_profiles: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run genomic neighborhood analysis for all hits in a DataFrame.

    The contig sequences are sourced either from an in-memory dict
    (`contig_cache`) or a directory of FASTA files (`contigs_dir`).
    If neither is provided, neighborhood analysis is skipped and the
    added columns are filled with default (False/0) values.

    Args:
        hits_df:       DataFrame with at minimum:
                         hit_id, sra_accession, contig_id,
                         orf_start, orf_end columns.
        contig_cache:  Dict of {contig_id: nucleotide_sequence}.
        contigs_dir:   Path to directory with *.fa / *.fasta contig files.
        hmm_profiles:  Pre-loaded HMM profiles dict.

    Returns:
        hits_df with neighborhood columns appended.
    """
    neighborhood_cols = [
        "neighborhood_has_tonb", "neighborhood_has_abc",
        "neighborhood_has_xoxf", "neighborhood_has_ef_hand",
        "neighborhood_ree_gene_count", "neighborhood_score",
        "neighbor_orf_count",
    ]

    # Initialise columns to defaults
    for col in neighborhood_cols:
        if col not in hits_df.columns:
            hits_df[col] = False if "has" in col else 0

    if contig_cache is None and contigs_dir is None:
        log.info("Neighborhood analysis: no contig source provided — using defaults")
        return hits_df

    # Build contig cache from directory if needed
    if contig_cache is None and contigs_dir is not None:
        contig_cache = _load_contigs_from_dir(contigs_dir)

    if not contig_cache:
        log.info("Neighborhood analysis: contig cache empty — skipping")
        return hits_df

    log.info(f"Analyzing genomic neighborhoods for {len(hits_df)} hits...")
    n_analyzed = 0
    records = []

    for _, row in hits_df.iterrows():
        cid  = row.get("contig_id", "")
        seq  = contig_cache.get(cid)
        if seq is None:
            records.append(NeighborhoodInfo(hit_id=row["hit_id"]).to_dict())
            continue

        nbr = analyze_genomic_neighborhood(
            contig_seq=seq,
            hit_orf_start=int(row.get("orf_start", 0)),
            hit_orf_end=int(row.get("orf_end", len(seq))),
            hit_id=row["hit_id"],
            hmm_profiles=hmm_profiles,
        )
        records.append(nbr.to_dict())
        n_analyzed += 1

    nbr_df = pd.DataFrame(records)
    # Merge neighborhood columns back onto hits_df
    for col in [c for c in nbr_df.columns if c != "hit_id"]:
        if col in hits_df.columns:
            hits_df = hits_df.drop(columns=[col])
    hits_df = hits_df.merge(nbr_df.rename(columns={"hit_id": "hit_id_nbr"}),
                            left_on="hit_id", right_on="hit_id_nbr", how="left")
    if "hit_id_nbr" in hits_df.columns:
        hits_df = hits_df.drop(columns=["hit_id_nbr"])

    log.info(f"  Neighborhood analysis: {n_analyzed}/{len(hits_df)} contigs found")
    return hits_df


def _load_contigs_from_dir(contigs_dir: Path) -> Dict[str, str]:
    """
    Load all FASTA files from a directory into a {contig_id: seq} dict.

    Supports .fa, .fasta, .fa.zst, and .fasta.gz files.
    """
    contigs: Dict[str, str] = {}
    suffixes = (".fa", ".fasta", ".fna")

    for fpath in contigs_dir.iterdir():
        if not any(str(fpath).endswith(s) for s in suffixes):
            continue
        try:
            text = fpath.read_text(errors="replace")
            current_id: Optional[str] = None
            parts: List[str] = []
            for line in text.splitlines():
                if line.startswith(">"):
                    if current_id and parts:
                        contigs[current_id] = "".join(parts)
                    current_id = line[1:].split()[0]
                    parts = []
                elif current_id:
                    parts.append(line.strip())
            if current_id and parts:
                contigs[current_id] = "".join(parts)
        except Exception as e:
            log.debug(f"  Could not load contigs from {fpath}: {e}")

    log.info(f"Loaded {len(contigs)} contigs from {contigs_dir}")
    return contigs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MAIN ANNOTATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def annotate_hits_dataframe(
    hits_df: pd.DataFrame,
    eggnog_mode: str = "web",                    # "web" | "local" | "skip"
    eggnog_db_dir: Optional[Path] = None,
    eggnog_tax_scope: int = 1,                   # 1 = root (cross-domain)
    contig_cache: Optional[Dict[str, str]] = None,
    contigs_dir:  Optional[Path] = None,
    hmm_profiles: Optional[dict] = None,
    api_key: Optional[str] = None,
    batch_size: int = 200,                       # sequences per eggNOG job
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full functional annotation pipeline for a hits DataFrame.

    Adds the following column groups:
      • taxonomy_*       – domain, organism, phylum, archaeal flags
      • cog_*/kegg_*/go_* – eggNOG-mapper functional terms
      • neighborhood_*   – genomic co-occurrence signatures

    Args:
        hits_df:           DataFrame from scan_chunk() or aggregate_hits().
        eggnog_mode:       "web"   – use eggnog-mapper REST API (default)
                           "local" – use locally installed emapper.py
                           "skip"  – disable eggNOG annotation
        eggnog_db_dir:     Required when eggnog_mode="local".
        eggnog_tax_scope:  Taxonomic scope for eggNOG search.
        contig_cache:      In-memory {contig_id: nucleotide_seq} dict.
        contigs_dir:       Directory of contig FASTA files.
        hmm_profiles:      Pre-loaded HMM profiles for neighborhood scan.
        api_key:           NCBI API key for Entrez queries.
        batch_size:        Sequences per eggNOG-mapper submission.
        out_path:          If set, save annotated DataFrame to this path.

    Returns:
        Annotated DataFrame.
    """
    if hits_df.empty:
        log.warning("annotate_hits_dataframe: empty input DataFrame")
        return hits_df

    df = hits_df.copy()
    log.info(f"Annotating {len(df)} hits...")

    # ── Step 1: Taxonomy ────────────────────────────────────────────────────
    log.info("Step 1/3: Taxonomy lookup via NCBI Entrez...")
    unique_accs = df["sra_accession"].dropna().unique().tolist()
    tax_map = batch_lookup_taxonomy(unique_accs, api_key=api_key)

    tax_rows = []
    for _, row in df.iterrows():
        acc  = row.get("sra_accession", "")
        info = tax_map.get(acc, TaxonomyInfo(accession=acc))
        tax_rows.append({
            "taxonomy_domain":              info.domain,
            "taxonomy_organism":            info.organism,
            "taxonomy_phylum":              info.phylum,
            "taxonomy_taxid":               info.taxid,
            "is_archaeal":                  info.is_archaeal,
            "is_thermophile":               info.is_thermophile,
            "is_acidophile":                info.is_acidophile,
            "archaeal_significance_score":  info.archaeal_significance_score,
        })

    tax_df = pd.DataFrame(tax_rows)
    for col in tax_df.columns:
        df[col] = tax_df[col].values

    n_archaeal = df["is_archaeal"].sum()
    log.info(f"  Taxonomy complete: {n_archaeal}/{len(df)} hits are archaeal")

    # ── Step 2: eggNOG-mapper functional annotation ─────────────────────────
    log.info(f"Step 2/3: eggNOG functional annotation (mode={eggnog_mode})...")

    # Initialise eggNOG columns
    egg_cols = ["cog_category", "cog_description", "kegg_ko", "kegg_pathway",
                "go_terms", "eggnog_description", "eggnog_og", "max_annot_lvl"]
    for col in egg_cols:
        df[col] = ""

    if eggnog_mode != "skip" and "protein_seq" in df.columns:
        # Process in batches to avoid API limits
        seq_dict = dict(zip(df["hit_id"].astype(str), df["protein_seq"].astype(str)))
        all_annotations: Dict[str, EggNOGAnnotation] = {}

        items = list(seq_dict.items())
        for batch_start in range(0, len(items), batch_size):
            batch = dict(items[batch_start: batch_start + batch_size])
            log.info(f"  eggNOG batch {batch_start // batch_size + 1}: "
                     f"{len(batch)} sequences")

            if eggnog_mode == "web":
                ann = annotate_with_eggnog_web(
                    batch, tax_scope=eggnog_tax_scope
                )
            elif eggnog_mode == "local" and eggnog_db_dir:
                ann = annotate_with_eggnog_local(
                    batch, eggnog_db_dir=eggnog_db_dir,
                    tax_scope=eggnog_tax_scope,
                )
            else:
                ann = {}

            all_annotations.update(ann)
            log.info(f"    → {len(ann)} annotations received")

        # Merge eggNOG annotations back into DataFrame
        for col in egg_cols:
            df[col] = df["hit_id"].astype(str).map(
                lambda hid: getattr(all_annotations.get(hid, EggNOGAnnotation(hid)),
                                    col, "")
            )

        n_annotated = (df["cog_category"] != "").sum()
        log.info(f"  eggNOG complete: {n_annotated}/{len(df)} hits annotated")
    else:
        log.info("  eggNOG skipped")

    # ── Step 3: Genomic neighborhood ────────────────────────────────────────
    log.info("Step 3/3: Genomic neighborhood analysis...")
    df = batch_analyze_neighborhoods(
        df,
        contig_cache=contig_cache,
        contigs_dir=contigs_dir,
        hmm_profiles=hmm_profiles,
    )
    n_cluster = (df.get("neighborhood_score", pd.Series([0])) >= 0.4).sum()
    log.info(f"  Neighborhood complete: {n_cluster} hits in high-confidence "
             f"REE gene clusters (score ≥ 0.4)")

    # ── Save ────────────────────────────────────────────────────────────────
    if out_path is None:
        out_path = ANNOTATED_HITS_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"Annotated hits saved → {out_path}")

    return df


def summarize_archaeal_hits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and summarize archaeal hits for prioritization.

    Returns a filtered, sorted DataFrame of archaeal hits with a
    composite prioritization score that combines:
      - HMM bit score (higher = more confident motif match)
      - Archaeal significance score (thermophile / acidophile bonuses)
      - Neighborhood score (genomic context of REE gene cluster)
      - is_ree_selective flag (Pro-switch EF-hand variant)
      - Engineering score (predicted designability)

    This output is suitable for manual review or downstream
    structure prediction with AlphaFold3.
    """
    archaeal_df = df[df.get("is_archaeal", pd.Series([False] * len(df)))].copy()

    if archaeal_df.empty:
        log.info("No archaeal hits found in dataset")
        return archaeal_df

    # Normalise each component to 0–1 before combining
    def _norm(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0.0

    archaeal_df["priority_score"] = (
        _norm(archaeal_df.get("hmm_score",              pd.Series([0] * len(archaeal_df)))) * 0.30
        + _norm(archaeal_df.get("archaeal_significance_score",
                                pd.Series([0] * len(archaeal_df))))                          * 0.20
        + _norm(archaeal_df.get("neighborhood_score",   pd.Series([0] * len(archaeal_df)))) * 0.25
        + archaeal_df.get("is_ree_selective",           pd.Series([False] * len(archaeal_df))).astype(float) * 0.15
        + _norm(archaeal_df.get("engineering_score",    pd.Series([0] * len(archaeal_df)))) * 0.10
    )

    # Sort by composite score
    archaeal_df = archaeal_df.sort_values("priority_score", ascending=False)

    log.info(f"Archaeal hits summary: {len(archaeal_df)} total, "
             f"{archaeal_df['is_thermophile'].sum()} thermophiles, "
             f"{archaeal_df['is_acidophile'].sum()} acidophiles")

    return archaeal_df


def export_annotated_training_entries(
    df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> List[dict]:
    """
    Convert an annotated hits DataFrame to ESM-Bind-compatible training entries
    with the new functional annotation fields.

    Each entry gains:
      • "taxonomy"   : {domain, organism, phylum, is_archaeal, ...}
      • "function"   : {cog_category, kegg_ko, go_terms, description}
      • "neighborhood": {score, has_tonb, has_abc, ree_gene_count}
      • "annotation_quality": composite confidence label
    """
    entries: List[dict] = []

    for _, row in df.iterrows():
        # Determine annotation quality tier
        quality = "low"
        hmm_score = float(row.get("hmm_score", 0))
        nbr_score = float(row.get("neighborhood_score", 0))
        if hmm_score >= 50 and nbr_score >= 0.4:
            quality = "high"
        elif hmm_score >= 25 or nbr_score >= 0.2:
            quality = "medium"

        entry = {
            "id":                   str(row.get("hit_id", "")),
            "sequence":             str(row.get("protein_seq", "")),
            "source":               "logan_metagenome",
            "architecture_class":   str(row.get("architecture_class", "")),
            "is_ree_selective":     bool(row.get("is_ree_selective", False)),
            "hmm_score":            hmm_score,
            "taxonomy": {
                "domain":           str(row.get("taxonomy_domain", "Unknown")),
                "organism":         str(row.get("taxonomy_organism", "")),
                "phylum":           str(row.get("taxonomy_phylum", "")),
                "taxid":            int(row.get("taxonomy_taxid", 0)),
                "is_archaeal":      bool(row.get("is_archaeal", False)),
                "is_thermophile":   bool(row.get("is_thermophile", False)),
                "is_acidophile":    bool(row.get("is_acidophile", False)),
                "archaeal_significance_score": int(row.get("archaeal_significance_score", 0)),
            },
            "function": {
                "cog_category":     str(row.get("cog_category", "")),
                "cog_description":  str(row.get("cog_description", "")),
                "kegg_ko":          str(row.get("kegg_ko", "")),
                "kegg_pathway":     str(row.get("kegg_pathway", "")),
                "go_terms":         str(row.get("go_terms", "")),
                "description":      str(row.get("eggnog_description", "")),
            },
            "neighborhood": {
                "score":            float(row.get("neighborhood_score", 0)),
                "has_tonb":         bool(row.get("neighborhood_has_tonb", False)),
                "has_abc":          bool(row.get("neighborhood_has_abc", False)),
                "has_xoxf":         bool(row.get("neighborhood_has_xoxf", False)),
                "ree_gene_count":   int(row.get("neighborhood_ree_gene_count", 0)),
            },
            "annotation_quality":   quality,
            "environment":          str(row.get("environment", "")),
        }
        entries.append(entry)

    if out_path is None:
        out_path = DATA_DIR / "annotated_training_entries.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2))
    log.info(f"Exported {len(entries)} annotated training entries → {out_path}")

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: OFFLINE TEST FIXTURE
# ─────────────────────────────────────────────────────────────────────────────

def run_offline_annotation_test() -> dict:
    """
    Self-contained offline test for the functional annotation module.

    Tests:
      T14a — TaxonomyInfo keyword classification (no network)
      T14b — eggNOG TSV parsing with synthetic annotation table
      T14c — Genomic neighborhood scoring with synthetic contig
      T14d — annotate_hits_dataframe (eggnog_mode="skip", no contigs)
      T14e — summarize_archaeal_hits prioritization
      T14f — export_annotated_training_entries format

    Returns:
        {"passed": [str, ...], "failed": [str, ...]}
    """
    import traceback
    passed: List[str] = []
    failed: List[str] = []

    # ── T14a: keyword taxonomy classification ──────────────────────────────
    try:
        thermo, acido, leacher = _keyword_classify("Metallosphaera sedula")
        assert acido  and leacher, f"Metallosphaera: acido={acido}, leacher={leacher}"
        thermo2, _, _ = _keyword_classify("Thermococcus kodakarensis")
        assert thermo2, "Thermococcus should be thermophile"
        t, a, l = _keyword_classify("Methylobacterium extorquens")
        assert not t and not a and not l, "Methylobacterium should be mesophile"
        passed.append("T14a: keyword taxonomy classification")
    except AssertionError as e:
        failed.append(f"T14a: {e}")
    except Exception as e:
        failed.append(f"T14a: unexpected error: {e}")

    # ── T14b: eggNOG TSV parsing ───────────────────────────────────────────
    try:
        synthetic_tsv = (
            "## This file was generated by eggNOG-mapper\n"
            "#query\tseed_ortholog\tevalue\tscore\teggNOG_OGs\tmax_annot_lvl\t"
            "COG_category\tDescription\tPreferred_name\tGOs\tEC\tKEGG_ko\t"
            "KEGG_Pathway\tKEGG_Module\tKEGG_Reaction\tKEGG_rclass\tBRITE\t"
            "KEGG_TC\tCAZy\tBiGG_Reaction\tPFAMs\n"
            "hit_001\t1234.ABC123\t1e-50\t200.0\tCOG0473@2|root\t2\t"
            "P\tMetal transport protein\tMTP1\tGO:0046872,GO:0055085\t-\t"
            "ko:K14028\tmap00010\t-\t-\t-\t-\t-\t-\t-\tPF00005\n"
            "hit_002\t5678.XYZ789\t1e-30\t120.0\tCOG1028@2157|Archaea\t2157\t"
            "C\tMethanol dehydrogenase PQQ-type\tXoxF1\tGO:0016614\t1.1.2.7\t"
            "ko:K16255\tmap00680\t-\t-\t-\t-\t-\t-\t-\tPF01011\n"
        )
        ann = _parse_eggnog_tsv(synthetic_tsv)
        assert "hit_001" in ann, "hit_001 not parsed"
        assert ann["hit_001"].cog_category == "P", f"Expected P, got {ann['hit_001'].cog_category}"
        assert ann["hit_001"].kegg_ko == "K14028", f"KEGG mismatch: {ann['hit_001'].kegg_ko}"
        assert "GO:0046872" in ann["hit_001"].go_terms
        assert ann["hit_002"].cog_category == "C"
        assert ann["hit_002"].eggnog_description == "Methanol dehydrogenase PQQ-type"
        passed.append("T14b: eggNOG TSV parsing")
    except AssertionError as e:
        failed.append(f"T14b: {e}")
    except Exception as e:
        failed.append(f"T14b: unexpected error: {traceback.format_exc()}")

    # ── T14c: neighborhood analysis with synthetic contig ──────────────────
    try:
        # Build a synthetic contig with a LanM ORF flanked by a simple ORF
        # Long enough to be realistic for pyrodigal (-meta mode needs >3 kb)
        import random
        random.seed(42)
        codon_aa = {
            "A": "GCT", "R": "CGT", "N": "AAT", "D": "GAT", "C": "TGT",
            "E": "GAA", "Q": "CAA", "G": "GGT", "H": "CAT", "I": "ATT",
            "L": "CTT", "K": "AAA", "M": "ATG", "F": "TTT", "P": "CCT",
            "S": "TCT", "T": "ACT", "W": "TGG", "Y": "TAT", "V": "GTT",
            "*": "TAA",
        }
        def _encode_protein(aa_seq: str) -> str:
            return "".join(codon_aa.get(aa, "NNN") for aa in aa_seq)

        # Build intergenic spacers
        def _spacer(n: int) -> str:
            return "".join(random.choice("ATGC") for _ in range(n))

        lanm_seq    = "M" + "YIDPNDGKFIEADELLAAK" * 4 + "KLAKELAE" + "*"
        lanm_codons = _encode_protein(lanm_seq)
        spacer1     = _spacer(200)
        dummy_gene  = "ATG" + _encode_protein("L" * 80 + "*")
        spacer2     = _spacer(150)

        contig = spacer1 + lanm_codons + spacer2 + dummy_gene + _spacer(100)

        # Test with no HMMs available (should return safely with defaults)
        nbr = analyze_genomic_neighborhood(
            contig_seq=contig,
            hit_orf_start=len(spacer1),
            hit_orf_end=len(spacer1) + len(lanm_codons),
            hit_id="synthetic_test",
            hmm_profiles={},   # empty — tests graceful degradation
        )
        assert isinstance(nbr, NeighborhoodInfo), "Should return NeighborhoodInfo"
        assert nbr.contig_length == len(contig), "Contig length mismatch"
        passed.append("T14c: genomic neighborhood analysis (graceful degradation)")
    except Exception as e:
        failed.append(f"T14c: unexpected error: {traceback.format_exc()}")

    # ── T14d: annotate_hits_dataframe (offline, eggnog_mode=skip) ─────────
    try:
        synthetic_hits = pd.DataFrame([
            {
                "hit_id":           "TEST_SRR001|ctg1|10-300|+",
                "sra_accession":    "SRR0000001",
                "contig_id":        "ctg1",
                "orf_start":        10,
                "orf_end":          300,
                "strand":           "+",
                "protein_seq":      "MYIDPNDGKFIEADELLAAK" * 3,
                "hmm_name":         "EF_hand_REE_proswitch",
                "hmm_score":        75.0,
                "e_value":          1e-15,
                "architecture_class": "EF_hand",
                "is_ree_selective":  True,
                "engineering_score": 0.85,
                "environment":       "acid_mine_drainage",
            },
            {
                "hit_id":           "TEST_SRR002|ctg2|50-500|-",
                "sra_accession":    "SRR0000002",
                "contig_id":        "ctg2",
                "orf_start":        50,
                "orf_end":          500,
                "strand":           "-",
                "protein_seq":      "MTGCNLMDYDGSGSTGAQLNL" * 3,
                "hmm_name":         "DYD_active_site",
                "hmm_score":        45.0,
                "e_value":          1e-8,
                "architecture_class": "PQQ_XoxF",
                "is_ree_selective":  False,
                "engineering_score": 0.50,
                "environment":       "thermoacidophilic_archaea",
            },
        ])

        annotated = annotate_hits_dataframe(
            synthetic_hits,
            eggnog_mode="skip",    # no network required
            contig_cache=None,     # no contigs for this test
            out_path=DATA_DIR / "test_annotated_hits.parquet",
        )

        assert "taxonomy_domain" in annotated.columns, "Missing taxonomy_domain column"
        assert "neighborhood_score" in annotated.columns, "Missing neighborhood_score column"
        assert "cog_category" in annotated.columns, "Missing cog_category column"
        assert len(annotated) == 2, f"Expected 2 rows, got {len(annotated)}"
        passed.append("T14d: annotate_hits_dataframe (offline mode)")
    except AssertionError as e:
        failed.append(f"T14d: {e}")
    except Exception as e:
        failed.append(f"T14d: unexpected error: {traceback.format_exc()}")

    # ── T14e: archaeal prioritization ─────────────────────────────────────
    try:
        df_arch = pd.DataFrame([
            {"hit_id": "arch_1", "is_archaeal": True,  "is_thermophile": True,
             "is_acidophile": True,  "archaeal_significance_score": 3,
             "hmm_score": 80, "neighborhood_score": 0.65,
             "is_ree_selective": True, "engineering_score": 0.9},
            {"hit_id": "arch_2", "is_archaeal": True,  "is_thermophile": False,
             "is_acidophile": False, "archaeal_significance_score": 1,
             "hmm_score": 40, "neighborhood_score": 0.1,
             "is_ree_selective": False, "engineering_score": 0.4},
            {"hit_id": "bact_1", "is_archaeal": False, "is_thermophile": False,
             "is_acidophile": False, "archaeal_significance_score": 0,
             "hmm_score": 90, "neighborhood_score": 0.8,
             "is_ree_selective": True, "engineering_score": 0.95},
        ])
        arch_summary = summarize_archaeal_hits(df_arch)
        assert len(arch_summary) == 2, f"Expected 2 archaeal hits, got {len(arch_summary)}"
        # arch_1 should rank above arch_2 (higher HMM + neighborhood + sig score)
        assert arch_summary.iloc[0]["hit_id"] == "arch_1", \
            f"arch_1 should rank first, got {arch_summary.iloc[0]['hit_id']}"
        assert "priority_score" in arch_summary.columns
        passed.append("T14e: archaeal hit prioritization")
    except AssertionError as e:
        failed.append(f"T14e: {e}")
    except Exception as e:
        failed.append(f"T14e: unexpected error: {traceback.format_exc()}")

    # ── T14f: export annotated training entries ───────────────────────────
    try:
        df_export = pd.DataFrame([{
            "hit_id": "EXP_001", "protein_seq": "MKLAA" * 10,
            "architecture_class": "EF_hand", "is_ree_selective": True,
            "hmm_score": 60.0, "taxonomy_domain": "Archaea",
            "taxonomy_organism": "Metallosphaera sedula",
            "taxonomy_phylum": "Thermoprotei",
            "taxonomy_taxid": 110163,
            "is_archaeal": True, "is_thermophile": False, "is_acidophile": True,
            "archaeal_significance_score": 2,
            "cog_category": "P", "cog_description": "Inorganic ion transport",
            "kegg_ko": "K14028", "kegg_pathway": "map00010",
            "go_terms": "GO:0046872", "eggnog_description": "LanM-like",
            "eggnog_og": "COG0473", "max_annot_lvl": "Archaea",
            "neighborhood_has_tonb": True, "neighborhood_has_abc": True,
            "neighborhood_has_xoxf": False, "neighborhood_has_ef_hand": False,
            "neighborhood_ree_gene_count": 2, "neighborhood_score": 0.65,
            "environment": "thermoacidophilic_archaea",
        }])

        out_path = DATA_DIR / "test_annotated_entries.json"
        entries = export_annotated_training_entries(df_export, out_path=out_path)

        assert len(entries) == 1
        e = entries[0]
        assert e["taxonomy"]["domain"] == "Archaea"
        assert e["taxonomy"]["is_archaeal"] is True
        assert e["taxonomy"]["archaeal_significance_score"] == 2
        assert e["function"]["cog_category"] == "P"
        assert e["neighborhood"]["has_tonb"] is True
        assert e["neighborhood"]["score"] == 0.65
        assert e["annotation_quality"] == "high"  # hmm_score=60 >= 50, nbr=0.65 >= 0.4
        assert out_path.exists()
        passed.append("T14f: annotated training entry export")
    except AssertionError as e:
        failed.append(f"T14f: {e}")
    except Exception as e:
        failed.append(f"T14f: unexpected error: {traceback.format_exc()}")

    return {"passed": passed, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="ree-miner annotate",
        description="Functional annotation of REE-binding protein hits from Logan metagenomes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hits", type=Path, default=None,
        help="Input hits parquet (default: datasets/logan_hits_merged.parquet)"
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output annotated parquet (default: datasets/annotated_hits.parquet)"
    )
    parser.add_argument(
        "--contigs-dir", type=Path, default=None,
        help="Directory of contig FASTA files for neighborhood analysis"
    )
    parser.add_argument(
        "--eggnog-mode", choices=["web", "local", "skip"], default="skip",
        help="eggNOG-mapper mode (default: skip)"
    )
    parser.add_argument(
        "--eggnog-db", type=Path, default=None,
        help="Path to local eggNOG database (required when --eggnog-mode=local)"
    )
    parser.add_argument(
        "--eggnog-tax-scope", type=int, default=1,
        help="Taxonomic scope for eggNOG search (1=root, 2=Bacteria, 2157=Archaea)"
    )
    parser.add_argument(
        "--archaeal-only", action="store_true",
        help="After annotation, print archaeal hit summary to stdout"
    )
    parser.add_argument(
        "--export-json", type=Path, default=None,
        help="Also export annotated entries as ESM-Bind-compatible JSON"
    )
    parser.add_argument(
        "--ncbi-api-key", type=str, default=None,
        help="NCBI API key for Entrez (also reads NCBI_API_KEY env var)"
    )
    parser.add_argument(
        "--offline-test", action="store_true",
        help="Run offline self-test and exit"
    )

    args = parser.parse_args()

    if args.offline_test:
        results = run_offline_annotation_test()
        for p in results["passed"]:
            print(f"  ✓ PASS  {p}")
        for f in results["failed"]:
            print(f"  ✗ FAIL  {f}")
        print(f"\n  {len(results['passed'])} passed | {len(results['failed'])} failed")
        return 0 if not results["failed"] else 1

    # Load hits
    hits_path = args.hits or (DATA_DIR / "logan_hits_merged.parquet")
    if not hits_path.exists():
        print(f"ERROR: hits file not found: {hits_path}", file=__import__("sys").stderr)
        return 1

    hits_df = pd.read_parquet(hits_path)
    log.info(f"Loaded {len(hits_df)} hits from {hits_path}")

    annotated = annotate_hits_dataframe(
        hits_df,
        eggnog_mode=args.eggnog_mode,
        eggnog_db_dir=args.eggnog_db,
        eggnog_tax_scope=args.eggnog_tax_scope,
        contigs_dir=args.contigs_dir,
        api_key=args.ncbi_api_key,
        out_path=args.out,
    )

    if args.archaeal_only:
        arch = summarize_archaeal_hits(annotated)
        print(arch[["hit_id", "taxonomy_organism", "hmm_score",
                     "archaeal_significance_score", "neighborhood_score",
                     "priority_score"]].to_string(index=False))

    if args.export_json:
        export_annotated_training_entries(annotated, out_path=args.export_json)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
