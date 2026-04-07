"""
REE Homolog Finder
==================
Discovers novel REE-binding protein architectures through three complementary
sequence-search strategies:

Strategy A — UniProt keyword search
    Query UniProt REST API for "lanthanide", "rare earth", "lanmodulin",
    "XoxF", "lanpepsy". These are proteins biologists have already annotated
    as REE-related. Returns sequences not in PDB (sequence-only positive set).

Strategy B — Sequence motif search (DYD + EF-hand variants)
    Use BioPython's PairwiseAligner to scan UniProt sequences for the DYD
    motif (REE-dependent MDH signature) and EF-hand loop variants with Pro
    at position 2. Flag hits by motif type — these are architecture candidates
    even without experimental binding data.

Strategy C — MLL biosynthetic cluster gene search
    Methylolanthanin (MLL) is the REE metallophore in methylotrophs. The
    biosynthetic gene cluster includes outer-membrane receptors and periplasmic
    binding proteins that must interact with Ln-MLL complexes — an entirely
    unexplored class of high-affinity REE-binding proteins. This strategy
    finds them by searching for genes co-occurring with the MLL synthase.

Output:
    datasets/uniprot_hits.csv
    datasets/motif_hits.csv
    datasets/mll_cluster_hits.csv
    datasets/all_homologs.csv   ← combined, deduplicated

Usage:
    python 03_homolog_finder.py [--strategy A|B|C|all]
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from Bio import Entrez, SeqIO
from Bio.Align import PairwiseAligner

from io import StringIO
from http.client import IncompleteRead

from ree_miner._workspace import DATA_DIR, LOG_DIR
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "homolog_finder.log"),
    ],
)
log = logging.getLogger("homolog_finder")

Entrez.email = "user@example.com"   # required by NCBI

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA_URL  = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
REQUEST_PAUSE      = 0.2
NCBI_REQUEST_PAUSE = 0.4 # NCBI pause - stays safely under 3 req/sec limit

# ─── Sequence motifs ─────────────────────────────────────────────────────────
# DYD motif: the Asp-Tyr-Asp triad in REE-dependent MDHs absent from Ca2+ MDHs.
# EF_HAND_REE: canonical EF-hand 12-residue loop with Pro at position 2 (LanM).
# RTX_REPEAT: calcium-binding repeat motif (GGXGXDXUX where U=hydrophobic).
MOTIFS = {
    # ── EXISTING: REE-specific motifs ──────────────────────────────────────
    # DYD triad in REE-dependent MDHs: Asp-Tyr-Asp with 0-2 gap residues
    "DYD_strict":     re.compile(r"DYD"),                           # exact Asp-Tyr-Asp
    "DYD_extended":   re.compile(r"D[A-Z]{0,3}Y[A-Z]{0,3}D"),     # 0-3 gap residues allowed
    # EF-hand with Pro at position 2 (REE-selective; blocks Ca2+)
    "EF_hand_REE":    re.compile(r"[DE]P[A-Z]{8,10}[EQ]"),
    # EF-hand WITHOUT Pro at pos 2 — Ca2+-binding (negative control motif)
    "EF_hand_Ca":     re.compile(r"[DE][ACDFGHIKLMNQRSTVWY][A-Z]{8,10}[EQ]"),
    # RTX calcium-binding repeat
    "RTX_repeat":     re.compile(r"[GA]G[A-Z]{0,2}D[GNA][A-Z]{0,3}G"),
    # PepSY acidic clusters (LanP architecture)
    "pepsy_core":     re.compile(r"[DE]{2,}[A-Z]{3,8}[DE]{2,}"),
    # LBT (Lanthanide Binding Tag) — phage-display sequence
    "LBT_like":       re.compile(r"Y[A-Z]{1,2}DT[A-Z]{1,3}DG[A-Z]YEG"),

    # ── NEW: C2 domain (β-sandwich, Asp-cluster Ca²⁺ sites) ───────────────
    # Tb³⁺ luminescence at C2 domain Ca²⁺ sites is experimentally documented.
    # CBR3 loop has clustered Asp/Asn in a 12-residue segment.
    "C2_asp_cluster":  re.compile(r"D[A-Z]{1,5}[DN][A-Z]{1,5}D"),          # triple Asp/Asn
    "C2_cbr1":         re.compile(r"DN[A-Z]{2,4}D"),                         # CBR1 signature
    "C2_dde_cluster":  re.compile(r"D[A-Z]{2,6}D[A-Z]{2,6}[DE]"),           # extended cluster

    # ── NEW: Annexin fold (endonexin repeat, type II Ca²⁺) ────────────────
    # La³⁺ inhibition of annexin A5 membrane binding demonstrated (Hofmann 1997).
    # The GXGT motif is the structural core of each 78-residue annexin repeat.
    "Annexin_GXGT":    re.compile(r"G[A-Z]GT"),                              # endonexin core
    "Annexin_type3":   re.compile(r"[DE][A-Z]{2,4}[DE][A-Z]{2,4}[DE]"),     # HAP Ca2+ site
    "Annexin_HAP":     re.compile(r"G[A-Z]GT[A-Z]{1,3}[DE]"),               # GXGT + acid

    # ── NEW: EGF-Ca²⁺ module (high-affinity Ca²⁺, Tb³⁺ in fibrillin) ─────
    # Each cbEGF module binds Ca²⁺ at ~10 nM affinity.
    "EGF_ca2_core":    re.compile(r"[DN][A-Z]{1,2}[DN][LIVMFY][A-Z]{3,6}[DE]"),  # key cluster
    "EGF_cysteine":    re.compile(r"[DE]{1,2}[A-Z]{0,3}C[A-Z]{1,5}C"),      # DEEC/DNEC pattern

    # ── NEW: Gla domain (γ-carboxyglutamate, Furie 1979 Ln³⁺ binding) ─────
    # γ-carboxyglutamate (Gla) appears as Glu in sequence; detect by Glu cluster.
    # Gla domains have 9-12 Gla residues in 40-aa N-terminal segment.
    "Gla_Glu_cluster": re.compile(r"E[A-Z]{0,4}E[A-Z]{0,4}E[A-Z]{0,4}E"),  # quad-Glu cluster
    "Gla_FLEEL":       re.compile(r"[FL][A-Z]{0,2}E[A-Z]{0,2}E[A-Z]{0,2}[LI]"),  # FLEEL core

    # ── NEW: Cadherin Ca²⁺ linker (DxD, DXXE, DxNDN) ────────────────────
    # 3 Ca²⁺ per EC-domain linker; O-donor rich; direct Ln³⁺ sub predicted.
    "Cadherin_DxD":    re.compile(r"D[A-Z]D[A-Z]{4,8}[DE]"),                # Ca²⁺ site 1
    "Cadherin_DxNDN":  re.compile(r"D[A-Z]NDN"),                             # most specific motif
    "Cadherin_DXXE":   re.compile(r"D[A-Z]{2}E[A-Z]{4,10}D"),               # sites 2-3 bridge
}

# ─── UniProt queries for Strategy A ──────────────────────────────────────────
UNIPROT_QUERIES = [
    # Annotated REE-binding proteins
    {"query": "protein_name:lanmodulin",                "label": "lanmodulin"},
    {"query": "gene:lanM",                              "label": "lanM_gene"},
    {"query": "protein_name:lanpepsy OR gene:lanP",     "label": "lanpepsy"},
    {"query": "protein_name:XoxF AND reviewed:true",    "label": "XoxF_reviewed"},
    {"query": "protein_name:XoxF AND reviewed:false",   "label": "XoxF_unreviewed"},
    {"query": "protein_name:PedH OR protein_name:ExaF", "label": "PedH_ExaF"},
    # Methylotrophs — the primary source organisms for REE proteins
    {"query": "organism_name:Methylorubrum AND cc_function:lanthanide",  "label": "Methylorubrum_Ln"},
    {"query": "organism_name:Methylobacterium AND cc_function:lanthanide","label": "Methylobacterium_Ln"},
    {"query": "cc_function:\"rare earth\" OR cc_function:lanthanide",    "label": "Ln_function_any"},
    # RTX domain proteins (for the beta-roll architecture from Paper 4)
    {"query": "ft_domain:RTX AND (taxonomy_id:562 OR taxonomy_id:1234)",    "label": "RTX_ecoli_related"},
    # De novo designs and engineered variants
    {"query": "protein_name:\"lanthanide binding tag\" OR protein_name:LBT", "label": "LBT"},
    # Lanthanide transport / metallophore-related
    {"query": "keyword:KW-0427 AND cc_function:lanthanide",              "label": "metalloprotein_Ln"},

    # ── Strategy D: Calmodulin-family (EF-hand Ca²⁺ proteins engineerable for Ln³⁺) ──
    # La³⁺/Ln³⁺ can replace Ca²⁺ in EF-hand sites (Horrocks 1979). A single D→P
    # substitution at position 2 of the 12-residue loop confers REE selectivity
    # (Cotruvo 2019). We collect the full EF-hand superfamily across all life.
    {"query": "protein_name:calmodulin AND reviewed:true",               "label": "cam_calmodulin"},
    {"query": "protein_name:\"calmodulin-like\" AND reviewed:true",      "label": "cam_cml"},
    {"query": "protein_name:parvalbumin AND reviewed:true",              "label": "cam_parvalbumin"},
    {"query": "protein_name:calbindin AND reviewed:true",               "label": "cam_calbindin"},
    {"query": "protein_name:S100 AND reviewed:true",                    "label": "cam_s100"},
    {"query": "protein_name:\"troponin C\" AND reviewed:true",          "label": "cam_troponin_c"},
    {"query": "protein_name:recoverin AND reviewed:true",               "label": "cam_recoverin"},
    {"query": "protein_name:\"neuronal calcium sensor\" AND reviewed:true", "label": "cam_ncs"},
    {"query": "protein_name:\"calcineurin\" AND reviewed:true",         "label": "cam_calcineurin"},
    {"query": "protein_name:sorcin AND reviewed:true",                  "label": "cam_sorcin"},
    # Extremophile/acid-stable variants — highest priority for bio-leaching
    {"query": "protein_name:calmodulin AND taxonomy_id:2285",           "label": "cam_sulfolobus"},  # Sulfolobus spp.
    {"query": "protein_name:calmodulin AND (taxonomy_name:thermophile OR taxonomy_name:acidophile)", "label": "cam_extremophile"},
    {"query": "family:\"EF-hand\" AND taxonomy_id:2157 AND reviewed:true", "label": "cam_archaea_efhand"},

    # ── Strategy E: Ca²⁺-binding folds with proven/predicted Ln³⁺ substitution ──
    # These fold families coordinate Ca²⁺ via O-donor residues and are known (or
    # strongly predicted) to bind Ln³⁺. They complement EF-hands with entirely
    # different structural architectures.
    #
    # C2 domain (β-sandwich, Asp-cluster): Tb³⁺ luminescence at Ca²⁺ sites
    # confirmed in synaptotagmin, PKCα, PLC-delta (Chapman 1998, Nalefski 2001)
    {"query": "protein_name:synaptotagmin AND reviewed:true",            "label": "c2_synaptotagmin"},
    {"query": "protein_name:\"protein kinase C\" AND reviewed:true",     "label": "c2_pkc"},
    {"query": "protein_name:dysferlin AND reviewed:true",                "label": "c2_dysferlin"},
    {"query": "protein_name:copine AND reviewed:true",                   "label": "c2_copine"},
    {"query": "protein_name:rabphilin AND reviewed:true",                "label": "c2_rabphilin"},
    # Annexin fold: La³⁺ inhibition of annexin A5 demonstrated (Hofmann 1997)
    # 4 repeats × 3 Ca²⁺ sites per repeat = up to 12 Ln³⁺ sites per protein
    {"query": "protein_name:annexin AND reviewed:true",                  "label": "annexin_reviewed"},
    {"query": "protein_name:annexin AND taxonomy_id:3702",               "label": "annexin_plant"},  # Arabidopsis
    # EGF-Ca²⁺ module: high-affinity Ca²⁺ (~10 nM), Tb³⁺ in fibrillin shown
    # 47 cbEGF modules in fibrillin-1 → huge sequence diversity
    {"query": "protein_name:fibrillin AND reviewed:true",                "label": "egf_fibrillin"},
    {"query": "protein_name:Notch AND reviewed:true",                    "label": "egf_notch"},
    {"query": "protein_name:EGF-like AND cc_function:calcium AND reviewed:true","label": "egf_ca2_any"},
    # Gla (γ-carboxyglutamate) domain: 9-12 Gla per domain, O-donor density
    # matches Ln³⁺ CN=8-9 preference; Furie 1979 showed Ln³⁺ substitution
    {"query": "protein_name:\"factor IX\" AND reviewed:true",            "label": "gla_fix"},
    {"query": "protein_name:\"factor VII\" AND reviewed:true",           "label": "gla_fvii"},
    {"query": "protein_name:prothrombin AND reviewed:true",              "label": "gla_prothrombin"},
    {"query": "protein_name:\"protein C\" AND taxonomy_id:9606",         "label": "gla_protc"},
    {"query": "protein_name:Gas6 AND reviewed:true",                     "label": "gla_gas6"},
    {"query": "protein_name:\"matrix Gla\" AND reviewed:true",           "label": "gla_mgp"},
    # Cadherin Ca²⁺ linker: 3 Ca²⁺ per linker, O-donor DxD+DXXE motifs
    {"query": "protein_name:cadherin AND reviewed:true AND length:[100 TO 1000]","label": "cadherin_short"},
    {"query": "protein_name:E-cadherin AND reviewed:true",               "label": "cadherin_ecad"},
    # PQQ-containing proteins beyond XoxF/PedH: PQQ is the only biologically
    # evolved Ln³⁺ cofactor; extend to all PQQ dehydrogenases
    {"query": "cc_cofactor:PQQ AND reviewed:true",                      "label": "pqq_all_reviewed"},
    {"query": "protein_name:\"glucose dehydrogenase\" AND cc_cofactor:PQQ","label": "pqq_gdh"},
]

# ─── MLL biosynthetic cluster genes (Strategy C) ─────────────────────────────
# Methylolanthanin (MLL) gene cluster in M. extorquens AM1 (GenBank CP001511).
# lanA = MLL synthetase (lanA/mxcL), lanB = transport, lanC = receptor.
# We search for co-occurring protein families.
MLL_CLUSTER_GENES = [
    "lanA", "mxcL", "mxcM",      # MLL biosynthesis
    "lanB", "mxcN",               # MLL export/transport
    "lanR", "mxcO",               # receptor/import
    "lanH",                        # hydrolase (MLL processing)
]
MLL_NCBI_QUERY = (
    '("methylolanthanin" OR "MLL" OR "lanthanide metallophore") '
    'AND ("biosynthesis" OR "transport" OR "receptor")'
)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY A: UniProt keyword search
# ═══════════════════════════════════════════════════════════════════════════════

def search_uniprot(query: str, max_results: int = 500) -> list[dict]:
    """Search UniProt REST API and return list of protein entries."""
    params = {
        "query":  query,
        "format": "json",
        "size":   min(max_results, 500),
        "fields": "accession,id,gene_names,organism_name,protein_name,sequence,length,"
                  "cc_function,xref_pdb,reviewed",
    }
    results = []
    url = UNIPROT_SEARCH_URL
    malformed_headers = []
    reconstructed = False
    
    while url:
        try:
            resp = requests.get(url, params=params if url == UNIPROT_SEARCH_URL else None,
                                timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            
            # ── Parse the Link header for the next page URL ──────────────────
            # UniProt returns pagination cursors in the HTTP Link header, e.g.:
            #   <https://rest.uniprot.org/uniprotkb/search?query=...&cursor=...>; rel="next"
            # We use re.findall to capture all <url>; rel="..." pairs in one pass,
            # then filter for rel="next". re.search is not used here because it
            # returns only the first match — which may be rel="prev" if that link
            # appears first in the header, causing us to paginate backwards.
            link_header = resp.headers.get("Link", "")
            next_url = None
            for candidate, rel in re.findall(r'<([^>]+)>\s*;\s*rel="([^"]+)"', link_header):
                if rel == "next":
                    # If the scheme+host prefix was trimmed (e.g. due to a comma
                    # in a query parameter splitting the URL), reconstruct the full
                    # URL using the known UniProt search base. This is safe because
                    # search_uniprot only ever paginates on UNIPROT_SEARCH_URL.
                    if not candidate.startswith(("http://", "https://")):
                        corrected = UNIPROT_SEARCH_URL + "?" + candidate
                        malformed_headers.append((candidate, corrected))
                        candidate = corrected
                        reconstructed = True
                    next_url = candidate
            # ─────────────────────────────────────────────────────────────────
            
            url = next_url
            params = None  # params only needed for the first request
            time.sleep(REQUEST_PAUSE)
        except Exception as e:
            log.warning(f"UniProt query failed: {e}")
            break

    # Log if any headers needed to be reconstructed, if the regex didn't work
    if reconstructed:
        for original, corrected in malformed_headers:
            log.warning(f"Malformed next URL detected: {original!r} → reconstructed as {corrected!r}")
            
    return results


def parse_uniprot_entry(entry: dict, label: str) -> dict:
    """Extract relevant fields from a UniProt JSON entry."""
    acc  = entry.get("primaryAccession", "")
    seqs = entry.get("sequence", {})
    xrefs = entry.get("uniProtKBCrossReferences", [])
    pdb_ids = [x.get("id", "") for x in xrefs if x.get("database") == "PDB"]
    return {
        "uniprot_acc": acc,
        "entry_name":  entry.get("uniProtkbId", ""),
        "gene_names":  "; ".join(
            g.get("geneName", {}).get("value", "")
            for g in entry.get("genes", [])
            if g.get("geneName")
        ),
        "organism":    entry.get("organism", {}).get("scientificName", ""),
        "protein_name": entry.get("proteinDescription", {})
                          .get("recommendedName", {})
                          .get("fullName", {})
                          .get("value", ""),
        "sequence":    seqs.get("value", ""),
        "seq_len":     seqs.get("length", 0),
        "reviewed":    entry.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)",
        "pdb_ids":     "; ".join(pdb_ids),
        "has_pdb":     len(pdb_ids) > 0,
        "query_label": label,
        "source":      "uniprot_keyword",
    }


def run_strategy_A() -> pd.DataFrame:
    """Strategy A: UniProt keyword search for annotated REE-binding proteins."""
    log.info("=" * 50)
    log.info("Strategy A: UniProt keyword search")
    log.info("=" * 50)
    all_rows = []
    seen_acc = set()
    for q in UNIPROT_QUERIES:
        log.info(f"  Query: {q['query'][:70]} ...")
        entries = search_uniprot(q["query"])
        new = 0
        for e in entries:
            row = parse_uniprot_entry(e, q["label"])
            if row["uniprot_acc"] not in seen_acc and row["sequence"]:
                seen_acc.add(row["uniprot_acc"])
                all_rows.append(row)
                new += 1
        log.info(f"    → {len(entries)} returned, {new} new unique")

    df = pd.DataFrame(all_rows)
    out = DATA_DIR / "uniprot_hits.csv"
    df.to_csv(out, index=False)
    log.info(f"Strategy A: {len(df)} unique proteins → {out}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY B: Sequence motif search
# ═══════════════════════════════════════════════════════════════════════════════

def scan_sequence_for_motifs(sequence: str) -> dict[str, list[tuple]]:
    """
    Scan a protein sequence for all defined motifs.
    Returns dict of {motif_name: [(start, end, match_str), ...]}
    """
    hits = {}
    for name, pattern in MOTIFS.items():
        matches = [(m.start(), m.end(), m.group()) for m in pattern.finditer(sequence)]
        if matches:
            hits[name] = matches
    return hits


def annotate_binding_residues_from_motif(seq: str, motif_hits: dict) -> list[int]:
    """
    Given motif hits, return list of residue positions likely to be binding residues.
    For DYD: Asp positions; for EF-hand REE: positions 1, 3, 5, 7, 9, 12 of the loop.
    """
    binding_pos = set()
    for motif_name, hits in motif_hits.items():
        for start, end, match in hits:
            if "DYD" in motif_name:
                # Asp residues at start and end of match
                for i, aa in enumerate(match):
                    if aa == "D":
                        binding_pos.add(start + i)
            elif "EF_hand" in motif_name:
                # Standard EF-hand binding positions: 1, 3, 5, 7, 9, 12
                for offset in [0, 2, 4, 6, 8, 11]:
                    if start + offset < len(seq):
                        binding_pos.add(start + offset)
            elif "RTX" in motif_name:
                # RTX D residue at position 6 of the motif
                for i, aa in enumerate(match):
                    if aa == "D":
                        binding_pos.add(start + i)
    return sorted(binding_pos)


def run_strategy_B(uniprot_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Strategy B: Scan UniProt hits (from Strategy A) + any additional sequences
    for sequence motifs indicative of novel REE-binding architectures.
    """
    log.info("=" * 50)
    log.info("Strategy B: Sequence motif scanning")
    log.info("=" * 50)

    if uniprot_df is None or uniprot_df.empty:
        uniprot_path = DATA_DIR / "uniprot_hits.csv"
        if uniprot_path.exists():
            uniprot_df = pd.read_csv(uniprot_path)
        else:
            log.error("No UniProt hits available. Run Strategy A first.")
            return pd.DataFrame()

    rows = []
    for _, row in uniprot_df.iterrows():
        seq = str(row.get("sequence", ""))
        if not seq or seq == "nan":
            continue
        motif_hits = scan_sequence_for_motifs(seq)
        if not motif_hits:
            continue
        binding_positions = annotate_binding_residues_from_motif(seq, motif_hits)
        motif_names = list(motif_hits.keys())

        # Architecture inference from motif combination
        arch_inferred = "unknown"
        if "DYD_strict" in motif_names or "DYD_extended" in motif_names:
            arch_inferred = "beta-propeller"  # MDH-like
        elif "EF_hand_REE" in motif_names and "EF_hand_Ca" not in motif_names:
            arch_inferred = "ef-hand"          # REE-selective EF-hand
        elif "RTX_repeat" in motif_names:
            arch_inferred = "beta-roll (RTX)"
        elif "LBT_like" in motif_names:
            arch_inferred = "ef-hand"          # LBT is EF-hand derived
        elif "pepsy_core" in motif_names:
            arch_inferred = "pepsy-domain"

        rows.append({
            "uniprot_acc":       row.get("uniprot_acc", ""),
            "sequence":          seq,
            "seq_len":           len(seq),
            "organism":          row.get("organism", ""),
            "gene_names":        row.get("gene_names", ""),
            "motifs_found":      "; ".join(motif_names),
            "n_motifs":          len(motif_names),
            "binding_positions": str(binding_positions),
            "n_binding_residues_predicted": len(binding_positions),
            "architecture_inferred": arch_inferred,
            "is_ef_hand":        arch_inferred == "ef-hand",
            "is_novel_arch":     arch_inferred not in ("ef-hand", "unknown"),
            "source":            "motif_scan",
            "query_label":       row.get("query_label", ""),
        })

    df = pd.DataFrame(rows)
    out = DATA_DIR / "motif_hits.csv"
    df.to_csv(out, index=False)
    log.info(f"Strategy B: {len(df)} sequences with motif hits → {out}")

    if not df.empty:
        log.info("Motif hit summary:")
        all_motifs = []
        for m_str in df["motifs_found"]:
            all_motifs.extend(m_str.split("; "))
        from collections import Counter
        for motif, count in Counter(all_motifs).most_common():
            log.info(f"  {motif}: {count} sequences")
        novel = df[df["is_novel_arch"]].shape[0]
        log.info(f"  Novel architecture predictions: {novel}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY C: MLL biosynthetic cluster gene search
# ═══════════════════════════════════════════════════════════════════════════════

def search_ncbi_protein(query: str, max_results: int = 200) -> list[str]:
    """Search NCBI Protein database and return list of accession IDs."""
    try:
        handle = Entrez.esearch(db="protein", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        time.sleep(NCBI_REQUEST_PAUSE) # stay under NCBI 3 req/sec limit
        return record.get("IdList", [])
    except Exception as e:
        log.warning(f"NCBI search failed: {e}")
        return []


def fetch_ncbi_fasta(id_list: list[str], max_retries: int = 3) -> list[dict]:
    """Fetch GenBank sequences from NCBI for a list of protein IDs."""
    if not id_list:
        return []
    rows = []
    log.info(f"  Fetching {len(id_list)} IDs from NCBI, sample: {id_list[:3]}")
    for attempt in range(max_retries):
        raw = None
        error = None
        error_raw = None
        handle = None
        try:
            handle = Entrez.efetch(db="protein", id=",".join(id_list),
                                   rettype="gb", retmode="text")
            raw = handle.read()
            if not raw.strip():
                raise ValueError("Empty response from NCBI")
            for record in SeqIO.parse(StringIO(raw), "genbank"):
                rows.append({
                    "ncbi_acc":    record.id,
                    "description": record.description,
                    "organism":    record.annotations.get("organism", ""),
                    "sequence":    str(record.seq),
                    "seq_len":     len(record.seq),
                    "source":      "ncbi_mll_cluster",
                    "query_label": "mll_cluster",
                })
        except Exception as e:
            # If NCBI dropped the connection mid-transfer, log the partial
            # bytes raw so we can inspect what was actually returned
            if isinstance(e, IncompleteRead) and e.partial:
                error_raw = e.partial.decode("utf-8", errors="ignore")
            error = e
        finally:
            if handle is not None:
                handle.close()

        # Log outside try/except so output is never swallowed by the except handler
        if raw is not None:
            log.info(f"  Raw response length: {len(raw)} bytes") #, preview: {raw[:200]!r}")
        if error_raw is not None:
            log.warning(f"  Partial response ({len(error_raw)} bytes): {error_raw!r}")
        if error is None:
            break  # fetch succeeded, stop retrying
        wait = 2 ** attempt
        log.warning(f"NCBI fetch failed (attempt {attempt + 1}/{max_retries}): {error} — retrying in {wait}s")
        time.sleep(wait)
        if attempt == max_retries - 1:
            log.warning(f"NCBI fetch gave up after {max_retries} attempts for {len(id_list)} IDs")

    return rows



def run_strategy_C() -> pd.DataFrame:
    """
    Strategy C: Mine NCBI for proteins encoded near MLL biosynthetic genes.
    These transport/receptor proteins are the least explored REE-binding class.
    """
    log.info("=" * 50)
    log.info("Strategy C: MLL biosynthetic cluster gene search")
    log.info("=" * 50)

    # Primary search: methylolanthanin-related proteins
    log.info("  Searching NCBI Protein for MLL cluster genes ...")
    id_list = search_ncbi_protein(MLL_NCBI_QUERY, max_results=300)
    log.info(f"  Found {len(id_list)} NCBI protein IDs")

    # Also search by individual gene names in methylotrophic organisms
    for gene in MLL_CLUSTER_GENES:
        gene_query = f"{gene}[Gene Name] AND Methylorubrum[Organism]"
        ids = search_ncbi_protein(gene_query, max_results=100)
        id_list.extend(ids)
        time.sleep(NCBI_REQUEST_PAUSE)

    # Additional search for lanthanide metallophore transporters
    lm_transport_query = (
        '(TonB-dependent OR "ABC transporter" OR "periplasmic binding") '
        'AND (lanthanide OR "rare earth" OR methylotrophic) '
        'AND Bacteria[Organism]'
    )
    transport_ids = search_ncbi_protein(lm_transport_query, max_results=500)
    id_list.extend(transport_ids)

    # Deduplicate
    id_list = list(dict.fromkeys(id_list))
    log.info(f"  Total unique NCBI IDs: {len(id_list)}")

    # Fetch sequences in batches
    all_rows = []
    batch_size = 20 # Reduced from 50 to reduce chances of IncompleteRead error
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i : i + batch_size]
        rows = fetch_ncbi_fasta(batch)
        all_rows.extend(rows)
        log.info(f"  Fetched batch {i//batch_size + 1}: {len(rows)} sequences")
        time.sleep(NCBI_REQUEST_PAUSE)

    # Scan fetched sequences for motifs to identify binding candidates
    for row in all_rows:
        motif_hits = scan_sequence_for_motifs(row["sequence"])
        row["motifs_found"] = "; ".join(motif_hits.keys())
        row["has_motif"] = len(motif_hits) > 0

        # Flag TonB-dependent receptors and ABC periplasmic proteins
        # as high-priority novel architecture candidates
        desc_lower = row["description"].lower()
        is_transport = any(kw in desc_lower for kw in
                           ["tonb", "abc transporter", "periplasmic", "receptor",
                            "import", "uptake", "binding protein"])
        row["is_transport_architecture"] = is_transport
        row["architecture_inferred"] = (
            "beta-barrel (TonB-receptor)" if "tonb" in desc_lower
            else "alpha-beta (ABC-transport)" if "abc" in desc_lower
            else "unknown"
        )
        row["is_novel_arch"] = row["architecture_inferred"] != "unknown"

    df = pd.DataFrame(all_rows)
    out = DATA_DIR / "mll_cluster_hits.csv"
    df.to_csv(out, index=False)
    log.info(f"Strategy C: {len(df)} proteins → {out}")

    novel = df[df.get("is_novel_arch", False)].shape[0] if not df.empty else 0
    log.info(f"  Transport/receptor architecture candidates: {novel}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def combine_homologs(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all homolog DataFrames, deduplicate by sequence, and output.
    Adds a unified `protein_id` and `architecture_class` column.
    """
    combined_rows = []
    for df in dfs:
        if df.empty:
            continue
        # Normalize column names across strategies
        for _, row in df.iterrows():
            seq = str(row.get("sequence", ""))
            if not seq or seq == "nan" or len(seq) < 10:
                continue
            combined_rows.append({
                "protein_id":          row.get("uniprot_acc") or row.get("ncbi_acc") or "",
                "gene_names":          row.get("gene_names", ""),
                "organism":            row.get("organism", ""),
                "sequence":            seq,
                "seq_len":             len(seq),
                "motifs_found":        row.get("motifs_found", ""),
                "architecture_class":  row.get("architecture_inferred", row.get("architecture_class", "unknown")),
                "is_ef_hand":          row.get("is_ef_hand", False),
                "is_novel_arch":       row.get("is_novel_arch", False),
                "binding_positions":   row.get("binding_positions", ""),
                "source":              row.get("source", ""),
                "query_label":         row.get("query_label", ""),
                "has_pdb":             row.get("has_pdb", False),
            })

    if not combined_rows:
        return pd.DataFrame()

    combined = pd.DataFrame(combined_rows)

    # Deduplicate by exact sequence
    before = len(combined)
    combined = combined.drop_duplicates(subset="sequence")
    log.info(f"Deduplication by sequence: {before} → {len(combined)} unique")

    out = DATA_DIR / "all_homologs.csv"
    combined.to_csv(out, index=False)
    log.info(f"Combined homologs: {len(combined)} proteins → {out}")

    # Summary
    arch_counts = combined["architecture_class"].value_counts()
    print("\nCombined homolog architecture summary:")
    print(arch_counts.to_string())
    print(f"\nTotal novel (non-EF-hand) architectures: {combined['is_novel_arch'].sum()}")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    df_A = run_strategy_A()
    df_B = run_strategy_B(df_A)
    df_C = run_strategy_C()
    combine_homologs(df_A, df_B, df_C)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REE Homolog Finder")
    parser.add_argument("--strategy", default="all",
                        choices=["A", "B", "C", "all"],
                        help="Which strategy to run (default: all)")
    args = parser.parse_args()

    df_A = df_B = df_C = pd.DataFrame()

    if args.strategy in ("A", "all"):
        df_A = run_strategy_A()
    elif (DATA_DIR / "uniprot_hits.csv").exists():
        df_A = pd.read_csv(DATA_DIR / "uniprot_hits.csv")

    if args.strategy in ("B", "all"):
        df_B = run_strategy_B(df_A)

    if args.strategy in ("C", "all"):
        df_C = run_strategy_C()

    if args.strategy == "all":
        combine_homologs(df_A, df_B, df_C)
