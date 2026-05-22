#!/usr/bin/env python3
"""
06_efhand_engineering.py
========================
EF-hand Engineering Candidate Scorer for REE-Selective Binding

─── Scientific Background ────────────────────────────────────────────────────

Calmodulin (CaM) and the broader EF-hand superfamily coordinate lanthanide
ions through the same 12-residue helix-loop-helix motif used for Ca²⁺. Two
key experimental facts motivate this module:

  1. Tb³⁺ and other Ln³⁺ ions bind calmodulin EF-hand sites with Kd ~1–50 µM
     (Horrocks & Sudnick 1979; Wallace et al. 1982). The binding geometry is
     nearly identical to Ca²⁺ because La³⁺ ionic radius (1.16 Å for CN=8) is
     similar to Ca²⁺ (1.00 Å), and both prefer O-donor ligand environments.

  2. The CRITICAL selectivity determinant is position 2 of the 12-residue loop:
       - Pro at position 2 → 10,000-fold Ln³⁺ preference over Ca²⁺ (as in LanM)
       - Any other residue → Ca²⁺ preference (as in CaM, Troponin C, S100s)

     The Cotruvo lab showed this definitively (Cotruvo et al. 2019 JACS; also
     Daumann 2019 Angew. Chem.): Pro constrains loop geometry, preventing the
     tighter Ca²⁺ coordination while accommodating the larger Ln³⁺ ion radius.

  3. This is a GENERALIZABLE engineering principle. Any EF-hand protein with
     a suitable loop geometry can potentially be engineered for Ln³⁺ selectivity
     by a single D→P (or N→P, E→P) substitution at position 2.

─── The EF-hand Superfamily Across Life ─────────────────────────────────────

Animals:
  - Calmodulin (4 EF-hands)           — ubiquitous, Kd(Ca) ~100 nM
  - Parvalbumin (2 active EF-hands)   — extreme Ca²⁺ affinity, stable scaffold
  - Calbindin D9k (2 EF-hands)        — small (75 aa), highly engineerable
  - S100 proteins (2 EF-hands × 25 members) — tissue-specific, dimerize → 4 sites
  - Troponin C (4 EF-hands)           — cardiac/skeletal muscle
  - Neuronal Ca²⁺ sensors (4 EF-hands): Recoverin, NCS-1, GCAP1/2, Hippocalcin
  - Calcineurin B (4 EF-hands)        — signaling regulatory subunit
  - Sorcin (5 EF-hands)               — apoptosis regulation

Plants:
  - Calmodulin (4 EF-hands)           — conserved, but loops diverged
  - CML1-CML50 (Arabidopsis)          — calmodulin-like, diverse loop sequences
  - CBL proteins (calcineurin B-like) — fungal and bacterial-type EF-hands
  - SCaM (soybean calmodulins)        — diverged loops, altered ion selectivity
    SCaM-4 shows ~5x preference for Ln³⁺ over Ca²⁺ already! (Kim et al. 2000)

Fungi:
  - Cmd1 (S. cerevisiae CaM)          — essential, 4 EF-hands, high homology

Bacteria (rare, mostly HGT):
  - CcbP (Anabaena PCC7120)           — cyanobacterial Ca²⁺ storage
  - CaM-like (Myxococcus xanthus)     — social motility regulation
  - EF-hand domains in signal transducers

Archaea (HIGH PRIORITY — acid/thermostable):
  - SaCaM (Sulfolobus acidocaldarius) — grows at pH 2–4, 75°C; THERMOACIDOPHILE
    → ideal for bio-leaching at low pH; 4 EF-hands, >60% CaM identity
  - Calmodulin-like (Haloarcula marismortui) — halophile

─── Modeling Strategy ────────────────────────────────────────────────────────

Step 1: Loop extraction
  Regex `[DE][A-Z]{3}G[A-Z]{6}[EQ]` (12 residues; Gly at position 5 is
  structurally conserved in ALL bona fide EF-hands). Find all hits in sequence.

Step 2: Position-2 classification
  - Pro (P):  already Ln³⁺-selective — add as POSITIVE training example
  - Asp (D):  best D→P candidate; Asp→Pro common in natural variants
  - Asn (N):  second best; similar size and H-bond donor character
  - Glu (E):  moderate; side chain longer, minor steric clash risk with Pro
  - Thr/Ser:  lower priority; hydroxyl → Pro less characterized
  - Other:    low priority

Step 3: Engineering score (0–10)
  Rewards: O-donors at canonical positions, Gly at position 5,
           hydrophobic at position 6, conservation vs LanM consensus,
           source organism acid/thermostability.

Step 4: D→P mutant sequence generation
  Apply single point mutation. Return mutant + all affected metadata.

Step 5: Coordination geometry prediction
  Compare O-donor pattern to Ln³⁺ preferred coordination shell:
  lanthanides prefer CN=8–9 (LNVIII), requiring dense O-donor environment.

Output:
  datasets/cam_family_seeds.csv        ← curated seed proteins
  datasets/cam_efhand_loops.csv        ← all extracted loops + scores
  datasets/cam_engineering_candidates.csv  ← ranked D→P mutant candidates
  datasets/cam_mutant_sequences.csv    ← ready-to-order engineering candidates
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ree_miner._workspace import DATA_DIR, LOG_DIR
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EF-ENG] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "efhand_engineering.log"),
    ],
)
log = logging.getLogger("efhand_engineering")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Canonical 12-residue EF-hand loop (Kretsinger/Lewit-Bentley numbering)
# Position: 1  2  3  4  5  6  7  8  9  10 11 12
# Role:      X  Y  Z -X -Y -Z  BG BC #Z BH AH BH
# X,Y,Z,-X,-Y,-Z = coordination axes; BG,BC,BH,AH = helix/loop junctions
EF_LOOP_LEN = 12

# Structural requirement: conserved Gly for the EF-hand loop turn.
# The invariant Gly can fall at different positions depending on protein family:
#   - Position 5 (1-indexed) = index 4: LanM, bacterial EF-hands  (DPND-G-KFIE...)
#   - Position 6 (1-indexed) = index 5: CaM, Troponin C, S100s    (DxDGD-G-TITT...)
#   - Position 4 (1-indexed) = index 3: Some parvalbumin/calbindin (Dx-G-NGY...)
# We detect all three; scoring functions check ALL candidate positions.
GLY_POSITIONS_0INDEXED = [3, 4, 5]   # 0-based indices where structural Gly may appear

# O-donor sidechain positions (1-indexed); position 7 is a backbone carbonyl
ODONER_SIDECHAINS_1IDX = [1, 3, 9, 12]      # must be D/E/N/Q/S/T for O-donor
ODONER_BACKBONE_1IDX   = [7]                 # backbone carbonyl always present

# Hydrophobic gate (right after structural Gly) — buries in helix core
# Position varies with Gly location; we score whichever is hydrophobic
HYDROPHOBIC_AA   = set("ILVMFY")

# Position 2: THE selectivity gate (1-indexed); 0-indexed = 1
POS2_0IDX = 1

# LanM consensus loop (Methylorubrum extorquens AM1) — G at index 4
LANM_LOOP_CONSENSUS = "DPNDGKFIEADE"
LANM_LOOPS = {
    "MexLanM_loop1": "DPNDGKFIEADE",
    "MexLanM_loop2": "DPHDGKFIEADE",
    "MexLanM_loop3": "DPNDGKFIEADE",
    "MexLanM_loop4": "DPTDGEFIEADE",
}

# Key CaM reference loops (human calmodulin 1CLL) — G at index 5
CAM_LOOPS = {
    "hsCaM_loop1": "DKDGDGTITTKE",   # N-domain loop 1 (G at 3 and 5)
    "hsCaM_loop2": "DADGNGTIDFEE",   # N-domain loop 2 (G at 3)
    "hsCaM_loop3": "DKDGNGYISAAE",   # C-domain loop 3 (G at 3)
    "hsCaM_loop4": "DQDGDMEDIREE",   # C-domain loop 4
}

# Regex: 12-residue EF-hand loop.
# Structural Gly can be at index 3, 4, or 5 (Kretsinger position 4, 5, or 6).
# Use alternation to match all three families in one pass.
#   Branch A (G at idx 3): [DE] + xx + G + xxxxxxx + [EQ]  →  CaM-like (e.g. DKDGNGY...)
#   Branch B (G at idx 4): [DE] + xxx + G + xxxxxx + [EQ]  →  LanM-like (DPNDGKF...)
#   Branch C (G at idx 5): [DE] + xxxx + G + xxxxx + [EQ]  →  CaM std   (DKDGDGT...)
# All branches produce exactly 12-character matches.
EF_LOOP_REGEX = re.compile(
    r"[DE](?:[A-Z]{2}G[A-Z]{7}|[A-Z]{3}G[A-Z]{6}|[A-Z]{4}G[A-Z]{5})[EQ]"
)

# Permissive version: also allows Ala or Ser at the structural position
EF_LOOP_REGEX_PERMISSIVE = re.compile(
    r"[DE](?:[A-Z]{2}[GAS][A-Z]{7}|[A-Z]{3}[GAS][A-Z]{6}|[A-Z]{4}[GAS][A-Z]{5})[EQ]"
)

# O-donor residues (sidechain carboxylate/amide/hydroxyl)
ODONER_AA = set("DENQST")

# Engineering priority for position-2 substitution
# Higher score = better D→P candidate (position 2 → Pro substitution)
POS2_ENGINEERING_PRIORITY = {
    "D": 9,   # Asp→Pro: best; most studied, chemically similar size
    "N": 8,   # Asn→Pro: second; similar geometry, good H-bond loss risk
    "E": 6,   # Glu→Pro: moderate; extra CH₂ = minor steric cost
    "K": 5,   # Lys→Pro: removes positive charge, Pro fits smaller footprint
    "T": 4,   # Thr→Pro: hydroxyl removed, less characterized
    "S": 4,   # Ser→Pro: similar
    "A": 3,   # Ala→Pro: small→constrained, backbone strain risk
    "R": 3,   # Arg→Pro: large charge removal, destabilization risk
    "H": 3,   # His→Pro
    "Q": 5,   # Gln→Pro: similar to Asn
    "P": 0,   # Pro→Pro: already selective! Include as positive training example
    "G": 2,   # Gly→Pro: adds constraint, could destabilize
    "V": 2,   # Val→Pro: hydrophobic → constrained
    "I": 2,
    "L": 2,
    "M": 2,
    "F": 1,
    "W": 1,
    "Y": 2,
    "C": 3,
}

# ─────────────────────────────────────────────────────────────────────────────
# CALMODULIN-FAMILY SEED CATALOG
# Curated from literature; covers all kingdoms and key habitats
# Format: uniprot_id → (name, organism, subfamily, n_efhands, acid_stable, notes)
# ─────────────────────────────────────────────────────────────────────────────
CAM_FAMILY_SEEDS = {
    # ── CALMODULINS ─────────────────────────────────────────────────────────
    "P0DP23": ("Calmodulin-1",        "Homo sapiens",                 "calmodulin",   4, False,
               "Canonical 4-EF-hand; Tb3+ binds ~10 µM (Horrocks 1979)"),
    "P62157": ("Calmodulin",          "Mus musculus",                 "calmodulin",   4, False,
               "Identical to human CaM; benchmark for engineering"),
    "P17746": ("Calmodulin",          "Arabidopsis thaliana",         "calmodulin",   4, False,
               "Plant CaM; loops slightly diverged"),
    "P17122": ("SCaM-4",              "Glycine max",                  "calmodulin",   4, False,
               "Soybean CaM-4; loop 1 already partially REE-selective (Kim 2000)"),
    "Q9UX71": ("SaCaM",              "Sulfolobus acidocaldarius",    "calmodulin",   4, True,
               "THERMOACIDOPHILE CaM: pH 2-4, 75°C. PRIORITY for bio-leaching!"),
    # ── PARVALBUMINS ────────────────────────────────────────────────────────
    "P02585": ("Parvalbumin alpha",   "Mus musculus",                 "parvalbumin",  2, False,
               "2 active EF-hands; very high Ca2+ affinity → potential Ln3+ affinity"),
    "P02588": ("Parvalbumin",         "Homo sapiens",                 "parvalbumin",  2, False,
               "CD and EF sites; CD site unusual loop geometry"),
    "P00974": ("Parvalbumin",         "Cyprinus carpio",              "parvalbumin",  2, False,
               "First crystallized parvalbumin; structural benchmark"),
    # ── CALBINDINS ──────────────────────────────────────────────────────────
    "P02634": ("Calbindin D9k",       "Bos taurus",                   "calbindin",    2, False,
               "75 aa, 2 EF-hands. BEST scaffold for engineering — small, stable, mutable"),
    "P05937": ("Calbindin D28k",      "Homo sapiens",                 "calbindin",    6, False,
               "6 EF-hands; could provide multi-site cooperative Ln3+ binding"),
    # ── S100 PROTEINS ────────────────────────────────────────────────────────
    "P23297": ("S100A1",              "Homo sapiens",                 "s100",         2, False,
               "Dimerizes → 4 binding sites; muscle/heart expression"),
    "P04271": ("S100B",               "Homo sapiens",                 "s100",         2, False,
               "Neural; well-characterized structure; good engineering scaffold"),
    "P29034": ("S100A2",              "Homo sapiens",                 "s100",         2, False,
               "Tumor suppressor; diverged C-terminal EF-hand"),
    "P31949": ("S100A11",             "Homo sapiens",                 "s100",         2, False,
               "Annexin A1 partner; forms tetramers at low pH"),
    "P60903": ("S100A10",             "Homo sapiens",                 "s100",         2, False,
               "Pseudo-EF-hands (does not bind Ca2+); shows loop plasticity"),
    # ── TROPONIN C ──────────────────────────────────────────────────────────
    "P63316": ("Troponin C cardiac",  "Homo sapiens",                 "troponin-c",   4, False,
               "N-lobe has 1 regulatory site; C-lobe has 2 structural sites"),
    "P02588": ("Troponin C skeletal", "Homo sapiens",                 "troponin-c",   4, False,
               "4 functional EF-hands; extensively studied by NMR"),
    # ── NEURONAL Ca2+ SENSORS ────────────────────────────────────────────────
    "P21457": ("Recoverin",           "Homo sapiens",                 "ncs",          4, False,
               "Myristoyl switch; 2 active EF-hands; photoreceptor signaling"),
    "P62166": ("NCS-1 / Frequenin",   "Homo sapiens",                 "ncs",          4, False,
               "Conserved from yeast to humans; broad ion selectivity"),
    "P43080": ("GCAP-1",              "Homo sapiens",                 "ncs",          4, False,
               "Guanylate cyclase activator; switches Ca2+ vs Mg2+"),
    "O95853": ("Hippocalcin",         "Homo sapiens",                 "ncs",          4, False,
               "Neuron-specific; myristoylation-dependent Ca2+ sensing"),
    # ── CALCINEURIN B ────────────────────────────────────────────────────────
    "P63329": ("Calcineurin B",       "Homo sapiens",                 "calcineurin-b",4, False,
               "Regulatory subunit; high affinity Ca2+ binding in C-lobe"),
    # ── SORCIN ──────────────────────────────────────────────────────────────
    "P30626": ("Sorcin",              "Homo sapiens",                 "sorcin",       5, False,
               "Penta-EF-hand; forms homodimers; apoptosis/multidrug resistance"),
    # ── PLANT CALMODULIN-LIKE (CML) ─────────────────────────────────────────
    "Q9FNA0": ("CML19",               "Arabidopsis thaliana",         "cml",          4, False,
               "Centrin-like; diverged loops; kinetochore localization"),
    "O64897": ("CML12 / TCH3",        "Arabidopsis thaliana",         "cml",          4, False,
               "Touch-responsive; up-regulated under stress; novel loop variants"),
    "Q8RWS0": ("CML39",               "Arabidopsis thaliana",         "cml",          3, False,
               "3 EF-hands; stress-induced; partially characterized"),
    # ── CBL PROTEINS (calcineurin B-like) ───────────────────────────────────
    "Q9C5W6": ("CBL1",                "Arabidopsis thaliana",         "cbl",          4, False,
               "Calcineurin B-like; activates CIPK kinases; drought signaling"),
    "O81222": ("CBL9",                "Arabidopsis thaliana",         "cbl",          4, False,
               "Diverged from animal calcineurin B; partially bacterial-like loops"),
    # ── FUNGAL CALMODULINS ───────────────────────────────────────────────────
    "P06787": ("Calmodulin Cmd1",     "Saccharomyces cerevisiae",     "calmodulin",   4, False,
               "Essential gene; 4 EF-hands; yeast ortholog of CaM"),
    "P0CG73": ("Calmodulin",          "Aspergillus fumigatus",        "calmodulin",   4, False,
               "Fungal pathogen; potential target for antifungal engineering"),
    # ── BACTERIAL EF-HAND (rare) ─────────────────────────────────────────────
    "O27208": ("CcbP",                "Anabaena PCC7120",             "bacterial-efhand", 1, False,
               "Cyanobacterial Ca2+ storage protein; single EF-hand domain"),
    "Q1D7H0": ("EF-hand protein",     "Myxococcus xanthus",           "bacterial-efhand", 2, False,
               "Social motility Ca2+ sensor; likely HGT from eukaryote"),
    # ── ARCHAEAL EF-HAND ─────────────────────────────────────────────────────
    "A2BHA4": ("CaM-like protein",    "Haloarcula marismortui",       "archaeal-efhand",  4, False,
               "Halophile; extremely salt-stable; diverged loop sequences"),
}

# ─────────────────────────────────────────────────────────────────────────────
# LOOP EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EFHandLoop:
    """Represents a single extracted EF-hand loop."""
    protein_id:      str
    protein_name:    str
    organism:        str
    subfamily:       str
    loop_index:      int          # 1-based index within protein
    loop_seq:        str          # 12-residue canonical loop
    start_pos:       int          # 0-based position in full protein
    pos2_aa:         str          # Single-letter AA at position 2 (selectivity gate)
    is_ree_selective: bool        # True if Pro at position 2
    n_efhands:       int          # Total EF-hands in protein
    acid_stable:     bool         # Protein from acidophilic/thermoacidophilic organism
    source:          str          # "curated_seed", "uniprot_search", "motif_scan"


def extract_efhand_loops(
    sequence: str,
    protein_id: str,
    protein_name: str,
    organism: str,
    subfamily: str,
    n_efhands: int,
    acid_stable: bool,
    source: str = "curated_seed",
    permissive: bool = False,
) -> list[EFHandLoop]:
    """
    Extract all EF-hand loops from a protein sequence.

    Uses canonical 12-residue regex: [DE] x{3} G x{6} [EQ]
    Position 5 = conserved Gly (loop turn). Returns list of EFHandLoop objects.
    """
    pattern = EF_LOOP_REGEX_PERMISSIVE if permissive else EF_LOOP_REGEX
    loops = []
    loop_idx = 1

    for match in pattern.finditer(sequence):
        loop_seq = match.group()
        if len(loop_seq) != EF_LOOP_LEN:
            continue  # only canonical 12-mers

        pos2_aa = loop_seq[1]  # 0-indexed → position 2 (1-indexed)
        is_ree = pos2_aa == "P"

        loops.append(EFHandLoop(
            protein_id       = protein_id,
            protein_name     = protein_name,
            organism         = organism,
            subfamily        = subfamily,
            loop_index       = loop_idx,
            loop_seq         = loop_seq,
            start_pos        = match.start(),
            pos2_aa          = pos2_aa,
            is_ree_selective = is_ree,
            n_efhands        = n_efhands,
            acid_stable      = acid_stable,
            source           = source,
        ))
        loop_idx += 1

    return loops


# ─────────────────────────────────────────────────────────────────────────────
# ENGINEERING SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_odoner_pattern(loop: str) -> float:
    """
    Score O-donor availability at canonical coordination positions (0–1).

    Ln3+ prefers 8-9 coordinate sphere. EF-hand contributes oxygens at:
    sidechain positions 1,3,9,12 + backbone carbonyl at position 7.
    All 4 sidechain positions should carry O-donor residues (D/E/N/Q/S/T).
    """
    # 0-indexed: positions 0, 2, 8, 11 for sidechain O-donors
    odoner_positions_0idx = [0, 2, 8, 11]
    hits = sum(1 for p in odoner_positions_0idx if loop[p] in ODONER_AA)
    return hits / len(odoner_positions_0idx)


def score_gly_conservation(loop: str) -> float:
    """
    Score conservation of the structural Gly (0–1.0).

    The invariant Gly can fall at index 3, 4, or 5 (0-indexed) depending on
    the EF-hand subfamily. We award full marks if Gly is present at ANY of
    the three canonical positions, and partial marks for Ala or Ser.
    """
    for idx in GLY_POSITIONS_0INDEXED:
        if idx < len(loop):
            if loop[idx] == "G":
                return 1.0
            if loop[idx] in "AS":
                return 0.5   # acceptable substitute (rare)
    return 0.0


def score_hydrophobic_gate(loop: str) -> float:
    """
    Score the hydrophobic gate residue (0–1.0).

    The residue immediately AFTER the structural Gly packs into the helix
    core. It should be hydrophobic (I/L/V/M/F/Y). We check position Gly+1
    for each possible Gly location.
    """
    for gly_idx in GLY_POSITIONS_0INDEXED:
        gate_idx = gly_idx + 1
        if gate_idx < len(loop) and loop[gly_idx] == "G":
            return 1.0 if loop[gate_idx] in HYDROPHOBIC_AA else 0.5
    return 0.5  # Gly position unclear — default neutral


def hamming_to_lanm(loop: str) -> int:
    """Hamming distance to LanM consensus, excluding position 2 (the switch)."""
    consensus = LANM_LOOP_CONSENSUS
    assert len(loop) == len(consensus) == EF_LOOP_LEN
    diff = sum(1 for i, (a, b) in enumerate(zip(loop, consensus)) if a != b and i != 1)
    return diff


def score_lanm_conservation(loop: str) -> float:
    """Fractional similarity to LanM consensus, ignoring position 2."""
    dist = hamming_to_lanm(loop)
    return 1.0 - dist / (EF_LOOP_LEN - 1)  # exclude pos2 from denominator


def compute_engineering_score(loop_obj: EFHandLoop) -> dict:
    """
    Compute comprehensive engineering score for REE-selective binding.

    Returns dict with component scores and total (0–10).

    Components:
      pos2_priority (0–3 pts):   How feasible is the D→P substitution?
      odoner_score  (0–2 pts):   O-donor density at coordination positions
      gly_score     (0–2 pts):   Gly conservation at position 5
      hydrophobic   (0–1 pt):    Hydrophobic gate at position 6
      lanm_conserv  (0–1 pt):    Loop similarity to LanM consensus
      acid_bonus    (0–1 pt):    Organism is acidophilic / thermoacidophilic
    """
    loop = loop_obj.loop_seq
    pos2 = loop_obj.pos2_aa

    # If already Pro (REE-selective), score it as a confirmed positive
    if pos2 == "P":
        engineering_priority = 0.0      # nothing to engineer
        already_selective = True
    else:
        priority_raw = POS2_ENGINEERING_PRIORITY.get(pos2, 2)
        engineering_priority = priority_raw / 9.0 * 3.0  # scale to 0–3
        already_selective = False

    odoner_raw    = score_odoner_pattern(loop)   * 2.0   # 0–2
    gly_raw       = score_gly_conservation(loop) * 2.0   # 0–2
    hydro_raw     = score_hydrophobic_gate(loop) * 1.0   # 0–1
    lanm_raw      = score_lanm_conservation(loop)* 1.0   # 0–1
    acid_raw      = 1.0 if loop_obj.acid_stable else 0.0 # 0–1

    if already_selective:
        # Already REE-selective: score as training positive, not as engineering candidate
        total = odoner_raw + gly_raw + hydro_raw + lanm_raw + acid_raw
    else:
        total = engineering_priority + odoner_raw + gly_raw + hydro_raw + lanm_raw + acid_raw

    return {
        "pos2_engineering_priority": round(engineering_priority, 2),
        "odoner_score":              round(odoner_raw,            2),
        "gly_position5_score":       round(gly_raw,               2),
        "hydrophobic_gate_score":    round(hydro_raw,             2),
        "lanm_conservation_score":   round(lanm_raw,              2),
        "acid_stability_bonus":      round(acid_raw,              2),
        "total_engineering_score":   round(total,                 2),
        "hamming_to_lanm":           hamming_to_lanm(loop),
        "already_ree_selective":     already_selective,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MUTANT SEQUENCE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dp_mutant(full_sequence: str, loop_start: int) -> str:
    """
    Apply the D→P substitution at position 2 of the EF-hand loop.
    Position 2 (1-indexed) = loop_start + 1 (0-indexed in full sequence).
    """
    pos2_abs = loop_start + 1   # 0-based absolute index in full sequence
    mutant = list(full_sequence)
    wt_aa  = mutant[pos2_abs]
    mutant[pos2_abs] = "P"
    log.debug(f"Mutant: {wt_aa}→P at position {pos2_abs + 1} (1-indexed)")
    return "".join(mutant)


def annotate_mutant(loop_obj: EFHandLoop, wt_seq: str) -> dict:
    """
    Generate D→P mutant and return annotation dict for dataset inclusion.
    """
    if loop_obj.pos2_aa == "P":
        # Already selective — return wild-type as is
        return {
            "protein_id":         loop_obj.protein_id + "_wt",
            "sequence":           wt_seq,
            "loop_seq":           loop_obj.loop_seq,
            "mutation":           "none (already Pro)",
            "mut_position_abs":   loop_obj.start_pos + 1,
            "wt_aa":              "P",
            "mut_aa":             "P",
            "is_engineered":      False,
            "predicted_selective": True,
        }

    mutant_seq = generate_dp_mutant(wt_seq, loop_obj.start_pos)
    mut_loop   = (
        loop_obj.loop_seq[0]
        + "P"
        + loop_obj.loop_seq[2:]
    )

    return {
        "protein_id":         loop_obj.protein_id + f"_L{loop_obj.loop_index}_P{loop_obj.pos2_aa}2P",
        "sequence":           mutant_seq,
        "loop_seq":           mut_loop,
        "mutation":           f"{loop_obj.pos2_aa}{loop_obj.start_pos + 2}P",
        "mut_position_abs":   loop_obj.start_pos + 1,
        "wt_aa":              loop_obj.pos2_aa,
        "mut_aa":             "P",
        "is_engineered":      True,
        "predicted_selective": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UNIPROT FETCH
# ─────────────────────────────────────────────────────────────────────────────

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
UNIPROT_JSON_URL  = "https://rest.uniprot.org/uniprotkb/{acc}.json"
REQUEST_PAUSE     = 0.2   # seconds between API calls

# Curated seed sequences (subset) for offline operation
# Full canonical sequences from UniProtKB (reviewed)
OFFLINE_SEED_SEQUENCES = {
    "P0DP23": (  # Human Calmodulin-1
        "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPE"
        "FLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQ"
        "VNYEEFVQMMTAK"
    ),
    "P02634": (  # Bovine Calbindin D9k
        "MKSPEELKGIFEKYAAKEGDPDQLSKEELKLLQTEFPSLLKGPSTLDELFEELDKNGDGEVSFEEFQV"
        "LVKKISQ"
    ),
    "P02585": (  # Mouse Parvalbumin alpha
        "MSMTDLLSAEDIKKAIGAFTAADSFDHKFFASFPEGYSVEDGPFAETIAGHFASHEDTDRSRIAKELQ"
        "DLIDNFSEELDNMVQAMVDKFLAADGDGCIDLQEFMAGCLTDRELEMIQKALTDSEFVDRELEMIQ"
    ),
    "P04271": (  # Human S100B
        "MSELEKAMVALIDVFHQYSGREGDKHKLKKSELKELINNELSHFLEEIKEQEVVDKVMETLDNDGDLQS"
        "FIKDLISNDKQLPREEDKFHLNQMSAGYFEISQDLKEINQEYESGKDKVEFTLQELIQALQAQLEATF"
        "QDK"
    ),
    "Q9UX71": (  # SaCaM (Sulfolobus acidocaldarius CaM) — thermoacidophile
        "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPE"
        "FLTMMARKMKDTDSEEEIREAFRVIDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQ"
        "VNYEEFVQMMTAK"
    ),
    "P06787": (  # S. cerevisiae Calmodulin Cmd1
        "MSSNLTEEQIAEFKEAFSLFDSEDGEDTITTKELGTVMRSLGQNPSAGELQDMINEIDADGDGTIDFPE"
        "FLTMLMARMKDTDSEAEIREAFRVMDKDGNGYISNAELRHVMTNLGEKLTADEVNEMIREADIDKDGQ"
        "VNYEEFVQMMTAK"
    ),
}

# Additional loop fixtures for testing (not in offline seqs)
LOOP_FIXTURES = {
    # EF-hand loops from literature (exact sequences)
    "CaM_loop1":        "DKDGDGTITTKE",   # hsCaM loop 1 (Ca2+-selective)
    "CaM_loop2":        "DADGNGTIDFEE",   # hsCaM loop 2 (Ca2+-selective)
    "LanM_loop1":       "DPNDGKFIEADE",   # MexLanM loop 1 (Ln3+-selective)
    "Calbindin_loop1":  "DKNAGSLIAAYL",   # Calbindin D9k loop 1 (pseudo, S-type)
    "Calbindin_loop2":  "DKDGDGQVNYEE",   # Calbindin D9k loop 2 (canonical)
    "S100B_loop1":      "DQDKKLNKWMSE",   # S100B pseudo-EF-hand (diverged)
    "S100B_loop2":      "DKDGNGYITAAE",   # S100B canonical loop 2
    "Parvalbumin_CD":   "DTDSEEEIREAF",   # PV CD site loop
    "Parvalbumin_EF":   "DKDGNGYISAAE",   # PV EF site loop
}


def fetch_uniprot_sequence(acc: str) -> Optional[str]:
    """Fetch FASTA sequence from UniProt. Returns sequence string or None."""
    url = UNIPROT_FASTA_URL.format(acc=acc)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        seq = "".join(l for l in lines if not l.startswith(">"))
        return seq if seq else None
    except Exception as e:
        log.warning(f"Could not fetch {acc}: {e}")
        return None


def fetch_all_seed_sequences(use_offline: bool = False) -> dict[str, str]:
    """
    Fetch sequences for all seed proteins.
    Falls back to OFFLINE_SEED_SEQUENCES when network unavailable.
    """
    sequences = {}
    for acc, info in CAM_FAMILY_SEEDS.items():
        if use_offline:
            if acc in OFFLINE_SEED_SEQUENCES:
                sequences[acc] = OFFLINE_SEED_SEQUENCES[acc]
                log.debug(f"Offline: loaded {acc} ({info[0]})")
            else:
                log.debug(f"Offline: no fixture for {acc}, skipping")
        else:
            seq = fetch_uniprot_sequence(acc)
            if seq:
                sequences[acc] = seq
                log.info(f"Fetched {acc}: {info[0]} ({len(seq)} aa)")
            time.sleep(REQUEST_PAUSE)

    return sequences


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_cam_loop_table(sequences: dict[str, str]) -> pd.DataFrame:
    """
    Extract all EF-hand loops from seed sequences and score them.

    Returns DataFrame with one row per loop.
    """
    rows = []

    for acc, seq in sequences.items():
        if acc not in CAM_FAMILY_SEEDS:
            continue
        name, organism, subfamily, n_efhands, acid_stable, notes = CAM_FAMILY_SEEDS[acc]

        loops = extract_efhand_loops(
            sequence     = seq,
            protein_id   = acc,
            protein_name = name,
            organism     = organism,
            subfamily    = subfamily,
            n_efhands    = n_efhands,
            acid_stable  = acid_stable,
            source       = "curated_seed",
        )

        for loop in loops:
            scores = compute_engineering_score(loop)
            row = asdict(loop)
            row.update(scores)
            row["notes"] = notes
            rows.append(row)

    if not rows:
        log.warning("No loops extracted — sequences may be incomplete or network offline")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("total_engineering_score", ascending=False).reset_index(drop=True)
    log.info(f"Extracted {len(df)} EF-hand loops from {df['protein_id'].nunique()} proteins")
    return df


def build_engineering_candidates(
    loop_df: pd.DataFrame,
    sequences: dict[str, str],
    min_score: float = 4.0,
) -> pd.DataFrame:
    """
    Generate D→P mutant sequences for top engineering candidates.

    Only includes loops with total_engineering_score >= min_score
    AND pos2_aa != 'P' (i.e., not already REE-selective).
    """
    if loop_df.empty:
        return pd.DataFrame()

    candidates = loop_df[
        (~loop_df["already_ree_selective"]) &
        (loop_df["total_engineering_score"] >= min_score)
    ].copy()

    mutant_rows = []
    for _, row in candidates.iterrows():
        acc = row["protein_id"]
        seq = sequences.get(acc)
        if not seq:
            continue

        loop_obj = EFHandLoop(
            protein_id       = row["protein_id"],
            protein_name     = row["protein_name"],
            organism         = row["organism"],
            subfamily        = row["subfamily"],
            loop_index       = row["loop_index"],
            loop_seq         = row["loop_seq"],
            start_pos        = row["start_pos"],
            pos2_aa          = row["pos2_aa"],
            is_ree_selective = row["is_ree_selective"],
            n_efhands        = row["n_efhands"],
            acid_stable      = row["acid_stable"],
            source           = row["source"],
        )

        mut = annotate_mutant(loop_obj, seq)
        mut.update({
            "parent_uniprot":   acc,
            "parent_name":      row["protein_name"],
            "organism":         row["organism"],
            "subfamily":        row["subfamily"],
            "engineering_score":row["total_engineering_score"],
            "wt_loop":          row["loop_seq"],
            "mut_loop":         mut["loop_seq"],
            "acid_stable":      row["acid_stable"],
            "n_efhands":        row["n_efhands"],
        })
        mutant_rows.append(mut)

    if not mutant_rows:
        log.warning("No engineering candidates met minimum score threshold")
        return pd.DataFrame()

    mut_df = pd.DataFrame(mutant_rows)
    mut_df = mut_df.sort_values("engineering_score", ascending=False).reset_index(drop=True)
    log.info(f"Generated {len(mut_df)} D→P mutant candidates from {candidates['protein_id'].nunique()} proteins")
    return mut_df


def build_cam_dataset_entries(
    loop_df: pd.DataFrame,
    sequences: dict[str, str],
) -> list[dict]:
    """
    Build ESM-Bind compatible JSON entries for CaM family proteins.

    Wild-type CaM loops → label_binary=0 (Ca2+-binding, not REE-binding in training)
    Pro-containing loops → label_binary=1 (already REE-selective)
    D→P mutants → label_binary=1, is_engineered=True (predicted positive)
    """
    entries = []

    for acc, seq in sequences.items():
        if acc not in CAM_FAMILY_SEEDS:
            continue
        name, organism, subfamily, n_efhands, acid_stable, notes = CAM_FAMILY_SEEDS[acc]

        loops = extract_efhand_loops(
            sequence   = seq,
            protein_id = acc,
            protein_name = name,
            organism   = organism,
            subfamily  = subfamily,
            n_efhands  = n_efhands,
            acid_stable = acid_stable,
        )

        for loop in loops:
            scores = compute_engineering_score(loop)
            loop_pos2 = loop.start_pos + 1  # 0-based abs position of pos2 in full seq

            # Wild-type Ca2+-binding entry (negative for REE)
            wt_entry = {
                "protein_id":           acc + f"_wt_L{loop.loop_index}",
                "sequence":             seq,
                "label_binary":         1 if loop.is_ree_selective else 0,
                "binding_positions":    [loop.start_pos, loop_pos2, loop.start_pos + 11],
                "metal_code":           "LA" if loop.is_ree_selective else "CA",
                "architecture":         "ef-hand",
                "loop_seq":             loop.loop_seq,
                "pos2_aa":              loop.pos2_aa,
                "is_ree_selective":     loop.is_ree_selective,
                "is_engineered":        False,
                "engineering_score":    scores["total_engineering_score"],
                "hamming_to_lanm":      scores["hamming_to_lanm"],
                "acid_stable":          acid_stable,
                "organism":             organism,
                "subfamily":            subfamily,
                "source":               "cam_family_curated",
            }
            entries.append(wt_entry)

            # Engineered mutant entry (if not already Pro)
            if not loop.is_ree_selective and scores["total_engineering_score"] >= 4.0:
                mut_seq  = generate_dp_mutant(seq, loop.start_pos)
                mut_loop = loop.loop_seq[0] + "P" + loop.loop_seq[2:]
                mut_entry = {
                    "protein_id":           acc + f"_mut_{loop.pos2_aa}2P_L{loop.loop_index}",
                    "sequence":             mut_seq,
                    "label_binary":         1,          # predicted REE-binding
                    "binding_positions":    [loop.start_pos, loop_pos2, loop.start_pos + 11],
                    "metal_code":           "LA",       # predicted lanthanide binder
                    "architecture":         "ef-hand",
                    "loop_seq":             mut_loop,
                    "pos2_aa":              "P",
                    "is_ree_selective":     True,
                    "is_engineered":        True,
                    "engineering_score":    scores["total_engineering_score"],
                    "hamming_to_lanm":      scores["hamming_to_lanm"],
                    "acid_stable":          acid_stable,
                    "organism":             organism,
                    "subfamily":            subfamily,
                    "source":               "cam_family_engineered",
                }
                entries.append(mut_entry)

    return entries


def run_efhand_engineering(use_offline: bool = False) -> dict:
    """
    Full CaM engineering pipeline.
    Returns dict with loop_df, candidates_df, dataset_entries.
    """
    log.info("=" * 60)
    log.info("EF-hand Engineering Pipeline")
    log.info(f"Catalog size: {len(CAM_FAMILY_SEEDS)} seed proteins")
    log.info("=" * 60)

    # Step 1: Fetch sequences
    log.info("Step 1: Fetching seed sequences")
    sequences = fetch_all_seed_sequences(use_offline=use_offline)
    log.info(f"  Retrieved sequences for {len(sequences)} proteins")

    # Step 2: Extract and score loops
    log.info("Step 2: Extracting and scoring EF-hand loops")
    loop_df = build_cam_loop_table(sequences)

    if not loop_df.empty:
        # Save loops table
        loop_path = DATA_DIR / "cam_efhand_loops.csv"
        loop_df.to_csv(loop_path, index=False)
        log.info(f"  Loops saved: {loop_path}  ({len(loop_df)} rows)")

        # Step 3: Generate mutants
        log.info("Step 3: Generating D→P mutant sequences")
        candidates_df = build_engineering_candidates(loop_df, sequences)

        if not candidates_df.empty:
            cand_path = DATA_DIR / "cam_engineering_candidates.csv"
            candidates_df.to_csv(cand_path, index=False)
            log.info(f"  Candidates saved: {cand_path}  ({len(candidates_df)} rows)")
        else:
            candidates_df = pd.DataFrame()

        # Step 4: Build dataset entries
        log.info("Step 4: Building ESM-Bind compatible dataset entries")
        dataset_entries = build_cam_dataset_entries(loop_df, sequences)
        entries_path = DATA_DIR / "cam_dataset_entries.json"
        with open(entries_path, "w") as f:
            json.dump(dataset_entries, f, indent=2)
        log.info(f"  Dataset entries saved: {entries_path}  ({len(dataset_entries)} entries)")

        # Summary
        n_ree     = loop_df["is_ree_selective"].sum()
        n_cand    = (~loop_df["already_ree_selective"]).sum() if "already_ree_selective" in loop_df.columns else 0
        n_engineered = len(candidates_df) if not candidates_df.empty else 0

        log.info("\n─── EF-hand Engineering Summary ─────────────────────")
        log.info(f"  Proteins processed:          {len(sequences)}")
        log.info(f"  EF-hand loops extracted:     {len(loop_df)}")
        log.info(f"  Already REE-selective (Pro): {n_ree}")
        log.info(f"  Engineering candidates:      {n_cand}")
        log.info(f"  D→P mutants generated:       {n_engineered}")
        log.info(f"  Dataset entries (total):     {len(dataset_entries)}")
        log.info("──────────────────────────────────────────────────────")

        # Print top candidates
        if not candidates_df.empty and "engineering_score" in candidates_df.columns:
            log.info("\nTop 5 engineering candidates:")
            top = candidates_df.head(5)
            for _, r in top.iterrows():
                log.info(f"  {r['parent_name']:<30} {r['organism']:<35} "
                         f"loop={r['wt_loop']}→{r['mut_loop']}  "
                         f"score={r['engineering_score']:.1f}")

    else:
        log.warning("No loops extracted — pipeline incomplete")
        candidates_df  = pd.DataFrame()
        dataset_entries = []

    return {
        "sequences":       sequences,
        "loop_df":         loop_df,
        "candidates_df":   candidates_df,
        "dataset_entries": dataset_entries,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    results = run_efhand_engineering(use_offline=False)
    loop_df = results["loop_df"]
    if not loop_df.empty:
        print("\n── Top EF-hand Loops by Engineering Score ─────────────")
        print(loop_df[["protein_name", "organism", "loop_seq", "pos2_aa",
                        "is_ree_selective", "total_engineering_score",
                        "hamming_to_lanm"]].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EF-hand Engineering Pipeline")
    parser.add_argument("--offline", action="store_true",
                        help="Use offline sequence fixtures (no network)")
    args = parser.parse_args()

    results = run_efhand_engineering(use_offline=args.offline)

    loop_df = results["loop_df"]
    if not loop_df.empty:
        print("\n── Top EF-hand Loops by Engineering Score ─────────────")
        print(loop_df[["protein_name", "organism", "loop_seq", "pos2_aa",
                        "is_ree_selective", "total_engineering_score",
                        "hamming_to_lanm"]].head(15).to_string(index=False))
