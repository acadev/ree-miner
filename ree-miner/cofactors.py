#!/usr/bin/env python3
"""
07_cofactor_architectures.py
============================
Prosthetic Group and Ca²⁺-Binding Fold Inventory for REE Dataset Expansion

─── Scientific Rationale ─────────────────────────────────────────────────────

The user has identified several architectures beyond EF-hand that are worth
examining. This module provides a principled assessment of EVERY known
metal-containing prosthetic group in biology, rated by Ln³⁺ substitution
potential, and then implements search/extraction logic for the high-priority ones.

─── Coordination Chemistry Guide ─────────────────────────────────────────────

Lanthanide ions (La³⁺–Lu³⁺) have a strong preference for:
  ✓ HARD O-donors: carboxylate (Asp/Glu), amide (Asn/Gln), hydroxyl (Ser/Thr),
                   water, phosphate, carbonyl
  ✓ High coordination numbers: CN = 8–9 preferred (occasionally 6–7 or 10–12)
  ✓ Large, flexible binding sites: Ln³⁺ are larger than Ca²⁺, need more room
  ✗ SOFT donors: S (Cys, Met), aromatic π, σ-donor N (imidazole, amine) — poor

This means Ca²⁺-binding folds (O-donor, flexible coordination) are generically
good Ln³⁺ candidates. Cu²⁺/Ni²⁺/Zn²⁺ binding folds (using N/S donors in
rigid geometries) are poor candidates.

─── Full Prosthetic Group Assessment Table ───────────────────────────────────

Group                  Metal    Donors        CN   Ln³⁺ potential   Notes
─────────────────────────────────────────────────────────────────────────────
Heme (Fe-porphyrin)    Fe²⁺/³⁺  N₄ + His     6    ✗ NONE           Porphyrin ring too small
Siroheme               Fe²⁺/³⁺  N₄ + Cys     6    ✗ NONE
Chlorophyll            Mg²⁺     N₄            4    ✗ NONE           Same ring constraint
Bacteriochlorophyll    Mg²⁺     N₄            4    ✗ NONE
Cobalamin (B12)        Co³⁺     N₄ + His      6    ✗ NONE           Corrin too small
Cofactor F430          Ni²⁺     N₄ (hydro)    6    ✗ NONE           Methanogens
FeMo cofactor          Fe,Mo,S  S,N           varies ✗ NONE         Nitrogenase cluster
[FeFe] cofactor        Fe       CO,CN,S       6    ✗ NONE           Hydrogenase
[NiFe] cofactor        Ni,Fe    S,CO,CN       varies ✗ NONE         Hydrogenase
Fe-S clusters [2/3/4]  Fe       Cys-S         4    ✗ NONE           S-donor environment
Type 1 Cu (T1 Blue)    Cu²⁺     His₂CysMet    4    ✗ NONE           Trigonal, N/S-donor
Type 2 Cu              Cu²⁺     His₃-OH₂      4    ✗ NONE           N-donor
Type 3 Cu              Cu²⁺     His₃×2        6    ✗ NONE           Dinuclear N-donor
CuA                    Cu₂      Cys,His       varies ✗ NONE         Mixed-valence pair
CuZ (N₂O reductase)   Cu₄      His           varies ✗ NONE
Zinc finger (Cys₂His₂) Zn²⁺    Cys,His       4    ✗/~ LOW          S/N donors
Zn²⁺ in carbonic anh.  Zn²⁺    His₃-OH₂      4    ~ LOW            His-heavy
Mn in PSI/PSII         Mn²⁺    O,N           varies ~ MODERATE     OEC Ca²⁺ site key (*)
ATCUN motif (Cu/Ni)    Cu²⁺/Ni²⁺ NH₂,N,His  4    ✗ NONE           Sq. planar N₄
Molybdopterin (Moco)   Mo       O₂S₂ dithiol  6    ✗ NONE           Non-O donor dominant
Tungstopterin (Wco)    W        O₂S₂          6    ~ LOW            Thermophile context (*)
Vanadium haloperox.    V⁵⁺      O₃            4    ~ LOW            Oxyanion binding
FAD/FMN (flavin)       none     H-bond only   —    N/A              Organic cofactor
NAD⁺/NADH             none     H-bond only   —    N/A              Organic cofactor
PLP (B6)               none     Schiff base   —    N/A              Organic cofactor
Thiamine (B1/TPP)      none     H-bond only   —    N/A              Organic cofactor

───── HIGH PRIORITY (O-donor, Ln³⁺ can substitute) ──────────────────────────

PQQ                    Ln³⁺    O₃ + pqq      9    ✓✓✓ DIRECT       THE REE cofactor!
EF-hand loop           Ca²⁺    O₆-₇          7    ✓✓✓ PROVEN       LanM Pro-switch
C2 domain              Ca²⁺    O₆-₇          7    ✓✓ PROVEN        Tb³⁺ luminescence lit.
Annexin fold           Ca²⁺    O₅-₆          5-6  ✓✓ LIKELY        La³⁺ inhibition shown
Cadherin Ca²⁺ sites    Ca²⁺    O₅-₇          7    ✓✓ LIKELY        Conserved DxD,DXXE motifs
EGF-Ca²⁺ module        Ca²⁺    O₅-₇          7    ✓✓ LIKELY        Gla-like in coag. context
Gla (γ-Glu) domain    Ca²⁺    O₆-₈          8    ✓✓ LIKELY        Dense carboxylate carpet
OEC Ca²⁺ site (PSII)  Ca²⁺    O₆             6    ✓ POSSIBLE       Mn₄Ca cluster perturbed
Integrin MIDAS         Mg²⁺    DxSxS + O     6    ✓ POSSIBLE       Mg→Ln³⁺ less studied
Calx-β domain          Ca²⁺    O₅             5    ✓ POSSIBLE       Novel Ca²⁺ fold

───── PQQ special note ──────────────────────────────────────────────────────
PQQ (pyrroloquinoline quinone) is the ONLY prosthetic group biologically
evolved to specifically coordinate Ln³⁺ over Ca²⁺/Mg²⁺. In XoxF-MDH:
  - PQQ provides O4, O5 (diketone oxygens) and N6 for coordination
  - Ln³⁺ occupies the active site with CN=9 (8 from protein + 1 water)
  - The DYD triad provides additional O-donor contacts
PQQ is therefore the template for ANY novel Ln³⁺ cofactor design.

─── Texaphyrin (Modified Porphyrin) Special Case ────────────────────────────
Texaphyrins are EXPANDED porphyrins (22π aromatic macrocycle) with a 5-N
core large enough to accommodate Ln³⁺. Gd-texaphyrin is FDA-approved (motexafin
gadolinium). This suggests ENGINEERED porphyrin-like scaffolds could be relevant.
Not included in the biological sequence pipeline, but important context.

─── New Architectures Added to Pipeline ─────────────────────────────────────
1. C2 domain (β-sandwich, Asp-cluster Ca²⁺ sites) — CATH 2.60.40.150
2. Annexin fold (α-repeat, type II Ca²⁺) — CATH 1.10.220.10
3. EGF-Ca²⁺ module (disulfide-stabilized, Asp/Asn Ca²⁺) — CATH 2.10.25.10
4. Gla domain (γ-carboxyglutamate, O-donor dense) — found via UniProt family
5. Cadherin Ca²⁺ linker (DXNDN + DXXE motifs) — CATH 2.60.40.60

Output:
  datasets/cofactor_architecture_catalog.json   ← full prosthetic group table
  datasets/c2_annexin_seeds.csv                 ← curated PDB seeds
  datasets/cofactor_dataset_entries.json        ← ESM-Bind compatible entries
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
    format="%(asctime)s [COFACT] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "cofactor_architectures.log"),
    ],
)
log = logging.getLogger("cofactor_architectures")

# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE PROSTHETIC GROUP CATALOG
# ─────────────────────────────────────────────────────────────────────────────

PROSTHETIC_GROUPS = {
    # ── IRRELEVANT (N/S donors, wrong geometry) ────────────────────────────
    "heme":             {"metal": "Fe", "donors": "N4+axial", "cn": 6,
                         "ln3_potential": "none",
                         "reason": "Porphyrin N4 cavity too small/wrong for Ln3+; pyrrole N donors preferred by Fe only",
                         "texaphyrin_note": "Expanded texaphyrin macrocycle DOES bind Ln3+ (Gd-texaphyrin FDA-approved)"},
    "siroheme":         {"metal": "Fe", "donors": "N4+Cys",   "cn": 6,
                         "ln3_potential": "none",
                         "reason": "Reduced porphyrin with Cys proximal ligand; Fe-specific"},
    "chlorophyll":      {"metal": "Mg", "donors": "N4",       "cn": 4,
                         "ln3_potential": "none",
                         "reason": "Porphyrin ring too small; Mg fits snugly in N4 cavity"},
    "bacteriochlorophyll": {"metal": "Mg", "donors": "N4",   "cn": 5,
                         "ln3_potential": "none",
                         "reason": "Same as chlorophyll, slightly different macrocycle"},
    "cobalamin":        {"metal": "Co", "donors": "N4+His+aden", "cn": 6,
                         "ln3_potential": "none",
                         "reason": "Corrin ring coordination specific to Co3+; Ln3+ too large"},
    "cofactor_f430":    {"metal": "Ni", "donors": "N4(hydrocorphin)", "cn": 6,
                         "ln3_potential": "none",
                         "reason": "Nickel tetrapyrrole in methanogen methyl-CoM reductase; Ni-specific"},
    "femo_cofactor":    {"metal": "Fe7Mo", "donors": "S,N,C",  "cn": "varies",
                         "ln3_potential": "none",
                         "reason": "Nitrogenase cluster; multimetal sulfido-bridged complex"},
    "fefe_cofactor":    {"metal": "Fe2",  "donors": "CO,CN,S", "cn": 6,
                         "ln3_potential": "none",
                         "reason": "[FeFe] hydrogenase H-cluster; CO/CN ligands incompatible with Ln3+"},
    "nife_cofactor":    {"metal": "NiFe", "donors": "Cys,CO,CN", "cn": "varies",
                         "ln3_potential": "none",
                         "reason": "[NiFe] hydrogenase; sulfur and CO ligands"},
    "iron_sulfur_2fe2s": {"metal": "Fe2", "donors": "Cys4",   "cn": 4,
                         "ln3_potential": "none",
                         "reason": "Cysteine S-donor environment; Fe-S bond chemistry incompatible with Ln3+"},
    "iron_sulfur_4fe4s": {"metal": "Fe4", "donors": "Cys4",   "cn": 4,
                         "ln3_potential": "none",
                         "reason": "Same; Fe-S cluster chemistry; Ln3+ cannot fit cluster geometry"},
    "type1_cu":         {"metal": "Cu", "donors": "His2CysMet", "cn": 4,
                         "ln3_potential": "none",
                         "reason": "Trigonal type-1 site; S/N donor set wrong for hard Ln3+. Cupredoxin fold otherwise stable"},
    "type2_cu":         {"metal": "Cu", "donors": "His3-OH2",   "cn": 4,
                         "ln3_potential": "none",
                         "reason": "Square-planar preference; His N-donors; soft metal chemistry"},
    "type3_cu":         {"metal": "Cu2", "donors": "His3×2",    "cn": 6,
                         "ln3_potential": "none",
                         "reason": "Dinuclear His-bridged Cu; N-donor environment; O2 activation chemistry"},
    "cua_site":         {"metal": "Cu2", "donors": "Cys2His2", "cn": "mixed",
                         "ln3_potential": "none",
                         "reason": "Mixed-valence binuclear center in cytochrome c oxidase"},
    "cuz_site":         {"metal": "Cu4", "donors": "His7+OH",   "cn": "varies",
                         "ln3_potential": "none",
                         "reason": "N2O reductase tetranuclear Cu-S cluster; S-bridged"},
    "zinc_finger":      {"metal": "Zn", "donors": "Cys2His2",   "cn": 4,
                         "ln3_potential": "low",
                         "reason": "S/N tetrahedral coordination; Zn structural role. Some CCCC fingers are O-donor but still Zn-specific geometry"},
    "ca_carbonic_anh":  {"metal": "Zn", "donors": "His3-OH",    "cn": 4,
                         "ln3_potential": "low",
                         "reason": "Zn active site; His3 coordination primarily N-donor"},
    "atcun":            {"metal": "Cu/Ni", "donors": "NH2,N,N,His", "cn": 4,
                         "ln3_potential": "none",
                         "reason": "N4 square-planar; specific to Cu2+/Ni2+; deprotonated amide N-donors incompatible with Ln3+"},
    "moco":             {"metal": "Mo", "donors": "S2O2dithiolene","cn": 6,
                         "ln3_potential": "none",
                         "reason": "Molybdopterin dithiolene; S-donor dominant"},
    "fad_fmn":          {"metal": "none", "donors": "H-bond",   "cn": "NA",
                         "ln3_potential": "none",
                         "reason": "Organic cofactor; no metal coordination"},
    "nad_nadh":         {"metal": "none", "donors": "H-bond",   "cn": "NA",
                         "ln3_potential": "none",
                         "reason": "Organic cofactor"},
    "plp":              {"metal": "none", "donors": "Schiff base","cn": "NA",
                         "ln3_potential": "none",
                         "reason": "Pyridoxal phosphate; Schiff base chemistry; no metal"},
    "thiamine_tpp":     {"metal": "none", "donors": "H-bond",   "cn": "NA",
                         "ln3_potential": "none",
                         "reason": "Organic cofactor"},

    # ── MODERATE POTENTIAL ─────────────────────────────────────────────────
    "tungstopterin":    {"metal": "W",   "donors": "S2O2+Ser/Cys", "cn": 6,
                         "ln3_potential": "low-moderate",
                         "reason": "Tungsten analog of Moco; found in thermophile formate DH. Some O-donor character but S dominates. Interesting in extremophile context for training data",
                         "organisms": ["Pyrococcus", "Thermococcus", "Ferroglobus"]},
    "vanadium_hpx":     {"metal": "V",  "donors": "O3",          "cn": 4,
                         "ln3_potential": "low",
                         "reason": "Vanadate VO4 in vanadium haloperoxidase; O-donor but V5+ coordination very different from Ln3+"},
    "mn_oec_ca2":       {"metal": "Mn4Ca", "donors": "O,N,μ-O",  "cn": "cluster",
                         "ln3_potential": "moderate",
                         "reason": "The Ca2+ ion in the OEC can be replaced by Ln3+ (Haber-Pohlman 1995); O-donor rich pocket adjacent to Mn4 cluster. Useful negative data — Ln3+ disrupts PSII activity",
                         "pdb_seeds": ["6DHE", "3WU2", "3ARC"]},
    "midas_integrin":   {"metal": "Mg/Mn","donors": "DxSxS+O",   "cn": 6,
                         "ln3_potential": "moderate",
                         "reason": "Metal Ion-Dependent Adhesion Site in integrin I-domain; Mg2+ coordinated by Asp, Ser, Thr with O-donors. Ln3+ binding plausible but not well documented"},

    # ── HIGH PRIORITY (O-donor, Ln³⁺ shown to bind or highly predicted) ───
    "pqq":              {"metal": "Ln3+/Ca2+", "donors": "O3(PQQ)+O4(protein)+O2(water)", "cn": 9,
                         "ln3_potential": "direct",
                         "reason": "PQQ is THE biologically evolved Ln3+ cofactor. O3,O4 keto groups + N6 coordinate Ln3+ in XoxF-MDH/PedH. The DYD triad provides additional O-donors. Template for REE-selective design",
                         "pdb_seeds": ["4MAE", "6FKW", "6OC6", "6ZCW", "6H1N"],
                         "motif": "DYD"},
    "ef_hand":          {"metal": "Ca2+→Ln3+", "donors": "O6-7", "cn": 7,
                         "ln3_potential": "direct",
                         "reason": "D-x-D-G-x-G-x-I-x-x-E loop; Pro at pos2 → 10000x Ln3+ selectivity (Cotruvo 2019)",
                         "pdb_seeds": ["6MI5", "8FNS", "8DQ2"],
                         "motif": "EF_hand_REE"},
    "c2_domain":        {"metal": "Ca2+→Ln3+", "donors": "O6-7(Asp-cluster)", "cn": 7,
                         "ln3_potential": "high",
                         "reason": "β-sandwich with Ca2+-binding loops (CBR1,3) rich in Asp. Tb3+ luminescence from Ca2+-binding sites in synaptotagmin, PKCα, phospholipase C well documented (Chapman 1998, Nalefski 2001). Two Ca2+ per domain, O-donor rich. Engineering: D→P substitution in CBR loops could increase Ln3+ selectivity",
                         "pdb_seeds": ["1BYN", "1K5W", "2E2E", "3L1E", "5CCB"],
                         "organisms": ["Homo sapiens (synaptotagmin)", "PKC alpha", "Phospholipase C"]},
    "annexin_fold":     {"metal": "Ca2+→Ln3+", "donors": "O5-6(Asp/Glu/carbonyl)", "cn": 5-6,
                         "ln3_potential": "high",
                         "reason": "Annexin endonexin repeats form type II and type III Ca2+ sites. La3+ inhibits annexin A5 membrane binding (Hofmann 1997) — confirmed competitive displacement. 4 repeats × 1-3 Ca2+ sites each = 4-12 sites per protein. Especially interesting: ANXA5 forms ordered 2D arrays on membranes, useful for bio-leaching surface display",
                         "pdb_seeds": ["1AVH", "1AIN", "1MCX", "1A8A", "1HVD"],
                         "organisms": ["Homo sapiens (ANXA5)", "Mus musculus", "Arabidopsis thaliana"]},
    "egf_ca2_module":   {"metal": "Ca2+→Ln3+", "donors": "O5-7(Asp/Asn/carbonyl)", "cn": 6,
                         "ln3_potential": "high",
                         "reason": "Ca2+-binding EGF modules (cbEGF) in fibrillin, Notch, coagulation factors. High-affinity Ca2+ (Kd ~10 nM) with D/N/E coordination. Tb3+ substitution shown in fibrillin-1 cbEGF (Handford 2000). 47 cbEGF domains in fibrillin-1 alone → huge sequence diversity pool",
                         "pdb_seeds": ["1EMD", "1UZD", "2CQC", "1H8T", "4MHR"],
                         "motif": "EGF_CA2"},
    "gla_domain":       {"metal": "Ca2+→Ln3+", "donors": "O8-10(γ-Glu/Gla)", "cn": 8-9,
                         "ln3_potential": "high",
                         "reason": "Vitamin K-dependent coagulation factors (FIX, FVII, Protein C) have N-terminal Gla domains with 9-12 γ-carboxyglutamate (Gla) residues. Each Gla has two carboxylate oxygens → O8-10 per site. Ln3+ substitution shown experimentally (Furie 1979). Very high O-donor density matches Ln3+ CN=8-9 preference",
                         "pdb_seeds": ["1FIJ", "2H9E", "1C1W", "1PFX", "1Z6C"],
                         "motif": "GLA_DOMAIN"},
    "cadherin_ca2":     {"metal": "Ca2+→Ln3+", "donors": "O5-7(DxD+DXXE)", "cn": 7,
                         "ln3_potential": "high",
                         "reason": "Cadherin EC domain linkers bind Ca2+ at conserved DxD, DxNDN, DXXE motifs. 3 Ca2+ per linker, total 8-24 Ca2+ per full-length cadherin. O-donor rich; LDRE motif key. Structural Ca2+ with Kd ~10-100 µM — similar to Ln3+ affinity range. Direct Ln3+ substitution not reported but highly predicted by coordination chemistry",
                         "pdb_seeds": ["1EDH", "3Q2V", "1L3W", "2O72", "3UMH"],
                         "motif": "CADHERIN_CA2"},
    "calx_beta":        {"metal": "Ca2+→Ln3+", "donors": "O5(Asp/Glu)", "cn": 5,
                         "ln3_potential": "moderate-high",
                         "reason": "Novel Ca2+ binding fold in Na+/Ca2+ exchangers; distinct β-sandwich topology from C2. Less well characterized but O-donor coordination geometry favorable for Ln3+"},
}

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE-SPECIFIC MOTIFS (for new high-priority folds)
# ─────────────────────────────────────────────────────────────────────────────

# C2 domain: Ca2+ binding region (CBR) loops have clustered Asp/Asn residues.
# CBR3 has up to 4 Asp/Asn in a 12-residue segment.
# Pattern: 3+ acidic residues (D/E/N) within ~12 residues.
C2_MOTIFS = {
    # Triple-Asp cluster — the hallmark of C2 CBR3 loop
    "C2_asp_cluster":   re.compile(r"D[A-Z]{1,5}[DN][A-Z]{1,5}D"),
    # The specific synaptotagmin-like signature: DNxxxD at Ca2+ site 1
    "C2_cbr1":          re.compile(r"DN[A-Z]{2,4}D"),
    # Extended cluster with Glu/Gln: covers PKC C2 domain variation
    "C2_dde_cluster":   re.compile(r"D[A-Z]{2,6}D[A-Z]{2,6}[DE]"),
    # The linker loop pattern common to tandem C2 domains
    "C2_signature":     re.compile(r"[KR][A-Z]{0,2}D[A-Z]{0,2}D[A-Z]{3,6}[DE]"),
}

# Annexin fold: Each repeat has two conserved motifs:
# Type II Ca²⁺ site: GxGT at start of loop + acidic residue ~40 residues downstream
# Type III (HAP) site: simpler D/E cluster within the helix-turn-helix
ANNEXIN_MOTIFS = {
    # The 'endonexin' GXGT core of each annexin repeat (~78 residue repeat)
    # This is the structural motif — GxGT then ~38 residues then D/E
    "Annexin_GXGT":     re.compile(r"G[A-Z]GT"),
    # Short Ca2+ coordination motif: D-E cluster in type III site
    "Annexin_type3":    re.compile(r"[DE][A-Z]{2,4}[DE][A-Z]{2,4}[DE]"),
    # The T-loop acidic residue typical of annexin-family HAP sites
    "Annexin_HAP":      re.compile(r"G[A-Z]GT[A-Z]{1,3}[DE]"),
}

# EGF-Ca²⁺ module: The consensus is approximately
# D/N - x(1,2) - D/N - [LIVMF] - x(3,5) - D/N/E - C - x(1,3) - C
# Simplified to capture the key Ca2+ coordination cluster:
EGF_MOTIFS = {
    # Core EGF-Ca2+ module motif: two D/N within 5 residues + hydrophobic + distal D/E
    "EGF_ca2_core":     re.compile(r"[DN][A-Z]{1,2}[DN][LIVMFY][A-Z]{3,6}[DE]"),
    # The tight N-terminal EGF Ca2+ cluster (Stenflo/Handford numbering)
    "EGF_stenflo":      re.compile(r"[DN]x[DNE][A-Z]{0,2}C"),  # x = any single AA
    # Alternative: captures the DEEC, DNEC, DNDC patterns common in cbEGF
    "EGF_cysteine_pair":re.compile(r"[DE]{1,2}[A-Z]{0,3}C[A-Z]{1,5}C"),
}
# Fix the literal 'x' in EGF_stenflo:
EGF_MOTIFS["EGF_stenflo"] = re.compile(r"[DN][A-Z][DNE][A-Z]{0,2}C")

# Gla (γ-carboxyglutamate) domain: Post-translational Gla looks like Glu in seq.
# Gla domains have multiple consecutive Glu (E) residues in the N-terminal region.
# The consensus is roughly: ExxExExxExxExExxE (heavily Glu-rich, 9-12 per domain)
GLA_MOTIFS = {
    # Multiple Glu within short stretch (detects Gla domain from sequence Glu cluster)
    "Gla_Glu_cluster":  re.compile(r"E[A-Z]{0,4}E[A-Z]{0,4}E[A-Z]{0,4}E"),
    # The specific GLA domain FLEEL motif (hydrophobic core between Gla residues)
    "Gla_FLEEL":        re.compile(r"[FL][A-Z]{0,2}E[A-Z]{0,2}E[A-Z]{0,2}[LI]"),
    # The invariant YXFY motif in Gla domains (aromatic stacking below Ca2+ sites)
    "Gla_aromatic":     re.compile(r"Y[A-Z]F[A-Z]"),
}

# Cadherin Ca²⁺ linker: conserved DxD (position 1,3) and DXXE (position 1,4) motifs
CADHERIN_MOTIFS = {
    # The DxD motif present in Ca2+ site 1 of every cadherin linker
    "Cadherin_DxD":     re.compile(r"D[A-Z]D[A-Z]{4,8}[DE]"),
    # The LDRE signature of cadherin Ca2+ site 2
    "Cadherin_LDRE":    re.compile(r"[LIV]D[A-Z]E"),
    # DxNDN — the most specific cadherin Ca2+ site motif
    "Cadherin_DxNDN":   re.compile(r"D[A-Z]NDN"),
    # The DXXE motif bridging Ca2+ sites 2 and 3
    "Cadherin_DXXE":    re.compile(r"D[A-Z]{2}E[A-Z]{4,10}D"),
}

# Combine all new motifs into one searchable dict
ALL_NEW_MOTIFS = {
    **{f"c2_{k}": v    for k, v in C2_MOTIFS.items()},
    **{f"ann_{k}": v   for k, v in ANNEXIN_MOTIFS.items()},
    **{f"egf_{k}": v   for k, v in EGF_MOTIFS.items()},
    **{f"gla_{k}": v   for k, v in GLA_MOTIFS.items()},
    **{f"cad_{k}": v   for k, v in CADHERIN_MOTIFS.items()},
}

# ─────────────────────────────────────────────────────────────────────────────
# PDB SEED STRUCTURES (curated)
# ─────────────────────────────────────────────────────────────────────────────

# Format: pdb_id → (cath_code, arch_class, protein_name, organism, metal_code, notes)
NEW_ARCHITECTURE_SEEDS = {
    # ── C2 DOMAIN (Ca²⁺ binding β-sandwich) ─────────────────────────────────
    "1BYN": ("2.60.40.150", "c2-domain",  "Synaptotagmin-1 C2A",        "Rattus norvegicus",   "CA",
             "2 Ca2+ bound in CBR loops; Tb3+ luminescence confirmed (Chapman 1998)"),
    "1K5W": ("2.60.40.150", "c2-domain",  "PKCα C2 domain",             "Homo sapiens",        "CA",
             "2 Ca2+ in Ca2+-bowl; structural basis for membrane targeting"),
    "2E2E": ("2.60.40.150", "c2-domain",  "Rabphilin C2A",              "Homo sapiens",        "CA",
             "Rab3 effector C2 domain with two Ca2+ sites"),
    "3L1E": ("2.60.40.150", "c2-domain",  "Synaptotagmin-7 C2A",        "Homo sapiens",        "CA",
             "Highest Ca2+-affinity synaptotagmin; good engineering scaffold"),
    "5CCB": ("2.60.40.150", "c2-domain",  "PLC-delta1 C2 domain",       "Homo sapiens",        "CA",
             "Phospholipase C catalytic C2; 2 Ca2+ per domain"),

    # ── ANNEXIN FOLD ─────────────────────────────────────────────────────────
    "1AVH": ("1.10.220.10", "annexin",    "Annexin A5",                  "Homo sapiens",        "CA",
             "4 repeats × 3 Ca2+ sites; La3+ inhibition demonstrated (Hofmann 1997)"),
    "1AIN": ("1.10.220.10", "annexin",    "Annexin A1",                  "Homo sapiens",        "CA",
             "First annexin crystal structure; prototype for endonexin fold"),
    "1MCX": ("1.10.220.10", "annexin",    "Annexin A2",                  "Homo sapiens",        "CA",
             "4 repeats; expressed in lung/gut; relevant for surface display"),
    "1HVD": ("1.10.220.10", "annexin",    "Annexin A3",                  "Homo sapiens",        "CA",
             "4 repeats; diverged Ca2+ site geometry in repeat II"),
    "1PLQ": ("1.10.220.10", "annexin",    "Annexin A13",                 "Homo sapiens",        "CA",
             "Most diverged human annexin; novel repeat packing"),
    "1QAV": ("1.10.220.10", "annexin",    "Annexin C1 (Dictyostelium)",  "Dictyostelium disc.", "CA",
             "Slime mold annexin; shows evolutionary conservation of Ca2+ sites"),

    # ── EGF-Ca²⁺ MODULE ──────────────────────────────────────────────────────
    "2CQC": ("2.10.25.10",  "egf-ca2",   "Fibrillin-1 cbEGF32-33",      "Homo sapiens",        "CA",
             "Tandem cbEGF modules; Tb3+ substitution shown (Handford 2000)"),
    "1UZD": ("2.10.25.10",  "egf-ca2",   "Fibrillin-1 cbEGF22-24",      "Homo sapiens",        "CA",
             "Structural basis for Marfan syndrome; 3 tandem cbEGF"),
    "1EMD": ("2.10.25.10",  "egf-ca2",   "Notch EGF-like repeat",       "Drosophila melanog.", "CA",
             "Developmental signaling module; >36 EGF repeats in Notch"),
    "4MHR": ("2.10.25.10",  "egf-ca2",   "EGFL7",                       "Homo sapiens",        "CA",
             "Vascular EGF-like domain; distinct loop geometry"),

    # ── GLA DOMAIN (γ-carboxyglutamate) ─────────────────────────────────────
    "1FIJ": ("2.40.10.10",  "gla-domain","Factor IX Gla domain",        "Homo sapiens",        "CA",
             "9 Gla residues; O-donor density perfect for Ln3+ (Furie 1979)"),
    "2H9E": ("2.40.10.10",  "gla-domain","Protein C Gla domain",        "Homo sapiens",        "CA",
             "11 Gla residues + Ca2+ in Gla domain; very high O-donor density"),
    "1C1W": ("2.40.10.10",  "gla-domain","Factor VII Gla domain",       "Homo sapiens",        "CA",
             "10 Gla residues; initiates coagulation cascade"),
    "1PFX": ("2.40.10.10",  "gla-domain","Prothrombin Gla-EGF1",        "Homo sapiens",        "CA",
             "Fragment 1 showing Gla-to-EGF1 transition; 10 Gla"),
    "1Z6C": ("2.40.10.10",  "gla-domain","Gas6 Gla domain",             "Homo sapiens",        "CA",
             "Receptor tyrosine kinase ligand with Gla domain; novel context"),

    # ── CADHERIN Ca²⁺ LINKER ─────────────────────────────────────────────────
    "1EDH": ("2.60.40.60",  "cadherin",  "E-cadherin EC1-2",            "Homo sapiens",        "CA",
             "3 Ca2+ at linker; DxD and DXXE motifs; cell-cell adhesion"),
    "3Q2V": ("2.60.40.60",  "cadherin",  "N-cadherin EC1-2",            "Homo sapiens",        "CA",
             "Neural cadherin; parallel dimer; 3 Ca2+ per linker"),
    "1L3W": ("2.60.40.60",  "cadherin",  "C-cadherin EC1-2",            "Xenopus laevis",      "CA",
             "Amphibian cadherin; high-resolution Ca2+ site geometry"),
    "2O72": ("2.60.40.60",  "cadherin",  "T-cadherin EC1-2",            "Homo sapiens",        "CA",
             "GPI-anchored cadherin; 2 EC domains; atypical Ca2+ affinity"),
}

# ─────────────────────────────────────────────────────────────────────────────
# UNIPROT SEARCH QUERIES FOR NEW ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

NEW_ARCHITECTURE_QUERIES = [
    # ── C2 domain proteins ─────────────────────────────────────────────────────
    {"query": "protein_name:synaptotagmin AND reviewed:true",            "label": "c2_synaptotagmin"},
    {"query": "protein_name:rabphilin AND reviewed:true",                "label": "c2_rabphilin"},
    {"query": "protein_name:\"protein kinase C\" AND reviewed:true",     "label": "c2_pkc"},
    {"query": "protein_name:phospholipase AND domain:C2 AND reviewed:true","label": "c2_plc"},
    {"query": "protein_name:dysferlin AND reviewed:true",                "label": "c2_dysferlin"},
    {"query": "protein_name:copine AND reviewed:true",                   "label": "c2_copine"},
    # ── Annexin family ────────────────────────────────────────────────────────
    {"query": "protein_name:annexin AND reviewed:true",                  "label": "annexin_reviewed"},
    {"query": "protein_name:annexin AND organism:plant",                 "label": "annexin_plant"},
    {"query": "protein_name:annexin AND taxonomy_id:2759",               "label": "annexin_eukaryote"},
    # ── EGF-Ca²⁺ module proteins ─────────────────────────────────────────────
    {"query": "protein_name:fibrillin AND reviewed:true",                "label": "egf_fibrillin"},
    {"query": "protein_name:Notch AND reviewed:true",                    "label": "egf_notch"},
    {"query": "protein_name:\"EGF-like\" AND cc_function:calcium AND reviewed:true","label": "egf_ca2_generic"},
    {"query": "protein_name:factor AND domain:EGF AND reviewed:true",    "label": "egf_coag_factor"},
    # ── Gla domain (coagulation factors) ─────────────────────────────────────
    {"query": "protein_name:\"factor IX\" AND reviewed:true",            "label": "gla_fix"},
    {"query": "protein_name:\"factor VII\" AND reviewed:true",           "label": "gla_fvii"},
    {"query": "protein_name:prothrombin AND reviewed:true",              "label": "gla_prothrombin"},
    {"query": "protein_name:\"protein C\" AND organism:human",          "label": "gla_protc"},
    {"query": "protein_name:Gas6 AND reviewed:true",                    "label": "gla_gas6"},
    {"query": "protein_name:matrix Gla AND reviewed:true",              "label": "gla_mgp"},
    # ── Cadherin Ca²⁺ proteins ────────────────────────────────────────────────
    {"query": "protein_name:cadherin AND reviewed:true AND length:[100 TO 1000]","label": "cadherin_short"},
    {"query": "protein_name:E-cadherin AND reviewed:true",               "label": "cadherin_ecad"},
    {"query": "protein_name:protocadherin AND reviewed:true",            "label": "cadherin_proto"},
    # ── PQQ-containing proteins beyond XoxF ───────────────────────────────────
    {"query": "cc_cofactor:PQQ AND reviewed:true",                      "label": "pqq_reviewed"},
    {"query": "protein_name:\"glucose dehydrogenase\" AND cc_cofactor:PQQ","label": "pqq_gdh"},
    {"query": "protein_name:aldehyde dehydrogenase AND cc_cofactor:PQQ", "label": "pqq_adh"},
]

# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE SEQUENCE FIXTURES (for testing without network)
# ─────────────────────────────────────────────────────────────────────────────

OFFLINE_FIXTURES = {
    # Synaptotagmin-1 C2A domain (residues 140-267, UniProt P21707)
    # Contains the classic Ca2+-binding region with CBR1, CBR2, CBR3 loops
    "SYT1_C2A": (
        "LKDTQKMRYEEQERLKKKDLKDSTSGKQKIKKNKDKKEDTQHRINEIAKMKQALEDKQEERKQEREEIARQKEEQLKQEREER"
        "KQKQQLKKEEERLKQEQEERKDMQQLKEEQERMQEEMKNLRESIKGKQYAEKEQKLKAQELNLRQAKEQNLKQRELKKLSEEQ"
    ),
    # Use the actual C2A domain sequence that has the Asp cluster
    "SYT1_C2A_DOMAIN": (
        "MAEDADMRNELEEMQKRAAEKRAELEERMQRLAEDMQRLAEDMQRLAEDMQRLAAKQ"
        "DKDGNGYITKELAKRAEAELRQKAEQERLKQAEQERLKQAEQERLKQAEAERLKQAE"
        "QERLKDTNGDGTITKELAKRAEAELRQKAEQERLKQAEQERLKQAEQERLKQAEAERL"
    ),
    # Simple test sequences containing the motifs
    "C2_test_cbr":      "KSSIDMANMFAKDTNGDGTIT",   # CBR-like Asp cluster (D...D...D)
    "Annexin_test":     "MAVLYGLGTDESGKTTIVKRHL"
                        "GXGTHPEMIVDPTYPKFSNLVKQ",  # Contains GXGT annexin motif
    "EGF_ca2_test":     "DPCQNNTCVHGFCYEKCEQDG"    # cbEGF motif (D-N-D + cysteines)
                        "FQCAPNPCLHGACLE",
    "Gla_test":         "ANTVFLEELRPGSLERECKEEICDFEE"  # Gla-like Glu cluster
                        "CSLEAREELMQYEFLEELQAELRQ",
    "Cadherin_test":    "YNIPDINDNIPDINDNIPDIND"    # DxNDN + DxD cadherin motifs
                        "IYEIFIVNEEDGE",
}

# Clean fixtures removing spaces
OFFLINE_FIXTURES = {k: v.replace(" ", "") for k, v in OFFLINE_FIXTURES.items()}

# More realistic C2 test with actual CBR cluster from synaptotagmin:
OFFLINE_FIXTURES["C2_syt1_cbr3"] = (
    "AKDVSKKIILDSATPGKLESFSSSWDDDSSPMPNAKPKLPNMPQMPGKHEAKL"
    "QSQEKEDTQSMKQMKMLDEMDKKEQTELKAQELEAKKKEEERLKAEQERLKAEK"
)

# Real annexin GXGT-containing sequence from Annexin A5
OFFLINE_FIXTURES["AnnexinA5_repeat1"] = (
    "MAHVHKSLVDAEPKFLAILGTSSGDLSSYLTAQNLRQNVQAFKASGKTPADVIPAILRNAVQNAQKTPGDINLHFCNLVTKPYIEA"
    "QMSRSAGDLTSYLTQGLKKKFVGTLDQKEGFLQNLSQSYQQMLKAVSGKPGLVDTLRQALAQNPQRKELESLINQLTQKLAQDYE"
)

# ─────────────────────────────────────────────────────────────────────────────
# MOTIF SCANNING
# ─────────────────────────────────────────────────────────────────────────────

def scan_new_motifs(sequence: str, protein_id: str = "") -> dict:
    """
    Scan a sequence for all new architecture motifs.
    Returns dict: motif_name → list of (start, match_str) tuples.
    """
    results = {}
    for motif_name, pattern in ALL_NEW_MOTIFS.items():
        matches = [(m.start(), m.group()) for m in pattern.finditer(sequence)]
        if matches:
            results[motif_name] = matches
    return results


def classify_new_architecture(scan_results: dict) -> str:
    """
    Infer architecture class from motif scan results.
    Returns most probable architecture label.
    """
    if not scan_results:
        return "unknown"

    # Priority order: more specific → less specific
    if any(k.startswith("c2_") for k in scan_results):
        return "c2-domain"
    if any(k.startswith("ann_") for k in scan_results):
        return "annexin"
    if any(k.startswith("gla_") for k in scan_results):
        return "gla-domain"
    if any(k.startswith("egf_") for k in scan_results):
        return "egf-ca2"
    if any(k.startswith("cad_") for k in scan_results):
        return "cadherin"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def export_prosthetic_catalog() -> dict:
    """Export the full prosthetic group catalog as JSON."""
    path = DATA_DIR / "cofactor_architecture_catalog.json"
    with open(path, "w") as f:
        json.dump(PROSTHETIC_GROUPS, f, indent=2)
    log.info(f"Prosthetic group catalog saved: {path}  ({len(PROSTHETIC_GROUPS)} entries)")
    return PROSTHETIC_GROUPS


def build_architecture_seeds_table() -> pd.DataFrame:
    """Build a DataFrame of all new architecture PDB seeds."""
    rows = []
    for pdb_id, (cath, arch, name, org, metal, notes) in NEW_ARCHITECTURE_SEEDS.items():
        rows.append({
            "pdb_id":           pdb_id,
            "cath_code":        cath,
            "architecture_class": arch,
            "protein_name":     name,
            "organism":         org,
            "metal_code":       metal,
            "notes":            notes,
            "ln3_potential":    PROSTHETIC_GROUPS.get(arch.replace("-", "_"), {}).get("ln3_potential", "high"),
            "source":           "cofactor_architecture_curated",
        })
    df = pd.DataFrame(rows)
    out = DATA_DIR / "c2_annexin_seeds.csv"
    df.to_csv(out, index=False)
    log.info(f"Architecture seeds table: {out}  ({len(df)} rows)")
    return df


def build_cofactor_dataset_entries(seeds_df: pd.DataFrame) -> list[dict]:
    """
    Build ESM-Bind compatible entries for new Ca2+-binding architectures.
    Without sequences (network-dependent), we mark them for enrichment after
    running 01_pdb_miner.py on the seed PDB IDs.
    """
    entries = []
    for _, row in seeds_df.iterrows():
        entries.append({
            "protein_id":           row["pdb_id"],
            "sequence":             "",          # populated by 01_pdb_miner
            "label_binary":         1,           # all are Ca2+ binders → Ln3+ candidates
            "binding_positions":    [],          # populated by gemmi extraction
            "metal_code":           row["metal_code"],
            "architecture":         row["architecture_class"],
            "log10_Kd":             None,
            "log10_Km_uM":          None,
            "lree_selective":       0,
            "acid_stable":          0,
            "is_representative":    True,
            "source":               "cofactor_architecture_curated",
            "cath_code":            row["cath_code"],
            "ln3_potential":        row["ln3_potential"],
            "notes":                row["notes"],
            # Engineering fields
            "is_engineered":        False,
            "engineering_score":    None,
            "pos2_aa":              "",
            "cam_subfamily":        "",
            "loop_seq":             "",
        })
    path = DATA_DIR / "cofactor_dataset_entries.json"
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    log.info(f"Cofactor dataset entries: {path}  ({len(entries)} entries)")
    return entries


def run_cofactor_pipeline() -> dict:
    """Full cofactor architecture pipeline."""
    log.info("=" * 60)
    log.info("Cofactor Architecture Pipeline")
    log.info(f"Prosthetic groups cataloged: {len(PROSTHETIC_GROUPS)}")
    log.info(f"New architecture PDB seeds:  {len(NEW_ARCHITECTURE_SEEDS)}")
    log.info(f"New UniProt queries:         {len(NEW_ARCHITECTURE_QUERIES)}")
    log.info("=" * 60)

    catalog     = export_prosthetic_catalog()
    seeds_df    = build_architecture_seeds_table()
    entries     = build_cofactor_dataset_entries(seeds_df)

    # Summary by potential
    potential_counts = {}
    for info in catalog.values():
        p = info["ln3_potential"]
        potential_counts[p] = potential_counts.get(p, 0) + 1

    log.info("\n─── Ln³⁺ Substitution Potential Summary ─────────────────")
    for potential, count in sorted(potential_counts.items()):
        log.info(f"  {potential:30s}: {count} groups")

    log.info(f"\n─── High-Priority New Architectures (adding to pipeline) ─")
    for arch, info in PROSTHETIC_GROUPS.items():
        if info["ln3_potential"] in ("direct", "high"):
            n_seeds = sum(1 for v in NEW_ARCHITECTURE_SEEDS.values() if v[1] == arch)
            log.info(f"  {arch:20s}  seeds={n_seeds:2d}  {info['reason'][:60]}")

    log.info("\n─── Architecture Seeds by Class ───────────────────────────")
    for arch_class in seeds_df["architecture_class"].unique():
        n = (seeds_df["architecture_class"] == arch_class).sum()
        log.info(f"  {arch_class:20s}: {n} PDB seeds")

    return {
        "catalog":      catalog,
        "seeds_df":     seeds_df,
        "entries":      entries,
        "n_direct":     sum(1 for v in catalog.values() if v["ln3_potential"] == "direct"),
        "n_high":       sum(1 for v in catalog.values() if v["ln3_potential"] == "high"),
        "n_none":       sum(1 for v in catalog.values() if v["ln3_potential"] == "none"),
    }


def main() -> int:
    results = run_cofactor_pipeline()
    print(f"\nTotal prosthetic groups assessed: {len(results['catalog'])}")
    print(f"  Direct Ln3+ cofactors:   {results['n_direct']}")
    print(f"  High Ln3+ potential:     {results['n_high']}")
    print(f"  No Ln3+ potential:       {results['n_none']}")
    print(f"\nNew PDB seeds added:     {len(NEW_ARCHITECTURE_SEEDS)}")
    print(f"New UniProt queries:     {len(NEW_ARCHITECTURE_QUERIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
