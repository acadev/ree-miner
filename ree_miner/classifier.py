"""
REE Architecture Classifier
============================
Assigns CATH and SCOP fold classifications to every REE-binding PDB chain.
This is the core tool for tracking architectural diversity — the central goal
of the dataset expansion effort.

Strategy:
  1. Query RCSB SIFTS API for CATH domain assignments per PDB chain
  2. Query SIFTS for SCOP assignments (fallback / cross-validation)
  3. Map each architecture to a human-readable fold family name
  4. Flag EF-hand (CATH 1.10.238) and DYD/beta-propeller (CATH 2.140.10)
     hits so we can track diversity relative to the known bias

Input:  datasets/pdb_sequences_raw.csv
Output: datasets/architecture_classified.csv
        figures/fold_diversity.png

Usage:
    python 02_architecture_classifier.py
"""

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests

from ree_miner._workspace import DATA_DIR, FIG_DIR, LOG_DIR
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "architecture_classifier.log"),
    ],
)
log = logging.getLogger("arch_classifier")

# ─── Known fold annotations for key REE-binding proteins ────────────────────
# These are manually curated ground-truth labels from the literature review.
KNOWN_ARCHITECTURES = {
    # PDB ID → (CATH code, fold_name, architecture_class)
    # ── Beta-propeller (PQQ-dependent MDH) ────────────────────────────────
    "4MAE": ("2.140.10.30",  "Methanol Dehydrogenase beta-propeller", "beta-propeller"),
    "6FKW": ("2.140.10.30",  "XoxF-MDH beta-propeller",              "beta-propeller"),
    "6OC6": ("2.140.10.30",  "XoxF-MDH beta-propeller",              "beta-propeller"),
    "7O6Z": ("2.140.10.30",  "XoxF-MDH beta-propeller",              "beta-propeller"),
    "6DAM": ("2.140.10.30",  "XoxF-MDH beta-propeller",              "beta-propeller"),
    "5LJR": ("2.140.10.30",  "XoxF-MDH beta-propeller",              "beta-propeller"),
    "6H1N": ("2.140.10.30",  "XoxF-MDH beta-propeller",              "beta-propeller"),
    "6ZCW": ("2.140.10.30",  "PedH-EDH beta-propeller",              "beta-propeller"),
    "1H4I": ("2.140.10.30",  "MxaF-MDH (Ca2+, negative ctrl)",       "beta-propeller"),
    "1W6S": ("2.140.10.30",  "MxaF-MDH (Ca2+, negative ctrl)",       "beta-propeller"),
    # ── EF-hand (lanmodulin family) ────────────────────────────────────────
    "6MI5": ("1.10.238.10",  "Lanmodulin EF-hand triple helix",      "ef-hand"),
    "8FNS": ("1.10.238.10",  "Mex-LanM EF-hand",                     "ef-hand"),
    "8DQ2": ("1.10.238.10",  "Hans-LanM EF-hand",                    "ef-hand"),
    "8FNR": ("1.10.238.10",  "Hans-LanM EF-hand variant",            "ef-hand"),
    "1GGZ": ("1.10.238.10",  "Calmodulin EF-hand (negative ctrl)",   "ef-hand"),
    # ── C2 domain (β-sandwich, Ca²⁺ via Asp-cluster) ──────────────────────
    # Tb³⁺ luminescence at Ca²⁺ sites confirmed experimentally
    "1BYN": ("2.60.40.150",  "Synaptotagmin-1 C2A domain",           "c2-domain"),
    "1K5W": ("2.60.40.150",  "PKCα C2 domain",                       "c2-domain"),
    "2E2E": ("2.60.40.150",  "Rabphilin C2A domain",                 "c2-domain"),
    "3L1E": ("2.60.40.150",  "Synaptotagmin-7 C2A domain",           "c2-domain"),
    "5CCB": ("2.60.40.150",  "PLC-delta1 C2 domain",                 "c2-domain"),
    # ── Annexin fold (α-repeat, type II Ca²⁺, La³⁺ inhibition shown) ─────
    "1AVH": ("1.10.220.10",  "Annexin A5",                           "annexin"),
    "1AIN": ("1.10.220.10",  "Annexin A1",                           "annexin"),
    "1MCX": ("1.10.220.10",  "Annexin A2",                           "annexin"),
    "1HVD": ("1.10.220.10",  "Annexin A3",                           "annexin"),
    "1PLQ": ("1.10.220.10",  "Annexin A13",                          "annexin"),
    "1QAV": ("1.10.220.10",  "Annexin C1 (Dictyostelium)",           "annexin"),
    # ── EGF-Ca²⁺ module (disulfide-stabilized, high-affinity Ca²⁺) ───────
    # Tb³⁺ substitution in fibrillin cbEGF demonstrated (Handford 2000)
    "2CQC": ("2.10.25.10",   "Fibrillin-1 cbEGF32-33",               "egf-ca2"),
    "1UZD": ("2.10.25.10",   "Fibrillin-1 cbEGF22-24",               "egf-ca2"),
    "1EMD": ("2.10.25.10",   "Notch EGF-like repeat",                "egf-ca2"),
    "4MHR": ("2.10.25.10",   "EGFL7",                                "egf-ca2"),
    # ── Gla domain (γ-carboxyglutamate, O-donor dense, Ln³⁺ binding proven) ──
    "1FIJ": ("2.40.10.10",   "Factor IX Gla domain",                 "gla-domain"),
    "2H9E": ("2.40.10.10",   "Protein C Gla domain",                 "gla-domain"),
    "1C1W": ("2.40.10.10",   "Factor VII Gla domain",                "gla-domain"),
    "1PFX": ("2.40.10.10",   "Prothrombin Gla-EGF1",                 "gla-domain"),
    "1Z6C": ("2.40.10.10",   "Gas6 Gla domain",                      "gla-domain"),
    # ── Cadherin Ca²⁺ linker (DxD + DXXE motifs, 3 Ca²⁺ per linker) ─────
    "1EDH": ("2.60.40.60",   "E-cadherin EC1-2",                     "cadherin"),
    "3Q2V": ("2.60.40.60",   "N-cadherin EC1-2",                     "cadherin"),
    "1L3W": ("2.60.40.60",   "C-cadherin EC1-2",                     "cadherin"),
    "2O72": ("2.60.40.60",   "T-cadherin EC1-2",                     "cadherin"),
}

# Architecture classes we care about distinguishing
ARCH_COLORS = {
    # ── EF-hand superfamily (position-2 engineering target) ──────────────
    "ef-hand":              "#E74C3C",   # red — the EF-hand bias we want to escape
    # ── PQQ-containing / beta-propeller (direct Ln3+ biology) ────────────
    "beta-propeller":       "#3498DB",   # blue — XoxF/PedH DYD-triad MDHs
    # ── Other previously identified architectures ─────────────────────────
    "beta-roll (RTX)":      "#27AE60",   # green
    "pepsy-domain":         "#F39C12",   # orange
    "tim-barrel":           "#9B59B6",   # purple
    "coiled-coil":          "#1ABC9C",   # teal
    "alpha-beta":           "#34495E",   # dark grey
    "all-alpha":            "#E91E63",   # pink
    "all-beta":             "#795548",   # brown
    "beta-barrel (TonB-receptor)": "#8E44AD",  # violet
    # ── NEW: Ca²⁺-binding folds with confirmed/predicted Ln³⁺ binding ────
    "c2-domain":            "#2ECC71",   # emerald — β-sandwich, Asp-cluster, Tb3+ proven
    "annexin":              "#F1C40F",   # yellow — α-repeat, endonexin Ca2+, La3+ inhibition
    "egf-ca2":              "#E67E22",   # deep orange — cbEGF, Tb3+ sub in fibrillin
    "gla-domain":           "#1A5276",   # dark blue — γ-carboxyglutamate, Furie 1979
    "cadherin":             "#5D6D7E",   # blue-grey — DxD/DXXE linker sites
    # ── Fallback ─────────────────────────────────────────────────────────
    "unknown / surrogate":  "#95A5A6",   # light grey
}

# CATH superfamily codes
EF_HAND_CATH_CODES    = {"1.10.238"}   # EF-hand helix-loop-helix
BETA_PROP_CATH_CODES  = {"2.140.10"}   # WD40 / beta-propeller (PQQ-MDH)
C2_DOMAIN_CATH_CODES  = {"2.60.40.150"}# C2 domain (β-sandwich, Ca2+ loops)
ANNEXIN_CATH_CODES    = {"1.10.220"}   # Annexin repeat fold
EGF_CA2_CATH_CODES    = {"2.10.25"}    # EGF-like Ca2+ module
GLA_CATH_CODES        = {"2.40.10"}    # Gla domain
CADHERIN_CATH_CODES   = {"2.60.40.60", "2.60.40.130"}  # Cadherin EC domain

SIFTS_CATH_URL   = "https://www.ebi.ac.uk/pdbe/api/mappings/cath"
SIFTS_SCOP_URL   = "https://www.ebi.ac.uk/pdbe/api/mappings/scop"
REQUEST_PAUSE = 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CATH CLASSIFICATION via SIFTS
# ═══════════════════════════════════════════════════════════════════════════════

def get_cath_domains(pdb_id: str) -> list[dict]:
    """
    Query SIFTS API for CATH domain assignments on a PDB entry.
    Returns list of {chain, cath_id, domain_name, superfamily_name}.
    """
    url = f"{SIFTS_CATH_URL}/{pdb_id.lower()}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        entry = data.get(pdb_id.lower(), {})
        domains = []
        for cath_code, cath_info in entry.get("CATH", {}).items():
            for mapping in cath_info.get("mappings", []):
                domains.append({
                    "pdb_id": pdb_id.upper(),
                    "chain_id": mapping.get("chain_id", ""),
                    "cath_id": cath_code,
                    "cath_class": cath_code.split(".")[0] if "." in cath_code else "",
                    "cath_arch": ".".join(cath_code.split(".")[:2]) if "." in cath_code else "",
                    "cath_topology": ".".join(cath_code.split(".")[:3]) if "." in cath_code else "",
                    "superfamily_name": cath_info.get("name", ""),
                    "domain_name": cath_info.get("domain", ""),
                })
        return domains
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            log.debug(f"No CATH entry for {pdb_id}")
        else:
            log.warning(f"CATH query failed for {pdb_id}: {e}")
    except Exception as e:
        log.warning(f"CATH query error for {pdb_id}: {e}")
    return []


def get_scop_domains(pdb_id: str) -> list[dict]:
    """Query SIFTS API for SCOP domain assignments."""
    url = f"{SIFTS_SCOP_URL}/{pdb_id.lower()}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        entry = data.get(pdb_id.lower(), {})
        domains = []
        for scop_code, scop_info in entry.get("SCOP", {}).items():
            for mapping in scop_info.get("mappings", []):
                domains.append({
                    "pdb_id": pdb_id.upper(),
                    "chain_id": mapping.get("chain_id", ""),
                    "scop_id": scop_code,
                    "scop_class": scop_info.get("sccs", ""),
                    "scop_name": scop_info.get("description", ""),
                })
        return domains
    except Exception as e:
        log.debug(f"SCOP query error for {pdb_id}: {e}")
    return []


def classify_architecture(cath_id: str, pdb_id: str) -> str:
    """
    Map a CATH code to a human-readable architecture class.
    Returns one of the keys in ARCH_COLORS.

    Priority order:
      1. Manually curated KNOWN_ARCHITECTURES (exact PDB match)
      2. CATH superfamily codes (sfam-level, 3 fields)
      3. CATH class-level fallback (single digit)
    """
    if not cath_id:
        return "unknown / surrogate"
    # Check manually curated known architectures first
    if pdb_id in KNOWN_ARCHITECTURES:
        return KNOWN_ARCHITECTURES[pdb_id][2]
    # CATH superfamily-level classification
    sfam = ".".join(cath_id.split(".")[:3])
    full = cath_id.strip()

    if any(sfam.startswith(c) or full.startswith(c) for c in EF_HAND_CATH_CODES):
        return "ef-hand"
    if any(sfam.startswith(c) or full.startswith(c) for c in BETA_PROP_CATH_CODES):
        return "beta-propeller"
    # ── New Ca²⁺-binding folds (added for Ln³⁺ dataset expansion) ────────
    if any(sfam.startswith(c) or full.startswith(c) for c in C2_DOMAIN_CATH_CODES):
        return "c2-domain"
    if any(sfam.startswith(c) or full.startswith(c) for c in ANNEXIN_CATH_CODES):
        return "annexin"
    if any(sfam.startswith(c) or full.startswith(c) for c in EGF_CA2_CATH_CODES):
        return "egf-ca2"
    if any(sfam.startswith(c) or full.startswith(c) for c in GLA_CATH_CODES):
        return "gla-domain"
    if any(sfam.startswith(c) or full.startswith(c) for c in CADHERIN_CATH_CODES):
        return "cadherin"
    # ── CATH class-level fallback ─────────────────────────────────────────
    cls = cath_id.split(".")[0] if "." in cath_id else cath_id
    cls_map = {
        "1": "all-alpha",
        "2": "all-beta",
        "3": "alpha-beta",
        "4": "few-secondary",
    }
    return cls_map.get(cls, "unknown / surrogate")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MAIN CLASSIFICATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def classify_all_structures(seqs_csv: Path = DATA_DIR / "pdb_sequences_raw.csv") -> pd.DataFrame:
    """
    For every PDB entry in the sequences CSV:
      1. Query CATH via SIFTS
      2. Fall back to manually curated KNOWN_ARCHITECTURES
      3. Assign architecture_class
      4. Flag EF-hand entries

    Returns enriched DataFrame.
    """
    if not seqs_csv.exists():
        log.error(f"Input file not found: {seqs_csv}")
        log.error("Run 01_pdb_miner.py first.")
        return pd.DataFrame()

    seqs_df = pd.read_csv(seqs_csv)
    unique_ids = seqs_df["pdb_id"].unique()
    log.info(f"Classifying architectures for {len(unique_ids)} PDB entries ...")

    all_domains = []
    for i, pdb_id in enumerate(unique_ids):
        # Try SIFTS CATH first
        domains = get_cath_domains(pdb_id)
        if not domains:
            # Fall back to known annotations
            if pdb_id in KNOWN_ARCHITECTURES:
                cath_code, name, arch = KNOWN_ARCHITECTURES[pdb_id]
                domains = [{
                    "pdb_id": pdb_id,
                    "chain_id": "*",
                    "cath_id": cath_code,
                    "cath_class": cath_code.split(".")[0],
                    "cath_arch": ".".join(cath_code.split(".")[:2]),
                    "cath_topology": ".".join(cath_code.split(".")[:3]),
                    "superfamily_name": name,
                    "domain_name": name,
                    "source": "manual",
                }]
            else:
                domains = [{
                    "pdb_id": pdb_id,
                    "chain_id": "*",
                    "cath_id": "",
                    "superfamily_name": "unknown",
                    "source": "no_cath",
                }]

        for d in domains:
            d["source"] = d.get("source", "sifts_cath")
            d["architecture_class"] = classify_architecture(
                d.get("cath_id", ""), pdb_id
            )
            d["is_ef_hand"] = d["architecture_class"] == "ef-hand"
            d["is_novel_arch"] = d["architecture_class"] not in ("ef-hand", "unknown / surrogate")

        all_domains.extend(domains)

        if (i + 1) % 20 == 0:
            log.info(f"  {i+1}/{len(unique_ids)} processed")
        time.sleep(REQUEST_PAUSE)

    domains_df = pd.DataFrame(all_domains)

    # Merge back with sequence data
    result = seqs_df.merge(
        domains_df[["pdb_id", "cath_id", "cath_arch", "cath_topology",
                    "superfamily_name", "architecture_class",
                    "is_ef_hand", "is_novel_arch", "source"]].drop_duplicates("pdb_id"),
        on="pdb_id",
        how="left",
    )
    result["architecture_class"] = result["architecture_class"].fillna("unknown / surrogate")
    result["is_ef_hand"] = result["is_ef_hand"].fillna(False)
    result["is_novel_arch"] = result["is_novel_arch"].fillna(False)

    out_path = DATA_DIR / "architecture_classified.csv"
    result.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}  ({len(result)} rows)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DIVERSITY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fold_diversity(df: pd.DataFrame, out_path: Path = FIG_DIR / "fold_diversity.png"):
    """
    Two-panel figure:
      Left:  Bar chart — count of unique PDB entries per architecture class
      Right: Stacked bar chart — metal type breakdown within each architecture
    """
    if df.empty:
        log.warning("No data to plot.")
        return

    # Merge with contacts if available to get metal info
    contacts_path = DATA_DIR / "pdb_contacts_raw.csv"
    if contacts_path.exists():
        contacts = pd.read_csv(contacts_path)[["pdb_id", "metal_type", "metal_name"]].drop_duplicates("pdb_id")
        df = df.merge(contacts, on="pdb_id", how="left")
        df["metal_type"] = df["metal_type"].fillna("unknown")
    else:
        df["metal_type"] = "unknown"

    # Per-architecture count
    arch_counts = (
        df.drop_duplicates("pdb_id")
        .groupby("architecture_class")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Left panel: architecture bar chart ──────────────────────────────────
    ax = axes[0]
    colors = [ARCH_COLORS.get(arch, "#95A5A6") for arch in arch_counts["architecture_class"]]
    bars = ax.barh(arch_counts["architecture_class"], arch_counts["count"],
                   color=colors, edgecolor="white", linewidth=1.2)
    for bar, count in zip(bars, arch_counts["count"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_xlabel("Number of unique PDB entries", fontsize=11)
    ax.set_title("REE-Binding Architectures\n(all PDB hits)", fontsize=13, fontweight="bold")
    ax.set_facecolor("#FAFAFA")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Annotate the EF-hand bar
    ef_idx = list(arch_counts["architecture_class"]).index("ef-hand") \
        if "ef-hand" in list(arch_counts["architecture_class"]) else -1
    if ef_idx >= 0:
        ax.get_yticklabels()[ef_idx].set_color("#E74C3C")
        ax.get_yticklabels()[ef_idx].set_fontweight("bold")

    # ── Right panel: metal type breakdown ───────────────────────────────────
    ax2 = axes[1]
    metal_colors = {
        "lanthanide": "#3498DB",
        "surrogate":  "#E67E22",
        "unknown":    "#BDC3C7",
    }
    pivot = (
        df.drop_duplicates("pdb_id")
        .groupby(["architecture_class", "metal_type"])
        .size()
        .unstack(fill_value=0)
    )
    pivot = pivot.reindex(arch_counts["architecture_class"])
    pivot.plot(
        kind="barh",
        ax=ax2,
        color=[metal_colors.get(c, "#95A5A6") for c in pivot.columns],
        edgecolor="white",
        linewidth=1.0,
    )
    ax2.set_xlabel("Number of unique PDB entries", fontsize=11)
    ax2.set_title("Metal Type by Architecture\n(lanthanide vs. surrogate)", fontsize=13, fontweight="bold")
    ax2.set_facecolor("#FAFAFA")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(title="Metal type", loc="lower right")

    fig.suptitle(
        "Architectural Diversity of REE-Binding Proteins in PDB\n"
        "Goal: move beyond EF-hand dominance",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved figure: {out_path}")

    # Print diversity summary
    n_total = len(df.drop_duplicates("pdb_id"))
    n_ef = int(df.drop_duplicates("pdb_id")["is_ef_hand"].sum())
    n_novel = int(df.drop_duplicates("pdb_id")["is_novel_arch"].sum())
    print(f"\nArchitecture diversity summary:")
    print(f"  Total unique PDB entries: {n_total}")
    print(f"  EF-hand (known bias):     {n_ef}  ({100*n_ef/max(n_total,1):.1f}%)")
    print(f"  Novel architectures:      {n_novel}  ({100*n_novel/max(n_total,1):.1f}%)")
    print(f"  Unknown (surrogates etc): {n_total - n_ef - n_novel}")
    print(f"\nBreakdown by architecture:")
    print(arch_counts.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    result_df = classify_all_structures()
    if not result_df.empty:
        plot_fold_diversity(result_df)
        print(f"\nTop architectures found:")
        print(result_df.drop_duplicates("pdb_id")[
            ["pdb_id", "architecture_class", "cath_id", "superfamily_name",
             "is_ef_hand", "is_novel_arch"]
        ].sort_values("architecture_class").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
