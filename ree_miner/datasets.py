"""
REE Training Dataset Builder
==============================
Integrates all sources into a single, ML-ready training dataset:

  Source 1: PDB binding contacts (01_pdb_miner.py)        — high confidence, structural labels
  Source 2: PDB sequences + architecture classes (02)      — fold diversity labels
  Source 3: UniProt homologs + motif hits (03)             — sequence-level expansion
  Source 4: CaM-family engineering entries (06)           — WT negatives + D→P predicted positives
  Source 5: Literature Kd/Km table                         — regression labels

Produces:
  datasets/training_dataset.csv       ← main ML training table
  datasets/training_dataset_esm.json  ← ESM-Bind compatible format
  datasets/label_summary.csv          ← label distribution report

Key design decisions (matching ESM-Bind methodology):
  - Cluster at 30% sequence identity (greedy algorithm, no cd-hit needed)
  - Positive class oversampled 3x (same as ESM-Bind for class imbalance)
  - AUPRC is the primary evaluation metric (not accuracy)
  - Both binary (binds/doesn't) AND architecture-level labels included
  - Kd values stored as log10(Kd/M) for regression tasks

Usage:
    python 04_dataset_builder.py
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.Align import substitution_matrices

from ree_miner._workspace import DATA_DIR, LOG_DIR
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "dataset_builder.log"),
    ],
)
log = logging.getLogger("dataset_builder")

BLOSUM62 = substitution_matrices.load("BLOSUM62")

# ─── Literature-curated Kd/Km labels ─────────────────────────────────────────
# Extracted directly from Papers 3 and 5. These provide regression labels
# (log10 Kd in M) for the training dataset.
# Format: {uniprot_acc_or_pdb: {metal: Kd_M}}
LITERATURE_KD = {
    # Hans-LanM (Paper 5, Table 2)
    "A0A0D6MGU0": {   # Hans-LanM UniProt approximate
        "La": 68e-12, "Nd": 91e-12, "Dy": 2600e-12,
        "Pr": 100e-12, "Sm": 150e-12,
    },
    # Mex-LanM (Papers 3, 5)
    "A6ULZ8": {
        "La": 100e-12, "Nd": 100e-12, "Dy": 200e-12,
        "Pr":  85e-12, "Eu": 120e-12,
    },
    # LBT (Lanthanide Binding Tag) — synthetic, Paper 3
    "LBT_synthetic": {
        "Tb": 57e-9,   # 57 nM
    },
    # TIM barrel de novo — picomolar, Paper 3
    "TIM_denovo": {
        "Tb": 1e-12,   # picomolar (approximate lower bound)
    },
}

# Literature Km values for XoxF-MDH subtypes (Paper 5, Table 1)
# Format: {pdb_id: {metal: Km_uM}}
LITERATURE_KM = {
    "4MAE": {"La": 0.56, "Ce": 0.43, "Pr": 0.48, "Nd": 0.61},
    "6FKW": {"La": 0.23, "Ce": 0.18, "Pr": 0.25},
    "6OC6": {"La": 1.2,  "Ce": 0.9},
    "6ZCW": {"La": 0.31, "Nd": 0.28, "Dy": 8.4},   # PedH
}

# Architecture novelty score (for ranking in training set curation)
ARCH_NOVELTY = {
    "ef-hand":              0,   # common — down-weight
    "beta-propeller":       1,   # important but known
    "beta-roll (RTX)":      3,   # novel, high priority
    "pepsy-domain":         3,   # novel, high priority
    "tim-barrel":           4,   # de novo — very novel
    "beta-barrel (TonB-receptor)": 4,  # unexplored class
    "alpha-beta (ABC-transport)":  3,  # novel architecture
    "coiled-coil":          2,
    "all-alpha":            2,
    "all-beta":             2,
    "alpha-beta":           1,
    "unknown / surrogate":  1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SEQUENCE CLUSTERING (greedy, 30% identity — mirrors ESM-Bind)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sequence_identity(seq_a: str, seq_b: str) -> float:
    """
    Compute pairwise sequence identity using global alignment (BLOSUM62).
    Returns fraction [0, 1]. Uses shorter sequence as denominator.
    """
    if not seq_a or not seq_b:
        return 0.0
    # Use local alignment for speed on long sequences
    try:
        alns = pairwise2.align.globalds(
            seq_a[:500], seq_b[:500],   # cap at 500aa for performance
            BLOSUM62, -10, -0.5,
            score_only=False, one_alignment_only=True,
        )
        if not alns:
            return 0.0
        aln = alns[0]
        aligned_a, aligned_b = aln.seqA, aln.seqB
        matches = sum(a == b and a != "-" for a, b in zip(aligned_a, aligned_b))
        aln_len = sum(1 for a, b in zip(aligned_a, aligned_b) if a != "-" or b != "-")
        return matches / max(aln_len, 1)
    except Exception:
        # Fallback: k-mer based identity estimate
        k = 3
        def kmers(s): return set(s[i:i+k] for i in range(len(s)-k+1))
        k_a, k_b = kmers(seq_a), kmers(seq_b)
        if not k_a or not k_b:
            return 0.0
        return len(k_a & k_b) / len(k_a | k_b)


def cluster_sequences(df: pd.DataFrame, identity_threshold: float = 0.30,
                       seq_col: str = "sequence") -> pd.DataFrame:
    """
    Greedy sequence clustering at `identity_threshold` (default 30%).
    Each sequence is either a cluster representative or assigned to the
    nearest cluster it exceeds the threshold with.

    Adds columns: cluster_id, is_representative.
    Performance note: O(n^2) — acceptable for datasets up to ~2000 sequences.
    For larger sets, use MMseqs2 or pre-filter by length.
    """
    seqs = df[seq_col].tolist()
    n = len(seqs)
    log.info(f"Clustering {n} sequences at {identity_threshold*100:.0f}% identity ...")

    representatives = []   # (rep_idx, rep_seq)
    cluster_id = [-1] * n

    for i, seq in enumerate(seqs):
        assigned = False
        for rep_idx, rep_seq in representatives:
            if abs(len(seq) - len(rep_seq)) / max(len(rep_seq), 1) > 0.5:
                continue  # skip obviously different lengths
            identity = compute_sequence_identity(seq, rep_seq)
            if identity >= identity_threshold:
                cluster_id[i] = rep_idx
                assigned = True
                break
        if not assigned:
            representatives.append((i, seq))
            cluster_id[i] = i

        if (i + 1) % 100 == 0:
            log.info(f"  {i+1}/{n} clustered, {len(representatives)} clusters so far")

    df = df.copy()
    df["cluster_id"] = cluster_id
    df["is_representative"] = [cid == i for i, cid in enumerate(cluster_id)]
    log.info(f"Clustering complete: {n} sequences → {len(representatives)} clusters")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LABEL ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign training labels to each sequence:
      label_binary:     1 = REE-binding, 0 = non-binding (negative control)
      label_arch:       architecture class string
      label_novelty:    novelty score (higher = rarer architecture)
      log10_Kd:         log10(Kd/M) if available, else NaN
      log10_Km_uM:      log10(Km/uM) if available, else NaN
      lree_selective:   1 if LREE-selective, -1 if HREE, 0 if unknown
      acid_stable:      1 if binding confirmed at pH < 3
    """
    df = df.copy()

    # Binary label
    df["label_binary"] = df.apply(
        lambda r: 0 if r.get("is_negative", False) else 1, axis=1
    )

    # Architecture label
    df["label_arch"] = df.get("architecture_class", df.get("architecture_inferred", "unknown"))

    # Novelty score
    df["label_novelty"] = df["label_arch"].map(
        lambda a: ARCH_NOVELTY.get(a, 1)
    )

    # Log10 Kd (from literature)
    def get_log_kd(row):
        pid = str(row.get("protein_id", row.get("uniprot_acc", "")))
        if pid in LITERATURE_KD:
            kds = list(LITERATURE_KD[pid].values())
            if kds:
                return round(math.log10(min(kds)), 3)
        return float("nan")

    df["log10_Kd"] = df.apply(get_log_kd, axis=1)

    # Log10 Km (from literature, per PDB ID)
    def get_log_km(row):
        pid = str(row.get("pdb_id", ""))
        if pid in LITERATURE_KM:
            kms = list(LITERATURE_KM[pid].values())
            if kms:
                return round(math.log10(min(kms)), 3)
        return float("nan")

    df["log10_Km_uM"] = df.apply(get_log_km, axis=1)

    # LREE vs HREE selectivity
    def get_lree_selectivity(row):
        pid = str(row.get("protein_id", row.get("uniprot_acc", "")))
        if pid in LITERATURE_KD:
            kds = LITERATURE_KD[pid]
            lree = {m: v for m, v in kds.items() if m in ("La", "Ce", "Pr", "Nd", "Sm")}
            hree = {m: v for m, v in kds.items() if m in ("Dy", "Ho", "Er", "Yb", "Lu")}
            if lree and hree:
                avg_lree = sum(lree.values()) / len(lree)
                avg_hree = sum(hree.values()) / len(hree)
                if avg_lree < avg_hree * 0.5:
                    return 1   # LREE selective
                elif avg_hree < avg_lree * 0.5:
                    return -1  # HREE selective
        return 0   # unknown or non-selective

    df["lree_selective"] = df.apply(get_lree_selectivity, axis=1)

    # Acid stability (from Paper 4 — lanmodulin and RTX in A. ferrooxidans at pH 1.8)
    acid_stable_accs = {"A6ULZ8", "A0A0D6MGU0"}  # Mex-LanM, Hans-LanM equivalents
    rtx_keywords = ["rtx", "repeat-in-toxin", "adenylate cyclase"]
    # Thermoacidophiles: proteins from Sulfolobus, Acidithiobacillus, Picrophilus etc.
    thermoacidophile_keywords = ["sulfolobus", "acidithiobacillus", "picrophilus",
                                  "thermoplasma", "ferroplasma"]
    def is_acid_stable(row):
        pid = str(row.get("protein_id", ""))
        if pid in acid_stable_accs:
            return 1
        gene = str(row.get("gene_names", "")).lower()
        arch = str(row.get("label_arch", "")).lower()
        org  = str(row.get("organism", "")).lower()
        if any(k in gene or k in arch for k in rtx_keywords):
            return 1
        # CaM-family from thermoacidophiles (e.g. SaCaM from Sulfolobus)
        if any(k in org for k in thermoacidophile_keywords):
            return 1
        # Propagate acid_stable flag from CaM engineering entries
        if row.get("acid_stable") in (True, 1, "True", "1"):
            return 1
        return 0
    df["acid_stable"] = df.apply(is_acid_stable, axis=1)

    # ── Engineering labels (for CaM-family entries from 06_efhand_engineering.py) ──
    # is_engineered:     True = D→P point mutant (predicted, not experimentally verified)
    # engineering_score: 0–10 float from the EF-hand engineering model
    # pos2_aa:           Wild-type residue at loop position 2 (P = already selective)
    # cam_subfamily:     EF-hand protein subfamily (calmodulin, parvalbumin, s100, etc.)
    if "is_engineered" not in df.columns:
        df["is_engineered"] = False
    if "engineering_score" not in df.columns:
        df["engineering_score"] = float("nan")
    if "pos2_aa" not in df.columns:
        df["pos2_aa"] = ""
    if "cam_subfamily" not in df.columns:
        df["cam_subfamily"] = ""

    # Propagate source-level cam fields
    def infer_cam_subfamily(row):
        src = str(row.get("source", "")).lower()
        sub = str(row.get("subfamily", "")).lower()
        if sub:
            return sub
        if "cam" in src:
            return "cam_family"
        return ""
    df["cam_subfamily"] = df.apply(
        lambda r: r.get("cam_subfamily") or infer_cam_subfamily(r), axis=1
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLASS BALANCING (mirrors ESM-Bind: 3x oversample positives)
# ═══════════════════════════════════════════════════════════════════════════════

def balance_dataset(df: pd.DataFrame, positive_oversample: int = 3) -> pd.DataFrame:
    """
    Oversample the positive (REE-binding) class to address class imbalance.
    ESM-Bind used 3x oversampling for minority (binding) class.
    Upweights novel architectures (higher novelty score = more copies).
    """
    pos = df[df["label_binary"] == 1].copy()
    neg = df[df["label_binary"] == 0].copy()

    if pos.empty:
        log.warning("No positive examples found — returning unbalanced dataset.")
        return df

    # Weight novel architectures more heavily
    pos["sample_weight"] = pos["label_novelty"].clip(1, 4)
    weighted_pos = pos.sample(
        n=len(pos) * positive_oversample,
        replace=True,
        weights="sample_weight",
        random_state=42,
    ).drop(columns="sample_weight")

    balanced = pd.concat([weighted_pos, neg], ignore_index=True).sample(
        frac=1, random_state=42
    )
    log.info(f"Dataset balanced: {len(pos)} pos → {len(weighted_pos)} (3x, novelty-weighted), "
             f"{len(neg)} neg → final {len(balanced)} rows")
    return balanced


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ESM-BIND COMPATIBLE JSON EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_esm_format(df: pd.DataFrame,
                      out_path: Path = DATA_DIR / "training_dataset_esm.json"):
    """
    Export dataset in the format expected by ESM-Bind's training code:
    List of dicts with:
      {sequence, label, binding_positions, metal, architecture, log10_Kd, ...}
    """
    records = []
    for _, row in df.iterrows():
        if not row.get("sequence"):
            continue
        binding_pos = row.get("binding_positions", "")
        if isinstance(binding_pos, str) and binding_pos.startswith("["):
            try:
                binding_pos = json.loads(binding_pos)
            except Exception:
                binding_pos = []
        records.append({
            "protein_id":       str(row.get("protein_id", row.get("pdb_id", ""))),
            "sequence":         str(row["sequence"]),
            "label_binary":     int(row.get("label_binary", 0)),
            "binding_positions": binding_pos if isinstance(binding_pos, list) else [],
            "metal_code":       str(row.get("metal_code", row.get("metal_name", ""))),
            "architecture":     str(row.get("label_arch", "unknown")),
            "log10_Kd":         None if pd.isna(row.get("log10_Kd", float("nan")))
                                     else float(row["log10_Kd"]),
            "log10_Km_uM":      None if pd.isna(row.get("log10_Km_uM", float("nan")))
                                     else float(row["log10_Km_uM"]),
            "lree_selective":   int(row.get("lree_selective", 0)),
            "acid_stable":      int(row.get("acid_stable", 0)),
            "is_representative":bool(row.get("is_representative", True)),
            "source":           str(row.get("source", "")),
            "resolution":       float(row["resolution"]) if row.get("resolution")
                                     and not pd.isna(row.get("resolution")) else None,
            # CaM engineering fields (populated for EF-hand superfamily entries)
            "is_engineered":    bool(row.get("is_engineered", False)),
            "engineering_score":float(row["engineering_score"])
                                     if row.get("engineering_score") is not None
                                     and not pd.isna(float(row.get("engineering_score", float("nan"))))
                                     else None,
            "pos2_aa":          str(row.get("pos2_aa", "")),
            "cam_subfamily":    str(row.get("cam_subfamily", "")),
            "loop_seq":         str(row.get("loop_seq", "")),
        })

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    log.info(f"ESM-Bind format exported: {out_path}  ({len(records)} records)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_dataset() -> pd.DataFrame:
    """
    Combine all sources, cluster, label, balance, and export the final
    training dataset.
    """
    log.info("=" * 60)
    log.info("TRAINING DATASET BUILDER")
    log.info("=" * 60)

    dfs = []

    # ── Source 1: PDB contacts (binding residue level) ────────────────────
    pdb_contacts_path = DATA_DIR / "pdb_contacts_raw.csv"
    pdb_seqs_path     = DATA_DIR / "pdb_sequences_raw.csv"
    arch_path         = DATA_DIR / "architecture_classified.csv"

    if arch_path.exists():
        pdb_df = pd.read_csv(arch_path)
        pdb_df["protein_id"] = pdb_df["pdb_id"] + "_" + pdb_df["chain_id"].astype(str)
        pdb_df["source"] = "pdb_structure"
        dfs.append(pdb_df)
        log.info(f"Source 1 (PDB/architecture): {len(pdb_df)} rows")
    elif pdb_seqs_path.exists():
        pdb_df = pd.read_csv(pdb_seqs_path)
        pdb_df["protein_id"] = pdb_df["pdb_id"] + "_" + pdb_df["chain_id"].astype(str)
        pdb_df["source"] = "pdb_structure"
        dfs.append(pdb_df)
        log.info(f"Source 1 (PDB sequences only): {len(pdb_df)} rows")
    else:
        log.warning("No PDB data found. Run 01_pdb_miner.py first.")

    # ── Source 2: Binding contacts annotation ────────────────────────────
    if pdb_contacts_path.exists():
        contacts_df = pd.read_csv(pdb_contacts_path)
        # Aggregate binding residues per PDB/chain pair
        bp_agg = (
            contacts_df.groupby("pdb_id")["binding_seqnum"]
            .apply(list)
            .reset_index()
            .rename(columns={"binding_seqnum": "binding_positions"})
        )
        if dfs:
            dfs[0] = dfs[0].merge(bp_agg, on="pdb_id", how="left")
        log.info(f"Binding positions merged from contacts")

    # ── Source 3: UniProt homologs ────────────────────────────────────────
    homologs_path = DATA_DIR / "all_homologs.csv"
    if homologs_path.exists():
        hom_df = pd.read_csv(homologs_path)
        dfs.append(hom_df)
        log.info(f"Source 3 (homologs): {len(hom_df)} rows")
    else:
        log.info("No homolog file found. Run 03_homolog_finder.py.")

    # ── Source 4: CaM-family engineering candidates (06_efhand_engineering.py) ──
    # Wild-type CaM-family EF-hand proteins (label_binary=0, negative for REE)
    # + D→P engineered mutants (label_binary=1, predicted REE-selective)
    cam_entries_path = DATA_DIR / "cam_dataset_entries.json"
    if cam_entries_path.exists():
        with open(cam_entries_path) as f:
            cam_raw = json.load(f)
        if cam_raw:
            cam_df = pd.DataFrame(cam_raw)
            # Map JSON fields to pipeline-standard column names
            if "label_binary" not in cam_df.columns:
                cam_df["label_binary"] = cam_df.get("is_ree_selective", 0).astype(int)
            if "architecture_class" not in cam_df.columns:
                cam_df["architecture_class"] = "ef-hand"
            if "architecture_inferred" not in cam_df.columns:
                cam_df["architecture_inferred"] = "ef-hand"
            cam_df["source"] = cam_df.get("source", "cam_family_curated")
            dfs.append(cam_df)
            log.info(f"Source 4 (CaM engineering): {len(cam_df)} rows "
                     f"({cam_df.get('is_engineered', pd.Series([False]*len(cam_df))).sum()} engineered)")
    else:
        log.info("No CaM dataset entries found. Run 06_efhand_engineering.py.")

    if not dfs:
        log.error("No data sources available. Run steps 01-03 first.")
        return pd.DataFrame()

    # ── Concatenate all sources ───────────────────────────────────────────
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # Require sequence to be present
    combined = combined[combined["sequence"].notna() & (combined["sequence"].str.len() > 20)]
    log.info(f"After merging all sources: {len(combined)} rows with sequences")

    # ── Cluster at 30% identity ───────────────────────────────────────────
    combined = cluster_sequences(combined, identity_threshold=0.30)

    # ── Assign labels ─────────────────────────────────────────────────────
    combined = assign_labels(combined)

    # ── Save full (unbalanced) dataset ────────────────────────────────────
    full_path = DATA_DIR / "training_dataset_full.csv"
    combined.to_csv(full_path, index=False)
    log.info(f"Full dataset (unbalanced): {full_path}  ({len(combined)} rows)")

    # ── Balance dataset ────────────────────────────────────────────────────
    balanced = balance_dataset(combined, positive_oversample=3)
    balanced_path = DATA_DIR / "training_dataset.csv"
    balanced.to_csv(balanced_path, index=False)
    log.info(f"Balanced training dataset: {balanced_path}  ({len(balanced)} rows)")

    # ── ESM-Bind format export ────────────────────────────────────────────
    export_esm_format(balanced)

    # ── Label summary ─────────────────────────────────────────────────────
    label_summary = {
        "total_rows": len(balanced),
        "positive_examples": int(balanced["label_binary"].sum()),
        "negative_examples": int((balanced["label_binary"] == 0).sum()),
        "unique_architectures": balanced["label_arch"].nunique(),
        "architecture_breakdown": balanced["label_arch"].value_counts().to_dict(),
        "has_kd_label": int(balanced["log10_Kd"].notna().sum()),
        "has_km_label": int(balanced["log10_Km_uM"].notna().sum()),
        "lree_selective": int((balanced["lree_selective"] == 1).sum()),
        "acid_stable": int(balanced["acid_stable"].sum()),
        "unique_clusters": int(combined["cluster_id"].nunique()),
        "ef_hand_fraction": round(
            (balanced["label_arch"] == "ef-hand").sum() / max(len(balanced), 1), 3
        ),
    }
    summary_df = pd.DataFrame([label_summary])
    summary_path = DATA_DIR / "label_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    log.info("\n" + "=" * 60)
    log.info("DATASET SUMMARY")
    log.info("=" * 60)
    for k, v in label_summary.items():
        if not isinstance(v, dict):
            log.info(f"  {k}: {v}")

    return balanced


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dataset = build_training_dataset()
    if not dataset.empty:
        print("\nFinal training dataset preview:")
        display_cols = [c for c in ["protein_id", "seq_len", "label_binary",
                                     "label_arch", "label_novelty", "log10_Kd",
                                     "lree_selective", "acid_stable", "source"]
                        if c in dataset.columns]
        print(dataset[display_cols].head(20).to_string(index=False))
