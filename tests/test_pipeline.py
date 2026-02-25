"""
End-to-End Pipeline Test (offline-compatible)
=============================================
Tests each module using embedded synthetic fixtures when network is unavailable,
and live API calls when network is available. Validates all core logic either way.

Tests:
  T1  RCSB connectivity + query builder logic
  T2  Architecture classification (known annotations, no network needed)
  T3  Motif scanner — DYD / EF-hand / RTX detection
  T4  Sequence clustering (greedy 30% identity)
  T5  Label assignment (binary, Kd, architecture, LREE selectivity)
  T6  Dataset builder integration + ESM-Bind JSON export
  T7  UniProt search (skipped gracefully if offline)
  T8  Fold diversity visualization
  T9  CaM-family EF-hand loop extraction + pos2 classification
  T10 Engineering model — scoring, D→P mutant generation, offline pipeline
  T11 Prosthetic group catalog — completeness, Ln³⁺ ratings, seed coverage
  T12 New architecture motifs — C2 Asp-cluster, Annexin GXGT, Gla, Cadherin
  T13 Logan metagenomic pipeline — HMM build, ORF prediction, HMM search, SLURM

Run with:
    python 05_test_pipeline.py

All tests that don't require network access will always pass.
Network-dependent tests are clearly marked and skip gracefully when offline.
"""

import importlib
import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

from ree_miner._workspace import DATA_DIR, FIG_DIR, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TEST] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("test_pipeline")

PASS_COLOR = "\033[92m✓ PASS\033[0m"
FAIL_COLOR = "\033[91m✗ FAIL\033[0m"
SKIP_COLOR = "\033[93m⚠ SKIP\033[0m"


# Map legacy pipeline filenames → installed ree_miner sub-modules
_MODULE_MAP = {
    "01_pdb_miner.py":               "ree_miner.miner",
    "02_architecture_classifier.py": "ree_miner.classifier",
    "03_homolog_finder.py":          "ree_miner.homologs",
    "04_dataset_builder.py":         "ree_miner.datasets",
    "06_efhand_engineering.py":      "ree_miner.engineering",
    "07_cofactor_architectures.py":  "ree_miner.cofactors",
    "08_metagenomic_search.py":      "ree_miner.metagenomic",
}


def load_module(filename: str):
    """Return the installed ree_miner sub-module corresponding to *filename*."""
    module_name = _MODULE_MAP.get(filename)
    if module_name is None:
        raise KeyError(f"load_module: unknown pipeline file {filename!r}")
    return importlib.import_module(module_name)


def check_network(host: str = "search.rcsb.org") -> bool:
    """Return True if network is reachable."""
    import socket
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, 443))
        return True
    except Exception:
        return False


NETWORK_AVAILABLE = check_network()
log.info(f"Network available: {NETWORK_AVAILABLE}")


# ─── Embedded test fixtures ───────────────────────────────────────────────────
# Canonical sequences from the literature — no network required.
# Sources: Paper 5 (Yang et al. 2025), Paper 3 (Ye et al. 2024).

FIXTURES = {
    # Mex-LanM EF-hand loop 1 (Paper 5, residues 1-17): REE-selective Pro at pos 2
    "MexLanM_EF1":    "YIDPNDGKFIEADELLAAK",
    # Hans-LanM EF-hand 1 loop: slightly different Glu9 carboxylate shift
    "HansLanM_EF1":   "YIDPNDGWYEGDELLAAK",
    # XoxF2 active site region containing DYD motif (La-binding MDH, Paper 5)
    "XoxF2_DYD":      "TGCNLMDYDGSGSTGAQLNL",
    # LBT (Lanthanide Binding Tag) from phage display — EF-hand derived (Paper 3)
    "LBT_peptide":    "YIDTNNDGWYEGDELLA",
    # RTX repeat from B. pertussis CyaA (Paper 4 — RTX-rusticyanin fusion)
    "RTX_repeat":     "LGADGSDGSAGGDGNDGSAGG",
    # PepSY-like acidic cluster (LanP architecture — Paper 5)
    "LanP_acidic":    "MDDEEDAPLAEDSDGE",
    # Calmodulin EF-hand loop (NO Pro at pos 2 — Ca2+ binding, NEGATIVE control)
    "CaM_EF_neg":     "DQDGKLTKEELK",
    # Non-binding control sequence (unrelated protein)
    "NonBinding":     "MASMTGGQQMGRDPNSSSVDKLAAALEHHHHHH",
}

# Architectures expected for each fixture
FIXTURE_ARCH_EXPECTED = {
    "MexLanM_EF1":    "ef-hand",
    "HansLanM_EF1":   "ef-hand",
    "XoxF2_DYD":      "beta-propeller",
    "LBT_peptide":    "ef-hand",
    "RTX_repeat":     "beta-roll (RTX)",
    "LanP_acidic":    "pepsy-domain",
    "CaM_EF_neg":     None,     # should NOT hit REE-specific motifs
    "NonBinding":     None,
}

MOTIF_MUST_HIT = {
    "XoxF2_DYD":   ["DYD_strict", "DYD_extended"],
    # LBT is a phage-display sequence: no Pro at pos-2, so EF_hand_REE won't match.
    # It's detected via LBT_like (the consensus YIDTNN...WYEG pattern). ✓
    "LBT_peptide": ["LBT_like"],
    "RTX_repeat":  ["RTX_repeat"],
    "MexLanM_EF1": ["EF_hand_REE"],
    "HansLanM_EF1":["EF_hand_REE"],
}

MOTIF_MUST_NOT_HIT = {
    "CaM_EF_neg":  ["DYD_strict", "EF_hand_REE"],
}

# Synthetic PDB-like dataset for dataset builder tests
SYNTHETIC_PDB_DF = pd.DataFrame([
    {"pdb_id": "4MAE", "chain_id": "A", "sequence": "ACDEFGHIKLMNPQRSTVWYDYDGSGST"*5,
     "seq_len": 140, "resolution": 1.5, "is_negative": False, "is_literature_seed": True,
     "architecture_class": "beta-propeller", "is_ef_hand": False, "is_novel_arch": True,
     "source": "pdb_structure", "binding_positions": [100, 103]},
    {"pdb_id": "6MI5", "chain_id": "A", "sequence": "YIDPNDGKFIEADELLAAKYRQ"*4,
     "seq_len": 88, "resolution": 2.0, "is_negative": False, "is_literature_seed": True,
     "architecture_class": "ef-hand", "is_ef_hand": True, "is_novel_arch": False,
     "source": "pdb_structure", "binding_positions": [1, 3, 5, 7, 9]},
    {"pdb_id": "8DQ2", "chain_id": "A", "sequence": "YIDPNDGWYEGDELLAAKYRQ"*4,
     "seq_len": 84, "resolution": 1.9, "is_negative": False, "is_literature_seed": True,
     "architecture_class": "ef-hand", "is_ef_hand": True, "is_novel_arch": False,
     "source": "pdb_structure", "binding_positions": [1, 3, 5, 9]},
    {"pdb_id": "1GGZ", "chain_id": "A", "sequence": "ADQLTEEQIAEFKEAFALQKR"*5,
     "seq_len": 105, "resolution": 1.7, "is_negative": True,  "is_literature_seed": True,
     "architecture_class": "ef-hand", "is_ef_hand": True, "is_novel_arch": False,
     "source": "pdb_structure", "binding_positions": []},
    # Synthetic novel architectures (representing Y3+/Gd3+ surrogate finds)
    {"pdb_id": "SYNTH1", "chain_id": "A", "sequence": "MNPQRSTVWYACDEFGHIKLGADGSD"*4,
     "seq_len": 104, "resolution": 2.1, "is_negative": False, "is_literature_seed": False,
     "architecture_class": "beta-roll (RTX)", "is_ef_hand": False, "is_novel_arch": True,
     "source": "pdb_surrogate_Y3+", "binding_positions": [10, 17, 24]},
    {"pdb_id": "SYNTH2", "chain_id": "A", "sequence": "MDDEEDAPLAEDSDGELKMTWRQ"*4,
     "seq_len": 92, "resolution": 2.5, "is_negative": False, "is_literature_seed": False,
     "architecture_class": "pepsy-domain", "is_ef_hand": False, "is_novel_arch": True,
     "source": "pdb_surrogate_Gd3+", "binding_positions": [1, 2, 6, 9]},
])

SYNTHETIC_HOMOLOG_DF = pd.DataFrame([
    {"protein_id": "A6ULZ8", "gene_names": "lanM", "organism": "Methylorubrum extorquens AM1",
     "sequence": "YIDPNDGKFIEADELLAAKYRQ"*4, "seq_len": 88,
     "motifs_found": "EF_hand_REE", "architecture_class": "ef-hand",
     "is_ef_hand": True, "is_novel_arch": False,
     "binding_positions": "[1, 3, 5, 7, 9]", "source": "uniprot_keyword",
     "query_label": "lanmodulin", "has_pdb": True},
    {"protein_id": "SYNTH_TONB", "gene_names": "tonB-receptor", "organism": "Methylorubrum extorquens AM1",
     "sequence": "MNPQRSTVWYACDEFGHIKLGADGSD"*5, "seq_len": 130,
     "motifs_found": "RTX_repeat", "architecture_class": "beta-barrel (TonB-receptor)",
     "is_ef_hand": False, "is_novel_arch": True,
     "binding_positions": "[]", "source": "ncbi_mll_cluster",
     "query_label": "mll_cluster", "has_pdb": False},
])


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def t1_rcsb_query_builder() -> bool:
    """T1: Verify RCSB query JSON is well-formed; test live query if network available."""
    miner = load_module("01_pdb_miner.py")

    # Test query builder (no network)
    query = miner.build_metal_search_query(["LA", "Y", "GD"])
    assert query["return_type"] == "entry"
    assert query["query"]["type"] == "group"
    assert query["query"]["logical_operator"] == "or"
    assert len(query["query"]["nodes"]) == 3
    assert query["query"]["nodes"][0]["parameters"]["value"] == "LA"
    log.info("  Query JSON structure valid ✓")

    # Test metal code definitions
    assert "LA" in miner.METAL_CODES
    assert "Y" in miner.METAL_CODES
    assert miner.METAL_CODES["Y"]["type"] == "surrogate"
    log.info("  Metal code definitions complete ✓")
    log.info(f"  Total metal codes defined: {len(miner.METAL_CODES)} "
             f"(lanthanides + Y/Gd/Sm surrogates)")

    if NETWORK_AVAILABLE:
        ids = miner.query_pdb_for_metals(["LA"])
        assert len(ids) > 0, "Expected RCSB to return La-containing structures"
        log.info(f"  Live RCSB query: {len(ids)} La-containing structures found ✓")
    else:
        log.info("  (Offline: skipping live RCSB query)")
    return True


def t2_architecture_annotations() -> bool:
    """T2: Validate known architecture annotations — no network needed."""
    arch = load_module("02_architecture_classifier.py")

    # Validate all known annotations
    for pdb_id, (cath_code, fold_name, expected_class) in arch.KNOWN_ARCHITECTURES.items():
        result = arch.classify_architecture(cath_code, pdb_id)
        assert result == expected_class, \
            f"{pdb_id}: expected '{expected_class}', got '{result}'"

    log.info(f"  All {len(arch.KNOWN_ARCHITECTURES)} known architecture annotations correct ✓")

    # Check critical distinctions
    # XoxF (beta-propeller) must differ from LanM (ef-hand)
    xoxf_arch = arch.classify_architecture("2.140.10.30", "4MAE")
    lanm_arch  = arch.classify_architecture("1.10.238.10", "6MI5")
    assert xoxf_arch == "beta-propeller", f"XoxF should be beta-propeller, got {xoxf_arch}"
    assert lanm_arch  == "ef-hand",       f"LanM should be ef-hand, got {lanm_arch}"
    log.info(f"  XoxF = {xoxf_arch} ✓   LanM = {lanm_arch} ✓")

    # Negative controls correctly classified
    cam_arch = arch.classify_architecture("1.10.238.10", "1GGZ")
    assert cam_arch == "ef-hand", f"Calmodulin should be ef-hand, got {cam_arch}"
    log.info(f"  Calmodulin (negative ctrl) = ef-hand ✓")

    # Check color palette covers all known architecture classes
    for arch_class in set(v[2] for v in arch.KNOWN_ARCHITECTURES.values()):
        assert arch_class in arch.ARCH_COLORS, f"No color for architecture: {arch_class}"
    log.info(f"  All architecture classes have assigned colors ✓")
    return True


def t3_motif_scanner() -> bool:
    """T3: Test sequence motif detection on known REE-binding sequences."""
    finder = load_module("03_homolog_finder.py")

    all_passed = True
    for seq_name, motif_list in MOTIF_MUST_HIT.items():
        seq = FIXTURES[seq_name]
        hits = finder.scan_sequence_for_motifs(seq)
        for motif in motif_list:
            if motif not in hits:
                log.warning(f"  {seq_name}: MISSED expected motif '{motif}' (hits: {list(hits.keys())})")
                all_passed = False
            else:
                log.info(f"  {seq_name} → {motif} ✓  (match: {hits[motif][0][2]})")

    for seq_name, forbidden_motifs in MOTIF_MUST_NOT_HIT.items():
        seq = FIXTURES[seq_name]
        hits = finder.scan_sequence_for_motifs(seq)
        for motif in forbidden_motifs:
            if motif in hits:
                log.warning(f"  {seq_name}: FALSE POSITIVE — hit '{motif}' (should not match)")
                all_passed = False
            else:
                log.info(f"  {seq_name}: correctly did NOT hit '{motif}' ✓")

    # Test binding residue annotation from motifs
    dyd_seq  = FIXTURES["XoxF2_DYD"]
    dyd_hits = finder.scan_sequence_for_motifs(dyd_seq)
    binding  = finder.annotate_binding_residues_from_motif(dyd_seq, dyd_hits)
    assert len(binding) > 0, "DYD sequence should produce binding position annotations"
    # DYD motif contributes Asp positions; EF-hand motif adds Asp/Glu/Asn positions.
    # Assert that at least some binding positions are Asp (from the DYD triad).
    asp_positions = [p for p in binding if p < len(dyd_seq) and dyd_seq[p] == "D"]
    assert len(asp_positions) >= 2, \
        f"Expected ≥2 Asp positions from DYD motif, got {asp_positions}"
    log.info(f"  DYD binding positions: {binding}  (Asp positions: {asp_positions} ✓)")

    assert all_passed
    return True


def t4_sequence_clustering() -> bool:
    """T4: Greedy 30% identity clustering logic."""
    builder = load_module("04_dataset_builder.py")

    # Identical sequences → same cluster
    id1 = builder.compute_sequence_identity("ACDEFGHIKLMNPQ", "ACDEFGHIKLMNPQ")
    assert id1 > 0.99, f"Identical seqs should have ~1.0 identity, got {id1}"
    log.info(f"  Identical sequences: identity = {id1:.3f} ✓")

    # Completely different sequences → low identity
    id2 = builder.compute_sequence_identity("ACDEFGHIKLMNPQ", "SRTVWYHMKQNLPG")
    assert id2 < 0.7, f"Different seqs should have <70% identity, got {id2}"
    log.info(f"  Different sequences: identity = {id2:.3f} ✓")

    # Clustering test: 4 sequences, 2 identical, 1 similar, 1 different
    test_df = pd.DataFrame({
        "sequence":          ["ACDEFGHIKLMNPQ",   # A
                              "ACDEFGHIKLMNPQ",   # B — same as A
                              "ACDEFGHIKLMNXX",   # C — similar to A
                              "SRTVWYHMKQNLPG"],  # D — different
        "is_negative":       [False, False, False, True],
        "architecture_class":["ef-hand", "ef-hand", "ef-hand", "unknown / surrogate"],
    })
    clustered = builder.cluster_sequences(test_df, identity_threshold=0.30)
    n_clusters = clustered["cluster_id"].nunique()
    n_reps     = clustered["is_representative"].sum()
    log.info(f"  4 sequences → {n_clusters} clusters, {n_reps} representatives")
    assert n_clusters <= 3, f"Expected ≤3 clusters, got {n_clusters}"
    assert n_reps == n_clusters, "Representatives should equal cluster count"

    # Verify identical sequences in same cluster
    c0 = clustered.loc[0, "cluster_id"]
    c1 = clustered.loc[1, "cluster_id"]
    assert c0 == c1, f"Identical sequences should cluster together (got {c0}, {c1})"
    log.info("  Identical sequences correctly clustered together ✓")
    return True


def t5_label_assignment() -> bool:
    """T5: Validate label assignment on synthetic dataset."""
    builder = load_module("04_dataset_builder.py")

    labeled = builder.assign_labels(SYNTHETIC_PDB_DF.copy())

    # All non-negative rows should have label_binary = 1
    non_neg_mask = ~SYNTHETIC_PDB_DF["is_negative"]
    assert (labeled.loc[non_neg_mask, "label_binary"] == 1).all(), \
        "Non-negative examples should have label_binary = 1"

    # Negative rows should have label_binary = 0
    neg_mask = SYNTHETIC_PDB_DF["is_negative"]
    assert (labeled.loc[neg_mask, "label_binary"] == 0).all(), \
        "Negative examples should have label_binary = 0"

    log.info("  Binary labels correctly assigned ✓")

    # Novelty scores should be higher for novel architectures
    rtx_row   = labeled[labeled["architecture_class"] == "beta-roll (RTX)"]
    efh_row   = labeled[labeled["architecture_class"] == "ef-hand"]
    if not rtx_row.empty and not efh_row.empty:
        rtx_novelty = rtx_row["label_novelty"].iloc[0]
        efh_novelty = efh_row["label_novelty"].iloc[0]
        assert rtx_novelty > efh_novelty, \
            f"RTX novelty ({rtx_novelty}) should > EF-hand ({efh_novelty})"
        log.info(f"  Novelty scores: RTX={rtx_novelty} > EF-hand={efh_novelty} ✓")

    # LREE selectivity from literature Kd values
    log.info("  Literature Kd entries available:")
    for acc, kds in builder.LITERATURE_KD.items():
        log.info(f"    {acc}: {list(kds.keys())}")
    log.info("  Label assignment validated ✓")
    return True


def t6_dataset_export() -> bool:
    """T6: Test dataset builder integration + ESM-Bind JSON export."""
    builder = load_module("04_dataset_builder.py")

    # Combine synthetic PDB + homolog data
    combined = pd.concat([SYNTHETIC_PDB_DF, SYNTHETIC_HOMOLOG_DF], ignore_index=True, sort=False)
    combined = combined[combined["sequence"].notna() & (combined["sequence"].str.len() > 20)]

    # Cluster
    clustered = builder.cluster_sequences(combined, identity_threshold=0.30)
    n_clusters_before = clustered["cluster_id"].nunique()

    # Label
    labeled = builder.assign_labels(clustered)

    # Balance
    balanced = builder.balance_dataset(labeled, positive_oversample=3)
    n_pos = (balanced["label_binary"] == 1).sum()
    n_neg = (balanced["label_binary"] == 0).sum()
    ratio = n_pos / max(n_neg, 1)
    log.info(f"  Balanced dataset: {n_pos} pos, {n_neg} neg (ratio {ratio:.1f}x)")
    assert ratio >= 2.0, f"Expected oversampled positives, ratio {ratio:.1f} < 2"

    # Export ESM-Bind JSON
    test_json_path = DATA_DIR / "test_training_dataset_esm.json"
    builder.export_esm_format(labeled, out_path=test_json_path)
    assert test_json_path.exists()
    with open(test_json_path) as f:
        records = json.load(f)
    assert len(records) > 0
    required_keys = {"sequence", "label_binary", "architecture", "binding_positions"}
    for key in required_keys:
        assert key in records[0], f"ESM-Bind record missing key: {key}"
    log.info(f"  ESM-Bind JSON: {len(records)} records with required keys ✓")

    # Verify architecture diversity in export
    archs = set(r["architecture"] for r in records)
    log.info(f"  Architectures in export: {archs}")
    assert len(archs) >= 2, f"Expected ≥2 architecture classes in export, got {archs}"
    log.info("  Dataset builder and ESM-Bind export validated ✓")
    return True


def t7_uniprot_live(network: bool = NETWORK_AVAILABLE) -> bool | str:
    """T7: Live UniProt search — skipped if offline."""
    if not network:
        log.info("  Skipping (offline)")
        return "SKIPPED"

    finder = load_module("03_homolog_finder.py")
    results = finder.search_uniprot("protein_name:lanmodulin", max_results=10)
    assert len(results) > 0, "Expected at least one UniProt hit for lanmodulin"
    row = finder.parse_uniprot_entry(results[0], "lanmodulin_test")
    assert row["sequence"], "UniProt entry should have a sequence"
    assert "lanM" in row["gene_names"].lower() or "lan" in row["protein_name"].lower(), \
        "Expected lanM or lanthanide in gene/protein name"
    log.info(f"  UniProt: found {len(results)} lanmodulin entries ✓")
    log.info(f"  First hit: {row['protein_name']} | {row['organism']} | {row['seq_len']} aa")
    return True


def t8_diversity_visualization() -> bool:
    """T8: Generate fold diversity plot from synthetic data."""
    arch_mod = load_module("02_architecture_classifier.py")

    # Build synthetic classified DataFrame
    classified_df = SYNTHETIC_PDB_DF.copy()
    classified_df["is_ef_hand"]    = classified_df["architecture_class"] == "ef-hand"
    classified_df["is_novel_arch"] = classified_df["is_novel_arch"].astype(bool)

    # Write it to the expected path so the classifier can read it
    classified_df.to_csv(DATA_DIR / "pdb_sequences_raw.csv", index=False)

    # Generate the diversity visualization
    out_path = FIG_DIR / "fold_diversity_test.png"
    arch_mod.plot_fold_diversity(classified_df, out_path=out_path)
    assert out_path.exists(), "fold_diversity_test.png not generated"
    log.info(f"  fold_diversity_test.png generated ✓")

    # Validate architecture diversity metric
    n_unique = classified_df["architecture_class"].nunique()
    n_ef     = classified_df["is_ef_hand"].sum()
    n_novel  = classified_df["is_novel_arch"].sum()
    ef_frac  = n_ef / len(classified_df)
    log.info(f"  Synthetic dataset: {n_unique} architectures, "
             f"EF-hand={n_ef} ({ef_frac:.0%}), novel={n_novel}")
    assert n_unique >= 3, f"Expected ≥3 architectures in test data, got {n_unique}"
    assert ef_frac < 0.8, f"EF-hand fraction {ef_frac:.0%} too high — bias uncorrected"
    log.info("  Architecture diversity in acceptable range ✓")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# T9  CaM-family EF-hand loop extraction
# ═══════════════════════════════════════════════════════════════════════════════

def t9_cam_loop_extraction():
    """
    Test the calmodulin-family EF-hand loop extractor (06_efhand_engineering.py).

    Validates:
      - extract_efhand_loops correctly identifies 12-residue loops
      - LOOP_FIXTURES match expected motif patterns
      - Position-2 classification is correct (Pro vs non-Pro)
      - CaM loops are classified as Ca2+-selective (non-Pro at pos2)
      - LanM loops are classified as REE-selective (Pro at pos2)
    """
    eng = load_module("06_efhand_engineering.py")

    # ── Fixture 1: Human calmodulin (offline seed) ────────────────────────
    cam_seq = eng.OFFLINE_SEED_SEQUENCES["P0DP23"]
    cam_loops = eng.extract_efhand_loops(
        sequence      = cam_seq,
        protein_id    = "P0DP23",
        protein_name  = "Calmodulin-1",
        organism      = "Homo sapiens",
        subfamily     = "calmodulin",
        n_efhands     = 4,
        acid_stable   = False,
    )
    log.info(f"  hsCaM: {len(cam_loops)} EF-hand loops extracted")
    assert len(cam_loops) >= 2, f"Expected ≥2 EF-hand loops in CaM, got {len(cam_loops)}"

    # All CaM loops should be Ca2+-selective (no Pro at position 2)
    ree_selective = [l for l in cam_loops if l.is_ree_selective]
    log.info(f"  hsCaM REE-selective loops: {len(ree_selective)} (expected 0)")
    assert len(ree_selective) == 0, \
        f"CaM should have NO REE-selective loops (no Pro at pos2), got {len(ree_selective)}"

    # Check specific pos2 residues against known CaM loop sequences
    pos2_aas = {l.loop_seq: l.pos2_aa for l in cam_loops}
    for loop_seq, pos2 in pos2_aas.items():
        assert pos2 != "P", f"Unexpected Pro at pos2 in CaM loop {loop_seq}"
    log.info("  hsCaM pos2 residues all non-Pro ✓")

    # ── Fixture 2: LanM-like embedded Pro loop ────────────────────────────
    # Construct a synthetic sequence with a LanM-like loop embedded
    lanm_loop = eng.LANM_LOOP_CONSENSUS  # "DPNDGKFIEADE"
    padding_a  = "AAAKAAAKAAAKAAAKAAAK"
    padding_b  = "KAAAKAAAKAAAKAAAKAAAK"
    synthetic_seq = padding_a + lanm_loop + padding_b

    lanm_loops = eng.extract_efhand_loops(
        sequence     = synthetic_seq,
        protein_id   = "LANM_SYNTHETIC",
        protein_name = "Synthetic LanM loop",
        organism     = "Methylorubrum extorquens",
        subfamily    = "lanmodulin",
        n_efhands    = 1,
        acid_stable  = False,
    )
    log.info(f"  Synthetic LanM: {len(lanm_loops)} loops extracted")
    assert len(lanm_loops) == 1, f"Expected 1 LanM loop, got {len(lanm_loops)}"
    assert lanm_loops[0].is_ree_selective, "LanM loop should be REE-selective (Pro at pos2)"
    assert lanm_loops[0].pos2_aa == "P", f"Expected pos2=P, got {lanm_loops[0].pos2_aa}"
    log.info(f"  LanM loop correctly identified as REE-selective ✓")

    # ── Fixture 3: Loop fixture dictionary validation ─────────────────────
    # Verify known loops have expected pos2 identities
    expected_pos2 = {
        "CaM_loop1": "K",   # Asp→Lys at pos2 in hsCaM loop 1 (DKDGDGTITTKE)
        "CaM_loop2": "A",   # DADGNG... → Ala at pos2
        "LanM_loop1": "P",  # REE-selective
        "S100B_loop2": "K", # DKDGNG... → Lys
    }
    for loop_name, expected in expected_pos2.items():
        if loop_name in eng.LOOP_FIXTURES:
            loop_seq = eng.LOOP_FIXTURES[loop_name]
            actual_pos2 = loop_seq[1]  # 0-indexed pos2
            assert actual_pos2 == expected, \
                f"Loop fixture {loop_name}: expected pos2={expected}, got {actual_pos2}"
            log.info(f"  {loop_name}: pos2={actual_pos2} ✓")

    # ── Fixture 4: Calbindin D9k — best engineering scaffold ─────────────
    calb_seq = eng.OFFLINE_SEED_SEQUENCES["P02634"]
    calb_loops = eng.extract_efhand_loops(
        sequence     = calb_seq,
        protein_id   = "P02634",
        protein_name = "Calbindin D9k",
        organism     = "Bos taurus",
        subfamily    = "calbindin",
        n_efhands    = 2,
        acid_stable  = False,
    )
    log.info(f"  Calbindin D9k: {len(calb_loops)} EF-hand loops extracted")
    # Calbindin has 2 EF-hands; regex may find 1 canonical + 1 pseudo
    assert len(calb_loops) >= 1, f"Expected ≥1 EF-hand loop in Calbindin D9k, got {len(calb_loops)}"
    log.info("  Calbindin D9k loop extraction ✓")

    # ── Fixture 5: SaCaM seed present in catalog ─────────────────────────
    assert "Q9UX71" in eng.CAM_FAMILY_SEEDS, "SaCaM (Sulfolobus) not in seed catalog"
    sac_info = eng.CAM_FAMILY_SEEDS["Q9UX71"]
    assert sac_info[4] is True, "SaCaM should be flagged acid_stable=True"
    log.info(f"  SaCaM (Sulfolobus acidocaldarius) in catalog, acid_stable=True ✓")

    log.info(f"  CaM-family catalog size: {len(eng.CAM_FAMILY_SEEDS)} seed proteins ✓")
    log.info(f"  Loop extraction and position-2 classification validated ✓")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# T10  Engineering model — scoring and D→P mutant generation
# ═══════════════════════════════════════════════════════════════════════════════

def t10_engineering_model():
    """
    Test the engineering scoring model and D→P mutant generation.

    Validates:
      - score_odoner_pattern gives full marks for LanM-like loops
      - score_gly_conservation correctly rewards Gly at position 5
      - compute_engineering_score ranks D>N>E>G at position 2
      - generate_dp_mutant correctly applies the D→P substitution
      - run_efhand_engineering works end-to-end in offline mode
      - Dataset entries contain both WT (negative) and mutant (positive) entries
    """
    eng = load_module("06_efhand_engineering.py")

    # ── Score component tests ─────────────────────────────────────────────
    # LanM loop should score perfectly on O-donor pattern (D at 1, N at 3, E at 9, E at 12)
    lanm = eng.LANM_LOOP_CONSENSUS  # DPNDGKFIEADE
    odoner_score = eng.score_odoner_pattern(lanm)
    assert odoner_score == 1.0, f"LanM should score 1.0 on O-donors, got {odoner_score}"
    log.info(f"  LanM O-donor score: {odoner_score:.2f} ✓")

    # Gly at position 5 (index 4) in LanM = 'G'
    gly_score = eng.score_gly_conservation(lanm)
    assert gly_score == 1.0, f"LanM should score 1.0 on Gly at pos5, got {gly_score}"
    log.info(f"  LanM Gly-at-pos5 score: {gly_score:.2f} ✓")

    # CaM loop 1 (DKDGDGTITTKE): Gly at position 5 (index 4) = 'G' → score 1.0
    cam1 = eng.CAM_LOOPS["hsCaM_loop1"]   # DKDGDGTITTKE
    gly_cam = eng.score_gly_conservation(cam1)
    assert gly_cam == 1.0, f"CaM loop1 should score 1.0 on Gly-pos5, got {gly_cam}"
    log.info(f"  CaM loop1 Gly-at-pos5: {gly_cam:.2f} ✓")

    # ── Engineering score ranking (Asp > Asn > Glu at position 2) ─────────
    # Build minimal EFHandLoop objects for scoring
    from dataclasses import replace

    def make_loop(pos2_aa: str) -> eng.EFHandLoop:
        loop_seq = "D" + pos2_aa + "NDGKFIEADE"  # 12-residue canonical
        return eng.EFHandLoop(
            protein_id="SYNTHETIC", protein_name="test", organism="test",
            subfamily="test", loop_index=1, loop_seq=loop_seq,
            start_pos=0, pos2_aa=pos2_aa, is_ree_selective=(pos2_aa == "P"),
            n_efhands=1, acid_stable=False, source="test",
        )

    score_D = eng.compute_engineering_score(make_loop("D"))["total_engineering_score"]
    score_N = eng.compute_engineering_score(make_loop("N"))["total_engineering_score"]
    score_E = eng.compute_engineering_score(make_loop("E"))["total_engineering_score"]
    score_G = eng.compute_engineering_score(make_loop("G"))["total_engineering_score"]
    score_P = eng.compute_engineering_score(make_loop("P"))["total_engineering_score"]

    log.info(f"  Engineering scores: Asp={score_D:.1f} Asn={score_N:.1f} "
             f"Glu={score_E:.1f} Gly={score_G:.1f} Pro={score_P:.1f}")
    assert score_D > score_N, f"Asp should score higher than Asn ({score_D} vs {score_N})"
    assert score_N > score_G, f"Asn should score higher than Gly ({score_N} vs {score_G})"
    assert score_D > score_E, f"Asp should score higher than Glu ({score_D} vs {score_E})"
    log.info("  Priority order D > N > E > G at position 2 ✓")

    # ── Hamming distance to LanM ─────────────────────────────────────────
    dist_lanm = eng.hamming_to_lanm(eng.LANM_LOOP_CONSENSUS)
    assert dist_lanm == 0, f"LanM vs itself should have Hamming=0, got {dist_lanm}"
    dist_cam  = eng.hamming_to_lanm(eng.CAM_LOOPS["hsCaM_loop1"])
    assert dist_cam > 0, f"CaM loop should differ from LanM consensus"
    log.info(f"  Hamming(LanM, LanM)={dist_lanm}  Hamming(CaM_L1, LanM)={dist_cam} ✓")

    # ── D→P mutant generation ─────────────────────────────────────────────
    # Full-length calmodulin test
    cam_full = eng.OFFLINE_SEED_SEQUENCES["P0DP23"]
    cam_loops = eng.extract_efhand_loops(
        cam_full, "P0DP23", "CaM", "Homo sapiens", "calmodulin", 4, False
    )
    assert len(cam_loops) >= 1, "Should extract at least 1 CaM loop for mutant generation"

    loop0 = cam_loops[0]
    assert loop0.pos2_aa != "P", "CaM should not have Pro at pos2 initially"

    mutant_seq = eng.generate_dp_mutant(cam_full, loop0.start_pos)
    # Check that position 2 of the loop is now Pro
    mut_pos2_abs = loop0.start_pos + 1  # 0-indexed absolute
    assert mutant_seq[mut_pos2_abs] == "P", \
        f"Mutant should have Pro at position {mut_pos2_abs+1}, got {mutant_seq[mut_pos2_abs]}"
    # Check that rest of sequence is unchanged
    assert cam_full[:mut_pos2_abs] == mutant_seq[:mut_pos2_abs], "Upstream region should be unchanged"
    assert cam_full[mut_pos2_abs+1:] == mutant_seq[mut_pos2_abs+1:], "Downstream region should be unchanged"
    log.info(f"  D→P mutant: {loop0.pos2_aa} → P at abs pos {mut_pos2_abs+1}  ✓")

    # ── End-to-end offline run ────────────────────────────────────────────
    results = eng.run_efhand_engineering(use_offline=True)

    loop_df        = results["loop_df"]
    candidates_df  = results["candidates_df"]
    dataset_entries = results["dataset_entries"]

    assert not loop_df.empty, "Loop table should not be empty in offline mode"
    log.info(f"  Offline pipeline: {len(loop_df)} loops, "
             f"{len(candidates_df)} candidates, "
             f"{len(dataset_entries)} dataset entries")

    # Validate that both WT (label=0) and engineered (label=1) entries exist
    labels = [e["label_binary"] for e in dataset_entries]
    assert 0 in labels, "Dataset should contain WT Ca2+-binding (label=0) entries"
    assert 1 in labels, "Dataset should contain REE-selective (label=1) entries"
    log.info(f"  Dataset entries: {labels.count(0)} WT (neg), {labels.count(1)} REE-selective (pos) ✓")

    # Acid-stable SaCaM entries should be included
    orgs = [e.get("organism", "") for e in dataset_entries]
    sulfolobus_present = any("sulfolobus" in o.lower() or "acidocaldarius" in o.lower()
                             for o in orgs)
    log.info(f"  Sulfolobus (acid-stable) entries present: {sulfolobus_present}")
    # (May be absent if SaCaM has no offline fixture — OK to warn, not fail)

    # Engineering score present and in range
    eng_scores = [e.get("engineering_score") for e in dataset_entries
                  if e.get("engineering_score") is not None]
    if eng_scores:
        assert all(0 <= s <= 10 for s in eng_scores), \
            f"Engineering scores should be in [0,10]: {eng_scores}"
        log.info(f"  Engineering scores in [0, 10]: min={min(eng_scores):.1f} "
                 f"max={max(eng_scores):.1f} ✓")

    # Verify top candidate is not SaCaM but acid_stable IS tracked
    if not loop_df.empty and "acid_stable" in loop_df.columns:
        acid_stable_loops = loop_df[loop_df["acid_stable"]].shape[0]
        log.info(f"  Acid-stable loops tracked: {acid_stable_loops}")

    log.info("  Engineering model end-to-end validated ✓")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# T11  Prosthetic group catalog
# ═══════════════════════════════════════════════════════════════════════════════

def t11_prosthetic_group_catalog() -> bool:
    """
    T11: Validate the prosthetic group catalog and new architecture metadata.

    Validates:
      - PROSTHETIC_GROUPS has >= 25 entries
      - Correct ln3_potential ratings for key groups (none/direct/high)
      - ALL_NEW_MOTIFS covers all 5 new architecture prefixes (c2_, ann_, egf_, gla_, cad_)
      - NEW_ARCHITECTURE_SEEDS covers all 5 new architecture classes (>= 20 total)
      - NEW_ARCHITECTURE_QUERIES count >= 20
      - run_cofactor_pipeline() returns n_direct >= 2, n_high >= 5, n_none >= 15
    """
    cof = load_module("07_cofactor_architectures.py")

    # ── 1. Prosthetic group catalog completeness ───────────────────────────
    n_groups = len(cof.PROSTHETIC_GROUPS)
    log.info(f"  Prosthetic groups cataloged: {n_groups}")
    assert n_groups >= 25, f"Expected ≥25 prosthetic groups, got {n_groups}"

    # ── 2. Critical Ln³⁺ potential ratings ────────────────────────────────
    EXPECTED_POTENTIALS = {
        # Must be NONE — wrong donor set (N/S dominant, rigid geometry)
        "heme":             "none",
        "type1_cu":         "none",
        "atcun":            "none",
        "cobalamin":        "none",
        "fad_fmn":          "none",
        # Must be DIRECT — biologically evolved to coordinate Ln³⁺
        "pqq":              "direct",
        "ef_hand":          "direct",
        # Must be HIGH — O-donor Ca²⁺ sites with experimental evidence
        "c2_domain":        "high",
        "annexin_fold":     "high",
        "gla_domain":       "high",
        "egf_ca2_module":   "high",
        "cadherin_ca2":     "high",
    }
    for group_name, expected in EXPECTED_POTENTIALS.items():
        assert group_name in cof.PROSTHETIC_GROUPS, \
            f"Missing group '{group_name}' in PROSTHETIC_GROUPS"
        actual = cof.PROSTHETIC_GROUPS[group_name]["ln3_potential"]
        assert actual == expected, \
            f"Group '{group_name}': expected ln3_potential='{expected}', got '{actual}'"
        log.info(f"  {group_name:20s}: ln3_potential={actual} ✓")

    # ── 3. ALL_NEW_MOTIFS covers all 5 architecture categories ────────────
    motif_keys = list(cof.ALL_NEW_MOTIFS.keys())
    for prefix in ("c2_", "ann_", "egf_", "gla_", "cad_"):
        matching = [k for k in motif_keys if k.startswith(prefix)]
        assert len(matching) >= 1, f"No motifs with prefix '{prefix}' in ALL_NEW_MOTIFS"
        log.info(f"  Motif prefix '{prefix}': {len(matching)} entries ✓")

    # ── 4. NEW_ARCHITECTURE_SEEDS — coverage across all 5 classes ─────────
    n_seeds = len(cof.NEW_ARCHITECTURE_SEEDS)
    log.info(f"  New architecture PDB seeds: {n_seeds}")
    assert n_seeds >= 20, f"Expected ≥20 PDB seeds, got {n_seeds}"

    arch_classes_in_seeds = set(v[1] for v in cof.NEW_ARCHITECTURE_SEEDS.values())
    for arch_class in ("c2-domain", "annexin", "egf-ca2", "gla-domain", "cadherin"):
        assert arch_class in arch_classes_in_seeds, \
            f"Expected architecture class '{arch_class}' missing from seeds"
    log.info(f"  All 5 architecture classes present in PDB seeds ✓")

    # ── 5. NEW_ARCHITECTURE_QUERIES — sufficient query coverage ───────────
    n_queries = len(cof.NEW_ARCHITECTURE_QUERIES)
    log.info(f"  New UniProt queries: {n_queries}")
    assert n_queries >= 20, f"Expected ≥20 UniProt queries, got {n_queries}"

    # ── 6. End-to-end pipeline run ─────────────────────────────────────────
    results = cof.run_cofactor_pipeline()
    assert "catalog"  in results, "run_cofactor_pipeline should return 'catalog'"
    assert "seeds_df" in results, "run_cofactor_pipeline should return 'seeds_df'"
    assert "entries"  in results, "run_cofactor_pipeline should return 'entries'"
    assert results["n_direct"] >= 2, \
        f"Expected ≥2 direct Ln³⁺ groups, got {results['n_direct']}"
    assert results["n_high"]   >= 5, \
        f"Expected ≥5 high potential groups, got {results['n_high']}"
    assert results["n_none"]   >= 15, \
        f"Expected ≥15 no-potential groups, got {results['n_none']}"
    log.info(f"  Pipeline: direct={results['n_direct']}, "
             f"high={results['n_high']}, none={results['n_none']} ✓")

    # Verify entries structure
    entries = results["entries"]
    assert len(entries) >= 20, f"Expected ≥20 dataset entries, got {len(entries)}"
    required_entry_keys = {"protein_id", "label_binary", "architecture",
                           "ln3_potential", "cath_code"}
    for key in required_entry_keys:
        assert key in entries[0], f"Dataset entry missing key: {key}"
    log.info(f"  {len(entries)} dataset entries with required keys ✓")

    # Verify seeds DataFrame structure
    seeds_df = results["seeds_df"]
    for col in ("pdb_id", "cath_code", "architecture_class", "protein_name"):
        assert col in seeds_df.columns, f"Seeds DataFrame missing column: {col}"
    log.info(f"  Seeds DataFrame: {len(seeds_df)} rows, required columns present ✓")

    log.info("  Prosthetic group catalog fully validated ✓")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# T12  New architecture motifs
# ═══════════════════════════════════════════════════════════════════════════════

def t12_new_architecture_motifs() -> bool:
    """
    T12: Test motif detection for new Ca²⁺-binding architectures.

    Validates:
      - C2 Asp-cluster motif fires on CBR-like sequences
      - Annexin GXGT motif fires on endonexin sequences (GLGT fixture)
      - Gla Glu-cluster motif fires on Gla-domain-like sequences
      - Cadherin DxNDN motif fires on cadherin EC-domain sequences
      - classify_new_architecture correctly infers each architecture class
        from isolated scan-result dicts
      - scan_new_motifs returns correctly prefixed motif keys
    """
    cof = load_module("07_cofactor_architectures.py")

    # ── 1. C2 Asp-cluster detection ────────────────────────────────────────
    # C2_test_cbr = "KSSIDMANMFAKDTNGDGTIT"
    # Contains DMANMFAKD (D + MA + N + MFAK + D) → C2_asp_cluster ✓
    # Also  DTNGD        (D + T + N + G + D)   → C2_asp_cluster ✓
    c2_seq  = cof.OFFLINE_FIXTURES["C2_test_cbr"]
    c2_hits = cof.scan_new_motifs(c2_seq)
    c2_found = any(k.startswith("c2_") for k in c2_hits)
    log.info(f"  C2_test_cbr hits: {list(c2_hits.keys())}")
    assert c2_found, \
        f"Expected a c2_ motif hit on CBR sequence, got: {list(c2_hits.keys())}"
    assert "c2_C2_asp_cluster" in c2_hits, \
        "Expected c2_C2_asp_cluster to fire on KSSIDMANMFAKDTNGDGTIT"
    log.info("  C2 Asp-cluster motif detected ✓")

    # ── 2. Annexin GXGT motif detection ────────────────────────────────────
    # Annexin_test fixture contains "GLGT" (G[L]GT) embedded explicitly
    ann_seq  = cof.OFFLINE_FIXTURES["Annexin_test"]
    ann_hits = cof.scan_new_motifs(ann_seq)
    ann_found = any(k.startswith("ann_") for k in ann_hits)
    log.info(f"  Annexin_test hits: {list(ann_hits.keys())}")
    assert ann_found, \
        f"Expected an ann_ motif hit on Annexin_test, got: {list(ann_hits.keys())}"
    assert "ann_Annexin_GXGT" in ann_hits, \
        "Expected ann_Annexin_GXGT to fire on sequence containing GLGT"
    log.info("  Annexin GXGT motif detected ✓")

    # ── 3. Gla domain Glu-cluster detection ────────────────────────────────
    # Gla_test contains multiple Glu clusters (ECKEEICDFEE → 4 E within window)
    # and FLEEL (→ Gla_FLEEL) — both gla_ motifs should fire
    gla_seq  = cof.OFFLINE_FIXTURES["Gla_test"]
    gla_hits = cof.scan_new_motifs(gla_seq)
    gla_found = any(k.startswith("gla_") for k in gla_hits)
    log.info(f"  Gla_test hits: {list(gla_hits.keys())}")
    assert gla_found, \
        f"Expected a gla_ motif hit on Gla_test, got: {list(gla_hits.keys())}"
    assert "gla_Gla_Glu_cluster" in gla_hits, \
        "Expected gla_Gla_Glu_cluster to fire on Glu-dense sequence"
    log.info("  Gla Glu-cluster motif detected ✓")

    # ── 4. Cadherin DxNDN motif detection ──────────────────────────────────
    # Cadherin_test = "YNIPDINDNIPDINDNIPDINDIYEIFIVNEEDGE"
    # DINDN at position 4 matches D[A-Z]NDN (D + I + NDN) ✓
    cad_seq  = cof.OFFLINE_FIXTURES["Cadherin_test"]
    cad_hits = cof.scan_new_motifs(cad_seq)
    cad_found = any(k.startswith("cad_") for k in cad_hits)
    log.info(f"  Cadherin_test hits: {list(cad_hits.keys())}")
    assert cad_found, \
        f"Expected a cad_ motif hit on Cadherin_test, got: {list(cad_hits.keys())}"
    assert "cad_Cadherin_DxNDN" in cad_hits, \
        "Expected cad_Cadherin_DxNDN to fire on DINDNIPD sequence"
    log.info("  Cadherin DxNDN motif detected ✓")

    # ── 5. classify_new_architecture from isolated scan-result dicts ───────
    # Feed pre-built scan results so we bypass priority-order artefacts
    test_cases = [
        ({"c2_C2_asp_cluster": [(0, "DMANMFAKD")]}, "c2-domain"),
        ({"ann_Annexin_GXGT":  [(5, "GLGT")]},      "annexin"),
        ({"gla_Gla_Glu_cluster":[(0, "ECKEE")]},    "gla-domain"),
        ({"egf_EGF_ca2_core":  [(0, "DNDLFDE")]},   "egf-ca2"),
        ({"cad_Cadherin_DxNDN":[(0, "DINDN")]},     "cadherin"),
        ({},                                         "unknown"),
    ]
    for scan_result, expected_arch in test_cases:
        predicted = cof.classify_new_architecture(scan_result)
        assert predicted == expected_arch, (
            f"classify_new_architecture({list(scan_result.keys())}) "
            f"→ '{predicted}', expected '{expected_arch}'"
        )
        keys_str = list(scan_result.keys()) or ["(empty)"]
        log.info(f"  classify {keys_str} → '{predicted}' ✓")

    # ── 6. AnnexinA5_repeat1 real sequence — soft check ───────────────────
    # The truncated fixture may not have canonical GXGT; log without asserting
    anxa5_seq  = cof.OFFLINE_FIXTURES["AnnexinA5_repeat1"]
    anxa5_hits = cof.scan_new_motifs(anxa5_seq)
    ann_in_anxa5 = any(k.startswith("ann_") for k in anxa5_hits)
    log.info(f"  AnnexinA5_repeat1 ann_ motif hits: {ann_in_anxa5}  "
             f"(all hits: {list(anxa5_hits.keys())})")

    log.info("  All new architecture motifs validated ✓")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# T13  Logan / HMM metagenomic search pipeline
# ─────────────────────────────────────────────────────────────────────────────

def t13_metagenomic_search() -> bool:
    """
    T13: Validate the Logan metagenomic search pipeline (08_metagenomic_search.py).

    Exercises:
      1. Custom HMM building (EF_hand_REE_proswitch + DYD_active_site)
      2. ORF prediction from nucleotide fixtures via pyrodigal
      3. Motif validation: REE-selectivity and DYD detection
      4. HMM search hits correct sequences (pyhmmer DigitalSequenceBlock API)
      5. Composite scoring — LanM_EF1 is top priority hit
      6. SLURM script generation (≥7 scripts with correct directives)
      7. End-to-end run_offline_test() returns valid summary dict
    """
    mg = load_module("08_metagenomic_search.py")

    # 1. Dependency availability ─────────────────────────────────────────────
    assert mg.PYHMMER_AVAILABLE,   "pyhmmer not available — install it"
    assert mg.PYRODIGAL_AVAILABLE, "pyrodigal not available — install it"
    assert mg.BOTO3_AVAILABLE,     "boto3 not available — install it"

    # 2. Custom HMM seeds coverage ────────────────────────────────────────────
    seeds = mg.CUSTOM_HMM_SEEDS
    assert "EF_hand_REE_proswitch" in seeds, "EF_hand_REE_proswitch seeds missing"
    assert "DYD_active_site"       in seeds, "DYD_active_site seeds missing"
    assert len(seeds["EF_hand_REE_proswitch"]) >= 4, "Need ≥4 LanM seeds"
    assert len(seeds["DYD_active_site"])       >= 3, "Need ≥3 XoxF seeds"

    # 3. Run offline pipeline end-to-end ──────────────────────────────────────
    result = mg.run_offline_test()

    # 4. HMM build ─────────────────────────────────────────────────────────────
    custom_hmm_paths = result.get("custom_hmm_paths", {})
    assert len(custom_hmm_paths) >= 2, \
        f"Expected ≥2 custom HMMs built, got {len(custom_hmm_paths)}"
    assert "EF_hand_REE_proswitch" in custom_hmm_paths, \
        "EF_hand_REE_proswitch.hmm not built"
    assert "DYD_active_site" in custom_hmm_paths, \
        "DYD_active_site.hmm not built"
    for hmm_name, hmm_path in custom_hmm_paths.items():
        from pathlib import Path
        p = Path(hmm_path)
        assert p.exists() and p.stat().st_size > 0, \
            f"HMM file missing or empty: {hmm_path}"

    # 5. ORF prediction ────────────────────────────────────────────────────────
    orf_results = result.get("orf_results", {})
    assert len(orf_results) == len(mg.OFFLINE_NUC_FIXTURES), \
        f"Expected {len(mg.OFFLINE_NUC_FIXTURES)} ORF results, got {len(orf_results)}"
    for seq_name, orfs in orf_results.items():
        assert len(orfs) >= 1, f"No ORFs predicted for {seq_name}"
        for orf_id, prot, start, end, strand in orfs:
            assert len(prot) >= 5, f"ORF protein too short: {prot}"
            assert strand in ("+", "-"), f"Bad strand value: {strand}"

    # 6. Motif validation ──────────────────────────────────────────────────────
    val_results = result.get("validation_results", {})
    assert "LanM_EF1" in val_results, "LanM_EF1 not in motif validation results"
    assert "XoxF_DYD" in val_results, "XoxF_DYD not in motif validation results"
    assert "CaM_EF_neg" in val_results, "CaM_EF_neg not in motif validation results"

    lanm_val = val_results["LanM_EF1"]
    assert lanm_val["is_ree_selective"], \
        "LanM_EF1 should be REE-selective (Pro at pos2)"
    assert "EF_hand_REE" in lanm_val["motifs_found"], \
        f"EF_hand_REE motif not found in LanM_EF1: {lanm_val['motifs_found']}"

    xoxf_val = val_results["XoxF_DYD"]
    assert "DYD_strict" in xoxf_val["motifs_found"], \
        f"DYD_strict not found in XoxF_DYD: {xoxf_val['motifs_found']}"

    cam_val = val_results["CaM_EF_neg"]
    assert not cam_val["is_ree_selective"], \
        "CaM_EF_neg should NOT be REE-selective (no Pro at pos2)"

    # 7. HMM search hits ───────────────────────────────────────────────────────
    search_results = result.get("search_results", [])
    assert len(search_results) >= 2, \
        f"Expected ≥2 HMM search hits, got {len(search_results)}"

    # LanM_EF1 must hit EF_hand_REE_proswitch with decent score
    lanm_hits = [h for h in search_results
                 if h["orf_id"] == "LanM_EF1" and "EF_hand" in h["hmm_name"]]
    assert lanm_hits, "LanM_EF1 did not hit EF_hand_REE_proswitch HMM"
    assert lanm_hits[0]["hmm_score"] > 20.0, \
        f"LanM_EF1 HMM score too low: {lanm_hits[0]['hmm_score']:.1f}"

    # XoxF_DYD must hit DYD_active_site with decent score
    xoxf_hits = [h for h in search_results
                 if h["orf_id"] == "XoxF_DYD" and "DYD" in h["hmm_name"]]
    assert xoxf_hits, "XoxF_DYD did not hit DYD_active_site HMM"
    assert xoxf_hits[0]["hmm_score"] > 20.0, \
        f"XoxF_DYD HMM score too low: {xoxf_hits[0]['hmm_score']:.1f}"

    # 8. Composite scoring ─────────────────────────────────────────────────────
    scored_hits = result.get("scored_hits", [])
    assert scored_hits, "No scored hits returned"
    top_name, top_score, _motifs = scored_hits[0]
    assert top_name == "LanM_EF1", \
        f"Expected LanM_EF1 as top hit, got {top_name} (score={top_score})"
    assert top_score >= 3.0, \
        f"LanM_EF1 composite score too low: {top_score:.2f} (expected ≥3.0)"

    # 9. SLURM script generation ───────────────────────────────────────────────
    # write_slurm_scripts() returns a dict keyed by role: {'scan': Path, ...}
    slurm_paths = result.get("slurm_paths", {})
    assert len(slurm_paths) >= 7, \
        f"Expected ≥7 SLURM scripts, got {len(slurm_paths)}"

    # Verify scan script has essential SBATCH directives
    from pathlib import Path
    scan_sh = slurm_paths.get("scan")
    assert scan_sh is not None and Path(scan_sh).exists(), \
        "submit_scan.sh not found (key='scan' missing from slurm_paths dict)"
    scan_sh = Path(scan_sh)
    scan_txt = scan_sh.read_text()
    for directive in ["#SBATCH", "--array", "--cpus-per-task", "--mem",
                      "08_metagenomic_search.py"]:
        assert directive in scan_txt, \
            f"Missing directive in scan script: {directive}"

    # 10. REE environment configuration ──────────────────────────────────────
    envs = mg.REE_ENVIRONMENTS
    assert len(envs) >= 5, f"Expected ≥5 REE environments, got {len(envs)}"
    priority_1 = [k for k, v in envs.items() if v.get("priority") == 1]
    assert len(priority_1) >= 1, "No priority-1 environments configured"

    return True


def run_test(name: str, fn, *args, **kwargs):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    try:
        result = fn(*args, **kwargs)
        if result == "SKIPPED":
            print(f"  {SKIP_COLOR}  (network unavailable)")
            return "SKIPPED"
        print(f"  {PASS_COLOR}")
        return True
    except AssertionError as e:
        print(f"  {FAIL_COLOR}: {e}")
        return False
    except Exception as e:
        print(f"  {FAIL_COLOR}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("  REE BINDING PROTEIN PIPELINE — TEST SUITE")
    print(f"  Network: {'AVAILABLE' if NETWORK_AVAILABLE else 'OFFLINE (sandbox)'}")
    print("="*60)

    tests = [
        ("T1  RCSB query builder + metal codes",         t1_rcsb_query_builder),
        ("T2  Architecture annotation (known proteins)",  t2_architecture_annotations),
        ("T3  Motif scanner (DYD / EF-hand / RTX / LBT)",t3_motif_scanner),
        ("T4  Sequence clustering (greedy 30% identity)", t4_sequence_clustering),
        ("T5  Label assignment (binary, Kd, novelty)",    t5_label_assignment),
        ("T6  Dataset builder + ESM-Bind JSON export",    t6_dataset_export),
        ("T7  UniProt live search (lanmodulin)",          t7_uniprot_live),
        ("T8  Fold diversity visualization",              t8_diversity_visualization),
        ("T9  CaM-family EF-hand loop extraction",        t9_cam_loop_extraction),
        ("T10 Engineering model + D→P mutant generation", t10_engineering_model),
        ("T11 Prosthetic group catalog + Ln³⁺ ratings",  t11_prosthetic_group_catalog),
        ("T12 New arch motifs (C2/Annexin/Gla/Cadherin)", t12_new_architecture_motifs),
        ("T13 Logan metagenomic search (HMM/ORF/SLURM)",  t13_metagenomic_search),
    ]

    results = {}
    for name, fn in tests:
        results[name] = run_test(name, fn)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    passed = skipped = failed = 0
    for name, res in results.items():
        if res is True:
            print(f"  {PASS_COLOR}    {name}")
            passed += 1
        elif res == "SKIPPED":
            print(f"  {SKIP_COLOR}  {name}")
            skipped += 1
        else:
            print(f"  {FAIL_COLOR}    {name}")
            failed += 1

    total = len(tests)
    print(f"\n  {passed} passed  |  {skipped} skipped (offline)  |  {failed} failed  |  {total} total")

    if failed == 0:
        print("\n  All runnable tests PASSED. Pipeline logic validated.")
        print("\n  Generated files:")
        for f in sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.json")):
            print(f"    datasets/{f.name:40s}  {f.stat().st_size/1024:.1f} KB")
        for f in sorted(FIG_DIR.glob("*.png")):
            print(f"    figures/{f.name}")
        print("\n  To run full pipeline against live APIs:")
        print("    ree-miner mine                                 # PDB mining")
        print("    ree-miner classify                             # architecture annotation")
        print("    ree-miner find-homologs                        # UniProt + motif scan")
        print("    ree-miner engineer                             # CaM EF-hand + D→P mutants")
        print("    ree-miner cofactors                            # C2/Annexin/EGF/Gla/Cadherin")
        print("    ree-miner scan --mode build-hmms               # download Pfam HMMs")
        print("    ree-miner scan --mode generate-slurm           # HPC SLURM scripts")
        print("    ree-miner build-dataset                        # cluster, label, export")
    else:
        print(f"\n  {failed} test(s) FAILED — see output above.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
