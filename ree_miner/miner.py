"""
REE-Binding Protein PDB Miner
==============================
Phase 1 (highest priority): Mine RCSB PDB for all structures containing:
  - Actual lanthanide ions (La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu)
  - Crystallographic surrogates (Y3+, Sm3+, Gd3+) routinely used as heavy-atom
    phasing agents — these ARE de facto REE-binding proteins never annotated as such.

For each hit: extracts protein chain sequences, identifies binding residues within
coordination distance, records metal identity, resolution, organism, PDB ID.

Output: datasets/pdb_hits_raw.csv  +  datasets/pdb_hits_annotated.csv

Usage:
    python 01_pdb_miner.py [--test]      # --test runs on a tiny subset (5 PDB IDs)
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import gemmi
import pandas as pd
import requests

# ─── CONFIG ──────────────────────────────────────────────────────────────────
from ree_miner._workspace import DATA_DIR, STRUCT_DIR, LOG_DIR
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "pdb_miner.log"),
    ],
)
log = logging.getLogger("pdb_miner")

# ─── Metal ion definitions ────────────────────────────────────────────────────
# PDB CCD (Chemical Component Dictionary) codes for lanthanides + surrogates.
# Y3+ and Gd3+ are routinely used as heavy-atom phasing agents; Sm3+ as an
# anomalous scatterer. Proteins crystallized with these are REE-binding proteins
# that were NEVER annotated as such — the key insight driving this phase.
METAL_CODES = {
    # True lanthanides
    "LA":  {"element": "La", "name": "Lanthanum",     "type": "lanthanide",   "z": 57},
    "CE":  {"element": "Ce", "name": "Cerium",        "type": "lanthanide",   "z": 58},
    "PR":  {"element": "Pr", "name": "Praseodymium",  "type": "lanthanide",   "z": 59},
    "ND":  {"element": "Nd", "name": "Neodymium",     "type": "lanthanide",   "z": 60},
    "SM":  {"element": "Sm", "name": "Samarium",      "type": "lanthanide",   "z": 62},
    "EU":  {"element": "Eu", "name": "Europium",      "type": "lanthanide",   "z": 63},
    "GD":  {"element": "Gd", "name": "Gadolinium",    "type": "surrogate",    "z": 64},  # phasing agent
    "TB":  {"element": "Tb", "name": "Terbium",       "type": "lanthanide",   "z": 65},
    "DY":  {"element": "Dy", "name": "Dysprosium",    "type": "lanthanide",   "z": 66},
    "HO":  {"element": "Ho", "name": "Holmium",       "type": "lanthanide",   "z": 67},
    "ER":  {"element": "Er", "name": "Erbium",        "type": "lanthanide",   "z": 68},
    "TM":  {"element": "Tm", "name": "Thulium",       "type": "lanthanide",   "z": 69},
    "YB":  {"element": "Yb", "name": "Ytterbium",     "type": "lanthanide",   "z": 70},
    "LU":  {"element": "Lu", "name": "Lutetium",      "type": "lanthanide",   "z": 71},
    # Crystallographic surrogates — underexplored source of REE-binding architectures
    "Y":   {"element": "Y",  "name": "Yttrium",       "type": "surrogate",    "z": 39},
    "YT3": {"element": "Y",  "name": "Yttrium(III)",  "type": "surrogate",    "z": 39},
}

BINDING_CUTOFF_ANGSTROM = 3.5   # coordination distance threshold
MAX_RESOLUTION = 3.0            # Å — matching ESM-Bind training criteria
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL   = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_FILE_URL   = "https://files.rcsb.org/download"
REQUEST_PAUSE   = 0.1           # polite delay between RCSB requests (seconds)

# ─── Known high-confidence REE-binding structures (seed set from literature) ─
LITERATURE_SEEDS = [
    # XoxF-MDH REE-dependent methanol dehydrogenases
    "4MAE", "6FKW", "6OC6", "7O6Z", "6DAM", "5LJR", "6H1N",
    # PedH / ExaF REE-dependent ethanol dehydrogenases
    "6ZCW",
    # Mex-LanM (lanmodulin from M. extorquens)
    "6MI5", "8FNS",
    # Hans-LanM (lanmodulin from H. quercus)
    "8DQ2", "8FNR",
    # MxaF-MDH (Ca2+-dependent — NEGATIVE CONTROLS)
    "1H4I", "1W6S",
    # Calmodulin (Ca-binding EF-hand — NEGATIVE CONTROL)
    "1GGZ",
]
NEGATIVE_SEEDS = {"1H4I", "1W6S", "1GGZ"}   # labeled negatives from literature


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RCSB QUERY
# ═══════════════════════════════════════════════════════════════════════════════

def build_metal_search_query(metal_codes: list[str], max_rows: int = 5000) -> dict:
    """
    Build RCSB Search API v2 JSON query for structures containing any of the
    specified metal CCD codes bound to polymer (protein) chains.

    API v2 changes vs v1:
      - URL:      /rcsbsearch/v2/query
      - operator: "exact_match" (replaces deprecated "equals")
      - "negation" field removed from terminal nodes
      - "results_content_type" removed from request_options
    """
    metal_nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_entity.comp_id",
                "operator": "exact_match",
                "value": code,
            },
        }
        for code in metal_codes
    ]

    return {
        "query": {
            "type": "group",
            "logical_operator": "or",
            "nodes": metal_nodes,
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_rows},
            "sort": [{"sort_by": "score", "direction": "descending"}],
        },
    }


def query_pdb_for_metals(metal_codes: list[str]) -> list[str]:
    """Query RCSB PDB and return a list of matching PDB IDs."""
    log.info(f"Querying RCSB for {len(metal_codes)} metal codes: {metal_codes}")
    query = build_metal_search_query(metal_codes)
    try:
        resp = requests.post(
            RCSB_SEARCH_URL,
            json=query,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        ids = [hit["identifier"] for hit in data.get("result_set", [])]
        log.info(f"  → Found {len(ids)} PDB entries")
        return ids
    except Exception as e:
        log.error(f"RCSB query failed: {e}")
        return []


def get_entry_metadata(pdb_id: str) -> dict:
    """Fetch resolution, organism, title, deposition date for one PDB entry."""
    url = f"{RCSB_DATA_URL}/{pdb_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        d = resp.json()
        info = d.get("rcsb_entry_info", {})
        struct = d.get("struct", {})
        return {
            "pdb_id": pdb_id,
            "title": struct.get("title", ""),
            "resolution": info.get("resolution_combined", [None])[0],
            "method": info.get("experimental_method", ""),
            "polymer_count": info.get("polymer_entity_count_protein", 0),
            "deposited": d.get("rcsb_accession_info", {}).get("deposit_date", ""),
        }
    except Exception as e:
        log.warning(f"Metadata fetch failed for {pdb_id}: {e}")
        return {"pdb_id": pdb_id}


def filter_by_resolution(pdb_ids: list[str], cutoff: float = MAX_RESOLUTION) -> list[str]:
    """Filter PDB entries to those at or better than cutoff resolution."""
    log.info(f"Filtering {len(pdb_ids)} entries by resolution ≤ {cutoff} Å ...")
    kept, skipped = [], 0
    for i, pid in enumerate(pdb_ids):
        meta = get_entry_metadata(pid)
        res = meta.get("resolution")
        if res is not None and float(res) <= cutoff:
            kept.append((pid, meta))
        else:
            skipped += 1
        if (i + 1) % 50 == 0:
            log.info(f"  {i+1}/{len(pdb_ids)} checked, kept {len(kept)}, skipped {skipped}")
        time.sleep(REQUEST_PAUSE)
    log.info(f"Resolution filter: kept {len(kept)}, removed {skipped}")
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STRUCTURE DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_structure(pdb_id: str, out_dir: Path = STRUCT_DIR) -> Path | None:
    """Download mmCIF file for a PDB entry. Returns path or None on failure."""
    out_path = out_dir / f"{pdb_id}.cif"
    if out_path.exists():
        return out_path
    url = f"{RCSB_FILE_URL}/{pdb_id}.cif"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        time.sleep(REQUEST_PAUSE)
        return out_path
    except Exception as e:
        log.warning(f"Download failed for {pdb_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BINDING RESIDUE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_binding_residues(
    cif_path: Path,
    metal_codes: set[str],
    cutoff: float = BINDING_CUTOFF_ANGSTROM,
) -> list[dict]:
    """
    Parse mmCIF with gemmi. For every lanthanide/surrogate ion found, identify
    all protein residues with any atom within `cutoff` Å.

    Returns list of dicts, one per (metal_site, chain, residue) contact.
    Includes:
        - pdb_id, chain_id, chain_seq (one-letter), seq_len
        - metal_code, metal_element, metal_type
        - binding_residue (one-letter AA), binding_position (seq number)
        - binding_atom, metal_atom, distance_A
        - coordination_number (total donors for that metal site)
        - organism (from _entity_src_gen)
    """
    pdb_id = cif_path.stem.upper()
    results = []

    try:
        st = gemmi.read_structure(str(cif_path))
        st.setup_entities()
    except Exception as e:
        log.warning(f"gemmi parse error {pdb_id}: {e}")
        return []

    # Build a fast neighbor search over all atoms
    ns = gemmi.NeighborSearch(st[0], st.cell, cutoff).populate(include_h=False)

    for model in st:
        for chain in model:
            for residue in chain:
                res_name = residue.name
                if res_name not in metal_codes:
                    continue
                if res_name not in METAL_CODES:
                    continue

                metal_info = METAL_CODES[res_name]

                # Find all protein atoms within cutoff of this metal
                for metal_atom in residue:
                    metal_pos = metal_atom.pos
                    neighbors = ns.find_atoms(metal_pos, "\0", radius=cutoff)
                    donors = []
                    for nb in neighbors:
                        nb_res = model[nb.chain_idx][nb.residue_idx]
                        # Skip self (the metal residue) and non-protein residues
                        if nb_res.name == res_name:
                            continue
                        if not gemmi.find_tabulated_residue(nb_res.name).is_amino_acid():
                            continue
                        nb_atom = nb_res[nb.atom_idx]
                        dist = metal_pos.dist(nb_atom.pos)
                        one_letter = gemmi.find_tabulated_residue(nb_res.name).one_letter_code
                        donors.append({
                            "binding_residue_3l": nb_res.name,
                            "binding_residue_1l": one_letter,
                            "binding_seqnum": nb_res.seqid.num,
                            "binding_atom": nb_atom.name,
                            "distance_A": round(dist, 3),
                            "chain_id": nb.chain_idx,
                        })

                    coord_number = len(donors)
                    for donor in donors:
                        results.append({
                            "pdb_id": pdb_id,
                            "metal_code": res_name,
                            "metal_element": metal_info["element"],
                            "metal_name": metal_info["name"],
                            "metal_type": metal_info["type"],
                            "metal_z": metal_info["z"],
                            "metal_chain": chain.name,
                            "metal_seqnum": residue.seqid.num,
                            "coordination_number": coord_number,
                            **donor,
                        })

    if not results:
        log.debug(f"No REE contacts found in {pdb_id}")
    return results


def extract_chain_sequences(cif_path: Path) -> dict[str, str]:
    """Return {chain_id: one_letter_sequence} for all protein chains."""
    seqs = {}
    try:
        st = gemmi.read_structure(str(cif_path))
        st.setup_entities()
        for entity in st.entities:
            if entity.entity_type == gemmi.EntityType.Polymer:
                if entity.polymer_type in (
                    gemmi.PolymerType.PeptideL,
                    gemmi.PolymerType.PeptideD,
                ):
                    seq_1l = gemmi.one_letter_code(entity.full_sequence)
                    for chain in st[0]:
                        if chain.get_entity_name() == entity.name:
                            seqs[chain.name] = seq_1l
    except Exception as e:
        log.warning(f"Sequence extraction failed for {cif_path.stem}: {e}")
    return seqs


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ORGANISM ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_organism(pdb_id: str) -> str:
    """Fetch the source organism for a PDB entry from RCSB."""
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        d = resp.json()
        src = d.get("rcsb_entity_source_organism", [{}])
        return src[0].get("ncbi_scientific_name", "") if src else ""
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def mine_all_ree_structures(test_mode: bool = False) -> pd.DataFrame:
    """
    Full pipeline:
      1. Query RCSB for all REE/surrogate structures
      2. Add literature seeds
      3. Filter by resolution
      4. Download mmCIF files
      5. Extract binding residues
      6. Annotate sequences and organisms
      7. Save raw + annotated datasets

    Returns annotated DataFrame.
    """
    log.info("=" * 60)
    log.info("REE PDB MINER — starting pipeline")
    log.info("=" * 60)

    # ── Step 1: query RCSB ──────────────────────────────────────────────────
    all_codes = list(METAL_CODES.keys())
    pdb_ids_from_query = query_pdb_for_metals(all_codes)

    # ── Step 2: add literature seeds ────────────────────────────────────────
    all_ids = list(dict.fromkeys(pdb_ids_from_query + LITERATURE_SEEDS))
    log.info(f"Total unique IDs (query + seeds): {len(all_ids)}")

    if test_mode:
        # In test mode, use only the well-characterised seed structures
        all_ids = LITERATURE_SEEDS
        log.info(f"TEST MODE: limiting to {len(all_ids)} seed structures")

    # ── Step 3: filter by resolution ────────────────────────────────────────
    kept = filter_by_resolution(all_ids, cutoff=MAX_RESOLUTION)
    meta_lookup = {pid: meta for pid, meta in kept}
    kept_ids = [pid for pid, _ in kept]
    log.info(f"After resolution filter: {len(kept_ids)} structures")

    # ── Steps 4-5: download + extract ───────────────────────────────────────
    all_contacts = []
    all_seq_rows = []
    failed = []

    for i, pdb_id in enumerate(kept_ids):
        log.info(f"[{i+1}/{len(kept_ids)}] Processing {pdb_id} ...")
        cif_path = download_structure(pdb_id)
        if cif_path is None:
            failed.append(pdb_id)
            continue

        # Binding contacts
        contacts = extract_binding_residues(cif_path, set(METAL_CODES.keys()))
        for c in contacts:
            c["resolution"] = meta_lookup.get(pdb_id, {}).get("resolution")
            c["title"] = meta_lookup.get(pdb_id, {}).get("title", "")
            c["is_negative"] = pdb_id in NEGATIVE_SEEDS
            c["is_literature_seed"] = pdb_id in set(LITERATURE_SEEDS)
        all_contacts.extend(contacts)

        # Sequences
        seqs = extract_chain_sequences(cif_path)
        for chain_id, seq in seqs.items():
            all_seq_rows.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "sequence": seq,
                "seq_len": len(seq),
                "resolution": meta_lookup.get(pdb_id, {}).get("resolution"),
                "is_negative": pdb_id in NEGATIVE_SEEDS,
                "is_literature_seed": pdb_id in set(LITERATURE_SEEDS),
            })

    log.info(f"Failed downloads: {failed}")
    log.info(f"Total binding contacts extracted: {len(all_contacts)}")
    log.info(f"Total chain sequences extracted: {len(all_seq_rows)}")

    # ── Step 6: save raw contacts ────────────────────────────────────────────
    contacts_df = pd.DataFrame(all_contacts)
    seqs_df = pd.DataFrame(all_seq_rows)

    raw_path = DATA_DIR / "pdb_contacts_raw.csv"
    seqs_path = DATA_DIR / "pdb_sequences_raw.csv"
    contacts_df.to_csv(raw_path, index=False)
    seqs_df.to_csv(seqs_path, index=False)
    log.info(f"Saved: {raw_path}  ({len(contacts_df)} rows)")
    log.info(f"Saved: {seqs_path}  ({len(seqs_df)} rows)")

    # ── Step 7: annotated summary per PDB entry ──────────────────────────────
    if not contacts_df.empty:
        summary = (
            contacts_df.groupby(["pdb_id", "metal_code", "metal_name", "metal_type"])
            .agg(
                n_binding_residues=("binding_residue_1l", "count"),
                unique_binding_residues=("binding_residue_1l", lambda x: ",".join(sorted(set(x)))),
                unique_binding_atoms=("binding_atom", lambda x: ",".join(sorted(set(x)))),
                avg_coordination_number=("coordination_number", "mean"),
                min_distance_A=("distance_A", "min"),
                resolution=("resolution", "first"),
                is_negative=("is_negative", "first"),
                is_literature_seed=("is_literature_seed", "first"),
            )
            .reset_index()
        )
        ann_path = DATA_DIR / "pdb_hits_annotated.csv"
        summary.to_csv(ann_path, index=False)
        log.info(f"Saved annotated summary: {ann_path}  ({len(summary)} rows)")
    else:
        summary = pd.DataFrame()

    log.info("Pipeline complete.")
    return contacts_df, seqs_df, summary


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    contacts_df, seqs_df, summary = mine_all_ree_structures()

    print("\n" + "=" * 60)
    print("MINING COMPLETE")
    print("=" * 60)
    if not contacts_df.empty:
        print(f"\nMetals found: {contacts_df['metal_code'].value_counts().to_dict()}")
        print(f"\nBinding residue frequency:")
        print(contacts_df["binding_residue_1l"].value_counts().head(10))
        print(f"\nCoordination number distribution:")
        print(contacts_df["coordination_number"].describe())
    if not summary.empty:
        print(f"\nTop PDB entries by binding residue count:")
        print(summary.nlargest(10, "n_binding_residues")[
            ["pdb_id", "metal_name", "n_binding_residues",
             "unique_binding_residues", "avg_coordination_number", "resolution"]
        ].to_string(index=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REE PDB Miner")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: run only on literature seed structures")
    args = parser.parse_args()
    contacts_df, seqs_df, summary = mine_all_ree_structures(test_mode=args.test)

    print("\n" + "=" * 60)
    print("MINING COMPLETE")
    print("=" * 60)
    if not contacts_df.empty:
        print(f"\nMetals found: {contacts_df['metal_code'].value_counts().to_dict()}")
        print(f"\nBinding residue frequency:")
        print(contacts_df["binding_residue_1l"].value_counts().head(10))
        print(f"\nCoordination number distribution:")
        print(contacts_df["coordination_number"].describe())
    if not summary.empty:
        print(f"\nTop PDB entries by binding residue count:")
        print(summary.nlargest(10, "n_binding_residues")[
            ["pdb_id", "metal_name", "n_binding_residues",
             "unique_binding_residues", "avg_coordination_number", "resolution"]
        ].to_string(index=False))
