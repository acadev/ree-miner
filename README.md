# REE-miner

**Rare-Earth-Element (REE) Binding Protein Discovery Pipeline**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)

REE-miner is a fully offline-capable, pip-installable Python toolkit for
discovering, classifying, and engineering proteins that selectively bind
lanthanide ions (La³⁺, Ce³⁺, Nd³⁺ and heavier REEs).  The pipeline scales
from a single laptop (offline test mode, no network required) all the way to
an HPC cluster scanning the entire [Logan/Serratus](https://serratus.io/)
metagenomic dataset (~5.7 million SRA experiments, billions of contigs).

---

## Features

| Module | What it does |
|--------|-------------|
| `miner` | RCSB PDB mining — pulls all structures with lanthanide/REE contact sites |
| `classifier` | Architecture annotation — EF-hand, β-propeller (XoxF/PQQ), RTX-repeat, LBT, … |
| `homologs` | UniProt REST + motif-scan based homolog discovery across proteomes |
| `datasets` | Sequence clustering (30 % identity), label assignment, ESM-Bind JSON export |
| `engineering` | CaM-family EF-hand loop extraction and D→P REE-selectivity mutant design |
| `metagenomic` | Logan S3 metagenome scan with pyhmmer profile HMMs + SLURM array job generation |

---
<!-- | `cofactors` | Cofactor / prosthetic-group architecture catalog (C2, Annexin, EGF-Ca², Gla, Cadherin) |  # Removing this for now --> 



## Installation

```bash
pip install ree-miner
```

For PDB structure mining (requires [gemmi](https://gemmi.readthedocs.io/)):

```bash
pip install "ree-miner[pdb]"
```

From source:

```bash
git clone https://github.com/ramanathana/ree-miner.git
cd ree-miner
pip install -e ".[dev]"
```

---

## Quick start

```bash
# Verify everything works offline (no network, no HPC required)
ree-miner test

# Full pipeline against live APIs
ree-miner mine                   # PDB mining
ree-miner classify               # architecture annotation
ree-miner find-homologs          # UniProt search + motif scan
ree-miner engineer               # CaM EF-hand engineering
ree-miner build-dataset          # cluster, label, export
<!-- ree-miner cofactors              # cofactor architecture catalog # Remove this for now -->

# Metagenome scan
ree-miner scan --mode build-hmms        # download Pfam + build custom HMMs
ree-miner scan --mode generate-slurm   # write SLURM array scripts
ree-miner scan --mode offline-test     # smoke test without S3 access
```

### Custom workspace

All outputs go to `./ree_miner_data/` by default.  Override with:

```bash
ree-miner --workspace /project/my_run mine
# or via environment variable:
export REE_MINER_WORKSPACE=/project/my_run
ree-miner mine
```

---

## Python API

```python
import ree_miner._workspace as ws
ws.set_workspace("/project/my_run")          # must be called first

from ree_miner.classifier import classify_architecture, ARCHITECTURE_RULES
from ree_miner.metagenomic import run_offline_test, build_custom_hmm, CUSTOM_HMM_SEEDS

# Test the HMM pipeline end-to-end
result = run_offline_test()
print(result["search_results"])             # [{orf_id, hmm_name, hmm_score, …}, …]

# Build a custom profile HMM from seed sequences
build_custom_hmm(
    "My_EF_hand",
    ["YIDPNDGKFIEADELLAAK", "YIDPNDGWYEGDELLAAK"],
    out_path="/tmp/my_ef.hmm",
)
```

---

## HPC / SLURM deployment

```bash
# 1. Build HMM profiles (once)
ree-miner --workspace /project/ree scan --mode build-hmms

# 2. Generate SLURM scripts targeting acid-mine-drainage metagenomes
ree-miner --workspace /project/ree scan --mode generate-slurm

# 3. Submit
sbatch /project/ree/slurm/submit_acid_mine_drainage.sh

# 4. Aggregate results
ree-miner --workspace /project/ree scan --mode aggregate
```

Each SLURM array task downloads a chunk of Logan S3 contigs, predicts ORFs
with pyrodigal (Prodigal metagenome mode), searches with profile HMMs, applies
motif validation, and writes a parquet file.  The aggregate step merges hits,
identifies novel families, and exports an ESM-Bind-compatible JSON for
downstream structure prediction and Kd scoring.

---

## Architecture scoring

Composite priority score (0–10):

| Component | Points |
|-----------|--------|
| HMM confidence: log₁₀(1/E-value) / 5, capped at 3 | 0–3 |
| Motif support: n_motifs × 0.5, capped at 2 | 0–2 |
| REE-selectivity bonus (Pro at EF-hand pos 2) | +2 |
| DYD active-site bonus (XoxF/PQQ) | +2 |
| Novel architecture bonus (non-EF-hand) | +1 |

---

## Test suite

```bash
ree-miner test          # CLI
# or
pytest tests/
```

13 tests covering all modules; 12 pass fully offline, 1 (UniProt live search)
is skipped when the sandbox has no network access.

---

## Citation

If you use REE-miner in your research, please cite:

> Ramanathan A. et al. *REE-miner: a pipeline for genome- and metagenome-scale
> discovery of rare-earth-element binding proteins.* (2024)

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
