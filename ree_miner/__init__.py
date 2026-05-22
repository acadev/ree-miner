"""
ree_miner — Rare-Earth-Element (REE) Binding Protein Discovery Pipeline
========================================================================

A fully-offline-capable, pip-installable Python toolkit for discovering,
classifying, and engineering proteins that selectively bind lanthanide ions
(La³⁺, Ce³⁺, Nd³⁺, …).  The package scales from a single laptop (offline
fixtures) to an HPC cluster (SLURM array jobs over the Logan/Serratus
metagenomic dataset, ~5.7 M SRA experiments).

Quick start
-----------
>>> pip install ree-miner
>>> ree-miner test               # smoke test — no network required
>>> ree-miner mine               # PDB mining (requires network)
>>> ree-miner scan --mode offline-test

Modules
-------
miner        — RCSB PDB mining of REE-coordinating structures
classifier   — Architecture annotation (EF-hand, β-propeller, RTX, …)
homologs     — UniProt / motif-based homolog discovery
datasets     — Sequence clustering, label assignment, ESM-Bind JSON export
engineering  — CaM-family EF-hand engineering and D→P mutant generation
cofactors    — Cofactor / prosthetic-group architecture catalog
metagenomic  — Logan S3 metagenome scan with pyhmmer profile HMMs + SLURM
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("ree-miner")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "__version__",
    "miner",
    "classifier",
    "homologs",
    "datasets",
    "engineering",
    "cofactors",
    "metagenomic",
]
