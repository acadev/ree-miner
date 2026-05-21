"""
ree_miner._workspace
====================
Centralised workspace / path management.

All output paths (datasets, figures, HMM profiles, SLURM scripts, logs)
are resolved relative to a *workspace directory*.  The workspace is chosen
in priority order:

  1. ``REE_MINER_WORKSPACE`` environment variable (absolute path).
  2. The ``--workspace`` flag parsed by ``ree_miner.cli`` (set via this module).
  3. ``./ree_miner_data/`` inside the current working directory.

Usage inside any sub-module::

    from ree_miner._workspace import DATA_DIR, FIG_DIR, HMM_DIR, SLURM_DIR

To override at runtime (e.g., from a script)::

    import ree_miner._workspace as ws
    ws.set_workspace("/project/myrun")
"""

import os
from pathlib import Path

# ── Mutable global so cli.py can override before importing sub-modules ────────
_workspace: Path | None = None


def set_workspace(path: str | Path) -> None:
    """Override the workspace at runtime (call before importing other modules)."""
    global _workspace, WORKSPACE_DIR, DATA_DIR, FIG_DIR, HMM_DIR, HITS_DIR, SLURM_DIR
    global STRUCT_DIR, LOG_DIR
    _workspace = Path(path).resolve()
    _refresh()


def get_workspace() -> Path:
    """Return the active workspace root, creating it if necessary."""
    global _workspace
    if _workspace is not None:
        return _workspace
    env = os.environ.get("REE_MINER_WORKSPACE")
    if env:
        _workspace = Path(env).resolve()
    else:
        _workspace = Path.cwd() / "ree_miner_data"
    return _workspace


# Public path constants — populated by _refresh() on first import ─────────────
WORKSPACE_DIR: Path
DATA_DIR:      Path
FIG_DIR:       Path
HMM_DIR:       Path
HITS_DIR:      Path
SLURM_DIR:     Path
STRUCT_DIR:    Path
LOG_DIR:       Path


def _refresh() -> None:
    """Recompute all path constants from the current workspace root."""
    global WORKSPACE_DIR, DATA_DIR, FIG_DIR, HMM_DIR, HITS_DIR, SLURM_DIR, STRUCT_DIR, LOG_DIR
    base          = get_workspace()
    WORKSPACE_DIR = base
    DATA_DIR  = base / "datasets"
    FIG_DIR   = base / "figures"
    HMM_DIR   = base / "datasets" / "hmm_profiles"
    HITS_DIR  = base / "datasets" / "metagenomic_hits"
    SLURM_DIR = base / "slurm"
    STRUCT_DIR = base / "structures"
    LOG_DIR   = base / "logs"
    for d in (DATA_DIR, FIG_DIR, HMM_DIR, HITS_DIR, SLURM_DIR, STRUCT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Initialise on first import
_refresh()
