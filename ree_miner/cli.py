"""
ree_miner.cli
=============
Unified command-line entry point for the REE-miner pipeline.

Usage
-----
    ree-miner [--workspace DIR] <subcommand> [subcommand-options]

Subcommands
-----------
    test            Run the offline test suite (no network required).
    mine            Mine RCSB PDB for REE-coordinating structures.
    classify        Annotate architecture class for each structure.
    find-homologs   Search UniProt / motif scan for REE-binding homologs.
    build-dataset   Cluster sequences, assign labels, export ESM-Bind JSON.
    engineer        Extract CaM-family EF-hand loops & generate D→P mutants.
    cofactors       Run cofactor / prosthetic-group architecture pipeline.
    scan            Search the Logan metagenome dataset with profile HMMs.
    annotate        Functional annotation of scan hits (taxonomy, eggNOG,
                    genomic neighborhood; archaeal prioritization).

Global flags
------------
    --workspace DIR     Root directory for all output files.
                        Overrides REE_MINER_WORKSPACE env var.
                        Default: ./ree_miner_data/
    --version           Print version and exit.
"""

import argparse
import os
import sys


def _set_workspace(path: str) -> None:
    """Set the workspace before any sub-module imports its path constants."""
    os.environ["REE_MINER_WORKSPACE"] = path
    import ree_miner._workspace as ws
    ws.set_workspace(path)


# ── sub-command handlers ──────────────────────────────────────────────────────

def cmd_test(args: argparse.Namespace) -> int:
    """Run the offline test suite."""
    from ree_miner import __version__
    print(f"ree-miner {__version__}  —  running test suite\n")
    # Import the test module that ships with the package
    import importlib.util, pathlib
    test_path = pathlib.Path(__file__).parent.parent / "tests" / "test_pipeline.py"
    if not test_path.exists():
        print(f"ERROR: test file not found: {test_path}", file=sys.stderr)
        return 1
    spec = importlib.util.spec_from_file_location("test_pipeline", test_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    success = mod.main()
    return 0 if success else 1


def cmd_mine(args: argparse.Namespace) -> int:
    from ree_miner import miner
    return miner.main() or 0


def cmd_classify(args: argparse.Namespace) -> int:
    from ree_miner import classifier
    return classifier.main() or 0


def cmd_find_homologs(args: argparse.Namespace) -> int:
    from ree_miner import homologs
    return homologs.main() or 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from ree_miner import datasets
    return datasets.main() or 0


def cmd_engineer(args: argparse.Namespace) -> int:
    from ree_miner import engineering
    return engineering.main() or 0


def cmd_cofactors(args: argparse.Namespace) -> int:
    from ree_miner import cofactors
    return cofactors.main() or 0


def cmd_annotate(args: argparse.Namespace) -> int:
    """Functional annotation of metagenome scan hits."""
    extra = []
    if hasattr(args, "hits")           and args.hits:           extra += ["--hits",             str(args.hits)]
    if hasattr(args, "out")            and args.out:            extra += ["--out",              str(args.out)]
    if hasattr(args, "contigs_dir")    and args.contigs_dir:    extra += ["--contigs-dir",      str(args.contigs_dir)]
    if hasattr(args, "eggnog_mode")    and args.eggnog_mode:    extra += ["--eggnog-mode",      args.eggnog_mode]
    if hasattr(args, "eggnog_db")      and args.eggnog_db:      extra += ["--eggnog-db",        str(args.eggnog_db)]
    if hasattr(args, "eggnog_tax_scope") and args.eggnog_tax_scope is not None:
        extra += ["--eggnog-tax-scope", str(args.eggnog_tax_scope)]
    if hasattr(args, "archaeal_only")  and args.archaeal_only:  extra += ["--archaeal-only"]
    if hasattr(args, "export_json")    and args.export_json:    extra += ["--export-json",      str(args.export_json)]
    if hasattr(args, "ncbi_api_key")   and args.ncbi_api_key:   extra += ["--ncbi-api-key",     args.ncbi_api_key]
    if hasattr(args, "offline_test")   and args.offline_test:   extra += ["--offline-test"]

    old_argv = sys.argv
    sys.argv = ["ree-miner annotate"] + extra
    try:
        from ree_miner import functional_annotation
        return functional_annotation.main() or 0
    finally:
        sys.argv = old_argv


def cmd_scan(args: argparse.Namespace) -> int:
    """Delegate to metagenomic.main(), forwarding the --mode flag."""
    # Inject the mode into sys.argv so argparse inside metagenomic.main() sees it
    extra = []
    if hasattr(args, "mode") and args.mode:
        extra = ["--mode", args.mode]
    if hasattr(args, "manifest") and args.manifest:
        extra += ["--manifest", args.manifest]
    if hasattr(args, "chunk_id") and args.chunk_id is not None:
        extra += ["--chunk-id", str(args.chunk_id)]
    old_argv = sys.argv
    sys.argv = ["ree-miner scan"] + extra
    try:
        from ree_miner import metagenomic
        return metagenomic.main() or 0
    finally:
        sys.argv = old_argv


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from ree_miner import __version__

    parser = argparse.ArgumentParser(
        prog="ree-miner",
        description="REE-miner: rare-earth-element binding protein discovery pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--workspace", metavar="DIR",
        help="Root directory for all output files (default: ./ree_miner_data/).",
    )

    sub = parser.add_subparsers(dest="command", metavar="<subcommand>")
    sub.required = True

    # test
    p_test = sub.add_parser("test", help="Run offline test suite (no network needed).")
    p_test.set_defaults(func=cmd_test)

    # mine
    p_mine = sub.add_parser("mine", help="Mine RCSB PDB for REE structures.")
    p_mine.set_defaults(func=cmd_mine)

    # classify
    p_cls = sub.add_parser("classify", help="Annotate architecture class for PDB hits.")
    p_cls.set_defaults(func=cmd_classify)

    # find-homologs
    p_hom = sub.add_parser("find-homologs", help="UniProt + motif-scan homolog search.")
    p_hom.set_defaults(func=cmd_find_homologs)

    # build-dataset
    p_ds = sub.add_parser("build-dataset", help="Cluster, label, and export ESM-Bind JSON.")
    p_ds.set_defaults(func=cmd_build_dataset)

    # engineer
    p_eng = sub.add_parser("engineer", help="CaM-family EF-hand loop engineering.")
    p_eng.set_defaults(func=cmd_engineer)

    # cofactors
    p_cof = sub.add_parser("cofactors", help="Cofactor / prosthetic-group architecture pipeline.")
    p_cof.set_defaults(func=cmd_cofactors)

    # scan
    p_scan = sub.add_parser("scan", help="Search Logan metagenome with profile HMMs.")
    p_scan.add_argument(
        "--mode",
        choices=["offline-test", "build-hmms", "generate-slurm", "scan-chunk",
                 "aggregate", "scan-environment"],
        default="offline-test",
        help="Pipeline stage to run (default: offline-test).",
    )
    p_scan.add_argument("--manifest",    metavar="FILE",  help="Logan manifest file path.")
    p_scan.add_argument("--chunk-id",    type=int, metavar="N", help="SLURM array task index.")
    p_scan.add_argument("--environment", metavar="ENV",  help="Target environment key.")
    p_scan.add_argument("--bioprojects", metavar="IDS",  help="Comma-separated BioProject IDs.")
    p_scan.set_defaults(func=cmd_scan)

    # annotate
    p_ann = sub.add_parser(
        "annotate",
        help="Functional annotation of scan hits (taxonomy, eggNOG, neighborhood).",
        description=(
            "Annotates hits from 'ree-miner scan' with:\n"
            "  • NCBI taxonomy (domain / phylum / archaeal flags)\n"
            "  • eggNOG-mapper COG/KEGG/GO terms\n"
            "  • Genomic neighborhood REE gene cluster scoring\n"
            "  • Archaeal hit prioritization\n\n"
            "Run 'ree-miner scan --mode aggregate' first to produce the merged parquet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ann.add_argument("--hits",         type=Path, default=None,
                       help="Input hits parquet (default: datasets/logan_hits_merged.parquet)")
    p_ann.add_argument("--out",          type=Path, default=None,
                       help="Output annotated parquet (default: datasets/annotated_hits.parquet)")
    p_ann.add_argument("--contigs-dir",  type=Path, default=None,  dest="contigs_dir",
                       help="Directory of contig FASTA files for neighborhood analysis")
    p_ann.add_argument("--eggnog-mode",  choices=["web", "local", "skip"], default="skip",
                       dest="eggnog_mode",
                       help="eggNOG-mapper mode: web API, local install, or skip (default: skip)")
    p_ann.add_argument("--eggnog-db",    type=Path, default=None,  dest="eggnog_db",
                       help="Path to local eggNOG database (required for --eggnog-mode=local)")
    p_ann.add_argument("--eggnog-tax-scope", type=int, default=1, dest="eggnog_tax_scope",
                       help="Taxonomic scope for eggNOG (1=root, 2=Bacteria, 2157=Archaea)")
    p_ann.add_argument("--archaeal-only", action="store_true", dest="archaeal_only",
                       help="After annotation, print sorted archaeal hit summary")
    p_ann.add_argument("--export-json",  type=Path, default=None,  dest="export_json",
                       help="Also export ESM-Bind-compatible JSON with full annotations")
    p_ann.add_argument("--ncbi-api-key", type=str, default=None,   dest="ncbi_api_key",
                       help="NCBI API key (also reads NCBI_API_KEY env var)")
    p_ann.add_argument("--offline-test", action="store_true",       dest="offline_test",
                       help="Run built-in offline self-test (T14a–T14f) and exit")
    p_ann.set_defaults(func=cmd_annotate)

    args = parser.parse_args()

    # Apply workspace override BEFORE any sub-module import
    if args.workspace:
        _set_workspace(args.workspace)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
