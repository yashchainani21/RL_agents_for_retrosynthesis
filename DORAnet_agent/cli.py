"""Command-line entry point for DORAnet MCTS retrosynthesis."""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def _resolve_repo_root() -> Path:
    """Find the repository root by looking for the data/ directory.

    Checks (in order):
    1. Parent of this package (works for editable installs / running from repo)
    2. Current working directory
    """
    # Editable install: DORAnet_agent/ is directly inside the repo root
    package_parent = Path(__file__).resolve().parent.parent
    if (package_parent / "data" / "processed").is_dir():
        return package_parent

    # Fallback: CWD
    cwd = Path.cwd()
    if (cwd / "data" / "processed").is_dir():
        return cwd

    raise FileNotFoundError(
        "Cannot locate the data/ directory. Run from the repository root "
        "or pass --data-dir explicitly."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run DORAnet MCTS retrosynthetic search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target_smiles", help="SMILES string of the target molecule")
    parser.add_argument("--name", default=None, help="Molecule name (used in output filenames)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of MCTS iterations")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum tree depth")
    parser.add_argument("--max-children", type=int, default=5, help="Max children per expansion")
    parser.add_argument("--output-dir", default=None, help="Results output directory")
    parser.add_argument("--output-subfolder", default=None, help="Subfolder within output dir")
    parser.add_argument("--strategy", choices=["mcts", "bfs"], default="mcts", help="Search strategy")
    parser.add_argument("--selection", choices=["UCB1", "depth_biased"], default="UCB1")
    parser.add_argument("--stop-on-first", action="store_true", help="Stop on first complete pathway")
    parser.add_argument("--no-retrotide", action="store_true", help="Disable RetroTide verification")
    parser.add_argument("--no-visualization", action="store_true", help="Skip HTML visualizations")
    parser.add_argument("--data-dir", default=None, help="Path to repository root containing data/")
    args = parser.parse_args(argv)

    # Resolve paths
    if args.data_dir:
        repo_root = Path(args.data_dir)
    else:
        repo_root = _resolve_repo_root()

    results_dir = Path(args.output_dir) if args.output_dir else repo_root / "results"
    if args.output_subfolder:
        results_dir = results_dir / args.output_subfolder
    results_dir.mkdir(parents=True, exist_ok=True)

    # Validate target
    target_molecule = Chem.MolFromSmiles(args.target_smiles)
    if target_molecule is None:
        parser.error(f"Could not parse SMILES: {args.target_smiles}")

    molecule_name = args.name or args.target_smiles
    print(f"Target molecule: {molecule_name} ({args.target_smiles})")

    # Late imports so --help is fast
    from DORAnet_agent import DORAnetMCTS, Node
    from DORAnet_agent.visualize import (
        create_enhanced_interactive_html,
        create_pathways_interactive_html,
    )
    from DORAnet_agent.policies import (
        NoOpTerminalDetector,
        SAScore_and_TerminalRewardPolicy,
        ThermodynamicScaledRewardPolicy,
        VerifyWithRetroTide,
    )

    # Policies
    if args.no_retrotide:
        terminal_detector = NoOpTerminalDetector()
    else:
        terminal_detector = VerifyWithRetroTide()

    reward_policy = ThermodynamicScaledRewardPolicy(
        base_policy=SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=1.0,
            pks_terminal_reward=1.0,
        ),
        feasibility_weight=1.0,
    )

    # Data files
    cofactors_files = [
        str(repo_root / "data" / "raw" / "all_cofactors.csv"),
        str(repo_root / "data" / "raw" / "chemistry_helpers.csv"),
    ]
    pks_library_file = str(repo_root / "data" / "processed" / "expanded_PKS_SMILES_V3.txt")
    sink_compounds_files = [
        str(repo_root / "data" / "processed" / "biological_building_blocks.txt"),
        str(repo_root / "data" / "processed" / "chemical_building_blocks.txt"),
    ]
    prohibited_chemicals_file = str(
        repo_root / "data" / "processed" / "prohibited_chemical_SMILES.txt"
    )

    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=args.iterations,
        max_depth=args.max_depth,
        max_children_per_expand=args.max_children,
        child_downselection_strategy="most_thermo_feasible",
        cofactors_files=cofactors_files,
        pks_library_file=pks_library_file,
        sink_compounds_files=sink_compounds_files,
        prohibited_chemicals_file=prohibited_chemicals_file,
        MW_multiple_to_exclude=1.5,
        terminal_detector=terminal_detector,
        reward_policy=reward_policy,
        retrotide_kwargs={"max_depth": 6, "total_iterations": 100, "maxPKSDesignsRetroTide": 500},
        sink_terminal_reward=1.0,
        selection_policy=args.selection,
        depth_bonus_coefficient=4.0,
        enable_frontier_fallback=False,
        enable_visualization=False,
        enable_interactive_viz=False,
        enable_iteration_visualizations=False,
        visualization_output_dir=str(results_dir),
        stop_on_first_pathway=args.stop_on_first,
        search_strategy=args.strategy,
    )

    start_time = time.time()
    agent.run()
    total_runtime = time.time() - start_time

    print("\n" + agent.get_tree_summary())

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = molecule_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
    filename_base = f"{safe_name}_{args.strategy}"

    agent.save_results(str(results_dir / f"doranet_results_{filename_base}_{timestamp}.txt"))
    agent.save_finalized_pathways(
        str(results_dir / f"finalized_pathways_{filename_base}_{timestamp}.txt"),
        total_runtime_seconds=total_runtime,
    )
    agent.save_successful_pathways(
        str(results_dir / f"successful_pathways_{filename_base}_{timestamp}.txt")
    )

    if not args.no_visualization:
        interactive_path = results_dir / f"doranet_interactive_{filename_base}_{timestamp}.html"
        pathways_path = results_dir / f"doranet_pathways_{filename_base}_{timestamp}.html"
        create_enhanced_interactive_html(agent=agent, output_path=str(interactive_path), auto_open=True)
        create_pathways_interactive_html(agent=agent, output_path=str(pathways_path), auto_open=True)

    # Cleanup .pgnet files
    cleanup_script = repo_root / "scripts" / "cleanup_pgnet_files.py"
    if cleanup_script.exists():
        try:
            subprocess.run([sys.executable, str(cleanup_script), "-y"], check=True)
        except subprocess.CalledProcessError:
            pass


if __name__ == "__main__":
    main()
