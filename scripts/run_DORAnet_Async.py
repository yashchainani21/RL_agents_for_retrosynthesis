"""
Runner for AsyncExpansionDORAnetMCTS (multiprocessing expansion).
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from rdkit import Chem
from rdkit import RDLogger

# Ensure the repository root is discoverable when running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent import AsyncExpansionDORAnetMCTS, Node
from DORAnet_agent.visualize import create_enhanced_interactive_html, create_pathways_interactive_html

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    # ---- Runner configuration (edit these in your IDE) ----
    create_interactive_visualization = True
    molecule_name = "nonanoic_acid"  # e.g., "cryptofolione"
    enable_iteration_viz = False
    iteration_interval = 1
    auto_open_iteration_viz = False
    num_workers = None  # None means "max available"
    max_inflight_expansions = None  # None means "same as num_workers"
    child_downselection_strategy = "first_N"  # "first_N" or "hybrid"

    # Example target molecule
    # target_smiles = "C1C=CC(=O)OC1C=CCC(CC(C=CC2=CC=CC=C2)O)O" # cryptofolione
    target_smiles = "CCCCCCCCC(=O)O"
    target_molecule = Chem.MolFromSmiles(target_smiles)

    if target_molecule is None:
        raise ValueError(f"Could not parse target SMILES: {target_smiles}")

    if molecule_name:
        print(f"Target molecule: {molecule_name} ({target_smiles})")
    else:
        print(f"Target molecule: {target_smiles}")

    # Paths to cofactor files (metabolites and chemistry helpers to exclude from the network)
    cofactors_files = [
        REPO_ROOT / "data" / "raw" / "all_cofactors.csv",
        REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv",
    ]

    # Path to PKS library file for reward calculation
    pks_library_file = REPO_ROOT / "data" / "processed" / "expanded_PKS_smiles.txt"

    # Paths to sink compounds files (commercially available building blocks)
    sink_compounds_files = [
        REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt",
        REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt",
    ]

    # Path to prohibited chemicals file (hazardous/controlled substances to avoid)
    prohibited_chemicals_file = REPO_ROOT / "data" / "processed" / "prohibited_chemical_SMILES.txt"

    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    agent = AsyncExpansionDORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=100,
        max_depth=3,
        use_enzymatic=True,
        use_synthetic=True,
        generations_per_expand=1,
        max_children_per_expand=30,
        child_downselection_strategy=child_downselection_strategy,
        cofactors_files=[str(f) for f in cofactors_files],
        pks_library_file=str(pks_library_file),
        sink_compounds_files=[str(f) for f in sink_compounds_files],
        prohibited_chemicals_file=str(prohibited_chemicals_file),
        MW_multiple_to_exclude=1.5,
        spawn_retrotide=True,
        retrotide_kwargs={
            "max_depth": 5,
            "total_iterations": 50,
            "maxPKSDesignsRetroTide": 500,
        },
        sink_terminal_reward=1.0,
        selection_policy="UCB1",
        depth_bonus_coefficient=4.0,
        enable_visualization=True,
        enable_interactive_viz=True,
        enable_iteration_visualizations=enable_iteration_viz,
        iteration_viz_interval=iteration_interval,
        auto_open_iteration_viz=auto_open_iteration_viz,
        visualization_output_dir=str(REPO_ROOT / "results"),
        num_workers=num_workers,
        max_inflight_expansions=max_inflight_expansions,
    )

    start_time = time.time()
    agent.run()
    total_runtime = time.time() - start_time

    print("\n" + agent.get_tree_summary())

    results_dir = REPO_ROOT / "results"
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")

    if molecule_name:
        safe_name = molecule_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
        filename_base = f"{safe_name}_async"
    else:
        safe_smiles = target_smiles.replace("/", "_").replace("\\", "_")[:20]
        filename_base = f"{safe_smiles}_async"

    results_path = results_dir / f"doranet_results_{filename_base}_{timestamp}.txt"
    agent.save_results(str(results_path))
    finalized_pathways_path = results_dir / f"finalized_pathways_{filename_base}_{timestamp}.txt"
    agent.save_finalized_pathways(str(finalized_pathways_path), total_runtime_seconds=total_runtime)
    successful_pathways_path = results_dir / f"successful_pathways_{filename_base}_{timestamp}.txt"
    agent.save_successful_pathways(str(successful_pathways_path))

    if create_interactive_visualization:
        results_dir.mkdir(parents=True, exist_ok=True)
        interactive_path = results_dir / f"doranet_interactive_{filename_base}_{timestamp}.html"
        pathways_path = results_dir / f"doranet_pathways_{filename_base}_{timestamp}.html"

        create_enhanced_interactive_html(
            agent=agent,
            output_path=str(interactive_path),
            molecule_img_size=(250, 250),
            auto_open=True,
        )

        create_pathways_interactive_html(
            agent=agent,
            output_path=str(pathways_path),
            molecule_img_size=(250, 250),
            auto_open=True,
        )

        print("\n" + "=" * 70)
        print("✓ Interactive visualizations complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
