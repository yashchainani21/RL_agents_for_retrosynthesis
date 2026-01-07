"""
Example runner for the simplified DORAnet MCTS agent.

This script launches a DORAnet tree search that fragments a target molecule
using retro-enzymatic and retro-synthetic transformations. For each fragment
discovered, a RetroTide forward MCTS search is spawned to attempt synthesis
from PKS building blocks.

Current implementation: Selection + Expansion only (no rollout/backprop yet).
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

from DORAnet_agent import DORAnetMCTS, Node
from DORAnet_agent.visualize import create_enhanced_interactive_html, create_pathways_interactive_html
RDLogger.DisableLog("rdApp.*")

def main() -> None:
    """
    Run the DORAnet MCTS agent 
    """
    create_interactive_visualization = True
    molecule_name = "target"  # e.g., "cryptofolione"
    enable_iteration_viz = False
    iteration_interval = 1
    auto_open_iteration_viz = False
    child_downselection_strategy = "first_N"  # "first_N" or "hybrid"

    # Example target molecule
    # target_smiles = "CCCC(C)=O"  # 3-pentanone (simple ketone)
    # target_smiles = "OCCCC(=O)O"  # 4-hydroxybutyric acid (gamma-hydroxybutyric acid)
    # target_smiles = "OCCCCO"  # 1,4-butanediol
    # target_smiles = "CCCCC(=O)O"  # pentanoic acid (valeric acid)
    # target_smiles = "CCCCCCCCC(=O)O"  # nonanoic acid (known PKS product)
    # target_smiles = "COC1=CC(OC(/C=C/C2=CC=CC=C2)C1)=O" # kavain
    # target_smiles = "CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O" # dronabinol
    # target_smiles = "CC(CC1=CC=C(C=C1)OC)NCC(C2=CC(=C(C=C2)O)NC=O)O" # arformoterol
    # target_smiles = "OC1C=CCC(C1)O" # basidalin
    # target_smiles = "CC1CCCCC(CC1)C" # DMCO
    # target_smiles = "C1C=CC(=O)OC1C=CCC(CC(C=CC2=CC=CC=C2)O)O" # cryptofolione
    # target_smiles = "OC23CCC(C1CC(CCC12C)C3(C)C)C" # patchoul
    target_smiles = "OC1C=CC=CC1"
    target_molecule = Chem.MolFromSmiles(target_smiles)
    
    if target_molecule is None:
        raise ValueError(f"Could not parse target SMILES: {target_smiles}")

    if molecule_name:
        print(f"Target molecule: {molecule_name} ({target_smiles})")
    else:
        print(f"Target molecule: {target_smiles}")

    # specify paths to cofactor files (metabolites and chemistry helpers to exclude from the network)
    cofactors_files = [
        REPO_ROOT / "data" / "raw" / "all_cofactors.csv",
        REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv",
    ]

    # specify path to PKS library file for reward calculation
    pks_library_file = REPO_ROOT / "data" / "processed" / "expanded_PKS_smiles.txt"

    # specify paths to sink compounds files (commercially available building blocks)
    # both biological and chemical building blocks are loaded as sink compounds
    sink_compounds_files = [
        REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt",
        REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt",
    ]

    # specify path to prohibited chemicals file (hazardous/controlled substances to avoid)
    prohibited_chemicals_file = REPO_ROOT / "data" / "processed" / "prohibited_chemical_SMILES.txt"

    # create root node with target
    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    # Common configuration for both sequential and parallel agents
    agent_kwargs = dict(
        root=root,
        target_molecule=target_molecule,
        total_iterations=10,        # more iterations for deeper exploration
        max_depth=3,        # deeper retrosynthetic search
        use_enzymatic=True,
        use_synthetic=True,
        generations_per_expand=1,
        max_children_per_expand=30,  # more children since only PKS matches trigger RetroTide
        child_downselection_strategy=child_downselection_strategy,  # "first_N" or "hybrid"
        cofactors_files=[str(f) for f in cofactors_files],  # exclude cofactors and chemistry helpers
        pks_library_file=str(pks_library_file),  # use PKS library for reward
        sink_compounds_files=[str(f) for f in sink_compounds_files],  # sink compounds (building blocks) that don't need expansion
        prohibited_chemicals_file=str(prohibited_chemicals_file),  # hazardous chemicals to avoid
        MW_multiple_to_exclude=1.5,
        spawn_retrotide=True,       # enable RetroTide for PKS library matches only
        retrotide_kwargs={
            "max_depth": 5,          # more PKS modules to try for exact matches
            "total_iterations": 50,  # more iterations to find exact matches
            "maxPKSDesignsRetroTide": 500,
        },
        sink_terminal_reward=1.0,  # bias selection toward terminal sink compounds
        selection_policy="UCB1",  # "UCB1" for standard or "depth_biased" for depth-first
        depth_bonus_coefficient=4.0,  # only used with depth_biased policy (higher = more depth-first)
        enable_visualization=True,
        enable_interactive_viz=True,  # enable interactive HTML visualizations
        enable_iteration_visualizations=enable_iteration_viz,  # generate visualizations per iteration
        iteration_viz_interval=iteration_interval,   # how often to generate iteration visualizations
        auto_open_iteration_viz=auto_open_iteration_viz,  # auto-open iteration visualizations in browser
        visualization_output_dir=str(REPO_ROOT / "results"),
    )

    print("[Runner] Using sequential DORAnetMCTS")
    agent = DORAnetMCTS(**agent_kwargs)

    # Run the search
    start_time = time.time()
    agent.run()
    total_runtime = time.time() - start_time

    # Print tree summary
    print("\n" + agent.get_tree_summary())

    # Check if we're in PKS library mode or RetroTide mode
    if agent.pks_library:
        # PKS library mode - show matches
        pks_matches = agent.get_pks_matches()
        sink_compounds = agent.get_sink_compounds()
        print(f"\n" + "=" * 70)
        print("PKS Library Match & Sink Compound Results")
        print("=" * 70)
        print(f"\nTotal nodes explored: {len(agent.nodes)}")
        print(f"PKS library matches: {len([n for n in pks_matches if not n.is_sink_compound])}")
        print(f"Sink compounds (building blocks): {len(sink_compounds)}")

        if sink_compounds:
            print(f"\n■ SINK COMPOUNDS (commercially available building blocks):")
            print("-" * 70)
            for node in sink_compounds:
                avg_value = node.value / node.visits if node.visits > 0 else 0
                print(f"  Node {node.node_id} ({node.provenance}): {node.smiles}")
                print(f"    Depth: {node.depth}, Visits: {node.visits}, Avg Value: {avg_value:.2f}")
                if node.reaction_name:
                    rxn_display = node.reaction_name[:60] + "..." if len(node.reaction_name) > 60 else node.reaction_name
                    print(f"    Reaction: {rxn_display}")

        # Show non-sink PKS matches
        non_sink_pks = [n for n in pks_matches if not n.is_sink_compound]
        if non_sink_pks:
            print(f"\n✅ FRAGMENTS MATCHING PKS LIBRARY (non-sink):")
            print("-" * 70)
            for node in non_sink_pks:
                avg_value = node.value / node.visits if node.visits > 0 else 0
                print(f"  Node {node.node_id} ({node.provenance}): {node.smiles}")
                print(f"    Depth: {node.depth}, Visits: {node.visits}, Avg Value: {avg_value:.2f}")
                if node.reaction_name:
                    rxn_display = node.reaction_name[:60] + "..." if len(node.reaction_name) > 60 else node.reaction_name
                    print(f"    Reaction: {rxn_display}")
    else:
        # RetroTide mode - show detailed results
        print(agent.get_results_summary())

    # Save detailed results to file (and generate visualizations if enabled)
    results_dir = REPO_ROOT / "results"
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use molecule name for filename if provided, otherwise use SMILES
    if molecule_name:
        # Sanitize molecule name for filename
        safe_name = molecule_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
        filename_base = f"{safe_name}_sequential"
    else:
        safe_smiles = target_smiles.replace("/", "_").replace("\\", "_")[:20]
        filename_base = f"{safe_smiles}_sequential"

    results_path = results_dir / f"doranet_results_{filename_base}_{timestamp}.txt"
    agent.save_results(str(results_path))
    finalized_pathways_path = results_dir / f"finalized_pathways_{filename_base}_{timestamp}.txt"
    agent.save_finalized_pathways(str(finalized_pathways_path), total_runtime_seconds=total_runtime)
    successful_pathways_path = results_dir / f"successful_pathways_{filename_base}_{timestamp}.txt"
    agent.save_successful_pathways(str(successful_pathways_path))

    # Print summary
    if agent.pks_library:
        pks_matches = agent.get_pks_matches()
        if pks_matches:
            print(f"\n🎉 Found {len(pks_matches)} PKS-synthesizable fragments!")
    else:
        successful = agent.get_successful_results()
        if successful:
            print(f"\n🎉 Found {len(successful)} successful PKS designs!")
            for r in successful:
                print(f"   Node {r.doranet_node_id} ({r.doranet_node_provenance}): {r.doranet_node_smiles}")

    if create_interactive_visualization:
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Use same filename base for interactive visualization
        interactive_path = results_dir / f"doranet_interactive_{filename_base}_{timestamp}.html"
        pathways_path = results_dir / f"doranet_pathways_{filename_base}_{timestamp}.html"

        # Create full tree interactive visualization
        create_enhanced_interactive_html(
            agent=agent,
            output_path=str(interactive_path),
            molecule_img_size=(250, 250),  # size of molecule images in pixels
            auto_open=True,  # automatically open in browser!
            )

        # Create pathways-only interactive visualization (filtered to PKS matches and sink compounds)
        create_pathways_interactive_html(
            agent=agent,
            output_path=str(pathways_path),
            molecule_img_size=(250, 250),
            auto_open=True,  # automatically open in browser!
            )

        print("\n" + "=" * 70)
        print("✓ Interactive visualizations complete!")
        print("=" * 70)
        print(f"\nTwo browser tabs opened:")
        print(f"  1. Full tree: file://{interactive_path}")
        print(f"  2. Pathways only: file://{pathways_path}")
        print("\nFeatures:")
        print("  • Hover over nodes to see molecule structures")
        print("  • View metadata: provenance, PKS match, visits, value")
        print("  • Hover over edges to see reaction SMARTS")
        print("  • Use mouse wheel to zoom")
        print("  • Drag to pan around the tree")
        print("  • Click reset button to restore original view")
        print("\nColor scheme:")
        print("  🟠 Orange = Target molecule")
        print("  🔵 Blue = Enzymatic pathway")
        print("  🟣 Purple = Synthetic pathway")
        print("  🟢 Green = PKS library match ✓")
        print("  ■ Square = Sink compound (building block)")
        print("\nPathways view shows ONLY paths that lead to PKS matches or sink compounds.")

if __name__ == "__main__":
    main()
