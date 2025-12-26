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

from rdkit import Chem
from rdkit import RDLogger

# Ensure the repository root is discoverable when running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent import DORAnetMCTS, Node

# Quiet down RDKit warnings.
RDLogger.DisableLog("rdApp.*")


def main() -> None:
    # Example target molecule
    # target_smiles = "CCCC(C)=O"  # 3-pentanone (simple ketone)
    # target_smiles = "OCCCC(=O)O"  # 4-hydroxybutyric acid (gamma-hydroxybutyric acid)
    # target_smiles = "OCCCCO"  # 1,4-butanediol
    # target_smiles = "CCCCC(=O)O"  # pentanoic acid (valeric acid)
    target_smiles = "CCCCCCCCC(=O)O"  # nonanoic acid (known PKS product)
    target_molecule = Chem.MolFromSmiles(target_smiles)
    if target_molecule is None:
        raise ValueError(f"Could not parse target SMILES: {target_smiles}")

    print(f"Target molecule: {target_smiles}")

    # Path to cofactors file (metabolites to exclude from the network)
    cofactors_file = REPO_ROOT / "data" / "raw" / "all_cofactors.csv"

    # Path to PKS library file for reward calculation
    pks_library_file = REPO_ROOT / "data" / "processed" / "PKS_smiles.txt"

    # Create root node with target
    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    # Configure the DORAnet agent
    # Enable RetroTide spawning to get PKS designs for PKS library matches
    agent = DORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=5,         # More iterations for thorough exploration
        max_depth=1,                # Shallow depth for proof of concept
        use_enzymatic=True,
        use_synthetic=True,
        generations_per_expand=1,
        max_children_per_expand=10,  # More children since only PKS matches trigger RetroTide
        cofactors_file=str(cofactors_file),  # Exclude cofactors from network
        pks_library_file=str(pks_library_file),  # Use PKS library for reward
        spawn_retrotide=True,       # Enable RetroTide for PKS library matches only
        retrotide_kwargs={
            "max_depth": 10,          # More PKS modules to try for exact matches
            "total_iterations": 200,  # More iterations to find exact matches
            "maxPKSDesignsRetroTide": 50,  # More designs per round
        },
    )

    # Run the search
    agent.run()

    # Print tree summary
    print("\n" + agent.get_tree_summary())

    # Check if we're in PKS library mode or RetroTide mode
    if agent.pks_library:
        # PKS library mode - show matches
        pks_matches = agent.get_pks_matches()
        print(f"\n" + "=" * 70)
        print("PKS Library Match Results")
        print("=" * 70)
        print(f"\nTotal nodes explored: {len(agent.nodes)}")
        print(f"PKS library matches: {len(pks_matches)}")

        if pks_matches:
            print(f"\n✅ FRAGMENTS MATCHING PKS LIBRARY:")
            print("-" * 70)
            for node in pks_matches:
                avg_value = node.value / node.visits if node.visits > 0 else 0
                print(f"  Node {node.node_id} ({node.provenance}): {node.smiles}")
                print(f"    Depth: {node.depth}, Visits: {node.visits}, Avg Value: {avg_value:.2f}")
                if node.reaction_name:
                    rxn_display = node.reaction_name[:60] + "..." if len(node.reaction_name) > 60 else node.reaction_name
                    print(f"    Reaction: {rxn_display}")
    else:
        # RetroTide mode - show detailed results
        print(agent.get_results_summary())

    # Save detailed results to file
    results_dir = REPO_ROOT / "results"
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_smiles = target_smiles.replace("/", "_").replace("\\", "_")[:20]
    results_path = results_dir / f"doranet_results_{safe_smiles}_{timestamp}.txt"
    agent.save_results(str(results_path))

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


if __name__ == "__main__":
    main()