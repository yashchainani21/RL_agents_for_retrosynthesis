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
    target_smiles = "CCCCC(=O)O"  # pentanoic acid (valeric acid)
    target_molecule = Chem.MolFromSmiles(target_smiles)
    if target_molecule is None:
        raise ValueError(f"Could not parse target SMILES: {target_smiles}")

    print(f"Target molecule: {target_smiles}")

    # Path to cofactors file (metabolites to exclude from the network)
    cofactors_file = REPO_ROOT / "data" / "raw" / "all_cofactors.csv"

    # Create root node with target
    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    # Configure the DORAnet agent
    # Set spawn_retrotide=False to test DORAnet fragmentation in isolation
    agent = DORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=3,        # Keep small for testing
        max_depth=1,               # Single step fragmentation
        use_enzymatic=True,
        use_synthetic=True,
        generations_per_expand=1,
        max_children_per_expand=5,  # Limit children per node
        cofactors_file=str(cofactors_file),  # Exclude cofactors from network
        spawn_retrotide=True,       # Enable RetroTide spawning
        retrotide_kwargs=dict(
            max_depth=5,
            total_iterations=100,   # Reduced for faster testing
            maxPKSDesignsRetroTide=50,
            selection_policy="UCB1",
            save_logs=False,
        ),
    )

    # Run the search
    agent.run()

    # Print tree summary
    print("\n" + agent.get_tree_summary())

    # Print detailed results summary
    print(agent.get_results_summary())

    # Save detailed results to file
    results_dir = REPO_ROOT / "results"
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_smiles = target_smiles.replace("/", "_").replace("\\", "_")[:20]
    results_path = results_dir / f"doranet_results_{safe_smiles}_{timestamp}.txt"
    agent.save_results(str(results_path))

    # Print summary of successful results
    successful = agent.get_successful_results()
    if successful:
        print(f"\n🎉 Found {len(successful)} successful PKS designs!")
        for r in successful:
            print(f"   Node {r.doranet_node_id} ({r.doranet_node_provenance}): {r.doranet_node_smiles}")


if __name__ == "__main__":
    main()