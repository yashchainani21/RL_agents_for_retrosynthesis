"""
Example runner for the DORAnet-guided MCTS agent.

The script launches a single tree search that applies retro-style
DORAnet enzymatic/synthetic steps to fragment a target molecule.
Whenever an intermediate fragment matches the polyketide library,
the agent records it as a success (and can optionally spawn a
RetroTide forward search—disabled by default here).
"""

from __future__ import annotations

from pathlib import Path
import sys

from rdkit import Chem
from rdkit import RDLogger

# Ensure the repository root (which contains the RL_agents_for_retrosynthesis package)
# is discoverable when this script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RL_agents_for_retrosynthesis.DORAnet_agent import DORAnetMCTS, Node

# Quiet down RDKit warnings for cleaner logs during runs.
RDLogger.DisableLog("rdApp.*")


def main() -> None:
    target_smiles = "CCCC(C)=O"  
    target_molecule = Chem.MolFromSmiles(target_smiles)
    if target_molecule is None:
        raise ValueError(f"Could not parse target SMILES: {target_smiles}")

    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    polyketide_library = (
        "RL_agents_for_retrosynthesis/data/processed/PKS_smiles.txt"
    )

    agent = DORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        polyketide_library_path=polyketide_library,
        total_iterations=250,
        max_depth=5,
        generations_per_expand=1,
        max_children_per_expand=50,
        use_enzymatic=True,
        use_synthetic=True,
        # Flip to True and pass RetroTide parameters if you want downstream exploration.
        spawn_retrotide_on_success=False,
        retrotide_kwargs=dict(
            max_depth=5,
            total_iterations=1500,
            maxPKSDesignsRetroTide=100,
            selection_policy="UCB1",
            save_logs=False,
        ),
    )

    agent.run()

    results_path = Path("RL_agents_for_retrosynthesis/results/doranet_successes.txt")
    agent.save_results(results_path)

    print(f"Search complete. Successful fragments saved to {results_path}")
    if agent.successful_nodes:
        print("\nDiscovered fragments:")
        for node in sorted(agent.successful_nodes, key=lambda n: n.node_id):
            print(f"- {node.smiles} (depth={node.depth}, via {node.provenance})")
    else:
        print("\nNo known polyketide fragments encountered in this run.")


if __name__ == "__main__":
    main()
