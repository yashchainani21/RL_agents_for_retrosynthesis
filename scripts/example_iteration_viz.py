"""
Example script demonstrating iteration-by-iteration visualization.

This script shows how to use the new iteration visualization feature to track
how the DORAnet MCTS tree grows over time and which nodes are being selected.
"""

from pathlib import Path
import sys
from rdkit import Chem
from rdkit import RDLogger

# Ensure the repository root is discoverable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent import DORAnetMCTS, Node

RDLogger.DisableLog("rdApp.*")


def main():
    """
    Run DORAnet with iteration visualizations enabled.

    This will create visualizations after each iteration showing:
    - Current state of the tree
    - Which nodes have been selected
    - Tree growth dynamics
    - UCB1 selection behavior
    """
    # Simple target molecule for testing
    target_smiles = "CCCCC(=O)O"  # pentanoic acid
    target_molecule = Chem.MolFromSmiles(target_smiles)

    print(f"Target molecule: {target_smiles}")
    print(f"Running MCTS with iteration visualizations...")

    # Paths to cofactor files
    cofactors_files = [
        REPO_ROOT / "data" / "raw" / "all_cofactors.csv",
        REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv",
    ]

    # Path to PKS library
    pks_library_file = REPO_ROOT / "data" / "processed" / "PKS_smiles.txt"

    # Create root node
    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    # Configure agent with iteration visualizations enabled
    agent = DORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=10,  # Small number for testing
        max_depth=2,
        use_enzymatic=True,
        use_synthetic=True,
        generations_per_expand=10,
        max_children_per_expand=5,
        cofactors_files=[str(f) for f in cofactors_files],
        pks_library_file=str(pks_library_file),
        spawn_retrotide=False,
        
        # Visualization settings
        enable_visualization=True,
        enable_interactive_viz=True,
        enable_iteration_visualizations=True,  # Enable iteration viz!
        iteration_viz_interval=1,  # Generate after every iteration
        auto_open_iteration_viz=False,  # Don't auto-open (set to True to open each iteration in browser)
        visualization_output_dir=str(REPO_ROOT / "results"),
    )

    # Run the search
    agent.run()

    print("\n" + "=" * 70)
    print("Iteration Visualizations Complete!")
    print("=" * 70)
    print(f"\nCheck the following directory for iteration visualizations:")
    print(f"  {REPO_ROOT / 'results' / 'iterations'}")
    print(f"\nYou should see files like:")
    print(f"  iteration_00000_tree.png")
    print(f"  iteration_00000_interactive.html")
    print(f"  iteration_00001_tree.png")
    print(f"  iteration_00001_interactive.html")
    print(f"  ...")
    print(f"\nThese visualizations show the tree state after each iteration,")
    print(f"allowing you to see:")
    print(f"  • How the tree grows over time")
    print(f"  • Which nodes are selected by the UCB1 policy")
    print(f"  • Node visit counts and values at each step")
    print(f"  • The exploration vs exploitation trade-off in action")


if __name__ == "__main__":
    main()
