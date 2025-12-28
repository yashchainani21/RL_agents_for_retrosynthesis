"""
Example demonstrating the visualization flag in DORAnet agent.

This script shows how to enable/disable automatic visualization generation
using the enable_visualization flag.
"""

from pathlib import Path
import sys

from rdkit import Chem
from rdkit import RDLogger

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent import DORAnetMCTS, Node

RDLogger.DisableLog("rdApp.*")


def example_with_visualization():
    """Example with automatic visualization enabled."""
    print("=" * 70)
    print("Example 1: DORAnet agent WITH automatic visualization")
    print("=" * 70)

    target_smiles = "CCCCC(=O)O"
    target_molecule = Chem.MolFromSmiles(target_smiles)

    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=5,
        max_depth=2,
        use_enzymatic=True,
        use_synthetic=True,
        pks_library_file=str(REPO_ROOT / "data" / "processed" / "PKS_smiles.txt"),
        enable_visualization=True,  # Enable auto-visualization
        visualization_output_dir=str(REPO_ROOT / "results"),  # Save to results dir
    )

    agent.run()

    # Results and visualizations will be saved automatically
    results_path = REPO_ROOT / "results" / "example_with_viz.txt"
    agent.save_results(str(results_path))
    print(f"\n✓ Results saved with automatic visualizations!")


def example_without_visualization():
    """Example with automatic visualization disabled."""
    print("\n" + "=" * 70)
    print("Example 2: DORAnet agent WITHOUT automatic visualization")
    print("=" * 70)

    target_smiles = "CCCCC(=O)O"
    target_molecule = Chem.MolFromSmiles(target_smiles)

    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=5,
        max_depth=2,
        use_enzymatic=True,
        use_synthetic=True,
        pks_library_file=str(REPO_ROOT / "data" / "processed" / "PKS_smiles.txt"),
        enable_visualization=False,  # Disable auto-visualization
    )

    agent.run()

    # Only save results, no visualizations
    results_path = REPO_ROOT / "results" / "example_without_viz.txt"
    agent.save_results(str(results_path))
    print(f"\n✓ Results saved without visualizations!")


if __name__ == "__main__":
    # Run both examples
    example_with_visualization()
    example_without_visualization()

    print("\n" + "=" * 70)
    print("Examples complete! Check the results directory for outputs.")
    print("=" * 70)
