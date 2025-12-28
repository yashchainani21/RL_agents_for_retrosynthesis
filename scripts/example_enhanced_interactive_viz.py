"""
Example script demonstrating the enhanced interactive visualization.

This creates an interactive HTML file where you can:
- Hover over nodes to see molecule structure images
- View node metadata (enzymatic/synthetic, PKS match, visits, value)
- Hover over edges to see reaction information (SMARTS, reaction names)
- Zoom and pan through the tree
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
from DORAnet_agent.visualize import create_enhanced_interactive_html

RDLogger.DisableLog("rdApp.*")


def main():
    """Run DORAnet agent and create enhanced interactive visualization."""
    print("=" * 70)
    print("DORAnet Agent with Enhanced Interactive Visualization")
    print("=" * 70)

    # Target molecule - use a PKS-synthesizable molecule
    target_smiles = "CCCCCCCCC(=O)O"  # nonanoic acid
    target_molecule = Chem.MolFromSmiles(target_smiles)

    if target_molecule is None:
        raise ValueError(f"Could not parse target SMILES: {target_smiles}")

    print(f"\nTarget molecule: {target_smiles}")
    print("Running DORAnet MCTS agent...")

    # Create root node
    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    # Configure paths
    cofactors_file = REPO_ROOT / "data" / "raw" / "all_cofactors.csv"
    pks_library_file = REPO_ROOT / "data" / "processed" / "PKS_smiles.txt"

    # Create and run agent
    agent = DORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=10,
        max_depth=2,
        use_enzymatic=True,
        use_synthetic=True,
        generations_per_expand=1,
        max_children_per_expand=8,
        cofactors_file=str(cofactors_file),
        pks_library_file=str(pks_library_file),
        spawn_retrotide=False,  # Disable RetroTide for faster demo
        enable_visualization=False,  # We'll create interactive viz manually
    )

    agent.run()

    print("\n" + "=" * 70)
    print("Creating Enhanced Interactive Visualization")
    print("=" * 70)

    # Create interactive HTML
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    interactive_path = results_dir / "doranet_interactive_enhanced.html"

    create_enhanced_interactive_html(
        agent=agent,
        output_path=str(interactive_path),
        molecule_img_size=(250, 250),  # Size of molecule images in pixels
        auto_open=True,  # Automatically open in browser!
    )

    print("\n" + "=" * 70)
    print("✓ Interactive visualization complete!")
    print("=" * 70)
    print(f"\nOpen in your browser:")
    print(f"  file://{interactive_path}")
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

    # Also save text results
    results_path = results_dir / "doranet_interactive_example.txt"
    agent.save_results(str(results_path))

    print(f"\nText results saved to: {results_path}")


if __name__ == "__main__":
    main()
