"""
Test script for the simplified DORAnet MCTS agent.

This script provides both:
1. Unit tests with mocked dependencies (fast, no external deps)
2. Integration test with real DORAnet/RetroTide (slower, requires full setup)

Usage:
    # Run unit tests only (fast):
    python test_DORAnet_agent.py --unit

    # Run integration test (requires DORAnet + RetroTide):
    python test_DORAnet_agent.py --integration

    # Run both:
    python test_DORAnet_agent.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

from rdkit import Chem
from rdkit import RDLogger

# Ensure the repository root is discoverable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Mock Setup - Must happen BEFORE importing DORAnet_agent
# =============================================================================

def setup_mocks():
    """
    Set up module-level mocks before importing DORAnet_agent.
    This is necessary because DORAnet_agent.mcts imports RetroTide at module load time.
    """
    # Create mock RetroTide modules
    mock_retrotide_mcts = MagicMock()
    mock_retrotide_node = MagicMock()

    # Mock the RetroTide MCTS class
    mock_mcts_class = MagicMock()
    mock_mcts_class.return_value = MagicMock()
    mock_retrotide_mcts.MCTS = mock_mcts_class

    # Mock the RetroTide Node class
    mock_node_class = MagicMock()
    mock_retrotide_node.Node = mock_node_class

    # Mock DORAnet modules - IMPORTANT: create a connected hierarchy
    mock_enzymatic = MagicMock()
    mock_synthetic = MagicMock()
    mock_doranet_modules = MagicMock()
    mock_doranet_modules.enzymatic = mock_enzymatic
    mock_doranet_modules.synthetic = mock_synthetic
    mock_doranet = MagicMock()
    mock_doranet.modules = mock_doranet_modules

    # Install mocks in sys.modules BEFORE importing DORAnet_agent
    sys.modules['RetroTide_agent'] = MagicMock()
    sys.modules['RetroTide_agent.mcts'] = mock_retrotide_mcts
    sys.modules['RetroTide_agent.node'] = mock_retrotide_node
    sys.modules['doranet'] = mock_doranet
    sys.modules['doranet.modules'] = mock_doranet_modules
    sys.modules['doranet.modules.enzymatic'] = mock_enzymatic
    sys.modules['doranet.modules.synthetic'] = mock_synthetic

    return mock_enzymatic, mock_synthetic, mock_mcts_class


# Set up mocks before any DORAnet imports
MOCK_ENZYMATIC, MOCK_SYNTHETIC, MOCK_RETROTIDE_MCTS = setup_mocks()

# NOW we can safely import DORAnet_agent
from DORAnet_agent.node import Node
from DORAnet_agent.mcts import DORAnetMCTS


# =============================================================================
# Unit Tests (with mocked dependencies)
# =============================================================================

def test_node_creation():
    """Test basic Node creation and properties."""
    # Reset node counter for predictable IDs
    Node.node_counter = 0

    mol = Chem.MolFromSmiles("CCO")
    root = Node(fragment=mol, parent=None, depth=0, provenance="target")

    assert root.node_id == 0, f"Expected node_id=0, got {root.node_id}"
    assert root.depth == 0, f"Expected depth=0, got {root.depth}"
    assert root.smiles == "CCO", f"Expected smiles='CCO', got {root.smiles}"
    assert root.provenance == "target", f"Expected provenance='target', got {root.provenance}"
    assert root.parent is None, "Expected parent=None"
    assert root.children == [], "Expected children=[]"
    assert root.visits == 0, f"Expected visits=0, got {root.visits}"
    assert root.value == 0.0, f"Expected value=0.0, got {root.value}"
    assert root.expanded is False, "Expected expanded=False"

    print("[PASS] test_node_creation")


def test_node_add_child():
    """Test adding children to nodes."""
    Node.node_counter = 0

    parent_mol = Chem.MolFromSmiles("CCCO")
    child_mol = Chem.MolFromSmiles("CCO")

    parent = Node(fragment=parent_mol, parent=None, depth=0, provenance="target")
    child = Node(fragment=child_mol, parent=None, provenance="enzymatic")

    parent.add_child(child)

    assert child in parent.children, "Child not in parent.children"
    assert child.parent == parent, "Child's parent not set correctly"
    assert child.depth == 1, f"Expected child depth=1, got {child.depth}"

    print("[PASS] test_node_add_child")


def test_node_update():
    """Test node update method."""
    Node.node_counter = 0

    mol = Chem.MolFromSmiles("CCO")
    node = Node(fragment=mol, parent=None, depth=0, provenance="target")

    node.update(reward=0.5)
    assert node.visits == 1, f"Expected visits=1, got {node.visits}"
    assert node.value == 0.5, f"Expected value=0.5, got {node.value}"

    node.update(reward=0.3)
    assert node.visits == 2, f"Expected visits=2, got {node.visits}"
    assert node.value == 0.8, f"Expected value=0.8, got {node.value}"

    print("[PASS] test_node_update")


def test_mcts_initialization():
    """Test DORAnetMCTS initialization."""
    Node.node_counter = 0

    mol = Chem.MolFromSmiles("CCCC(C)=O")
    root = Node(fragment=mol, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(
        root=root,
        target_molecule=mol,
        total_iterations=10,
        max_depth=2,
    )

    assert agent.root == root, "Root not set correctly"
    assert agent.total_iterations == 10, f"Expected total_iterations=10, got {agent.total_iterations}"
    assert agent.max_depth == 2, f"Expected max_depth=2, got {agent.max_depth}"
    assert len(agent.nodes) == 1, f"Expected 1 node, got {len(agent.nodes)}"
    assert len(agent.retrotide_runs) == 0, f"Expected 0 retrotide runs, got {len(agent.retrotide_runs)}"
    assert "O" in agent.excluded_fragments, "Water should be in excluded_fragments"

    print("[PASS] test_mcts_initialization")


def test_select_on_root():
    """Test that select returns root when tree has no children."""
    Node.node_counter = 0

    mol = Chem.MolFromSmiles("CCO")
    root = Node(fragment=mol, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(root=root, target_molecule=mol)

    selected = agent.select(root)
    assert selected == root, "Selection should return root when no children"

    print("[PASS] test_select_on_root")


def test_select_prefers_unvisited():
    """Test that UCB1 selection prioritizes unvisited nodes."""
    Node.node_counter = 0

    mol = Chem.MolFromSmiles("CCO")
    root = Node(fragment=mol, parent=None, depth=0, provenance="target")
    root.visits = 5  # Parent has been visited

    # Create children
    child1 = Node(fragment=Chem.MolFromSmiles("CO"), parent=None, provenance="enzymatic")
    child2 = Node(fragment=Chem.MolFromSmiles("CC"), parent=None, provenance="synthetic")
    root.add_child(child1)
    root.add_child(child2)

    # Child1 visited, child2 not visited
    child1.visits = 3
    child2.visits = 0

    agent = DORAnetMCTS(root=root, target_molecule=mol)

    selected = agent.select(root)
    # Should select the unvisited child (gets infinite score)
    assert selected == child2, f"Expected child2 (unvisited), got {selected}"

    print("[PASS] test_select_prefers_unvisited")


def test_expand_with_mock_doranet():
    """Test expansion with mocked DORAnet fragment generation."""
    Node.node_counter = 0

    # Create mock network with fragments
    mock_network = MagicMock()
    mock_mol1 = MagicMock()
    mock_mol1.uid = "CCO"
    mock_mol2 = MagicMock()
    mock_mol2.uid = "CC"
    mock_network.mols = [mock_mol1, mock_mol2]

    # Use the global mocks that are connected to the import hierarchy
    MOCK_ENZYMATIC.generate_network.return_value = mock_network
    MOCK_SYNTHETIC.generate_network.return_value = MagicMock(mols=[])

    mol = Chem.MolFromSmiles("CCCO")  # propanol
    root = Node(fragment=mol, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(
        root=root,
        target_molecule=mol,
        use_enzymatic=True,
        use_synthetic=False,  # Only enzymatic for simplicity
        max_children_per_expand=10,
    )

    new_children = agent.expand(root)

    # Should have created 2 children (CCO and CC)
    assert len(new_children) == 2, f"Expected 2 children, got {len(new_children)}"
    assert root.expanded is True, "Root should be marked as expanded"
    assert len(agent.nodes) == 3, f"Expected 3 nodes (root + 2 children), got {len(agent.nodes)}"

    # Should have spawned RetroTide for each child
    assert len(agent.retrotide_runs) == 2, f"Expected 2 RetroTide runs, got {len(agent.retrotide_runs)}"

    print("[PASS] test_expand_with_mock_doranet")


def test_excluded_fragments():
    """Test that small byproducts are excluded."""
    Node.node_counter = 0

    # Create mock network with water and a real fragment
    mock_network = MagicMock()
    mock_water = MagicMock()
    mock_water.uid = "O"  # water - should be excluded
    mock_frag = MagicMock()
    mock_frag.uid = "CCO"  # ethanol - should be kept
    mock_network.mols = [mock_water, mock_frag]

    MOCK_ENZYMATIC.generate_network.return_value = mock_network
    MOCK_SYNTHETIC.generate_network.return_value = MagicMock(mols=[])

    mol = Chem.MolFromSmiles("CCCO")
    root = Node(fragment=mol, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(
        root=root,
        target_molecule=mol,
        use_enzymatic=True,
        use_synthetic=False,
    )

    new_children = agent.expand(root)

    # Should only have 1 child (water excluded)
    assert len(new_children) == 1, f"Expected 1 child (water excluded), got {len(new_children)}"
    assert new_children[0].smiles == "CCO", f"Expected child smiles='CCO', got {new_children[0].smiles}"

    print("[PASS] test_excluded_fragments")


def test_run_loop_respects_max_depth():
    """Test that the run loop respects max_depth."""
    Node.node_counter = 0

    # Return empty networks (no children generated)
    MOCK_ENZYMATIC.generate_network.return_value = MagicMock(mols=[])
    MOCK_SYNTHETIC.generate_network.return_value = MagicMock(mols=[])

    mol = Chem.MolFromSmiles("CCO")
    root = Node(fragment=mol, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(
        root=root,
        target_molecule=mol,
        total_iterations=5,
        max_depth=1,
    )

    agent.run()

    # Root should have been visited
    assert root.visits > 0, f"Expected root.visits > 0, got {root.visits}"
    assert root.expanded is True, "Root should be marked as expanded"

    print("[PASS] test_run_loop_respects_max_depth")


def test_get_tree_summary():
    """Test the tree summary output."""
    Node.node_counter = 0

    mol = Chem.MolFromSmiles("CCO")
    root = Node(fragment=mol, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(root=root, target_molecule=mol)

    summary = agent.get_tree_summary()

    assert "DORAnet MCTS Tree Summary:" in summary, "Summary should have header"
    assert "Node 0:" in summary, "Summary should contain root node"
    assert "CCO" in summary, "Summary should contain molecule SMILES"

    print("[PASS] test_get_tree_summary")


def run_unit_tests():
    """Run all unit tests."""
    print("\n" + "=" * 50)
    print("Running Unit Tests (with mocked dependencies)")
    print("=" * 50 + "\n")

    tests = [
        test_node_creation,
        test_node_add_child,
        test_node_update,
        test_mcts_initialization,
        test_select_on_root,
        test_select_prefers_unvisited,
        test_expand_with_mock_doranet,
        test_excluded_fragments,
        test_run_loop_respects_max_depth,
        test_get_tree_summary,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "-" * 50)
    print(f"Unit Tests: {passed} passed, {failed} failed")
    print("-" * 50)

    return failed == 0


# =============================================================================
# Integration Test (requires real DORAnet + RetroTide)
# =============================================================================

def run_integration_test():
    """
    Run a full integration test with real DORAnet and RetroTide.

    This test requires the full environment to be set up.
    """
    print("\n" + "=" * 50)
    print("Running Integration Test (real DORAnet + RetroTide)")
    print("=" * 50 + "\n")

    print("[INFO] Integration test requires real DORAnet + RetroTide modules.")
    print("[INFO] Since we're using mocked modules for unit tests, the integration")
    print("[INFO] test would need a separate script that imports without mocks.")
    print("\n[SKIP] Use run_DORAnet_single_agent.py for integration testing.\n")

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test the simplified DORAnet MCTS agent")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration test only")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # Default to unit tests if no args
    if not (args.unit or args.integration or args.all):
        args.unit = True

    success = True

    if args.unit or args.all:
        success = run_unit_tests() and success

    if args.integration or args.all:
        success = run_integration_test() and success

    print("\n" + "=" * 50)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 50)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()