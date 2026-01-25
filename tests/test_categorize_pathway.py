"""
Unit tests for DORAnetMCTS._categorize_pathway method.

Tests verify that pathways are correctly categorized based on:
1. Terminal node type (PKS vs sink compound)
2. PKS-synthesizable byproducts along the pathway
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch
import sys

import pytest
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent.node import Node
from DORAnet_agent.mcts import DORAnetMCTS, RetroTideResult


@dataclass
class MockRetroTideResult:
    """Mock RetroTide result for testing."""
    doranet_node_id: int
    doranet_node_smiles: str
    doranet_node_depth: int
    doranet_node_provenance: str
    doranet_reaction_name: Optional[str] = None
    doranet_reaction_smarts: Optional[str] = None
    doranet_reactants_smiles: List[str] = field(default_factory=list)
    doranet_products_smiles: List[str] = field(default_factory=list)
    retrotide_target_smiles: str = ""
    retrotide_successful: bool = False
    retrotide_num_successful_nodes: int = 0
    retrotide_best_score: float = 0.0
    retrotide_total_nodes: int = 0
    retrotide_agent: Any = field(default=None, repr=False)


class TestCategorizePathway:
    """Tests for _categorize_pathway with PKS byproduct detection."""

    @pytest.fixture
    def sample_molecule(self):
        return Chem.MolFromSmiles("CCCCC(=O)O")

    @pytest.fixture
    def mcts_agent(self, sample_molecule):
        """Create a minimal DORAnetMCTS agent for testing."""
        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        with patch.object(DORAnetMCTS, '__init__', lambda self, **kwargs: None):
            agent = DORAnetMCTS.__new__(DORAnetMCTS)
            agent.root = root
            agent.nodes = [root]
            agent.retrotide_results = []
            agent.excluded_fragments = set()
            # Bind required methods
            agent.get_pathway_to_node = DORAnetMCTS.get_pathway_to_node.__get__(agent)
            agent._categorize_pathway = DORAnetMCTS._categorize_pathway.__get__(agent)
        return agent

    def test_purely_enzymatic_no_pks(self, mcts_agent, sample_molecule):
        """Test pathway with enzymatic steps and sink terminal (no PKS)."""
        root = mcts_agent.root

        # Create enzymatic child node (sink compound terminal)
        child = Node(
            fragment=Chem.MolFromSmiles("CCO"),
            parent=root,
            depth=1,
            provenance="enzymatic"
        )
        child.is_terminal = True
        child.is_sink_compound = True
        child.is_pks_terminal = False
        root.add_child(child)
        mcts_agent.nodes.append(child)

        # Mock _collect_pks_byproducts_for_pathway to return empty list
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(return_value=[])

        result = mcts_agent._categorize_pathway(child)
        assert result == "purely_enzymatic"

    def test_purely_synthetic_no_pks(self, mcts_agent, sample_molecule):
        """Test pathway with synthetic steps and sink terminal (no PKS)."""
        root = mcts_agent.root

        # Create synthetic child node (sink compound terminal)
        child = Node(
            fragment=Chem.MolFromSmiles("CCO"),
            parent=root,
            depth=1,
            provenance="synthetic"
        )
        child.is_terminal = True
        child.is_sink_compound = True
        child.is_pks_terminal = False
        root.add_child(child)
        mcts_agent.nodes.append(child)

        # Mock _collect_pks_byproducts_for_pathway to return empty list
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(return_value=[])

        result = mcts_agent._categorize_pathway(child)
        assert result == "purely_synthetic"

    def test_enzymatic_pks_terminal(self, mcts_agent, sample_molecule):
        """Test pathway with enzymatic steps and PKS terminal."""
        root = mcts_agent.root

        # Create enzymatic child node (PKS terminal)
        child = Node(
            fragment=Chem.MolFromSmiles("CCCC(=O)O"),
            parent=root,
            depth=1,
            provenance="enzymatic"
        )
        child.is_terminal = True
        child.is_pks_terminal = True
        root.add_child(child)
        mcts_agent.nodes.append(child)

        # No need to call _collect_pks_byproducts since terminal is already PKS
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(return_value=[])

        result = mcts_agent._categorize_pathway(child)
        assert result == "enzymatic_pks"

    def test_enzymatic_with_pks_byproduct(self, mcts_agent, sample_molecule):
        """
        Test pathway with enzymatic steps, sink terminal, but PKS byproduct.

        This is the key bug fix: pathways with PKS-synthesizable byproducts
        should be categorized as *_pks even if the terminal is a sink compound.
        """
        root = mcts_agent.root

        # Create enzymatic child node (sink compound terminal)
        child = Node(
            fragment=Chem.MolFromSmiles("C=CC(=O)O"),  # Acrylic acid - sink
            parent=root,
            depth=1,
            provenance="enzymatic"
        )
        child.is_terminal = True
        child.is_sink_compound = True
        child.is_pks_terminal = False
        child.products_smiles = ["C=CC(=O)O", "CCC=CC(=O)O"]  # Second is PKS byproduct
        root.add_child(child)
        mcts_agent.nodes.append(child)

        # Mock _collect_pks_byproducts_for_pathway to return a PKS byproduct
        mock_retrotide_result = MockRetroTideResult(
            doranet_node_id=999,
            doranet_node_smiles="CCC=CC(=O)O",
            doranet_node_depth=1,
            doranet_node_provenance="enzymatic",
            retrotide_successful=True,
            retrotide_best_score=1.0
        )
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(
            return_value=[(1, "CCC=CC(=O)O", mock_retrotide_result)]
        )

        result = mcts_agent._categorize_pathway(child)
        # Should be enzymatic_pks because of the PKS byproduct
        assert result == "enzymatic_pks"

    def test_synthetic_with_pks_byproduct(self, mcts_agent, sample_molecule):
        """Test pathway with synthetic steps, sink terminal, but PKS byproduct."""
        root = mcts_agent.root

        # Create synthetic child node (sink compound terminal)
        child = Node(
            fragment=Chem.MolFromSmiles("CCO"),
            parent=root,
            depth=1,
            provenance="synthetic"
        )
        child.is_terminal = True
        child.is_sink_compound = True
        child.is_pks_terminal = False
        root.add_child(child)
        mcts_agent.nodes.append(child)

        # Mock _collect_pks_byproducts_for_pathway to return a PKS byproduct
        mock_retrotide_result = MockRetroTideResult(
            doranet_node_id=999,
            doranet_node_smiles="CCCCC(=O)O",
            doranet_node_depth=1,
            doranet_node_provenance="synthetic",
            retrotide_successful=True
        )
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(
            return_value=[(1, "CCCCC(=O)O", mock_retrotide_result)]
        )

        result = mcts_agent._categorize_pathway(child)
        # Should be synthetic_pks because of the PKS byproduct
        assert result == "synthetic_pks"

    def test_mixed_with_pks_byproduct(self, mcts_agent, sample_molecule):
        """Test pathway with mixed chemistry, sink terminal, but PKS byproduct."""
        root = mcts_agent.root

        # Create first enzymatic child
        child1 = Node(
            fragment=Chem.MolFromSmiles("CCCC(=O)O"),
            parent=root,
            depth=1,
            provenance="enzymatic"
        )
        root.add_child(child1)
        mcts_agent.nodes.append(child1)

        # Create second synthetic child (terminal)
        child2 = Node(
            fragment=Chem.MolFromSmiles("CCO"),
            parent=child1,
            depth=2,
            provenance="synthetic"
        )
        child2.is_terminal = True
        child2.is_sink_compound = True
        child2.is_pks_terminal = False
        child1.add_child(child2)
        mcts_agent.nodes.append(child2)

        # Mock _collect_pks_byproducts_for_pathway to return a PKS byproduct
        mock_retrotide_result = MockRetroTideResult(
            doranet_node_id=999,
            doranet_node_smiles="CCCCC(=O)O",
            doranet_node_depth=1,
            doranet_node_provenance="enzymatic",
            retrotide_successful=True
        )
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(
            return_value=[(1, "CCCCC(=O)O", mock_retrotide_result)]
        )

        result = mcts_agent._categorize_pathway(child2)
        # Should be synthetic_enzymatic_pks because of mixed chemistry + PKS byproduct
        assert result == "synthetic_enzymatic_pks"

    def test_direct_pks(self, mcts_agent, sample_molecule):
        """Test pathway with no chemistry steps (direct PKS match)."""
        root = mcts_agent.root
        root.is_terminal = True
        root.is_pks_terminal = True

        # Mock _collect_pks_byproducts_for_pathway - shouldn't be called
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(return_value=[])

        result = mcts_agent._categorize_pathway(root)
        assert result == "direct_pks"

    def test_sink_terminal_with_sink_byproduct_no_pks(self, mcts_agent, sample_molecule):
        """
        Test pathway with sink terminal and sink byproduct (no PKS involvement).

        If all byproducts are also sink compounds, the pathway should NOT be
        categorized as PKS.
        """
        root = mcts_agent.root

        # Create enzymatic child node (sink compound terminal)
        child = Node(
            fragment=Chem.MolFromSmiles("CCO"),
            parent=root,
            depth=1,
            provenance="enzymatic"
        )
        child.is_terminal = True
        child.is_sink_compound = True
        child.is_pks_terminal = False
        root.add_child(child)
        mcts_agent.nodes.append(child)

        # Mock _collect_pks_byproducts_for_pathway to return empty list
        # (sink byproducts are filtered out by _collect_pks_byproducts_for_pathway)
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(return_value=[])

        result = mcts_agent._categorize_pathway(child)
        assert result == "purely_enzymatic"

    def test_retrotide_results_check_for_terminal(self, mcts_agent, sample_molecule):
        """Test that retrotide_results are checked for terminal PKS status."""
        root = mcts_agent.root

        # Create enzymatic child node (not marked as PKS terminal)
        child = Node(
            fragment=Chem.MolFromSmiles("CCCC(=O)O"),
            parent=root,
            depth=1,
            provenance="enzymatic"
        )
        child.node_id = 1
        child.is_terminal = True
        child.is_pks_terminal = False  # Not marked as PKS terminal
        root.add_child(child)
        mcts_agent.nodes.append(child)

        # Add a successful retrotide result for this node
        retrotide_result = MockRetroTideResult(
            doranet_node_id=1,
            doranet_node_smiles="CCCC(=O)O",
            doranet_node_depth=1,
            doranet_node_provenance="enzymatic",
            retrotide_successful=True
        )
        mcts_agent.retrotide_results = [retrotide_result]

        # Mock _collect_pks_byproducts_for_pathway - shouldn't be called
        # since terminal check should find PKS first
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(return_value=[])

        result = mcts_agent._categorize_pathway(child)
        # Should be enzymatic_pks because of retrotide_results match
        assert result == "enzymatic_pks"

    def test_multiple_pks_byproducts(self, mcts_agent, sample_molecule):
        """Test pathway with multiple PKS byproducts (should still work correctly)."""
        root = mcts_agent.root

        # Create enzymatic child node (sink compound terminal)
        child = Node(
            fragment=Chem.MolFromSmiles("CCO"),
            parent=root,
            depth=1,
            provenance="enzymatic"
        )
        child.is_terminal = True
        child.is_sink_compound = True
        child.is_pks_terminal = False
        root.add_child(child)
        mcts_agent.nodes.append(child)

        # Mock _collect_pks_byproducts_for_pathway to return multiple PKS byproducts
        mock_result1 = MockRetroTideResult(
            doranet_node_id=999,
            doranet_node_smiles="CCCC(=O)O",
            doranet_node_depth=1,
            doranet_node_provenance="enzymatic",
            retrotide_successful=True
        )
        mock_result2 = MockRetroTideResult(
            doranet_node_id=998,
            doranet_node_smiles="CCCCC(=O)O",
            doranet_node_depth=1,
            doranet_node_provenance="enzymatic",
            retrotide_successful=True
        )
        mcts_agent._collect_pks_byproducts_for_pathway = MagicMock(
            return_value=[
                (1, "CCCC(=O)O", mock_result1),
                (1, "CCCCC(=O)O", mock_result2)
            ]
        )

        result = mcts_agent._categorize_pathway(child)
        # Should be enzymatic_pks because of the PKS byproducts
        assert result == "enzymatic_pks"
