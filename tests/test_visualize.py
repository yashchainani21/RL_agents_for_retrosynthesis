"""
Unit tests for the DORAnet visualization module.

Tests cover:
- Tree graph creation (create_tree_graph)
- Hierarchical layout computation (get_hierarchical_pos)
- Molecule image generation (_generate_molecule_image_base64)
- Reaction image generation (_generate_reaction_image_base64)
- UCB1 score computation (_compute_ucb1_scores)
- Interactive HTML generation (create_enhanced_interactive_html, create_pathways_interactive_html)
- Static visualization (visualize_doranet_tree)

Most tests use mocked agents and nodes to avoid dependencies on real MCTS runs.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings during tests
RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Mock Node class for testing
# =============================================================================

class MockNode:
    """
    Mock Node class that mimics DORAnet_agent.node.Node for testing.
    """
    node_counter = 0

    def __init__(
        self,
        smiles: Optional[str] = None,
        parent: Optional["MockNode"] = None,
        depth: int = 0,
        provenance: Optional[str] = None,
        visits: int = 0,
        value: float = 0.0,
        is_sink_compound: bool = False,
        sink_compound_type: Optional[str] = None,
        expanded: bool = False,
        reaction_smarts: Optional[str] = None,
        reaction_name: Optional[str] = None,
        reactants_smiles: Optional[List[str]] = None,
        products_smiles: Optional[List[str]] = None,
    ):
        self._smiles = smiles
        self.fragment = Chem.MolFromSmiles(smiles) if smiles else None
        self.parent = parent
        self.children: List[MockNode] = []
        self.depth = depth
        self.provenance = provenance
        self.visits = visits
        self.value = value
        self.is_sink_compound = is_sink_compound
        self.sink_compound_type = sink_compound_type
        self.expanded = expanded
        self.reaction_smarts = reaction_smarts
        self.reaction_name = reaction_name
        self.reactants_smiles = reactants_smiles or []
        self.products_smiles = products_smiles or []
        self.created_at_iteration = 0
        self.expanded_at_iteration = None
        self.selected_at_iterations = []
        self.node_id = MockNode.node_counter
        self.parent_id = parent.node_id if parent else None
        MockNode.node_counter += 1

    @property
    def smiles(self) -> Optional[str]:
        return self._smiles

    def add_child(self, child: "MockNode") -> None:
        if child not in self.children:
            child.parent = self
            child.depth = self.depth + 1
            self.children.append(child)


def reset_mock_node_counter():
    """Reset MockNode counter for predictable node IDs."""
    MockNode.node_counter = 0


# =============================================================================
# Mock Agent class for testing
# =============================================================================

class MockAgent:
    """
    Mock DORAnetMCTS agent for testing visualization functions.
    """
    def __init__(
        self,
        nodes: Optional[List[MockNode]] = None,
        edges: Optional[List[tuple]] = None,
        selection_policy: str = "UCB1",
        depth_bonus_coefficient: float = 0.0,
        pks_library_smiles: Optional[set] = None,
        target_smiles: Optional[str] = "CCO",
        total_iterations: int = 10,
        max_depth: int = 10,
    ):
        self.nodes = nodes or []
        self.edges = edges or []
        self.selection_policy = selection_policy
        self.depth_bonus_coefficient = depth_bonus_coefficient
        self._pks_library_smiles = pks_library_smiles or set()
        self.target_molecule = Chem.MolFromSmiles(target_smiles) if target_smiles else None
        self.total_iterations = total_iterations
        self.max_depth = max_depth
        # Additional attributes needed by visualization functions
        self.pks_library = list(pks_library_smiles) if pks_library_smiles else []
        self.retrotide_results = {}

    def _is_in_pks_library(self, smiles: str) -> bool:
        return smiles in self._pks_library_smiles

    def get_sink_compounds(self) -> List[MockNode]:
        """Return nodes that are sink compounds."""
        return [n for n in self.nodes if n.is_sink_compound]


def create_simple_tree() -> MockAgent:
    """
    Create a simple tree structure for testing:

    Root (ethanol, CCO)
    ├── Child1 (methanol, CO) - biological sink
    └── Child2 (methane, C) - chemical sink
    """
    reset_mock_node_counter()

    root = MockNode(smiles="CCO", depth=0, provenance="target", visits=10, value=5.0)
    child1 = MockNode(
        smiles="CO",
        depth=1,
        provenance="enzymatic",
        visits=5,
        value=3.0,
        is_sink_compound=True,
        sink_compound_type="biological",
    )
    child2 = MockNode(
        smiles="C",
        depth=1,
        provenance="synthetic",
        visits=3,
        value=2.0,
        is_sink_compound=True,
        sink_compound_type="chemical",
    )

    root.add_child(child1)
    root.add_child(child2)

    edges = [(root.node_id, child1.node_id), (root.node_id, child2.node_id)]

    return MockAgent(nodes=[root, child1, child2], edges=edges)


def create_deeper_tree() -> MockAgent:
    """
    Create a deeper tree structure for testing hierarchical layout:

    Root (CCO)
    ├── Child1 (CO)
    │   ├── Grandchild1 (C)
    │   └── Grandchild2 (O)
    └── Child2 (CC)
        └── Grandchild3 (C)
    """
    reset_mock_node_counter()

    root = MockNode(smiles="CCO", depth=0, provenance="target", visits=20, value=10.0)

    child1 = MockNode(smiles="CO", depth=1, provenance="enzymatic", visits=10, value=5.0)
    child2 = MockNode(smiles="CC", depth=1, provenance="synthetic", visits=8, value=4.0)

    grandchild1 = MockNode(
        smiles="C", depth=2, provenance="enzymatic", visits=3, value=1.5,
        is_sink_compound=True, sink_compound_type="biological"
    )
    grandchild2 = MockNode(
        smiles="O", depth=2, provenance="synthetic", visits=4, value=2.0,
        is_sink_compound=True, sink_compound_type="chemical"
    )
    grandchild3 = MockNode(
        smiles="C", depth=2, provenance="enzymatic", visits=5, value=2.5,
        is_sink_compound=True, sink_compound_type="biological"
    )

    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(grandchild1)
    child1.add_child(grandchild2)
    child2.add_child(grandchild3)

    edges = [
        (root.node_id, child1.node_id),
        (root.node_id, child2.node_id),
        (child1.node_id, grandchild1.node_id),
        (child1.node_id, grandchild2.node_id),
        (child2.node_id, grandchild3.node_id),
    ]

    return MockAgent(nodes=[root, child1, child2, grandchild1, grandchild2, grandchild3], edges=edges)


# =============================================================================
# Tests for create_tree_graph
# =============================================================================

class TestCreateTreeGraph:
    """Tests for the create_tree_graph function."""

    def test_creates_directed_graph(self):
        """Test that create_tree_graph returns a NetworkX DiGraph."""
        from DORAnet_agent.visualize import create_tree_graph

        agent = create_simple_tree()
        G = create_tree_graph(agent)

        assert isinstance(G, nx.DiGraph)

    def test_correct_number_of_nodes(self):
        """Test that the graph has the correct number of nodes."""
        from DORAnet_agent.visualize import create_tree_graph

        agent = create_simple_tree()
        G = create_tree_graph(agent)

        assert len(G.nodes) == 3

    def test_correct_number_of_edges(self):
        """Test that the graph has the correct number of edges."""
        from DORAnet_agent.visualize import create_tree_graph

        agent = create_simple_tree()
        G = create_tree_graph(agent)

        assert len(G.edges) == 2

    def test_node_attributes_present(self):
        """Test that nodes have the expected attributes."""
        from DORAnet_agent.visualize import create_tree_graph

        agent = create_simple_tree()
        G = create_tree_graph(agent)

        # Check root node attributes
        root_attrs = G.nodes[0]
        assert "smiles" in root_attrs
        assert "depth" in root_attrs
        assert "visits" in root_attrs
        assert "value" in root_attrs
        assert "provenance" in root_attrs
        assert "is_sink_compound" in root_attrs
        assert "sink_compound_type" in root_attrs

    def test_node_attribute_values(self):
        """Test that node attributes have correct values."""
        from DORAnet_agent.visualize import create_tree_graph

        agent = create_simple_tree()
        G = create_tree_graph(agent)

        root_attrs = G.nodes[0]
        assert root_attrs["smiles"] == "CCO"
        assert root_attrs["depth"] == 0
        assert root_attrs["visits"] == 10
        assert root_attrs["provenance"] == "target"

    def test_empty_agent(self):
        """Test handling of an agent with no nodes."""
        from DORAnet_agent.visualize import create_tree_graph

        agent = MockAgent(nodes=[], edges=[])
        G = create_tree_graph(agent)

        assert len(G.nodes) == 0
        assert len(G.edges) == 0

    def test_avg_value_calculation(self):
        """Test that average value is correctly calculated."""
        from DORAnet_agent.visualize import create_tree_graph

        agent = create_simple_tree()
        G = create_tree_graph(agent)

        root_attrs = G.nodes[0]
        expected_avg = 5.0 / 10  # value / visits
        assert root_attrs["avg_value"] == expected_avg

    def test_smiles_truncation(self):
        """Test that long SMILES strings are truncated."""
        from DORAnet_agent.visualize import create_tree_graph

        reset_mock_node_counter()
        long_smiles = "C" * 30  # 30 characters
        node = MockNode(smiles=long_smiles, depth=0, provenance="target")
        agent = MockAgent(nodes=[node], edges=[])

        G = create_tree_graph(agent)

        # smiles_short should be truncated
        assert len(G.nodes[0]["smiles_short"]) == 23  # 20 chars + "..."


# =============================================================================
# Tests for get_hierarchical_pos
# =============================================================================

class TestGetHierarchicalPos:
    """Tests for the get_hierarchical_pos function."""

    def test_returns_dict(self):
        """Test that get_hierarchical_pos returns a dictionary."""
        from DORAnet_agent.visualize import create_tree_graph, get_hierarchical_pos

        agent = create_simple_tree()
        G = create_tree_graph(agent)
        pos = get_hierarchical_pos(G, root=0)

        assert isinstance(pos, dict)

    def test_all_nodes_have_positions(self):
        """Test that all nodes have position coordinates."""
        from DORAnet_agent.visualize import create_tree_graph, get_hierarchical_pos

        agent = create_simple_tree()
        G = create_tree_graph(agent)
        pos = get_hierarchical_pos(G, root=0)

        for node_id in G.nodes:
            assert node_id in pos
            assert isinstance(pos[node_id], tuple)
            assert len(pos[node_id]) == 2

    def test_root_at_top(self):
        """Test that root node is at the top (y=0 or max y)."""
        from DORAnet_agent.visualize import create_tree_graph, get_hierarchical_pos

        agent = create_simple_tree()
        G = create_tree_graph(agent)
        pos = get_hierarchical_pos(G, root=0)

        root_y = pos[0][1]
        # Root should have y=0 (top of tree which grows downward)
        assert root_y == 0

    def test_children_below_parent(self):
        """Test that children are positioned below their parent."""
        from DORAnet_agent.visualize import create_tree_graph, get_hierarchical_pos

        agent = create_simple_tree()
        G = create_tree_graph(agent)
        pos = get_hierarchical_pos(G, root=0)

        root_y = pos[0][1]
        child1_y = pos[1][1]
        child2_y = pos[2][1]

        # Children should be below root (more negative y)
        assert child1_y < root_y
        assert child2_y < root_y

    def test_deeper_tree_layout(self):
        """Test hierarchical layout with a deeper tree."""
        from DORAnet_agent.visualize import create_tree_graph, get_hierarchical_pos

        agent = create_deeper_tree()
        G = create_tree_graph(agent)
        pos = get_hierarchical_pos(G, root=0)

        # All 6 nodes should have positions
        assert len(pos) == 6


# =============================================================================
# Tests for _generate_molecule_image_base64
# =============================================================================

class TestGenerateMoleculeImageBase64:
    """Tests for the _generate_molecule_image_base64 function."""

    def test_valid_smiles_returns_base64(self):
        """Test that valid SMILES returns a base64 image."""
        from DORAnet_agent.visualize import _generate_molecule_image_base64

        result = _generate_molecule_image_base64("CCO")

        assert result is not None
        assert result.startswith("data:image/png;base64,")

    def test_invalid_smiles_returns_none(self):
        """Test that invalid SMILES returns None."""
        from DORAnet_agent.visualize import _generate_molecule_image_base64

        result = _generate_molecule_image_base64("invalid_smiles_xyz")

        assert result is None

    def test_empty_smiles_returns_none(self):
        """Test that empty SMILES returns None or a blank image."""
        from DORAnet_agent.visualize import _generate_molecule_image_base64

        result = _generate_molecule_image_base64("")

        # RDKit may generate a blank image for empty SMILES, which is acceptable
        # The key is that it doesn't crash
        assert result is None or result.startswith("data:image/png;base64,")

    def test_custom_size(self):
        """Test that custom size parameter is accepted."""
        from DORAnet_agent.visualize import _generate_molecule_image_base64

        result = _generate_molecule_image_base64("CCO", size=(100, 100))

        assert result is not None
        assert result.startswith("data:image/png;base64,")

    def test_complex_molecule(self):
        """Test image generation for a more complex molecule."""
        from DORAnet_agent.visualize import _generate_molecule_image_base64

        # Aspirin
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        result = _generate_molecule_image_base64(aspirin_smiles)

        assert result is not None
        assert result.startswith("data:image/png;base64,")


# =============================================================================
# Tests for _generate_reaction_image_base64
# =============================================================================

class TestGenerateReactionImageBase64:
    """Tests for the _generate_reaction_image_base64 function."""

    def test_valid_reaction_returns_base64(self):
        """Test that valid reaction returns a base64 image."""
        from DORAnet_agent.visualize import _generate_reaction_image_base64

        reactants = ["CC", "O"]
        products = ["CCO"]
        result = _generate_reaction_image_base64(reactants, products)

        assert result is not None
        assert result.startswith("data:image/png;base64,")

    def test_empty_reactants_and_products_returns_none(self):
        """Test that empty reactants and products returns None."""
        from DORAnet_agent.visualize import _generate_reaction_image_base64

        result = _generate_reaction_image_base64([], [])

        assert result is None

    def test_single_reactant_single_product(self):
        """Test reaction with single reactant and product."""
        from DORAnet_agent.visualize import _generate_reaction_image_base64

        result = _generate_reaction_image_base64(["CCO"], ["CC=O"])

        assert result is not None

    def test_multiple_reactants(self):
        """Test reaction with multiple reactants."""
        from DORAnet_agent.visualize import _generate_reaction_image_base64

        result = _generate_reaction_image_base64(["CC", "O", "N"], ["CCO"])

        assert result is not None


# =============================================================================
# Tests for _compute_ucb1_scores
# =============================================================================

class TestComputeUCB1Scores:
    """Tests for the _compute_ucb1_scores function."""

    def test_returns_dict(self):
        """Test that _compute_ucb1_scores returns a dictionary."""
        from DORAnet_agent.visualize import _compute_ucb1_scores

        agent = create_simple_tree()
        scores = _compute_ucb1_scores(agent)

        assert isinstance(scores, dict)

    def test_all_nodes_have_scores(self):
        """Test that all nodes have UCB1 scores."""
        from DORAnet_agent.visualize import _compute_ucb1_scores

        agent = create_simple_tree()
        scores = _compute_ucb1_scores(agent)

        for node in agent.nodes:
            assert node.node_id in scores

    def test_root_score_is_na(self):
        """Test that root node score is 'N/A'."""
        from DORAnet_agent.visualize import _compute_ucb1_scores

        agent = create_simple_tree()
        scores = _compute_ucb1_scores(agent)

        # Root node (node_id 0) should have N/A
        assert scores[0] == "N/A"

    def test_child_scores_are_numeric(self):
        """Test that child node scores are numeric strings."""
        from DORAnet_agent.visualize import _compute_ucb1_scores

        agent = create_simple_tree()
        scores = _compute_ucb1_scores(agent)

        # Child nodes should have numeric scores
        for node in agent.nodes:
            if node.parent is not None:
                score_str = scores[node.node_id]
                # Should be a valid float or infinity symbol
                assert score_str == "∞" or float(score_str)

    def test_unvisited_node_score(self):
        """Test that unvisited nodes have high/infinite scores."""
        from DORAnet_agent.visualize import _compute_ucb1_scores

        reset_mock_node_counter()
        root = MockNode(smiles="CCO", depth=0, provenance="target", visits=10, value=5.0)
        unvisited_child = MockNode(smiles="CO", depth=1, provenance="enzymatic", visits=0, value=0.0)
        root.add_child(unvisited_child)

        agent = MockAgent(nodes=[root, unvisited_child], edges=[(0, 1)])
        scores = _compute_ucb1_scores(agent)

        # Unvisited node should have infinite or very high score
        assert scores[1] == "∞" or float(scores[1]) >= 1000

    def test_depth_biased_policy(self):
        """Test UCB1 computation with depth-biased policy."""
        from DORAnet_agent.visualize import _compute_ucb1_scores

        agent = create_simple_tree()
        agent.selection_policy = "depth_biased"
        agent.depth_bonus_coefficient = 0.5

        scores = _compute_ucb1_scores(agent)

        # Should still return valid scores
        assert all(s == "N/A" or s == "∞" or float(s) for s in scores.values())


# =============================================================================
# Tests for visualize_doranet_tree (static matplotlib visualization)
# =============================================================================

class TestVisualizeDORAnetTree:
    """Tests for the visualize_doranet_tree function."""

    def test_returns_figure(self):
        """Test that visualize_doranet_tree returns a matplotlib Figure."""
        import matplotlib.pyplot as plt
        from DORAnet_agent.visualize import visualize_doranet_tree

        agent = create_simple_tree()
        fig = visualize_doranet_tree(agent)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self):
        """Test that the figure can be saved to a file."""
        import matplotlib.pyplot as plt
        from DORAnet_agent.visualize import visualize_doranet_tree

        agent = create_simple_tree()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            fig = visualize_doranet_tree(agent, output_path=output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            plt.close(fig)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_custom_figsize(self):
        """Test that custom figsize is respected."""
        import matplotlib.pyplot as plt
        from DORAnet_agent.visualize import visualize_doranet_tree

        agent = create_simple_tree()
        fig = visualize_doranet_tree(agent, figsize=(10, 8))

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 8
        plt.close(fig)

    def test_custom_title(self):
        """Test that custom title is applied."""
        import matplotlib.pyplot as plt
        from DORAnet_agent.visualize import visualize_doranet_tree

        agent = create_simple_tree()
        fig = visualize_doranet_tree(agent, title="Test Tree")

        # Figure should have been created successfully
        assert fig is not None
        plt.close(fig)

    def test_empty_agent(self):
        """Test handling of empty agent."""
        import matplotlib.pyplot as plt
        from DORAnet_agent.visualize import visualize_doranet_tree

        agent = MockAgent(nodes=[], edges=[])
        fig = visualize_doranet_tree(agent)

        # Should create a figure even with no nodes
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Tests for create_enhanced_interactive_html
# =============================================================================

# Check if Bokeh is available
try:
    import bokeh
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False


class TestCreateEnhancedInteractiveHTML:
    """Tests for the create_enhanced_interactive_html function."""

    @pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not installed")
    def test_creates_html_file(self):
        """Test that HTML file is created."""
        from DORAnet_agent.visualize import create_enhanced_interactive_html

        agent = create_simple_tree()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            create_enhanced_interactive_html(agent, output_path, auto_open=False)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    @pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not installed")
    def test_html_contains_bokeh_script(self):
        """Test that generated HTML contains Bokeh visualization."""
        from DORAnet_agent.visualize import create_enhanced_interactive_html

        agent = create_simple_tree()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            create_enhanced_interactive_html(agent, output_path, auto_open=False)

            with open(output_path, 'r') as f:
                content = f.read()

            # Should contain Bokeh-specific content
            assert "bokeh" in content.lower() or "Bokeh" in content
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    @pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not installed")
    def test_deeper_tree(self):
        """Test with a deeper tree structure."""
        from DORAnet_agent.visualize import create_enhanced_interactive_html

        agent = create_deeper_tree()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            create_enhanced_interactive_html(agent, output_path, auto_open=False)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_empty_agent_does_not_crash(self):
        """Test that empty agent doesn't cause crash."""
        from DORAnet_agent.visualize import create_enhanced_interactive_html

        agent = MockAgent(nodes=[], edges=[])

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            # Should not raise exception (may print warning if Bokeh not installed)
            create_enhanced_interactive_html(agent, output_path, auto_open=False)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


# =============================================================================
# Tests for create_pathways_interactive_html
# =============================================================================

class TestCreatePathwaysInteractiveHTML:
    """Tests for the create_pathways_interactive_html function."""

    @pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not installed")
    def test_creates_html_file(self):
        """Test that HTML file is created."""
        from DORAnet_agent.visualize import create_pathways_interactive_html

        agent = create_simple_tree()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            create_pathways_interactive_html(agent, output_path, auto_open=False)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    @pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not installed")
    def test_with_sink_compounds(self):
        """Test that pathways to sink compounds are visualized."""
        from DORAnet_agent.visualize import create_pathways_interactive_html

        agent = create_simple_tree()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            create_pathways_interactive_html(agent, output_path, auto_open=False)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


# =============================================================================
# Tests for PKS library matching in visualizations
# =============================================================================

class TestPKSLibraryVisualization:
    """Tests for PKS library matching in visualizations."""

    def test_pks_match_detected_in_graph(self):
        """Test that PKS library matches are detected in graph creation."""
        from DORAnet_agent.visualize import create_tree_graph

        reset_mock_node_counter()
        root = MockNode(smiles="CCO", depth=0, provenance="target", visits=10, value=5.0)
        pks_child = MockNode(smiles="CCCC", depth=1, provenance="enzymatic", visits=5, value=2.5)
        root.add_child(pks_child)

        # Create agent with CCCC in PKS library
        agent = MockAgent(
            nodes=[root, pks_child],
            edges=[(0, 1)],
            pks_library_smiles={"CCCC"}
        )

        G = create_tree_graph(agent)

        # Child node should be marked as PKS match
        assert G.nodes[1]["is_pks_match"] is True
        # Root should not be PKS match
        assert G.nodes[0]["is_pks_match"] is False


# =============================================================================
# Tests for edge cases and error handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_node_with_none_smiles(self):
        """Test handling of nodes with None SMILES."""
        from DORAnet_agent.visualize import create_tree_graph

        reset_mock_node_counter()
        root = MockNode(smiles=None, depth=0, provenance="target", visits=1, value=0.0)
        agent = MockAgent(nodes=[root], edges=[])

        G = create_tree_graph(agent)

        assert G.nodes[0]["smiles"] is None

    def test_single_node_tree(self):
        """Test visualization with only a root node."""
        from DORAnet_agent.visualize import create_tree_graph, get_hierarchical_pos

        reset_mock_node_counter()
        root = MockNode(smiles="CCO", depth=0, provenance="target", visits=1, value=0.0)
        agent = MockAgent(nodes=[root], edges=[])

        G = create_tree_graph(agent)
        pos = get_hierarchical_pos(G, root=0)

        assert len(G.nodes) == 1
        assert 0 in pos

    def test_large_tree(self):
        """Test with a larger tree (performance check)."""
        from DORAnet_agent.visualize import create_tree_graph, get_hierarchical_pos

        reset_mock_node_counter()

        # Create a tree with 50 nodes
        nodes = []
        edges = []

        root = MockNode(smiles="C" * 10, depth=0, provenance="target", visits=100, value=50.0)
        nodes.append(root)

        for i in range(49):
            parent_idx = i // 3  # Each node has up to 3 children
            parent = nodes[parent_idx]
            child = MockNode(
                smiles=f"C{i}",
                depth=parent.depth + 1,
                provenance="enzymatic" if i % 2 == 0 else "synthetic",
                visits=max(1, 50 - i),
                value=float(50 - i),
            )
            parent.add_child(child)
            nodes.append(child)
            edges.append((parent.node_id, child.node_id))

        agent = MockAgent(nodes=nodes, edges=edges)

        G = create_tree_graph(agent)
        pos = get_hierarchical_pos(G, root=0)

        assert len(G.nodes) == 50
        assert len(pos) == 50


# =============================================================================
# Integration test - requires actual DORAnet_agent imports
# =============================================================================

class TestIntegrationWithRealNode:
    """
    Integration tests using real DORAnet_agent.node.Node class.
    These tests verify compatibility with the actual implementation.
    """

    @pytest.fixture
    def real_node_tree(self):
        """Create a tree using real Node class."""
        try:
            from DORAnet_agent.node import Node
            Node.node_counter = 0

            root = Node(
                fragment=Chem.MolFromSmiles("CCO"),
                parent=None,
                depth=0,
                provenance="target"
            )
            root.visits = 10
            root.value = 5.0
            root.created_at_iteration = 0

            child = Node(
                fragment=Chem.MolFromSmiles("CO"),
                parent=root,
                depth=1,
                provenance="enzymatic"
            )
            child.visits = 5
            child.value = 2.5
            child.is_sink_compound = True
            child.sink_compound_type = "biological"
            child.created_at_iteration = 1

            root.add_child(child)

            return root, child
        except ImportError:
            pytest.skip("DORAnet_agent.node not available")

    def test_create_tree_graph_with_real_nodes(self, real_node_tree):
        """Test create_tree_graph with real Node instances."""
        from DORAnet_agent.visualize import create_tree_graph

        root, child = real_node_tree

        # Create a mock agent with real nodes
        class RealNodeAgent:
            def __init__(self, nodes, edges):
                self.nodes = nodes
                self.edges = edges

            def _is_in_pks_library(self, smiles):
                return False

        agent = RealNodeAgent(nodes=[root, child], edges=[(0, 1)])

        G = create_tree_graph(agent)

        assert len(G.nodes) == 2
        assert G.nodes[0]["smiles"] == "CCO"
        assert G.nodes[1]["is_sink_compound"] is True


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
