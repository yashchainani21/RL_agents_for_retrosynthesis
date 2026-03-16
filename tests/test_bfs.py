"""
Tests for BFS (breadth-first search) mode in DORAnetMCTS.

Tests cover:
- search_strategy parameter validation
- BFS construction and defaults
- BFS run behavior (level-by-level expansion)
- BFS depth limits (total_iterations as depth levels, max_depth ceiling)
- BFS early stopping (stop_on_first_pathway)
- BFS output compatibility (save_results, get_tree_summary)
- MCTS regression (existing behavior unaffected)
"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from rdkit import Chem

from DORAnet_agent.node import Node
from DORAnet_agent.mcts import DORAnetMCTS
from DORAnet_agent.policies import (
    NoOpTerminalDetector,
    SAScore_and_TerminalRewardPolicy,
    SparseTerminalRewardPolicy,
)


# --- Fixtures ---

@pytest.fixture
def sample_molecule():
    """Simple molecule for testing (ethanol)."""
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def root_node(sample_molecule):
    """Root node for testing."""
    return Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")


def _make_agent(root_node, sample_molecule, search_strategy="mcts", **kwargs):
    """Helper to construct a minimal DORAnetMCTS agent for testing."""
    defaults = dict(
        root=root_node,
        target_molecule=sample_molecule,
        total_iterations=2,
        max_depth=3,
        use_enzymatic=False,
        use_synthetic=False,
        terminal_detector=NoOpTerminalDetector(),
        reward_policy=SparseTerminalRewardPolicy(sink_terminal_reward=1.0),
        search_strategy=search_strategy,
    )
    defaults.update(kwargs)
    return DORAnetMCTS(**defaults)


# --- Parameter Validation Tests ---

class TestBFSParameterValidation:
    """Tests for search_strategy parameter validation."""

    def test_invalid_search_strategy_raises_value_error(self, root_node, sample_molecule):
        """search_strategy='invalid' should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid search_strategy"):
            _make_agent(root_node, sample_molecule, search_strategy="invalid")

    def test_bfs_search_strategy_accepted(self, root_node, sample_molecule):
        """search_strategy='bfs' should be accepted without error."""
        agent = _make_agent(root_node, sample_molecule, search_strategy="bfs")
        assert agent.search_strategy == "bfs"

    def test_mcts_search_strategy_accepted(self, root_node, sample_molecule):
        """search_strategy='mcts' should be accepted without error."""
        agent = _make_agent(root_node, sample_molecule, search_strategy="mcts")
        assert agent.search_strategy == "mcts"

    def test_default_search_strategy_is_mcts(self, root_node, sample_molecule):
        """Default search_strategy should be 'mcts'."""
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=3,
            use_enzymatic=False,
            use_synthetic=False,
        )
        assert agent.search_strategy == "mcts"

    def test_bfs_disables_frontier_fallback(self, root_node, sample_molecule):
        """BFS mode should automatically disable frontier fallback."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            enable_frontier_fallback=True,  # Explicitly enable - BFS should override
        )
        assert agent.enable_frontier_fallback is False

    def test_mcts_preserves_frontier_fallback(self, root_node, sample_molecule):
        """MCTS mode should preserve the frontier fallback setting."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="mcts",
            enable_frontier_fallback=True,
        )
        assert agent.enable_frontier_fallback is True


# --- BFS Run Behavior Tests ---

class TestBFSRunBehavior:
    """Tests for BFS run execution and level-by-level expansion."""

    def test_bfs_run_completes(self, root_node, sample_molecule):
        """BFS run() should complete without error and populate self.nodes."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            total_iterations=2,
            max_depth=3,
        )
        # The root is the only node; with no enzymatic/synthetic operators,
        # expand() will produce no children, so BFS stops immediately.
        agent.run()
        # At minimum, the root node should be present
        assert len(agent.nodes) >= 1

    def test_bfs_expands_all_nodes_at_level(self, root_node, sample_molecule):
        """BFS should expand every non-terminal node at each depth level."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            total_iterations=2,
            max_depth=3,
        )

        # Mock expand to create 2 children per node
        original_expand = agent.expand
        call_count = {"value": 0}

        def mock_expand(node):
            call_count["value"] += 1
            child1 = Node(
                fragment=Chem.MolFromSmiles("C"),
                parent=node,
                depth=node.depth + 1,
                provenance="mock",
            )
            child2 = Node(
                fragment=Chem.MolFromSmiles("CC"),
                parent=node,
                depth=node.depth + 1,
                provenance="mock",
            )
            agent.nodes.extend([child1, child2])
            node.children.extend([child1, child2])
            node.expanded = True
            return [child1, child2]

        agent.expand = mock_expand
        agent.run()

        # Level 0: expand root (1 expansion) → 2 children
        # Level 1: expand both children (2 expansions) → 4 grandchildren
        # total_iterations=2 means 2 depth levels
        assert call_count["value"] == 3  # 1 (root) + 2 (depth-1 children)

    def test_bfs_respects_max_depth(self, root_node, sample_molecule):
        """No nodes should be created beyond max_depth."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            total_iterations=10,  # Would go deep if allowed
            max_depth=2,  # But this is the ceiling
        )

        def mock_expand(node):
            child = Node(
                fragment=Chem.MolFromSmiles("C"),
                parent=node,
                depth=node.depth + 1,
                provenance="mock",
            )
            agent.nodes.append(child)
            node.children.append(child)
            node.expanded = True
            return [child]

        agent.expand = mock_expand
        agent.run()

        # Check no node exceeds max_depth
        for node in agent.nodes:
            assert node.depth <= agent.max_depth

    def test_bfs_respects_total_iterations_as_depth_levels(self, root_node, sample_molecule):
        """With total_iterations=1 and max_depth=5, only depth 0 should be expanded."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            total_iterations=1,  # Only expand 1 depth level (root)
            max_depth=5,
        )

        expansion_depths = []

        def mock_expand(node):
            expansion_depths.append(node.depth)
            child = Node(
                fragment=Chem.MolFromSmiles("C"),
                parent=node,
                depth=node.depth + 1,
                provenance="mock",
            )
            agent.nodes.append(child)
            node.children.append(child)
            node.expanded = True
            return [child]

        agent.expand = mock_expand
        agent.run()

        # Only depth-0 node (root) should have been expanded
        assert expansion_depths == [0]
        # Children at depth 1 exist but were NOT expanded
        depth_1_nodes = [n for n in agent.nodes if n.depth == 1]
        assert len(depth_1_nodes) == 1
        assert not depth_1_nodes[0].expanded

    def test_bfs_skips_sink_compounds(self, root_node, sample_molecule):
        """BFS should not expand sink compounds (they are terminal)."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            total_iterations=3,
            max_depth=5,
        )

        expansion_count = {"value": 0}

        def mock_expand(node):
            expansion_count["value"] += 1
            # Create one sink compound and one normal child
            sink_child = Node(
                fragment=Chem.MolFromSmiles("C"),
                parent=node,
                depth=node.depth + 1,
                provenance="mock_sink",
            )
            sink_child.is_sink_compound = True
            sink_child.expanded = True  # Sink compounds are pre-expanded

            normal_child = Node(
                fragment=Chem.MolFromSmiles("CCC"),
                parent=node,
                depth=node.depth + 1,
                provenance="mock_normal",
            )

            agent.nodes.extend([sink_child, normal_child])
            node.children.extend([sink_child, normal_child])
            node.expanded = True
            return [sink_child, normal_child]

        agent.expand = mock_expand
        agent.run()

        # Level 0: expand root → sink + normal
        # Level 1: expand normal only (sink skipped) → sink + normal
        # Level 2: expand normal only → sink + normal
        assert expansion_count["value"] == 3


# --- BFS Early Stopping Tests ---

class TestBFSEarlyStopping:
    """Tests for BFS early stopping (stop_on_first_pathway)."""

    def test_bfs_stop_on_first_pathway(self, root_node, sample_molecule):
        """BFS should stop when a complete pathway is found if stop_on_first_pathway=True."""
        # Add a sink compound to the agent's sink_compounds set so nodes
        # can be recognized as terminal
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            total_iterations=5,
            max_depth=5,
            stop_on_first_pathway=True,
        )

        # Add "C" as a sink compound so the child is recognized as terminal
        canonical_c = Chem.MolToSmiles(Chem.MolFromSmiles("C"))
        agent.sink_compounds.add(canonical_c)

        expansion_count = {"value": 0}

        def mock_expand(node):
            expansion_count["value"] += 1
            # Create a sink compound child (completing a trivial pathway)
            child = Node(
                fragment=Chem.MolFromSmiles("C"),
                parent=node,
                depth=node.depth + 1,
                provenance="mock",
            )
            child.is_sink_compound = True
            child.expanded = True

            agent.nodes.append(child)
            node.children.append(child)
            node.expanded = True
            return [child]

        agent.expand = mock_expand

        # Mock _is_complete_pathway to return True for our sink compound
        agent._is_complete_pathway = lambda node: node.is_sink_compound

        agent.run()

        # Should have stopped after finding the first pathway
        assert agent.first_pathway_found is True
        # Should have expanded at most 1 node (the root)
        assert expansion_count["value"] == 1


# --- BFS Output Compatibility Tests ---

class TestBFSOutputCompatibility:
    """Tests for BFS output compatibility with MCTS output methods."""

    def test_bfs_get_tree_summary(self, root_node, sample_molecule):
        """get_tree_summary() should work in BFS mode and include 'BFS' label."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            total_iterations=1,
            max_depth=3,
        )
        agent.run()

        summary = agent.get_tree_summary()
        assert "BFS" in summary
        assert "Tree Summary" in summary

    def test_mcts_get_tree_summary_label(self, root_node, sample_molecule):
        """get_tree_summary() should include 'MCTS' label in MCTS mode."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="mcts",
            total_iterations=0,
            max_depth=3,
        )
        summary = agent.get_tree_summary()
        assert "MCTS" in summary

    def test_bfs_save_results(self, root_node, sample_molecule):
        """save_results() should work in BFS mode and include search strategy."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            total_iterations=1,
            max_depth=3,
        )
        agent.run()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            tmp_path = f.name

        try:
            agent.save_results(tmp_path)
            with open(tmp_path, "r") as f:
                content = f.read()
            assert "Search strategy: bfs" in content
            assert "BFS mode" in content
        finally:
            os.unlink(tmp_path)

    def test_mcts_save_results_includes_strategy(self, root_node, sample_molecule):
        """save_results() should include search strategy in MCTS mode too."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="mcts",
            total_iterations=0,
            max_depth=3,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            tmp_path = f.name

        try:
            agent.save_results(tmp_path)
            with open(tmp_path, "r") as f:
                content = f.read()
            assert "Search strategy: mcts" in content
        finally:
            os.unlink(tmp_path)


# --- MCTS Regression Tests ---

class TestMCTSRegression:
    """Regression tests to ensure MCTS behavior is unaffected by BFS changes."""

    def test_mcts_run_still_works(self, root_node, sample_molecule):
        """MCTS run() should still work correctly after the refactor."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="mcts",
            total_iterations=2,
            max_depth=3,
        )
        # With no enzymatic/synthetic operators, run() should complete
        # without error (no expansions, but no crashes)
        agent.run()
        assert len(agent.nodes) >= 1

    def test_mcts_default_policies_unchanged(self, root_node, sample_molecule):
        """Default policies should be unchanged after adding search_strategy."""
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=3,
            use_enzymatic=False,
            use_synthetic=False,
        )
        assert isinstance(agent.terminal_detector, NoOpTerminalDetector)
        assert isinstance(agent.reward_policy, SAScore_and_TerminalRewardPolicy)
        assert agent.search_strategy == "mcts"

    def test_mcts_dispatch_calls_run_mcts(self, root_node, sample_molecule):
        """run() with search_strategy='mcts' should dispatch to _run_mcts()."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="mcts",
            total_iterations=0,
        )

        mcts_called = {"value": False}
        bfs_called = {"value": False}

        original_run_mcts = agent._run_mcts
        original_run_bfs = agent._run_bfs

        def mock_run_mcts():
            mcts_called["value"] = True
            original_run_mcts()

        def mock_run_bfs():
            bfs_called["value"] = True
            original_run_bfs()

        agent._run_mcts = mock_run_mcts
        agent._run_bfs = mock_run_bfs

        agent.run()

        assert mcts_called["value"] is True
        assert bfs_called["value"] is False

    def test_bfs_dispatch_calls_run_bfs(self, root_node, sample_molecule):
        """run() with search_strategy='bfs' should dispatch to _run_bfs()."""
        agent = _make_agent(
            root_node, sample_molecule,
            search_strategy="bfs",
            total_iterations=0,
        )

        mcts_called = {"value": False}
        bfs_called = {"value": False}

        original_run_mcts = agent._run_mcts
        original_run_bfs = agent._run_bfs

        def mock_run_mcts():
            mcts_called["value"] = True
            original_run_mcts()

        def mock_run_bfs():
            bfs_called["value"] = True
            original_run_bfs()

        agent._run_mcts = mock_run_mcts
        agent._run_bfs = mock_run_bfs

        agent.run()

        assert bfs_called["value"] is True
        assert mcts_called["value"] is False
