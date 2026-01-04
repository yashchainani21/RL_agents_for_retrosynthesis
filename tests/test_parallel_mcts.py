"""
Unit tests for parallel MCTS with virtual loss.

Tests cover:
- Virtual loss application and removal on Node
- Thread safety of parallel operations
- ParallelDORAnetMCTS class functionality
- Result consistency between parallel and sequential execution
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch
from rdkit import Chem

from DORAnet_agent.node import Node
from DORAnet_agent.parallel_mcts import ParallelDORAnetMCTS


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_molecule():
    """Create a sample molecule for testing."""
    return Chem.MolFromSmiles("CCCCC(=O)O")  # pentanoic acid


@pytest.fixture
def root_node(sample_molecule):
    """Create a root node for testing."""
    # Reset node counter for consistent test IDs
    Node.node_counter = 0
    return Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")


@pytest.fixture
def tree_with_children(sample_molecule):
    """Create a small tree for testing virtual loss path operations."""
    Node.node_counter = 0

    root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")
    root.visits = 10
    root.value = 5.0

    child1 = Node(fragment=sample_molecule, parent=root, depth=1, provenance="enzymatic")
    child1.visits = 5
    child1.value = 2.5
    root.add_child(child1)

    child2 = Node(fragment=sample_molecule, parent=root, depth=1, provenance="synthetic")
    child2.visits = 3
    child2.value = 1.5
    root.add_child(child2)

    grandchild = Node(fragment=sample_molecule, parent=child1, depth=2, provenance="enzymatic")
    grandchild.visits = 2
    grandchild.value = 1.0
    child1.add_child(grandchild)

    return root, child1, child2, grandchild


# =============================================================================
# Virtual Loss Tests on Node
# =============================================================================

class TestNodeVirtualLoss:
    """Tests for virtual loss methods on Node class."""

    def test_apply_virtual_loss_increments_visits(self, sample_molecule):
        """Virtual loss should increment visit count."""
        node = Node(fragment=sample_molecule, parent=None, depth=0)
        node.visits = 10

        node.apply_virtual_loss(1.0)

        assert node.visits == 11

    def test_apply_virtual_loss_decrements_value(self, sample_molecule):
        """Virtual loss should decrement node value."""
        node = Node(fragment=sample_molecule, parent=None, depth=0)
        node.value = 5.0

        node.apply_virtual_loss(1.0)

        assert node.value == 4.0

    def test_remove_virtual_loss_decrements_visits(self, sample_molecule):
        """Removing virtual loss should decrement visit count."""
        node = Node(fragment=sample_molecule, parent=None, depth=0)
        node.visits = 10

        node.remove_virtual_loss(1.0)

        assert node.visits == 9

    def test_remove_virtual_loss_increments_value(self, sample_molecule):
        """Removing virtual loss should increment node value."""
        node = Node(fragment=sample_molecule, parent=None, depth=0)
        node.value = 5.0

        node.remove_virtual_loss(1.0)

        assert node.value == 6.0

    def test_virtual_loss_cancellation(self, sample_molecule):
        """Virtual loss should completely cancel out when applied and removed."""
        node = Node(fragment=sample_molecule, parent=None, depth=0)
        original_visits = 10
        original_value = 5.0
        node.visits = original_visits
        node.value = original_value

        # Apply and remove virtual loss
        node.apply_virtual_loss(1.0)
        node.remove_virtual_loss(1.0)

        assert node.visits == original_visits
        assert node.value == original_value

    def test_virtual_loss_with_custom_penalty(self, sample_molecule):
        """Virtual loss should work with custom penalty values."""
        node = Node(fragment=sample_molecule, parent=None, depth=0)
        node.visits = 10
        node.value = 5.0

        node.apply_virtual_loss(2.5)

        assert node.visits == 11
        assert node.value == 2.5

        node.remove_virtual_loss(2.5)

        assert node.visits == 10
        assert node.value == 5.0

    def test_multiple_virtual_losses(self, sample_molecule):
        """Multiple virtual losses should accumulate correctly."""
        node = Node(fragment=sample_molecule, parent=None, depth=0)
        node.visits = 10
        node.value = 10.0

        # Apply virtual loss 3 times
        for _ in range(3):
            node.apply_virtual_loss(1.0)

        assert node.visits == 13
        assert node.value == 7.0

        # Remove virtual loss 3 times
        for _ in range(3):
            node.remove_virtual_loss(1.0)

        assert node.visits == 10
        assert node.value == 10.0


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety of parallel operations."""

    def test_concurrent_virtual_loss_application(self, sample_molecule):
        """Virtual loss should be safe under concurrent access."""
        node = Node(fragment=sample_molecule, parent=None, depth=0)
        node.visits = 0
        node.value = 0.0

        num_threads = 100
        barrier = threading.Barrier(num_threads)

        def apply_and_remove():
            barrier.wait()  # Synchronize start
            node.apply_virtual_loss(1.0)
            time.sleep(0.001)  # Small delay to increase contention
            node.remove_virtual_loss(1.0)

        threads = [threading.Thread(target=apply_and_remove) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After all apply/remove pairs, should be back to original
        assert node.visits == 0
        assert node.value == 0.0

    def test_node_id_uniqueness_under_concurrency(self, sample_molecule, root_node):
        """Node IDs should remain unique under concurrent node creation."""
        # We'll test this through the ParallelDORAnetMCTS's _get_next_node_id method
        # Create a mock parallel agent
        with patch.object(ParallelDORAnetMCTS, '__init__', lambda self: None):
            agent = ParallelDORAnetMCTS.__new__(ParallelDORAnetMCTS)
            agent._node_id_lock = threading.Lock()
            agent._next_node_id = 0

            num_threads = 100
            ids = []
            ids_lock = threading.Lock()

            def get_id():
                node_id = agent._get_next_node_id()
                with ids_lock:
                    ids.append(node_id)

            threads = [threading.Thread(target=get_id) for _ in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All IDs should be unique
            assert len(ids) == num_threads
            assert len(set(ids)) == num_threads


# =============================================================================
# ParallelDORAnetMCTS Tests
# =============================================================================

class TestParallelDORAnetMCTS:
    """Tests for ParallelDORAnetMCTS class."""

    def test_optimal_workers_respects_cpu_count(self, sample_molecule, root_node):
        """Worker count should be capped by CPU count."""
        with patch('os.cpu_count', return_value=4):
            agent = ParallelDORAnetMCTS.__new__(ParallelDORAnetMCTS)
            optimal = agent._get_optimal_workers(10)
            assert optimal <= 3  # At most cpu_count - 1

    def test_optimal_workers_respects_requested(self, sample_molecule, root_node):
        """Worker count should not exceed requested count."""
        with patch('os.cpu_count', return_value=16):
            agent = ParallelDORAnetMCTS.__new__(ParallelDORAnetMCTS)
            optimal = agent._get_optimal_workers(4)
            assert optimal == 4

    def test_optimal_workers_minimum_one(self, sample_molecule, root_node):
        """Worker count should be at least 1."""
        with patch('os.cpu_count', return_value=1):
            agent = ParallelDORAnetMCTS.__new__(ParallelDORAnetMCTS)
            optimal = agent._get_optimal_workers(4)
            assert optimal >= 1

    def test_virtual_loss_path_application(self, tree_with_children):
        """Virtual loss should be applied to entire path from leaf to root."""
        root, child1, child2, grandchild = tree_with_children

        # Store original values
        original_root_visits = root.visits
        original_root_value = root.value
        original_child1_visits = child1.visits
        original_child1_value = child1.value
        original_grandchild_visits = grandchild.visits
        original_grandchild_value = grandchild.value

        # Create mock agent with virtual loss method
        with patch.object(ParallelDORAnetMCTS, '__init__', lambda self: None):
            agent = ParallelDORAnetMCTS.__new__(ParallelDORAnetMCTS)
            agent.virtual_loss = 1.0

            # Apply virtual loss to grandchild path
            agent._apply_virtual_loss_to_path(grandchild)

            # Grandchild, child1, and root should all be affected
            assert grandchild.visits == original_grandchild_visits + 1
            assert grandchild.value == original_grandchild_value - 1.0
            assert child1.visits == original_child1_visits + 1
            assert child1.value == original_child1_value - 1.0
            assert root.visits == original_root_visits + 1
            assert root.value == original_root_value - 1.0

            # child2 should NOT be affected (not on path)
            assert child2.visits == 3
            assert child2.value == 1.5

    def test_virtual_loss_path_removal(self, tree_with_children):
        """Virtual loss should be removed from entire path."""
        root, child1, child2, grandchild = tree_with_children

        # Store original values
        original_values = {
            'root_visits': root.visits,
            'root_value': root.value,
            'child1_visits': child1.visits,
            'child1_value': child1.value,
            'grandchild_visits': grandchild.visits,
            'grandchild_value': grandchild.value,
        }

        with patch.object(ParallelDORAnetMCTS, '__init__', lambda self: None):
            agent = ParallelDORAnetMCTS.__new__(ParallelDORAnetMCTS)
            agent.virtual_loss = 1.0

            # Apply then remove virtual loss
            agent._apply_virtual_loss_to_path(grandchild)
            agent._remove_virtual_loss_from_path(grandchild)

            # All values should be restored
            assert grandchild.visits == original_values['grandchild_visits']
            assert grandchild.value == original_values['grandchild_value']
            assert child1.visits == original_values['child1_visits']
            assert child1.value == original_values['child1_value']
            assert root.visits == original_values['root_visits']
            assert root.value == original_values['root_value']

    def test_get_parallel_stats_returns_expected_keys(self, sample_molecule, root_node):
        """get_parallel_stats should return dictionary with expected keys."""
        with patch.object(ParallelDORAnetMCTS, '__init__', lambda self: None):
            agent = ParallelDORAnetMCTS.__new__(ParallelDORAnetMCTS)
            agent.num_workers = 4
            agent.virtual_loss = 1.0
            agent._completed_iterations = 100
            agent._failed_iterations = 0
            agent.nodes = [root_node]
            agent.pks_library = set()
            agent.sink_compounds = set()

            stats = agent.get_parallel_stats()

            assert 'num_workers' in stats
            assert 'virtual_loss' in stats
            assert 'completed_iterations' in stats
            assert 'failed_iterations' in stats
            assert 'total_nodes' in stats


# =============================================================================
# Integration-style Tests (mocked DORAnet)
# =============================================================================

class TestParallelMCTSIntegration:
    """Integration tests for parallel MCTS (with mocked DORAnet)."""

    def test_sequential_fallback_when_one_worker(self, sample_molecule, root_node):
        """With num_workers=1, should fall back to sequential execution."""
        with patch.object(ParallelDORAnetMCTS, '__init__', lambda self: None):
            agent = ParallelDORAnetMCTS.__new__(ParallelDORAnetMCTS)
            agent.num_workers = 1
            agent.root = root_node
            agent.pks_library = set()

            # Mock the parent class run method
            with patch.object(ParallelDORAnetMCTS.__bases__[0], 'run') as mock_run:
                agent.run()
                mock_run.assert_called_once()

    def test_parallel_execution_with_multiple_workers(self, sample_molecule, root_node):
        """With multiple workers, should use thread pool."""
        with patch.object(ParallelDORAnetMCTS, '__init__', lambda self: None):
            agent = ParallelDORAnetMCTS.__new__(ParallelDORAnetMCTS)
            agent.num_workers = 4
            agent.virtual_loss = 1.0
            agent.total_iterations = 10
            agent.max_depth = 3
            agent.root = root_node
            agent.nodes = [root_node]
            agent.edges = []
            agent.pks_library = set()
            agent.sink_compounds = set()
            agent._tree_lock = threading.Lock()
            agent._node_id_lock = threading.Lock()
            agent._results_lock = threading.Lock()
            agent._next_node_id = 1
            agent._completed_iterations = 0
            agent._failed_iterations = 0
            agent.retrotide_results = []

            # Mock the parallel iteration to always succeed
            with patch.object(agent, '_parallel_iteration', return_value=True):
                with patch.object(agent, 'get_sink_compounds', return_value=[]):
                    with patch.object(agent, 'get_pks_terminal_nodes', return_value=[]):
                        # This should not raise and should complete
                        agent.run()

            assert agent._completed_iterations == 10
            assert agent._failed_iterations == 0


# =============================================================================
# UCB1 Score Impact Tests
# =============================================================================

class TestVirtualLossUCB1Impact:
    """Tests verifying virtual loss affects UCB1 scores correctly."""

    def test_virtual_loss_lowers_ucb1_score(self, sample_molecule):
        """Applying virtual loss should lower the effective UCB1 score."""
        import math

        node = Node(fragment=sample_molecule, parent=None, depth=0)
        node.visits = 10
        node.value = 20.0  # Average value = 2.0

        parent_visits = 100

        # Calculate UCB1 before virtual loss
        exploit_before = node.value / node.visits
        explore_before = math.sqrt(2 * math.log(parent_visits) / node.visits)
        ucb1_before = exploit_before + explore_before

        # Apply virtual loss
        node.apply_virtual_loss(1.0)

        # Calculate UCB1 after virtual loss
        exploit_after = node.value / node.visits
        explore_after = math.sqrt(2 * math.log(parent_visits) / node.visits)
        ucb1_after = exploit_after + explore_after

        # UCB1 should be lower after virtual loss
        assert ucb1_after < ucb1_before

        # Both exploitation and exploration terms should decrease
        assert exploit_after < exploit_before  # Value decreased, visits increased
        assert explore_after < explore_before  # Visits increased

    def test_multiple_virtual_losses_further_lower_score(self, sample_molecule):
        """Multiple virtual losses should progressively lower UCB1 score."""
        import math

        node = Node(fragment=sample_molecule, parent=None, depth=0)
        node.visits = 10
        node.value = 20.0

        parent_visits = 100
        scores = []

        # Record UCB1 after each virtual loss application
        for _ in range(5):
            exploit = node.value / node.visits
            explore = math.sqrt(2 * math.log(parent_visits) / node.visits)
            scores.append(exploit + explore)
            node.apply_virtual_loss(1.0)

        # Each subsequent score should be lower
        for i in range(1, len(scores)):
            assert scores[i] < scores[i-1]
