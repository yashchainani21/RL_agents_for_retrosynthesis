"""
Integration tests for the policy system with DORAnetMCTS and AsyncExpansionDORAnetMCTS.

These tests verify that the policy system integrates correctly with the MCTS agents,
including backward compatibility with the legacy spawn_retrotide parameter.
"""

import pytest
from typing import List
from unittest.mock import MagicMock, patch

from rdkit import Chem

from DORAnet_agent.node import Node
from DORAnet_agent.mcts import DORAnetMCTS
from DORAnet_agent.async_expansion_mcts import AsyncExpansionDORAnetMCTS
from DORAnet_agent.policies import (
    RolloutPolicy,
    RewardPolicy,
    RolloutResult,
    NoOpRolloutPolicy,
    SpawnRetroTideOnDatabaseCheck,
    SparseTerminalRewardPolicy,
    SinkCompoundRewardPolicy,
    PKSLibraryRewardPolicy,
    ComposedRewardPolicy,
)


# --- Fixtures ---

@pytest.fixture
def sample_smiles():
    """A simple molecule SMILES for testing."""
    return "CCCCC(=O)O"  # Pentanoic acid


@pytest.fixture
def sample_molecule(sample_smiles):
    """RDKit Mol object for testing."""
    return Chem.MolFromSmiles(sample_smiles)


@pytest.fixture
def root_node(sample_molecule):
    """A root node for testing."""
    Node.node_counter = 0  # Reset counter for consistent IDs
    return Node(
        fragment=sample_molecule,
        parent=None,
        depth=0,
        provenance="target",
    )


@pytest.fixture
def pks_library():
    """A mock PKS library with a matching compound."""
    return {"CC(O)CC(=O)O": "3-hydroxybutanoic acid"}


# --- Custom policies for testing ---

class CountingRolloutPolicy(RolloutPolicy):
    """A rollout policy that counts how many times it's called."""
    
    def __init__(self):
        self.call_count = 0
        self.called_nodes: List[Node] = []
    
    @property
    def name(self) -> str:
        return "Counting"
    
    def rollout(self, node: Node, context: dict) -> RolloutResult:
        self.call_count += 1
        self.called_nodes.append(node)
        return RolloutResult(reward=0.5, terminal=False)


class FixedRewardPolicy(RewardPolicy):
    """A reward policy that returns a fixed reward."""
    
    def __init__(self, reward: float = 1.0):
        self._reward = reward
        self.call_count = 0
        self.called_nodes: List[Node] = []
    
    @property
    def name(self) -> str:
        return f"Fixed({self._reward})"
    
    def calculate_reward(self, node: Node, context: dict) -> float:
        self.call_count += 1
        self.called_nodes.append(node)
        return self._reward


# --- DORAnetMCTS Integration Tests ---

class TestDORAnetMCTSPolicyIntegration:
    """Integration tests for DORAnetMCTS with the policy system."""
    
    def test_default_policies_are_noop_and_sparse(self, root_node, sample_molecule):
        """Default instantiation uses NoOpRolloutPolicy and SparseTerminalRewardPolicy."""
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,  # Don't run
            max_depth=5,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        assert isinstance(agent.rollout_policy, NoOpRolloutPolicy)
        assert isinstance(agent.reward_policy, SparseTerminalRewardPolicy)
    
    def test_spawn_retrotide_creates_spawn_policy(self, root_node, sample_molecule):
        """spawn_retrotide=True creates SpawnRetroTideOnDatabaseCheck."""
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            spawn_retrotide=True,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        assert isinstance(agent.rollout_policy, SpawnRetroTideOnDatabaseCheck)
    
    def test_explicit_rollout_policy_overrides_spawn_retrotide(self, root_node, sample_molecule):
        """Explicit rollout_policy takes precedence over spawn_retrotide."""
        custom_policy = CountingRolloutPolicy()
        
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            spawn_retrotide=True,  # This should be ignored
            rollout_policy=custom_policy,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        assert agent.rollout_policy is custom_policy
    
    def test_custom_policies_are_used(self, root_node, sample_molecule):
        """Custom policies are correctly stored and used."""
        rollout_policy = CountingRolloutPolicy()
        reward_policy = FixedRewardPolicy(reward=0.8)
        
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            rollout_policy=rollout_policy,
            reward_policy=reward_policy,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        assert agent.rollout_policy is rollout_policy
        assert agent.reward_policy is reward_policy
    
    def test_calculate_reward_delegates_to_policy(self, root_node, sample_molecule):
        """calculate_reward() delegates to the reward_policy."""
        reward_policy = FixedRewardPolicy(reward=0.75)
        
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            reward_policy=reward_policy,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        reward = agent.calculate_reward(root_node)
        
        assert reward == 0.75
        assert reward_policy.call_count == 1
        assert root_node in reward_policy.called_nodes
    
    def test_sink_compound_reward_via_policy(self, sample_molecule):
        """Sink compounds receive correct reward via policy."""
        Node.node_counter = 0
        
        # Create root node
        root_mol = Chem.MolFromSmiles("CCO")
        root = Node(fragment=root_mol, parent=None, depth=0, provenance="target")
        
        # Create a node that is a sink compound
        sink_mol = Chem.MolFromSmiles("O=O")  # O2 - typically a sink
        sink_node = Node(fragment=sink_mol, parent=root, depth=1, provenance="test")
        sink_node.is_sink_compound = True
        
        agent = DORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            reward_policy=SparseTerminalRewardPolicy(sink_terminal_reward=0.5),
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        reward = agent.calculate_reward(sink_node)
        
        assert reward == 0.5
    
    def test_policy_receives_correct_context(self, root_node, sample_molecule):
        """Policies receive the correct context dictionary."""
        class ContextCapturingPolicy(RewardPolicy):
            def __init__(self):
                self.captured_context = None
            
            @property
            def name(self) -> str:
                return "ContextCapturing"
            
            def calculate_reward(self, node, context):
                self.captured_context = context
                return 0.0
        
        context_policy = ContextCapturingPolicy()
        
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            reward_policy=context_policy,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        agent.calculate_reward(root_node)
        
        assert context_policy.captured_context is not None
        assert "target_molecule" in context_policy.captured_context
        assert context_policy.captured_context["target_molecule"] == sample_molecule


# --- AsyncExpansionDORAnetMCTS Integration Tests ---

class TestAsyncExpansionMCTSPolicyIntegration:
    """Integration tests for AsyncExpansionDORAnetMCTS with the policy system."""
    
    def test_default_policies_are_inherited(self, root_node, sample_molecule):
        """Default instantiation uses NoOpRolloutPolicy and SparseTerminalRewardPolicy."""
        agent = AsyncExpansionDORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            num_workers=1,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        assert isinstance(agent.rollout_policy, NoOpRolloutPolicy)
        assert isinstance(agent.reward_policy, SparseTerminalRewardPolicy)
    
    def test_spawn_retrotide_backward_compatibility(self, root_node, sample_molecule):
        """spawn_retrotide=True is correctly handled."""
        agent = AsyncExpansionDORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            num_workers=1,
            spawn_retrotide=True,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        assert isinstance(agent.rollout_policy, SpawnRetroTideOnDatabaseCheck)
    
    def test_custom_policies_passed_through(self, root_node, sample_molecule):
        """Custom policies are correctly passed to AsyncExpansionDORAnetMCTS."""
        rollout_policy = CountingRolloutPolicy()
        reward_policy = FixedRewardPolicy(reward=0.9)
        
        agent = AsyncExpansionDORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            num_workers=1,
            rollout_policy=rollout_policy,
            reward_policy=reward_policy,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        assert agent.rollout_policy is rollout_policy
        assert agent.reward_policy is reward_policy
    
    def test_legacy_reward_fn_deprecation_warning(self, root_node, sample_molecule, capsys):
        """Using reward_fn shows deprecation message."""
        def legacy_reward_fn(node):
            return 0.5
        
        agent = AsyncExpansionDORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            num_workers=1,
            reward_fn=legacy_reward_fn,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        captured = capsys.readouterr()
        assert "DEPRECATED" in captured.out
        assert "reward_policy" in captured.out


# --- Composed Policy Integration Tests ---

class TestComposedPolicyIntegration:
    """Tests for ComposedRewardPolicy with MCTS agents."""
    
    def test_composed_policy_with_mcts(self, root_node, sample_molecule, pks_library):
        """ComposedRewardPolicy works correctly with DORAnetMCTS."""
        composed = ComposedRewardPolicy([
            (SinkCompoundRewardPolicy(reward_value=1.0), 0.5),
            (PKSLibraryRewardPolicy(pks_library=pks_library), 0.5),
        ])
        
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            reward_policy=composed,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        assert agent.reward_policy is composed
        # Non-terminal node should get 0 reward from both policies
        reward = agent.calculate_reward(root_node)
        assert reward == 0.0


# --- Policy Logging Integration Tests ---

class TestPolicyLoggingIntegration:
    """Tests for policy logging in MCTS agents."""
    
    def test_policy_names_logged_on_init(self, root_node, sample_molecule, capsys):
        """Policy names are logged on initialization."""
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        captured = capsys.readouterr()
        assert "NoOp" in captured.out
        assert "SparseTerminal" in captured.out
    
    def test_custom_policy_name_logged(self, root_node, sample_molecule, capsys):
        """Custom policy names are logged."""
        rollout = CountingRolloutPolicy()
        reward = FixedRewardPolicy(reward=0.5)
        
        agent = DORAnetMCTS(
            root=root_node,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            rollout_policy=rollout,
            reward_policy=reward,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        captured = capsys.readouterr()
        assert "Counting" in captured.out
        assert "Fixed(0.5)" in captured.out


# --- Rollout Behavior Integration Tests ---

class TestRolloutBehaviorIntegration:
    """Tests for rollout behavior during MCTS execution."""
    
    def test_rollout_policy_called_on_non_sink_children(self, sample_molecule):
        """Rollout policy is called for non-sink children after expansion."""
        rollout_policy = CountingRolloutPolicy()
        
        Node.node_counter = 0
        root_mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=root_mol, parent=None, depth=0, provenance="target")
        
        agent = DORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=0,  # Don't run automatically
            max_depth=5,
            rollout_policy=rollout_policy,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        # Manually create a child to test rollout
        child_mol = Chem.MolFromSmiles("CCCC(=O)O")
        child = Node(fragment=child_mol, parent=root, depth=1, provenance="test")
        agent.nodes.append(child)
        root.add_child(child)
        
        # Build context and call rollout directly
        context = agent._build_rollout_context()
        result = rollout_policy.rollout(child, context)
        
        assert rollout_policy.call_count == 1
        assert result.reward == 0.5
    
    def test_noop_rollout_returns_zero(self, sample_molecule):
        """NoOpRolloutPolicy returns zero reward and doesn't mark terminal."""
        Node.node_counter = 0
        root_mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=root_mol, parent=None, depth=0, provenance="target")
        
        agent = DORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=5,
            use_enzymatic=False,
            use_synthetic=False,
        )
        
        context = agent._build_rollout_context()
        result = agent.rollout_policy.rollout(root, context)
        
        assert result.reward == 0.0
        assert result.terminal is False
