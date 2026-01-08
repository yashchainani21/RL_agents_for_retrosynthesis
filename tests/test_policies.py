"""
Tests for DORAnet MCTS policies module.

Tests cover:
- Base classes and dataclasses
- Reward policies (Sink, PKS, Sparse, Composed)
- Rollout policies (NoOp, SpawnRetroTideOnDatabaseCheck, SAScore_and_SpawnRetroTideOnDatabaseCheck)
"""

import pytest
from unittest.mock import MagicMock, patch
from rdkit import Chem

from DORAnet_agent.policies import (
    # Base classes
    RolloutPolicy,
    RewardPolicy,
    RolloutResult,
    # Rollout policies
    NoOpRolloutPolicy,
    SpawnRetroTideOnDatabaseCheck,
    SAScore_and_SpawnRetroTideOnDatabaseCheck,
    # Reward policies
    SparseTerminalRewardPolicy,
    SinkCompoundRewardPolicy,
    PKSLibraryRewardPolicy,
    ComposedRewardPolicy,
)
from DORAnet_agent.node import Node


class TestRolloutResult:
    """Tests for RolloutResult dataclass."""

    def test_basic_result(self):
        result = RolloutResult(reward=0.5)
        assert result.reward == 0.5
        assert result.terminal is False
        assert result.terminal_type is None
        assert result.metadata == {}

    def test_terminal_result(self):
        result = RolloutResult(
            reward=1.0,
            terminal=True,
            terminal_type="pks_terminal",
            metadata={"agent": "test"},
        )
        assert result.reward == 1.0
        assert result.terminal is True
        assert result.terminal_type == "pks_terminal"
        assert result.metadata == {"agent": "test"}

    def test_repr(self):
        result = RolloutResult(reward=0.5)
        assert "0.500" in repr(result)

        terminal_result = RolloutResult(reward=1.0, terminal=True, terminal_type="pks_terminal")
        assert "pks_terminal" in repr(terminal_result)


class TestNoOpRolloutPolicy:
    """Tests for NoOpRolloutPolicy."""

    def test_rollout_returns_zero_reward(self):
        policy = NoOpRolloutPolicy()
        node = MagicMock()
        context = {}

        result = policy.rollout(node, context)

        assert result.reward == 0.0
        assert result.terminal is False
        assert result.metadata == {}

    def test_name(self):
        policy = NoOpRolloutPolicy()
        assert policy.name == "NoOp"


class TestSinkCompoundRewardPolicy:
    """Tests for SinkCompoundRewardPolicy."""

    def test_sink_compound_returns_reward(self):
        policy = SinkCompoundRewardPolicy(reward_value=0.8)
        node = MagicMock()
        node.is_sink_compound = True

        reward = policy.calculate_reward(node, {})
        assert reward == 0.8

    def test_non_sink_returns_zero(self):
        policy = SinkCompoundRewardPolicy(reward_value=0.8)
        node = MagicMock()
        node.is_sink_compound = False

        reward = policy.calculate_reward(node, {})
        assert reward == 0.0

    def test_name(self):
        policy = SinkCompoundRewardPolicy(reward_value=0.8)
        assert "0.8" in policy.name


class TestPKSLibraryRewardPolicy:
    """Tests for PKSLibraryRewardPolicy."""

    def test_pks_match_returns_reward(self):
        # Create a real node with a molecule
        mol = Chem.MolFromSmiles("CCO")  # ethanol
        node = Node(fragment=mol, parent=None)

        # PKS library contains canonical ethanol
        pks_library = {"CCO"}
        policy = PKSLibraryRewardPolicy(pks_library=pks_library, reward_value=1.0)

        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_non_match_returns_zero(self):
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)

        pks_library = {"CCC"}  # propane, not ethanol
        policy = PKSLibraryRewardPolicy(pks_library=pks_library, reward_value=1.0)

        reward = policy.calculate_reward(node, {})
        assert reward == 0.0

    def test_empty_library_returns_zero(self):
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)

        policy = PKSLibraryRewardPolicy(pks_library=set(), reward_value=1.0)

        reward = policy.calculate_reward(node, {})
        assert reward == 0.0

    def test_uses_context_library_if_not_provided(self):
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)

        # Policy has no library, but context does
        policy = PKSLibraryRewardPolicy(pks_library=None, reward_value=1.0)
        context = {"pks_library": {"CCO"}}

        reward = policy.calculate_reward(node, context)
        assert reward == 1.0


class TestSparseTerminalRewardPolicy:
    """Tests for SparseTerminalRewardPolicy."""

    def test_sink_compound_returns_sink_reward(self):
        policy = SparseTerminalRewardPolicy(sink_terminal_reward=0.5)
        node = MagicMock()
        node.is_sink_compound = True
        node.is_pks_terminal = False
        node.smiles = "CCO"

        reward = policy.calculate_reward(node, {})
        assert reward == 0.5

    def test_pks_terminal_returns_one(self):
        policy = SparseTerminalRewardPolicy(sink_terminal_reward=0.5)
        node = MagicMock()
        node.is_sink_compound = False
        node.is_pks_terminal = True
        node.smiles = "CCO"

        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_pks_library_match_returns_one(self):
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False

        policy = SparseTerminalRewardPolicy(pks_library={"CCO"})
        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_non_terminal_returns_zero(self):
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False

        policy = SparseTerminalRewardPolicy(pks_library={"CCC"})  # no match
        reward = policy.calculate_reward(node, {})
        assert reward == 0.0


class TestComposedRewardPolicy:
    """Tests for ComposedRewardPolicy."""

    def test_weighted_sum(self):
        sink_policy = SinkCompoundRewardPolicy(reward_value=1.0)
        pks_policy = PKSLibraryRewardPolicy(pks_library={"CCO"}, reward_value=1.0)

        composed = ComposedRewardPolicy([
            (sink_policy, 0.3),
            (pks_policy, 0.7),
        ])

        # Node is both sink and PKS match
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = True

        reward = composed.calculate_reward(node, {})
        assert reward == pytest.approx(1.0)  # 0.3 * 1.0 + 0.7 * 1.0

    def test_partial_match(self):
        sink_policy = SinkCompoundRewardPolicy(reward_value=1.0)
        pks_policy = PKSLibraryRewardPolicy(pks_library={"CCC"}, reward_value=1.0)

        composed = ComposedRewardPolicy([
            (sink_policy, 0.5),
            (pks_policy, 0.5),
        ])

        # Node is sink but not PKS match
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = True

        reward = composed.calculate_reward(node, {})
        assert reward == pytest.approx(0.5)  # 0.5 * 1.0 + 0.5 * 0.0

    def test_empty_policies_raises(self):
        with pytest.raises(ValueError):
            ComposedRewardPolicy([])

    def test_name(self):
        sink_policy = SinkCompoundRewardPolicy(reward_value=1.0)
        composed = ComposedRewardPolicy([(sink_policy, 0.5)])
        assert "Composed" in composed.name
        assert "0.5" in composed.name


class TestSpawnRetroTideOnDatabaseCheck:
    """Tests for SpawnRetroTideOnDatabaseCheck rollout policy."""

    def test_non_pks_match_returns_failure_reward(self):
        policy = SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCC"},  # propane
            success_reward=1.0,
            failure_reward=0.0,
        )

        # Node is ethanol, not in PKS library
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        context = {"pks_library": {"CCC"}, "target_molecule": mol}

        result = policy.rollout(node, context)

        assert result.reward == 0.0
        assert result.terminal is False

    def test_already_attempted_returns_failure_reward(self):
        policy = SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCO"},
            success_reward=1.0,
            failure_reward=0.0,
        )

        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.retrotide_attempted = True  # Already attempted
        context = {"pks_library": {"CCO"}, "target_molecule": mol}

        result = policy.rollout(node, context)

        assert result.reward == 0.0
        assert result.terminal is False

    def test_pks_match_without_target_molecule_returns_failure(self):
        policy = SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCO"},
            success_reward=1.0,
            failure_reward=0.0,
        )

        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        context = {"pks_library": {"CCO"}}  # No target_molecule

        result = policy.rollout(node, context)

        assert result.reward == 0.0
        assert result.terminal is False

    def test_name(self):
        policy = SpawnRetroTideOnDatabaseCheck(success_reward=0.9)
        assert "0.9" in policy.name
        assert "SpawnRetroTide" in policy.name

    def test_repr_shows_availability(self):
        policy = SpawnRetroTideOnDatabaseCheck()
        repr_str = repr(policy)
        assert "retrotide_available" in repr_str


class TestPolicyInheritance:
    """Test that policies properly inherit from base classes."""

    def test_rollout_policies_inherit_from_base(self):
        assert issubclass(NoOpRolloutPolicy, RolloutPolicy)
        assert issubclass(SpawnRetroTideOnDatabaseCheck, RolloutPolicy)
        assert issubclass(SAScore_and_SpawnRetroTideOnDatabaseCheck, RolloutPolicy)

    def test_reward_policies_inherit_from_base(self):
        assert issubclass(SinkCompoundRewardPolicy, RewardPolicy)
        assert issubclass(PKSLibraryRewardPolicy, RewardPolicy)
        assert issubclass(SparseTerminalRewardPolicy, RewardPolicy)
        assert issubclass(ComposedRewardPolicy, RewardPolicy)


class TestSAScoreHelpers:
    """Tests for SA Score helper functions."""
    
    def test_sa_score_to_reward_formula(self):
        """Test the SA Score to reward conversion formula."""
        from DORAnet_agent.policies.rollout import _sa_score_to_reward
        
        # Easy synthesis (SA Score = 1) -> high reward
        assert _sa_score_to_reward(1.0) == pytest.approx(0.9)
        
        # Moderate synthesis (SA Score = 5) -> medium reward
        assert _sa_score_to_reward(5.0) == pytest.approx(0.5)
        
        # Hard synthesis (SA Score = 10) -> low reward
        assert _sa_score_to_reward(10.0) == pytest.approx(0.0)
        
        # Typical range
        assert _sa_score_to_reward(3.0) == pytest.approx(0.7)
        assert _sa_score_to_reward(8.0) == pytest.approx(0.2)
    
    def test_sa_score_to_reward_with_max(self):
        """Test SA Score reward capping with sa_max_reward."""
        from DORAnet_agent.policies.rollout import _sa_score_to_reward
        
        # With default max_reward=1.0, no cap effect
        assert _sa_score_to_reward(1.0, max_reward=1.0) == pytest.approx(0.9)
        
        # With max_reward=0.5, cap takes effect
        assert _sa_score_to_reward(1.0, max_reward=0.5) == pytest.approx(0.5)
        assert _sa_score_to_reward(5.0, max_reward=0.5) == pytest.approx(0.5)
        
        # With max_reward=0.3
        assert _sa_score_to_reward(1.0, max_reward=0.3) == pytest.approx(0.3)
        assert _sa_score_to_reward(8.0, max_reward=0.3) == pytest.approx(0.2)  # below cap
    
    def test_calculate_sa_score_for_simple_molecules(self):
        """Test SA Score calculation for simple molecules."""
        from DORAnet_agent.policies.rollout import _calculate_sa_score, _SA_SCORE_AVAILABLE
        
        if not _SA_SCORE_AVAILABLE:
            pytest.skip("SA Score not available in this environment")
        
        # Simple molecules should have low SA scores (easy to synthesize)
        mol = Chem.MolFromSmiles("CCCCCCCCC(=O)O")  # nonanoic acid
        sa_score = _calculate_sa_score(mol)
        assert sa_score is not None
        assert 1.0 <= sa_score <= 10.0
        assert sa_score < 3.0  # Should be easy to synthesize
        
        # Ethanol - very simple
        mol = Chem.MolFromSmiles("CCO")
        sa_score = _calculate_sa_score(mol)
        assert sa_score is not None
        assert sa_score < 2.5  # Very easy


class TestSAScoreAndSpawnRetroTideOnDatabaseCheck:
    """Tests for SAScore_and_SpawnRetroTideOnDatabaseCheck rollout policy."""

    def test_initialization_defaults(self):
        """Test default initialization parameters."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck()
        
        assert policy.success_reward == 1.0
        assert policy.sa_max_reward == 1.0
        assert policy._pks_library is None
        assert policy._sa_score_available is True  # Assuming RDKit with SA Score

    def test_initialization_custom_params(self):
        """Test custom initialization parameters."""
        pks_lib = {"CCO", "CCC"}
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            pks_library=pks_lib,
            success_reward=0.9,
            sa_max_reward=0.8,
        )
        
        assert policy.success_reward == 0.9
        assert policy.sa_max_reward == 0.8
        assert policy._pks_library == pks_lib

    def test_non_pks_match_returns_sa_reward(self):
        """Test that non-PKS matches return SA Score reward."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCC"},  # propane - not our molecule
            success_reward=1.0,
        )

        # Node is ethanol, not in PKS library
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        context = {"pks_library": {"CCC"}, "target_molecule": mol}

        result = policy.rollout(node, context)

        # Should return SA Score reward, not 0.0
        assert result.reward > 0.0
        assert result.reward < 1.0  # SA reward range is 0.0-0.9
        assert result.terminal is False
        assert result.metadata.get("pks_match") is False
        assert result.metadata.get("sa_score") is not None
        assert result.metadata.get("sa_reward") is not None

    def test_already_attempted_returns_sa_reward(self):
        """Test that already attempted nodes return SA Score reward."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCO"},
            success_reward=1.0,
        )

        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.retrotide_attempted = True  # Already attempted
        context = {"pks_library": {"CCO"}, "target_molecule": mol}

        result = policy.rollout(node, context)

        # Should return SA reward, not zero
        assert result.reward > 0.0
        assert result.terminal is False
        assert result.metadata.get("sa_reward") is not None

    def test_pks_match_without_target_molecule_returns_sa_reward(self):
        """Test PKS match without target_molecule returns SA reward."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCO"},
            success_reward=1.0,
        )

        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        context = {"pks_library": {"CCO"}}  # No target_molecule

        result = policy.rollout(node, context)

        # Should return SA reward since RetroTide can't run
        assert result.reward > 0.0
        assert result.terminal is False
        assert result.metadata.get("pks_match") is True
        assert result.metadata.get("retrotide_skipped") == "no_target_molecule"

    def test_metadata_contains_sa_info(self):
        """Test that metadata always contains SA Score information."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCC"},
        )

        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        context = {"pks_library": {"CCC"}}

        result = policy.rollout(node, context)

        # Metadata should contain SA info
        assert "sa_score" in result.metadata
        assert "sa_reward" in result.metadata
        # SA score should be reasonable for ethanol
        assert result.metadata["sa_score"] is None or 1.0 <= result.metadata["sa_score"] <= 10.0

    def test_sa_max_reward_caps_sa_reward(self):
        """Test that sa_max_reward parameter caps SA rewards."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCC"},  # Not our molecule
            sa_max_reward=0.5,
        )

        mol = Chem.MolFromSmiles("CCO")  # Easy molecule, would get ~0.8 SA reward
        node = Node(fragment=mol, parent=None)
        context = {"pks_library": {"CCC"}}

        result = policy.rollout(node, context)

        # Reward should be capped at 0.5
        assert result.reward <= 0.5
        assert result.terminal is False

    def test_simple_molecule_gets_high_sa_reward(self):
        """Test that simple molecules get high SA rewards."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            pks_library={"IMPOSSIBLE_SMILES"},  # No match
        )

        # Simple molecule - should have low SA score (easy synthesis)
        mol = Chem.MolFromSmiles("CCCCCCCCC(=O)O")  # nonanoic acid
        node = Node(fragment=mol, parent=None)
        context = {}

        result = policy.rollout(node, context)

        # Simple molecules should get high SA rewards (>0.7)
        assert result.reward > 0.7

    def test_complex_molecule_gets_lower_sa_reward(self):
        """Test that complex molecules get lower SA rewards."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            pks_library={"IMPOSSIBLE_SMILES"},  # No match
        )

        # More complex molecule
        mol = Chem.MolFromSmiles("CC(C)CC(=O)NC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2")
        if mol is None:
            pytest.skip("Complex molecule could not be parsed")
            
        node = Node(fragment=mol, parent=None)
        context = {}

        result = policy.rollout(node, context)

        # Complex molecules should get lower SA rewards
        # But still a valid reward (not 0)
        assert 0.0 < result.reward < 0.9

    def test_name_includes_parameters(self):
        """Test that policy name includes key parameters."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            success_reward=0.9,
            sa_max_reward=0.7,
        )
        
        name = policy.name
        assert "SAScore" in name
        assert "0.9" in name
        assert "0.7" in name

    def test_repr_shows_availability(self):
        """Test that repr shows SA Score and RetroTide availability."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck()
        repr_str = repr(policy)
        
        assert "sa_score_available" in repr_str
        assert "retrotide_available" in repr_str
        assert "success_reward" in repr_str
        assert "sa_max_reward" in repr_str

    def test_get_sa_reward_fallback_for_none_molecule(self):
        """Test SA reward fallback when molecule is None."""
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck()
        
        # Node with no fragment and invalid smiles
        node = MagicMock()
        node.fragment = None
        node.smiles = None
        
        # Should return 0.0 as fallback
        reward = policy._get_sa_reward(node)
        assert reward == 0.0

    def test_dense_vs_sparse_reward_comparison(self):
        """Compare SAScore policy (dense) vs SpawnRetroTide (sparse) for non-PKS nodes."""
        sparse_policy = SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCC"},
            failure_reward=0.0,
        )
        dense_policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
            pks_library={"CCC"},
        )

        mol = Chem.MolFromSmiles("CCO")  # Not in PKS library
        node = Node(fragment=mol, parent=None)
        context = {}

        sparse_result = sparse_policy.rollout(node, context)
        dense_result = dense_policy.rollout(node, context)

        # Sparse should return 0.0 (failure_reward)
        assert sparse_result.reward == 0.0
        
        # Dense should return SA reward > 0
        assert dense_result.reward > 0.0
        assert dense_result.metadata.get("sa_score") is not None

    def test_inherits_from_rollout_policy(self):
        """Test that SAScore policy inherits from RolloutPolicy."""
        assert issubclass(SAScore_and_SpawnRetroTideOnDatabaseCheck, RolloutPolicy)
        
        policy = SAScore_and_SpawnRetroTideOnDatabaseCheck()
        assert isinstance(policy, RolloutPolicy)
