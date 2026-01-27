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
    PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
    # Reward policies
    SparseTerminalRewardPolicy,
    SinkCompoundRewardPolicy,
    PKSLibraryRewardPolicy,
    ComposedRewardPolicy,
    PKSSimilarityRewardPolicy,
    SAScore_and_TerminalRewardPolicy,
    # Thermodynamic scaling
    ThermodynamicScaledRewardPolicy,
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


class TestSAScore_and_TerminalRewardPolicy:
    """Tests for SAScore_and_TerminalRewardPolicy."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        policy = SAScore_and_TerminalRewardPolicy()
        assert policy.sink_terminal_reward == 1.0
        assert policy.pks_terminal_reward == 1.0
        assert policy.sa_max_reward == 1.0
        assert policy.sa_fallback_reward == 0.0
        assert "SAScore" in policy.name
        assert "Terminal" in policy.name

    def test_sink_compound_gets_terminal_reward(self):
        """Sink compounds should get full terminal reward."""
        policy = SAScore_and_TerminalRewardPolicy(sink_terminal_reward=1.0)
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = True
        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_pks_terminal_gets_terminal_reward(self):
        """PKS terminals should get full terminal reward."""
        policy = SAScore_and_TerminalRewardPolicy(pks_terminal_reward=1.0)
        mol = Chem.MolFromSmiles("CCCCCC(=O)O")
        node = Node(fragment=mol, parent=None)
        node.is_pks_terminal = True
        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_non_terminal_gets_sa_score(self):
        """Non-terminal compounds should get SA score reward."""
        policy = SAScore_and_TerminalRewardPolicy()
        mol = Chem.MolFromSmiles("CCO")  # Ethanol - easy to synthesize
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False
        reward = policy.calculate_reward(node, {})
        # SA score for ethanol is ~1.5, so reward should be ~0.85
        assert 0.5 < reward < 1.0

    def test_complex_non_terminal_lower_reward(self):
        """Complex non-terminals should get lower SA score reward."""
        policy = SAScore_and_TerminalRewardPolicy()
        # Complex natural product-like structure
        mol = Chem.MolFromSmiles("CC1=C2C(=O)C3=C(C=CC=C3O)C(=O)C2=CC=C1")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False
        reward = policy.calculate_reward(node, {})
        assert 0.0 <= reward <= 1.0

    def test_priority_sink_over_pks(self):
        """Sink compound takes priority over PKS terminal."""
        policy = SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=0.8,
            pks_terminal_reward=0.9,
        )
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = True
        node.is_pks_terminal = True  # Both flags set
        reward = policy.calculate_reward(node, {})
        assert reward == 0.8  # Sink reward takes priority

    def test_pks_library_match_fallback(self):
        """PKS library match should give terminal reward."""
        pks_library = {"CCO"}  # Ethanol in PKS library
        policy = SAScore_and_TerminalRewardPolicy(
            pks_terminal_reward=1.0,
            pks_library=pks_library,
        )
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        # node.smiles is computed from fragment, so it should be "CCO"
        node.is_sink_compound = False
        node.is_pks_terminal = False  # Not marked, but in library
        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_sa_max_reward_cap(self):
        """SA score should be capped at sa_max_reward."""
        policy = SAScore_and_TerminalRewardPolicy(sa_max_reward=0.5)
        mol = Chem.MolFromSmiles("C")  # Methane - trivially easy
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False
        reward = policy.calculate_reward(node, {})
        assert reward <= 0.5

    def test_inherits_from_reward_policy(self):
        """Test proper inheritance."""
        policy = SAScore_and_TerminalRewardPolicy()
        assert isinstance(policy, RewardPolicy)

    def test_thermodynamic_scaling_wrapper(self):
        """Test wrapping with ThermodynamicScaledRewardPolicy."""
        base = SAScore_and_TerminalRewardPolicy()
        scaled = ThermodynamicScaledRewardPolicy(
            base_policy=base,
            feasibility_weight=0.5,
        )
        assert "ThermodynamicScaled" in scaled.name
        assert "SAScore" in scaled.name


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


class TestPKSSimScoreAndSpawnRetroTideOnDatabaseCheck:
    """Tests for PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck rollout policy."""

    def test_initialization_defaults(self):
        """Test default initialization parameters including new ones."""
        # Use a mock to avoid loading actual PKS building blocks
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck()

        assert policy.success_reward == 1.0
        assert policy.similarity_threshold == 0.9
        assert policy.similarity_method == "tanimoto"
        assert policy.fingerprint_radius == 2
        assert policy.fingerprint_bits == 2048
        # New parameters
        assert policy.retrotide_spawn_threshold == 0.9
        assert policy.similarity_reward_exponent == 2.0

    def test_initialization_custom_new_params(self):
        """Test custom initialization of new parameters."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                retrotide_spawn_threshold=0.85,
                similarity_reward_exponent=3.0,
            )

        assert policy.retrotide_spawn_threshold == 0.85
        assert policy.similarity_reward_exponent == 3.0

    def test_initialization_exact_match_only_behavior(self):
        """Test initialization for exact-match-only behavior (old behavior)."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                retrotide_spawn_threshold=1.0,  # Only exact matches
                similarity_reward_exponent=1.0,  # Linear rewards
            )

        assert policy.retrotide_spawn_threshold == 1.0
        assert policy.similarity_reward_exponent == 1.0

    def test_exponential_scaling_in_get_pks_similarity_reward(self):
        """Test that _get_pks_similarity_reward applies exponential scaling."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                similarity_reward_exponent=2.0,
            )

        # Mock the _compute_tanimoto_similarity method to return a known value
        with patch.object(policy, '_compute_tanimoto_similarity') as mock_compute:
            mock_compute.return_value = (0.9, {"best_similarity": 0.9, "similarity_method": "tanimoto"})

            node = MagicMock()
            reward, metadata = policy._get_pks_similarity_reward(node)

            # 0.9^2 = 0.81
            assert reward == pytest.approx(0.81)
            assert metadata["raw_similarity"] == 0.9
            assert metadata["similarity_exponent"] == 2.0

    def test_exponential_scaling_with_different_exponents(self):
        """Test exponential scaling with various exponent values."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            # Test with exponent=3.0
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                similarity_reward_exponent=3.0,
            )

        with patch.object(policy, '_compute_tanimoto_similarity') as mock_compute:
            mock_compute.return_value = (0.5, {"best_similarity": 0.5, "similarity_method": "tanimoto"})

            node = MagicMock()
            reward, metadata = policy._get_pks_similarity_reward(node)

            # 0.5^3 = 0.125
            assert reward == pytest.approx(0.125)

    def test_no_scaling_with_exponent_one(self):
        """Test that exponent=1.0 means no scaling."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                similarity_reward_exponent=1.0,
            )

        with patch.object(policy, '_compute_tanimoto_similarity') as mock_compute:
            mock_compute.return_value = (0.7, {"best_similarity": 0.7, "similarity_method": "tanimoto"})

            node = MagicMock()
            reward, metadata = policy._get_pks_similarity_reward(node)

            # No scaling
            assert reward == pytest.approx(0.7)
            # Should not add raw_similarity or similarity_exponent when exponent=1.0
            assert "raw_similarity" not in metadata

    def test_high_similarity_triggers_retrotide_spawn(self):
        """Test that high similarity (>= threshold) triggers RetroTide spawn logic."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_library=set(),  # Empty library = no exact match
                retrotide_spawn_threshold=0.9,
                similarity_reward_exponent=1.0,
            )

        # Mock similarity to return 0.95 (above threshold)
        with patch.object(policy, '_get_pks_similarity_reward') as mock_sim:
            mock_sim.return_value = (0.95, {"best_similarity": 0.95, "similarity_method": "tanimoto"})

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)
            context = {"pks_library": set()}  # Empty library

            # Without RetroTide available, it should still recognize high similarity
            result = policy.rollout(node, context)

            # The metadata should indicate high_similarity_match (even if RetroTide fails/unavailable)
            assert result.metadata.get("high_similarity_match") is True or \
                   result.metadata.get("retrotide_available") is False

    def test_low_similarity_does_not_trigger_retrotide(self):
        """Test that low similarity (< threshold) does not trigger RetroTide."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_library=set(),  # Empty library = no exact match
                retrotide_spawn_threshold=0.9,
                similarity_reward_exponent=2.0,
            )

        # Mock similarity to return 0.7 (below threshold)
        with patch.object(policy, '_get_pks_similarity_reward') as mock_sim:
            mock_sim.return_value = (0.49, {  # 0.7^2 = 0.49 (scaled)
                "best_similarity": 0.7,
                "raw_similarity": 0.7,
                "similarity_exponent": 2.0,
                "similarity_method": "tanimoto"
            })

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)
            context = {"pks_library": set()}

            result = policy.rollout(node, context)

            # Should return the scaled similarity reward without attempting RetroTide
            assert result.reward == pytest.approx(0.49)
            assert result.terminal is False
            assert result.metadata.get("pks_match") is False
            assert result.metadata.get("high_similarity_match") is False

    def test_metadata_includes_new_fields(self):
        """Test that metadata includes new tracking fields."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_library=set(),
                retrotide_spawn_threshold=0.9,
                similarity_reward_exponent=2.0,
            )

        with patch.object(policy, '_get_pks_similarity_reward') as mock_sim:
            mock_sim.return_value = (0.36, {
                "best_similarity": 0.6,
                "raw_similarity": 0.6,
                "similarity_exponent": 2.0,
                "similarity_method": "tanimoto"
            })

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)
            context = {}

            result = policy.rollout(node, context)

            # Check for new metadata fields
            assert "retrotide_spawn_threshold" in result.metadata
            assert result.metadata["retrotide_spawn_threshold"] == 0.9
            assert "raw_similarity" in result.metadata
            assert result.metadata["raw_similarity"] == 0.6

    def test_name_includes_new_params(self):
        """Test that policy name includes new parameters."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                retrotide_spawn_threshold=0.85,
                similarity_reward_exponent=2.5,
            )

        name = policy.name
        assert "spawn_thresh=0.85" in name
        assert "exp=2.5" in name

    def test_repr_includes_new_params(self):
        """Test that repr includes new parameters."""
        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                retrotide_spawn_threshold=0.85,
                similarity_reward_exponent=2.5,
            )

        repr_str = repr(policy)
        assert "retrotide_spawn_threshold=0.85" in repr_str
        assert "similarity_reward_exponent=2.5" in repr_str

    def test_inherits_from_rollout_policy(self):
        """Test that PKS Sim Score policy inherits from RolloutPolicy."""
        assert issubclass(PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck, RolloutPolicy)

        with patch.object(
            PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
            '_load_pks_building_blocks'
        ):
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck()
        assert isinstance(policy, RolloutPolicy)


class TestPKSSimilarityRewardPolicy:
    """Tests for PKSSimilarityRewardPolicy."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy()

        assert policy.similarity_threshold == 0.95
        assert policy.fingerprint_radius == 2
        assert policy.fingerprint_bits == 2048
        assert policy.similarity_exponent == 2.0

    def test_initialization_custom_params(self):
        """Test custom initialization parameters."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(
                similarity_threshold=0.9,
                fingerprint_radius=3,
                fingerprint_bits=1024,
                similarity_exponent=3.0,
            )

        assert policy.similarity_threshold == 0.9
        assert policy.fingerprint_radius == 3
        assert policy.fingerprint_bits == 1024
        assert policy.similarity_exponent == 3.0

    def test_pks_exact_match_returns_one(self):
        """PKS library exact matches should return ~1.0 (high similarity)."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(similarity_exponent=1.0)

        # Mock similarity to return 1.0 (exact match)
        with patch.object(policy, '_compute_pks_similarity') as mock_sim:
            mock_sim.return_value = (1.0, {"best_similarity": 1.0})

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)

            reward = policy.calculate_reward(node, {})
            assert reward == 1.0

    def test_high_similarity_returns_high_reward(self):
        """High PKS similarity should return high reward."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(similarity_exponent=1.0)

        # Mock similarity to return 0.9
        with patch.object(policy, '_compute_pks_similarity') as mock_sim:
            mock_sim.return_value = (0.9, {"best_similarity": 0.9})

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)

            reward = policy.calculate_reward(node, {})
            assert reward == pytest.approx(0.9)

    def test_low_similarity_returns_low_reward(self):
        """Low PKS similarity should return low reward."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(similarity_exponent=1.0)

        # Mock similarity to return 0.2
        with patch.object(policy, '_compute_pks_similarity') as mock_sim:
            mock_sim.return_value = (0.2, {"best_similarity": 0.2})

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)

            reward = policy.calculate_reward(node, {})
            assert reward == pytest.approx(0.2)

    def test_exponential_scaling(self):
        """Test that similarity^exponent is applied correctly."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(similarity_exponent=2.0)

        # Mock similarity to return 0.8
        with patch.object(policy, '_compute_pks_similarity') as mock_sim:
            mock_sim.return_value = (0.8, {"best_similarity": 0.8})

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)

            reward = policy.calculate_reward(node, {})
            # 0.8^2 = 0.64
            assert reward == pytest.approx(0.64)

    def test_exponential_scaling_various_exponents(self):
        """Test exponential scaling with different exponent values."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            # Test with exponent=3.0
            policy = PKSSimilarityRewardPolicy(similarity_exponent=3.0)

        with patch.object(policy, '_compute_pks_similarity') as mock_sim:
            mock_sim.return_value = (0.5, {"best_similarity": 0.5})

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)

            reward = policy.calculate_reward(node, {})
            # 0.5^3 = 0.125
            assert reward == pytest.approx(0.125)

    def test_sink_compound_gets_similarity_reward(self):
        """Sink compounds should get PKS similarity, not 1.0."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(similarity_exponent=2.0)

        # Mock similarity to return 0.7 for a "sink compound"
        with patch.object(policy, '_compute_pks_similarity') as mock_sim:
            mock_sim.return_value = (0.7, {"best_similarity": 0.7})

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)
            node.is_sink_compound = True  # Mark as sink compound

            reward = policy.calculate_reward(node, {})
            # Even though it's a sink compound, it gets PKS similarity reward
            # 0.7^2 = 0.49
            assert reward == pytest.approx(0.49)
            assert reward != 1.0  # NOT the flat 1.0 that SparseTerminalRewardPolicy would give

    def test_non_sink_gets_similarity_reward(self):
        """Non-terminal nodes should get PKS similarity reward."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(similarity_exponent=2.0)

        # Mock similarity to return 0.6 for a non-terminal node
        with patch.object(policy, '_compute_pks_similarity') as mock_sim:
            mock_sim.return_value = (0.6, {"best_similarity": 0.6})

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)
            node.is_sink_compound = False
            node.is_pks_terminal = False

            reward = policy.calculate_reward(node, {})
            # Non-terminal gets similarity reward, NOT 0.0
            # 0.6^2 = 0.36
            assert reward == pytest.approx(0.36)
            assert reward > 0.0

    def test_inherits_from_reward_policy(self):
        """Test proper inheritance."""
        assert issubclass(PKSSimilarityRewardPolicy, RewardPolicy)

        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy()
        assert isinstance(policy, RewardPolicy)

    def test_name_includes_exponent(self):
        """Test that policy name includes exponent parameter."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(similarity_exponent=2.5)

        name = policy.name
        assert "PKSSimilarity" in name
        assert "exp=2.5" in name

    def test_repr_includes_params(self):
        """Test that repr includes key parameters."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(
                similarity_exponent=2.5,
                similarity_threshold=0.9,
                fingerprint_radius=3,
                fingerprint_bits=1024,
            )

        repr_str = repr(policy)
        assert "exponent=2.5" in repr_str
        assert "threshold=0.9" in repr_str
        assert "radius=3" in repr_str
        assert "bits=1024" in repr_str

    def test_no_scaling_with_exponent_one(self):
        """Test that exponent=1.0 means no scaling."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy(similarity_exponent=1.0)

        with patch.object(policy, '_compute_pks_similarity') as mock_sim:
            mock_sim.return_value = (0.7, {"best_similarity": 0.7})

            mol = Chem.MolFromSmiles("CCO")
            node = Node(fragment=mol, parent=None)

            reward = policy.calculate_reward(node, {})
            # No scaling with exponent=1.0
            assert reward == pytest.approx(0.7)

    def test_comparison_with_sparse_terminal_policy(self):
        """Compare PKSSimilarityRewardPolicy vs SparseTerminalRewardPolicy behavior."""
        # Setup PKSSimilarityRewardPolicy
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            pks_sim_policy = PKSSimilarityRewardPolicy(similarity_exponent=2.0)

        # Setup SparseTerminalRewardPolicy
        sparse_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)

        # Create a sink compound node
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = True

        # Sparse policy gives flat 1.0 for sink compounds
        sparse_reward = sparse_policy.calculate_reward(node, {})
        assert sparse_reward == 1.0

        # PKS similarity policy gives similarity-based reward
        with patch.object(pks_sim_policy, '_compute_pks_similarity') as mock_sim:
            mock_sim.return_value = (0.5, {"best_similarity": 0.5})

            pks_reward = pks_sim_policy.calculate_reward(node, {})
            # 0.5^2 = 0.25, not 1.0
            assert pks_reward == pytest.approx(0.25)
            assert pks_reward != sparse_reward

    def test_none_molecule_returns_zero(self):
        """Test that None molecule returns 0.0 reward."""
        with patch.object(
            PKSSimilarityRewardPolicy,
            '_load_pks_fingerprints'
        ):
            policy = PKSSimilarityRewardPolicy()

        node = MagicMock()
        node.fragment = None
        node.smiles = None

        reward = policy.calculate_reward(node, {})
        assert reward == 0.0


class TestPKSDatabaseContents:
    """
    Tests to verify specific molecules are present in the PKS database.

    These tests use the actual PKS database (expanded_PKS_SMILES_V3.txt)
    to verify that key target molecules are included and receive appropriate
    similarity rewards.
    """

    @pytest.fixture
    def pks_policy(self):
        """Create a PKSSimilarityRewardPolicy with the real database."""
        return PKSSimilarityRewardPolicy(similarity_exponent=1.0)

    def test_4_hydroxybutyric_acid_in_database(self, pks_policy):
        """Test that 4-hydroxybutyric acid is in the PKS database."""
        # 4-hydroxybutyric acid (gamma-hydroxybutyric acid)
        mol = Chem.MolFromSmiles("OCCCC(=O)O")
        node = Node(fragment=mol, parent=None)

        reward = pks_policy.calculate_reward(node, {})

        # Should have perfect similarity (1.0) since it's in the database
        assert reward == pytest.approx(1.0), \
            f"4-hydroxybutyric acid should be in PKS database, got reward {reward}"

    def test_5_ketohexanoic_acid_in_database(self, pks_policy):
        """Test that 5-ketohexanoic acid is in the PKS database."""
        # 5-ketohexanoic acid (5-oxohexanoic acid)
        mol = Chem.MolFromSmiles("CC(=O)CCCC(=O)O")
        node = Node(fragment=mol, parent=None)

        reward = pks_policy.calculate_reward(node, {})

        # Should have perfect similarity (1.0) since it's in the database
        assert reward == pytest.approx(1.0), \
            f"5-ketohexanoic acid should be in PKS database, got reward {reward}"

    def test_tiglic_acid_in_database(self, pks_policy):
        """Test that tiglic acid is in the PKS database."""
        # Tiglic acid (trans-2-methyl-2-butenoic acid)
        mol = Chem.MolFromSmiles("CC=CC(=O)O")
        node = Node(fragment=mol, parent=None)

        reward = pks_policy.calculate_reward(node, {})

        # Should have perfect similarity (1.0) since it's in the database
        assert reward == pytest.approx(1.0), \
            f"Tiglic acid should be in PKS database, got reward {reward}"

    def test_gamma_valerolactone_not_in_database(self, pks_policy):
        """Test that gamma-valerolactone is NOT in the PKS database."""
        # Gamma-valerolactone (5-methyl-2-oxolanone)
        mol = Chem.MolFromSmiles("CC1CCC(=O)O1")
        node = Node(fragment=mol, parent=None)

        reward = pks_policy.calculate_reward(node, {})

        # Should NOT have perfect similarity since it's not in the database
        assert reward < 1.0, \
            f"Gamma-valerolactone should NOT be in PKS database, got reward {reward}"

    def test_hydroxyethyl_furanone_in_database(self, pks_policy):
        """Test that 5-(2-hydroxyethylidene)furan-2(5H)-one is in the PKS database."""
        # This molecule can be synthesized by RetroTide via PKS + cyclization
        mol = Chem.MolFromSmiles("O=C1C=CC(=CCO)O1")
        node = Node(fragment=mol, parent=None)

        reward = pks_policy.calculate_reward(node, {})

        # Should have perfect similarity (1.0) since it's in the database
        assert reward == pytest.approx(1.0), \
            f"Hydroxyethyl furanone should be in PKS database, got reward {reward}"

    def test_styryl_lactone_in_database(self, pks_policy):
        """Test that 6-styryl-5,6-dihydro-2H-pyran-2-one is in the PKS database."""
        # 13-carbon styryl lactone (pyranone with phenyl group)
        mol = Chem.MolFromSmiles("O=C1C=CCC(C=Cc2ccccc2)O1")
        node = Node(fragment=mol, parent=None)

        reward = pks_policy.calculate_reward(node, {})

        # Should have perfect similarity (1.0) since it's in the database
        assert reward == pytest.approx(1.0), \
            f"Styryl lactone should be in PKS database, got reward {reward}"

    def test_database_loads_correct_molecule_count(self, pks_policy):
        """Test that the PKS database loads the expected number of molecules."""
        # The expanded_PKS_SMILES_V3.txt database should have ~962k molecules
        num_molecules = len(pks_policy._pks_building_blocks)

        # Allow some tolerance in case the database is updated
        assert num_molecules > 900000, \
            f"PKS database should have >900k molecules, got {num_molecules}"
        assert num_molecules < 1100000, \
            f"PKS database should have <1.1M molecules, got {num_molecules}"


class TestPreprocessTargetMolecule:
    """Tests for preprocess_target_molecule helper function."""

    def test_removes_stereochemistry(self):
        """Test that stereochemistry is removed from molecules."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Molecule with chiral center (2-butanol)
        mol_with_stereo = Chem.MolFromSmiles("C[C@H](O)CC")
        preprocessed, smiles = preprocess_target_molecule(mol_with_stereo)

        # Resulting SMILES should not contain @ symbols
        assert "@" not in smiles
        # The canonical SMILES for 2-butanol without stereo is "CCC(C)O"
        assert smiles == "CCC(C)O"

    def test_removes_ez_stereochemistry(self):
        """Test that E/Z double bond stereochemistry is removed."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Molecule with E/Z stereochemistry
        mol_with_ez = Chem.MolFromSmiles("C/C=C/C")
        preprocessed, smiles = preprocess_target_molecule(mol_with_ez)

        # Resulting SMILES should not contain / or \ symbols
        assert "/" not in smiles
        assert "\\" not in smiles

    def test_produces_canonical_smiles(self):
        """Test that output SMILES is canonical."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Two equivalent but different SMILES representations
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("OCC")

        _, smiles1 = preprocess_target_molecule(mol1)
        _, smiles2 = preprocess_target_molecule(mol2)

        # Both should produce the same canonical SMILES
        assert smiles1 == smiles2

    def test_sanitizes_molecule(self):
        """Test that molecule is sanitized."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Valid molecule
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        preprocessed, smiles = preprocess_target_molecule(mol)

        # Should produce valid aromatic SMILES
        assert preprocessed is not None
        assert "c" in smiles.lower()  # Should contain aromatic carbons

    def test_raises_on_none_molecule(self):
        """Test that None molecule raises ValueError."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        with pytest.raises(ValueError, match="Cannot preprocess None"):
            preprocess_target_molecule(None)

    def test_returns_valid_mol_object(self):
        """Test that returned molecule is a valid RDKit Mol."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        mol = Chem.MolFromSmiles("CCO")
        preprocessed, smiles = preprocess_target_molecule(mol)

        # Should be a valid Mol object
        assert preprocessed is not None
        assert isinstance(preprocessed, Chem.Mol)
        assert preprocessed.GetNumAtoms() == mol.GetNumAtoms()

    def test_complex_molecule_with_multiple_stereocenters(self):
        """Test preprocessing of complex molecule with multiple stereocenters."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Molecule with multiple chiral centers (e.g., a sugar-like structure)
        mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](O)[C@H](O)C")
        preprocessed, smiles = preprocess_target_molecule(mol)

        # No stereochemistry in output
        assert "@" not in smiles

        # Atom count should be preserved
        assert preprocessed.GetNumAtoms() == mol.GetNumAtoms()

    def test_kavalactone_with_stereo(self):
        """Test preprocessing of kavalactone-like structure with stereochemistry."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Kavain with added stereochemistry
        mol = Chem.MolFromSmiles("COC1=CC(OC(/C=C/C2=CC=CC=C2)C1)=O")
        preprocessed, smiles = preprocess_target_molecule(mol)

        # Should succeed and remove E/Z stereochemistry
        assert preprocessed is not None
        assert "/" not in smiles
        assert "\\" not in smiles


class TestGetSinkCompoundType:
    """Tests for _get_sink_compound_type method which labels byproducts in pathway output."""

    @pytest.fixture
    def mcts_agent(self):
        """Create a minimal DORAnetMCTS agent for testing sink compound type labeling."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        mol = Chem.MolFromSmiles("CCCCC(=O)O")  # pentanoic acid
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        # Manually set up test data for the various categories
        agent.biological_sink_compounds = {"CCO", "CCC"}  # ethanol, propane
        agent.chemical_sink_compounds = {"C", "CC"}  # methane, ethane
        agent.bio_cofactors = {
            "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CSCC[C@H](N)C(=O)O)[C@@H](O)[C@H]1O"  # SAH (canonical)
        }
        agent.chemistry_helpers = {"O", "[H][H]"}  # water, hydrogen

        return agent

    def test_biological_sink_compound(self, mcts_agent):
        """Test that biological building blocks return 'biological'."""
        result = mcts_agent._get_sink_compound_type("CCO")  # ethanol
        assert result == "biological"

    def test_chemical_sink_compound(self, mcts_agent):
        """Test that chemical building blocks return 'chemical'."""
        result = mcts_agent._get_sink_compound_type("C")  # methane
        assert result == "chemical"

    def test_bio_cofactor(self, mcts_agent):
        """Test that biology cofactors return 'bio_cofactor'."""
        # SAH (S-adenosyl-L-homocysteine) - a common enzymatic cofactor byproduct
        sah_smiles = "NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](CSCC[C@H](N)C(=O)O)[C@@H](O)[C@H]1O"
        result = mcts_agent._get_sink_compound_type(sah_smiles)
        assert result == "bio_cofactor"

    def test_chemistry_helper(self, mcts_agent):
        """Test that chemistry helpers return 'chem_helper'."""
        result = mcts_agent._get_sink_compound_type("O")  # water
        assert result == "chem_helper"

        result = mcts_agent._get_sink_compound_type("[H][H]")  # hydrogen
        assert result == "chem_helper"

    def test_unknown_compound_returns_none(self, mcts_agent):
        """Test that unknown compounds return None."""
        result = mcts_agent._get_sink_compound_type("CCCCCCCCCC")  # decane
        assert result is None

    def test_invalid_smiles_returns_none(self, mcts_agent):
        """Test that invalid SMILES return None."""
        result = mcts_agent._get_sink_compound_type("invalid_smiles_xyz")
        assert result is None

    def test_priority_biological_over_bio_cofactor(self, mcts_agent):
        """Test that biological sink compounds take priority over bio_cofactors."""
        # Add the same compound to both sets
        test_smiles = "CCCC"
        mcts_agent.biological_sink_compounds.add(test_smiles)
        mcts_agent.bio_cofactors.add(test_smiles)

        result = mcts_agent._get_sink_compound_type(test_smiles)
        assert result == "biological"  # biological takes priority

    def test_priority_chemical_over_chem_helper(self, mcts_agent):
        """Test that chemical sink compounds take priority over chem_helpers."""
        # Add the same compound to both sets
        test_smiles = "CCCCC"
        mcts_agent.chemical_sink_compounds.add(test_smiles)
        mcts_agent.chemistry_helpers.add(test_smiles)

        result = mcts_agent._get_sink_compound_type(test_smiles)
        assert result == "chemical"  # chemical takes priority

    def test_pks_library_match(self, mcts_agent):
        """Test that PKS library matches return 'pks'."""
        # Add a test SMILES to pks_library
        mcts_agent.pks_library = {"CCCCC(=O)O"}  # pentanoic acid
        result = mcts_agent._get_sink_compound_type("CCCCC(=O)O")
        assert result == "pks"

    def test_pks_library_lower_priority_than_building_blocks(self, mcts_agent):
        """Test that PKS library has lower priority than building blocks."""
        # Add the same compound to biological and pks_library
        test_smiles = "CCCCCC"  # hexane
        mcts_agent.biological_sink_compounds.add(test_smiles)
        mcts_agent.pks_library = {test_smiles}

        result = mcts_agent._get_sink_compound_type(test_smiles)
        assert result == "biological"  # building block takes priority over pks

    def test_pks_library_lower_priority_than_cofactors(self, mcts_agent):
        """Test that PKS library has lower priority than cofactors."""
        # Add the same compound to bio_cofactors and pks_library
        test_smiles = "CCCCCCC"  # heptane
        mcts_agent.bio_cofactors.add(test_smiles)
        mcts_agent.pks_library = {test_smiles}

        result = mcts_agent._get_sink_compound_type(test_smiles)
        assert result == "bio_cofactor"  # cofactor takes priority over pks


class TestBioCofactorsTracking:
    """Tests for bio_cofactors set initialization and tracking."""

    def test_bio_cofactors_set_exists(self):
        """Test that bio_cofactors set is created during initialization."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        assert hasattr(agent, 'bio_cofactors')
        assert isinstance(agent.bio_cofactors, set)

    def test_bio_cofactors_populated_from_cofactors_file(self, tmp_path):
        """Test that cofactor files populate bio_cofactors set."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        # Create a temporary cofactors CSV file (name must contain 'cofactors')
        # Note: CSV reader expects column named "SMILES" (uppercase)
        cofactors_file = tmp_path / "biology_cofactors.csv"
        cofactors_file.write_text("SMILES,name\nCCO,ethanol\nCCC,propane\n")

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
            cofactors_files=[str(cofactors_file)],
        )

        # bio_cofactors should contain the loaded cofactors
        assert "CCO" in agent.bio_cofactors
        assert "CCC" in agent.bio_cofactors
        # They should also be in excluded_fragments
        assert "CCO" in agent.excluded_fragments
        assert "CCC" in agent.excluded_fragments

    def test_chemistry_helpers_not_in_bio_cofactors(self, tmp_path):
        """Test that chemistry_helpers file populates chemistry_helpers, not bio_cofactors."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        # Create a temporary chemistry_helpers CSV file
        # Note: CSV reader expects column named "SMILES" (uppercase)
        helpers_file = tmp_path / "test_chemistry_helpers.csv"
        helpers_file.write_text("SMILES,name\nO,water\n[H][H],hydrogen\n")

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
            cofactors_files=[str(helpers_file)],
        )

        # chemistry_helpers should contain the loaded helpers
        assert "O" in agent.chemistry_helpers
        # They should NOT be in bio_cofactors (file name has 'chemistry_helpers')
        assert "O" not in agent.bio_cofactors
        # They should also be in excluded_fragments
        assert "O" in agent.excluded_fragments


class TestSaveSuccessfulPathways:
    """Tests for save_successful_pathways method."""

    def test_leaf_node_without_sink_or_pks_excluded(self, tmp_path):
        """Test that leaf nodes without sink/PKS status are excluded from successful pathways."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        # Create a leaf node that is NOT a sink compound
        leaf_mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC")  # Long chain, not in any library
        leaf = Node(fragment=leaf_mol, parent=root, depth=1, provenance="enzymatic")
        leaf.is_sink_compound = False
        leaf.is_pks_terminal = False
        root.children.append(leaf)
        agent.nodes.append(leaf)

        # Save successful pathways
        output_file = tmp_path / "test_successful.txt"
        agent.save_successful_pathways(str(output_file))

        # Read the file and verify the leaf node is NOT included
        content = output_file.read_text()
        assert "CCCCCCCCCCCCCCCC" not in content

    def test_leaf_node_with_sink_compound_included(self, tmp_path):
        """Test that leaf nodes with sink compound status ARE included in successful pathways."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        # Add a known sink compound to the biological_sink_compounds set
        sink_smiles = "CCO"  # ethanol
        agent.biological_sink_compounds.add(sink_smiles)

        # Create a leaf node that IS a sink compound
        leaf_mol = Chem.MolFromSmiles(sink_smiles)
        leaf = Node(fragment=leaf_mol, parent=root, depth=1, provenance="enzymatic")
        leaf.is_sink_compound = True
        root.children.append(leaf)
        agent.nodes.append(leaf)

        # Save successful pathways
        output_file = tmp_path / "test_successful.txt"
        agent.save_successful_pathways(str(output_file))

        # Read the file and verify the leaf node IS included
        content = output_file.read_text()
        assert "CCO" in content or "OCC" in content  # canonical SMILES may vary

    def test_leaf_node_with_pks_terminal_included(self, tmp_path):
        """Test that leaf nodes with PKS terminal status ARE included in successful pathways."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node
        from DORAnet_agent.mcts import RetroTideResult

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        # Create a leaf node that is a PKS terminal
        pks_smiles = "CCCC(=O)O"  # butyric acid
        leaf_mol = Chem.MolFromSmiles(pks_smiles)
        leaf = Node(fragment=leaf_mol, parent=root, depth=1, provenance="enzymatic")
        leaf.is_sink_compound = False
        leaf.is_pks_terminal = True
        root.children.append(leaf)
        agent.nodes.append(leaf)

        # Add a successful RetroTide result for this node
        agent.retrotide_results.append(
            RetroTideResult(
                doranet_node_id=1,
                doranet_node_smiles=pks_smiles,
                doranet_node_depth=1,
                doranet_node_provenance="enzymatic",
                retrotide_successful=True,
            )
        )

        # Save successful pathways
        output_file = tmp_path / "test_successful.txt"
        agent.save_successful_pathways(str(output_file))

        # Read the file and verify the leaf node IS included
        content = output_file.read_text()
        assert "CCCC(=O)O" in content or pks_smiles in content

    @pytest.mark.slow
    def test_pentanoic_acid_integration_all_modalities(self, tmp_path):
        """
        Integration test: Run minimal MCTS on pentanoic acid and verify
        all successful pathways have covered terminal fragments and byproducts.

        This test runs a real (but minimal) search to verify end-to-end
        pathway validation logic.
        """
        from pathlib import Path
        import re
        from DORAnet_agent.async_expansion_mcts import AsyncExpansionDORAnetMCTS
        from DORAnet_agent.node import Node

        REPO_ROOT = Path(__file__).resolve().parents[1]

        # Target: pentanoic acid
        target_smiles = "CCCCC(=O)O"
        mol = Chem.MolFromSmiles(target_smiles)
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        # Minimal MCTS configuration
        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=20,
            max_depth=2,
            max_children_per_expand=10,
            use_enzymatic=True,
            use_synthetic=True,
            use_chem_building_blocksDB=True,
            use_bio_building_blocksDB=True,
            use_PKS_building_blocksDB=True,
            cofactors_files=[
                str(REPO_ROOT / "data" / "raw" / "all_cofactors.csv"),
                str(REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv"),
            ],
            pks_library_file=str(REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V3.txt"),
            sink_compounds_files=[
                str(REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt"),
                str(REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt"),
            ],
            rollout_policy=NoOpRolloutPolicy(),
            reward_policy=SparseTerminalRewardPolicy(sink_terminal_reward=1.0),
            spawn_retrotide=False,  # Disable RetroTide for speed
            num_workers=2,
        )

        # Run the search
        agent.run()

        # Save successful pathways
        output_file = tmp_path / "successful_pathways.txt"
        agent.save_successful_pathways(str(output_file))

        # Read and parse the output
        content = output_file.read_text()

        # === VERIFICATION 1: File format ===
        assert "SUCCESSFUL PATHWAYS (PKS OR SINK PRODUCTS ONLY)" in content
        assert "RUN CONFIGURATION" in content
        assert "PATHWAY TYPE BREAKDOWN" in content

        # Extract total pathways count from header
        total_match = re.search(r"Total pathways:\s*(\d+)", content)
        assert total_match is not None, "Could not find 'Total pathways' in output"
        total_pathways = int(total_match.group(1))

        # Count actual pathway blocks
        pathway_blocks = re.findall(r"PATHWAY #(\d+):", content)
        assert len(pathway_blocks) == total_pathways, \
            f"Header says {total_pathways} pathways but found {len(pathway_blocks)} blocks"

        # If no pathways found, test passes (nothing to validate)
        if total_pathways == 0:
            return

        # === VERIFICATION 2: All terminal fragments are covered ===
        # Build the coverage sets
        coverage_sets = (
            agent.biological_sink_compounds |
            agent.chemical_sink_compounds |
            agent.pks_library |
            agent.excluded_fragments
        )

        # Extract terminal fragments from each pathway
        terminal_pattern = r"Terminal Fragment:\s*(\S+)"
        terminals = re.findall(terminal_pattern, content)

        for terminal_smiles in terminals:
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(terminal_smiles))
            assert canonical in coverage_sets, \
                f"Terminal fragment '{terminal_smiles}' (canonical: {canonical}) is not covered"

        # === VERIFICATION 3: All byproducts are covered ===
        # Extract products from each step, check non-primary ones
        # Pattern matches lines like: "- SMILES [branch, sink=chemical]"
        product_pattern = r"-\s*(\S+)\s*\[(\w+),\s*sink=(\w+)\]"
        products = re.findall(product_pattern, content)

        for smiles, role, sink_type in products:
            if role == "primary":
                continue  # Primary products continue along pathway
            # Byproducts must be covered
            if sink_type == "No":
                # If marked as sink=No, it should NOT be in successful pathways
                # This would indicate a bug
                pytest.fail(f"Byproduct '{smiles}' marked as sink=No in successful pathway")
