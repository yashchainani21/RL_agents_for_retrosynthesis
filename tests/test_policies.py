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
