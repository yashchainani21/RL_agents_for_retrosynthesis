"""
Unit tests for thermodynamic-scaled rollout and reward policies.

These tests verify:
1. Sigmoid transformation function
2. Single node feasibility scoring
3. Pathway feasibility aggregation
4. ThermodynamicScaledRolloutPolicy wrapper
5. ThermodynamicScaledRewardPolicy wrapper
"""

import pytest
import math
from typing import Optional
from unittest.mock import MagicMock

from DORAnet_agent.policies.thermodynamic import (
    sigmoid_transform,
    get_node_feasibility_score,
    get_pathway_feasibility,
    ThermodynamicScaledRolloutPolicy,
    ThermodynamicScaledRewardPolicy,
)
from DORAnet_agent.policies import (
    RolloutResult,
    NoOpRolloutPolicy,
    SparseTerminalRewardPolicy,
)


class MockNode:
    """Mock node for testing."""
    def __init__(
        self,
        provenance: Optional[str] = None,
        feasibility_score: Optional[float] = None,
        enthalpy_of_reaction: Optional[float] = None,
        parent: Optional["MockNode"] = None,
        is_sink_compound: bool = False,
        is_pks_terminal: bool = False,
    ):
        self.provenance = provenance
        self.feasibility_score = feasibility_score
        self.enthalpy_of_reaction = enthalpy_of_reaction
        self.parent = parent
        self.is_sink_compound = is_sink_compound
        self.is_pks_terminal = is_pks_terminal
        self.smiles = "CCO"


class TestSigmoidTransform:
    """Tests for sigmoid transformation function."""

    def test_favorable_reaction(self):
        """Highly favorable reactions should have score near 1.0."""
        score = sigmoid_transform(-20.0)
        assert score > 0.99

    def test_unfavorable_reaction(self):
        """Highly unfavorable reactions should have score near 0.0."""
        score = sigmoid_transform(50.0)
        assert score < 0.01

    def test_threshold_reaction(self):
        """Reaction at threshold should have score of 0.5."""
        score = sigmoid_transform(15.0, threshold=15.0)
        assert score == pytest.approx(0.5, rel=0.01)

    def test_zero_enthalpy(self):
        """Zero enthalpy should have high score (exergonic)."""
        score = sigmoid_transform(0.0)
        # With k=0.2, threshold=15: 1/(1+exp(-3)) ≈ 0.953
        assert score == pytest.approx(0.953, rel=0.01)

    def test_custom_parameters(self):
        """Custom k and threshold should work correctly."""
        score = sigmoid_transform(20.0, k=0.1, threshold=20.0)
        assert score == pytest.approx(0.5, rel=0.01)

    def test_steepness_parameter(self):
        """Higher k should produce steeper transition."""
        # With k=0.2, ΔH=30: 1/(1+exp(0.2*(30-15))) = 1/(1+exp(3)) ≈ 0.047
        score_low_k = sigmoid_transform(30.0, k=0.2)
        # With k=0.5, ΔH=30: 1/(1+exp(0.5*(30-15))) = 1/(1+exp(7.5)) ≈ 0.0006
        score_high_k = sigmoid_transform(30.0, k=0.5)

        assert score_high_k < score_low_k

    def test_expected_mappings(self):
        """Test specific example mappings from docstring."""
        # ΔH = -20 → score ≈ 0.999
        assert sigmoid_transform(-20.0) == pytest.approx(0.999, rel=0.01)
        # ΔH = 0   → score ≈ 0.953
        assert sigmoid_transform(0.0) == pytest.approx(0.953, rel=0.01)
        # ΔH = 15  → score = 0.500
        assert sigmoid_transform(15.0) == pytest.approx(0.5, rel=0.01)
        # ΔH = 30  → score ≈ 0.047
        assert sigmoid_transform(30.0) == pytest.approx(0.047, rel=0.01)


class TestGetNodeFeasibilityScore:
    """Tests for single node feasibility scoring."""

    def test_enzymatic_with_dora_xgb(self):
        """Enzymatic node should use DORA-XGB score when available."""
        node = MockNode(provenance="enzymatic", feasibility_score=0.85)
        score = get_node_feasibility_score(node)
        assert score == 0.85

    def test_enzymatic_fallback_to_thermo(self):
        """Enzymatic node should fall back to thermo if no DORA-XGB."""
        node = MockNode(provenance="enzymatic", enthalpy_of_reaction=0.0)
        score = get_node_feasibility_score(node)
        assert score > 0.9  # Should be high for ΔH=0

    def test_enzymatic_no_dora_xgb_flag(self):
        """Should use thermo when use_dora_xgb_for_enzymatic=False."""
        node = MockNode(
            provenance="enzymatic",
            feasibility_score=0.85,
            enthalpy_of_reaction=15.0,
        )
        score = get_node_feasibility_score(node, use_dora_xgb_for_enzymatic=False)
        # Should use sigmoid(15) ≈ 0.5, not DORA-XGB score of 0.85
        assert score == pytest.approx(0.5, rel=0.01)

    def test_synthetic_uses_thermo(self):
        """Synthetic node should use thermodynamic score."""
        node = MockNode(provenance="synthetic", enthalpy_of_reaction=15.0)
        score = get_node_feasibility_score(node)
        assert score == pytest.approx(0.5, rel=0.01)

    def test_synthetic_ignores_dora_xgb(self):
        """Synthetic node should ignore DORA-XGB score even if present."""
        node = MockNode(
            provenance="synthetic",
            feasibility_score=0.85,  # Should be ignored
            enthalpy_of_reaction=30.0,
        )
        score = get_node_feasibility_score(node)
        # Should use sigmoid(30) ≈ 0.047, not DORA-XGB score
        assert score == pytest.approx(0.047, rel=0.01)

    def test_unknown_provenance(self):
        """Unknown provenance should return 1.0."""
        node = MockNode(provenance="target")
        score = get_node_feasibility_score(node)
        assert score == 1.0

    def test_none_provenance(self):
        """None provenance should return 1.0."""
        node = MockNode(provenance=None)
        score = get_node_feasibility_score(node)
        assert score == 1.0

    def test_missing_values_enzymatic(self):
        """Missing thermodynamic values for enzymatic should return 1.0."""
        node = MockNode(provenance="enzymatic")
        score = get_node_feasibility_score(node)
        assert score == 1.0

    def test_missing_values_synthetic(self):
        """Missing thermodynamic values for synthetic should return 1.0."""
        node = MockNode(provenance="synthetic")
        score = get_node_feasibility_score(node)
        assert score == 1.0

    def test_custom_sigmoid_parameters(self):
        """Should respect custom sigmoid parameters."""
        node = MockNode(provenance="synthetic", enthalpy_of_reaction=20.0)
        score = get_node_feasibility_score(node, sigmoid_k=0.1, sigmoid_threshold=20.0)
        assert score == pytest.approx(0.5, rel=0.01)


class TestGetPathwayFeasibility:
    """Tests for pathway feasibility aggregation."""

    def test_single_node(self):
        """Single node pathway should return that node's score."""
        node = MockNode(provenance="synthetic", enthalpy_of_reaction=0.0)
        feas, scores = get_pathway_feasibility(node)
        assert len(scores) == 1
        assert feas == scores[0]

    def test_two_node_pathway(self):
        """Two-node pathway should compute geometric mean."""
        root = MockNode(provenance="target")
        child = MockNode(provenance="enzymatic", feasibility_score=0.64, parent=root)

        feas, scores = get_pathway_feasibility(child)

        assert len(scores) == 2
        # root = 1.0, child = 0.64
        # geometric mean = (1.0 * 0.64)^(1/2) = 0.8
        assert feas == pytest.approx(0.8, rel=0.01)

    def test_multi_node_pathway(self):
        """Multi-node pathway should aggregate scores."""
        root = MockNode(provenance="target")
        child1 = MockNode(provenance="enzymatic", feasibility_score=0.9, parent=root)
        child2 = MockNode(provenance="synthetic", enthalpy_of_reaction=0.0, parent=child1)

        feas, scores = get_pathway_feasibility(child2)

        assert len(scores) == 3
        # Geometric mean of scores
        expected = math.prod(scores) ** (1.0 / len(scores))
        assert feas == pytest.approx(expected, rel=0.01)

    def test_aggregation_product(self):
        """Product aggregation should multiply all scores."""
        root = MockNode(provenance="target")
        child = MockNode(provenance="enzymatic", feasibility_score=0.5, parent=root)

        feas, scores = get_pathway_feasibility(child, aggregation="product")
        # 1.0 * 0.5 = 0.5
        assert feas == pytest.approx(0.5, rel=0.01)

    def test_aggregation_minimum(self):
        """Minimum aggregation should return lowest score."""
        root = MockNode(provenance="target")
        child1 = MockNode(provenance="enzymatic", feasibility_score=0.9, parent=root)
        child2 = MockNode(provenance="enzymatic", feasibility_score=0.3, parent=child1)

        feas, scores = get_pathway_feasibility(child2, aggregation="minimum")
        assert feas == pytest.approx(0.3, rel=0.01)

    def test_aggregation_arithmetic_mean(self):
        """Arithmetic mean aggregation should average scores."""
        root = MockNode(provenance="target")
        child = MockNode(provenance="enzymatic", feasibility_score=0.5, parent=root)

        feas, scores = get_pathway_feasibility(child, aggregation="arithmetic_mean")
        # (1.0 + 0.5) / 2 = 0.75
        assert feas == pytest.approx(0.75, rel=0.01)

    def test_unknown_aggregation_raises(self):
        """Unknown aggregation method should raise ValueError."""
        node = MockNode(provenance="target")
        with pytest.raises(ValueError) as exc_info:
            get_pathway_feasibility(node, aggregation="unknown")
        assert "Unknown aggregation method" in str(exc_info.value)

    def test_empty_pathway(self):
        """Empty pathway (None node) should return 1.0."""
        # This tests internal behavior when no scores collected
        # In practice, we always have at least one node
        node = MockNode(provenance="target")
        feas, scores = get_pathway_feasibility(node)
        assert feas == 1.0
        assert len(scores) == 1


class TestThermodynamicScaledRolloutPolicy:
    """Tests for rollout policy wrapper."""

    def test_scales_reward(self):
        """Should scale base reward by pathway feasibility."""
        # Create a mock base policy that returns fixed reward
        base_policy = MagicMock()
        base_policy.rollout.return_value = RolloutResult(reward=1.0, terminal=False)
        base_policy.name = "MockPolicy"

        scaled_policy = ThermodynamicScaledRolloutPolicy(
            base_policy=base_policy,
            feasibility_weight=1.0,
        )

        # Create node with known feasibility
        node = MockNode(provenance="enzymatic", feasibility_score=0.5)

        result = scaled_policy.rollout(node, {})

        # With weight=1.0 and pathway_feas=0.5, reward should be 1.0 * 0.5 = 0.5
        assert result.reward == pytest.approx(0.5, rel=0.01)
        assert result.metadata["base_reward"] == 1.0
        assert result.metadata["pathway_feasibility"] == pytest.approx(0.5, rel=0.01)

    def test_zero_weight_returns_unchanged(self):
        """Weight=0 should return base reward unchanged."""
        base_policy = MagicMock()
        base_policy.rollout.return_value = RolloutResult(reward=0.8, terminal=False)
        base_policy.name = "MockPolicy"

        scaled_policy = ThermodynamicScaledRolloutPolicy(
            base_policy=base_policy,
            feasibility_weight=0.0,
        )

        node = MockNode(provenance="enzymatic", feasibility_score=0.1)
        result = scaled_policy.rollout(node, {})

        assert result.reward == pytest.approx(0.8, rel=0.01)

    def test_half_weight_blends(self):
        """Weight=0.5 should blend base reward with feasibility."""
        base_policy = MagicMock()
        base_policy.rollout.return_value = RolloutResult(reward=1.0, terminal=False)
        base_policy.name = "MockPolicy"

        scaled_policy = ThermodynamicScaledRolloutPolicy(
            base_policy=base_policy,
            feasibility_weight=0.5,
        )

        # Node with 0.5 feasibility
        node = MockNode(provenance="enzymatic", feasibility_score=0.5)
        result = scaled_policy.rollout(node, {})

        # scaling_factor = 0.5 + 0.5 * 0.5 = 0.75
        # reward = 1.0 * 0.75 = 0.75
        assert result.reward == pytest.approx(0.75, rel=0.01)
        assert result.metadata["feasibility_scaling_factor"] == pytest.approx(0.75, rel=0.01)

    def test_zero_base_reward_unchanged(self):
        """Zero base reward should remain zero without scaling."""
        base_policy = MagicMock()
        base_policy.rollout.return_value = RolloutResult(reward=0.0, terminal=False)
        base_policy.name = "MockPolicy"

        scaled_policy = ThermodynamicScaledRolloutPolicy(
            base_policy=base_policy,
            feasibility_weight=1.0,
        )

        node = MockNode(provenance="enzymatic", feasibility_score=0.5)
        result = scaled_policy.rollout(node, {})

        assert result.reward == 0.0
        # Metadata should not have scaling info since we early-return
        assert "base_reward" not in result.metadata

    def test_preserves_terminal_status(self):
        """Should preserve terminal flag from base policy."""
        base_policy = MagicMock()
        base_policy.rollout.return_value = RolloutResult(
            reward=1.0, terminal=True, terminal_type="pks_terminal"
        )
        base_policy.name = "MockPolicy"

        scaled_policy = ThermodynamicScaledRolloutPolicy(base_policy=base_policy)
        node = MockNode(provenance="target")
        result = scaled_policy.rollout(node, {})

        assert result.terminal is True
        assert result.terminal_type == "pks_terminal"

    def test_preserves_base_metadata(self):
        """Should preserve metadata from base policy."""
        base_policy = MagicMock()
        base_policy.rollout.return_value = RolloutResult(
            reward=1.0, terminal=False, metadata={"base_key": "base_value"}
        )
        base_policy.name = "MockPolicy"

        scaled_policy = ThermodynamicScaledRolloutPolicy(base_policy=base_policy)
        node = MockNode(provenance="target")
        result = scaled_policy.rollout(node, {})

        assert result.metadata["base_key"] == "base_value"
        assert "pathway_feasibility" in result.metadata

    def test_name_property(self):
        """Should return descriptive name."""
        base_policy = NoOpRolloutPolicy()
        scaled_policy = ThermodynamicScaledRolloutPolicy(
            base_policy=base_policy,
            feasibility_weight=0.8,
        )

        name = scaled_policy.name
        assert "ThermodynamicScaled" in name
        assert "NoOp" in name
        assert "0.8" in name

    def test_repr(self):
        """Should return informative repr."""
        base_policy = NoOpRolloutPolicy()
        scaled_policy = ThermodynamicScaledRolloutPolicy(
            base_policy=base_policy,
            feasibility_weight=0.8,
            sigmoid_k=0.3,
            sigmoid_threshold=20.0,
            aggregation="minimum",
        )

        repr_str = repr(scaled_policy)
        assert "ThermodynamicScaledRolloutPolicy" in repr_str
        assert "0.8" in repr_str
        assert "0.3" in repr_str
        assert "20.0" in repr_str
        assert "minimum" in repr_str

    def test_multi_node_pathway_scaling(self):
        """Should correctly scale using multi-node pathway feasibility."""
        base_policy = MagicMock()
        base_policy.rollout.return_value = RolloutResult(reward=1.0, terminal=False)
        base_policy.name = "MockPolicy"

        scaled_policy = ThermodynamicScaledRolloutPolicy(
            base_policy=base_policy,
            feasibility_weight=1.0,
        )

        # Create a 3-node pathway
        root = MockNode(provenance="target")
        child1 = MockNode(provenance="enzymatic", feasibility_score=0.81, parent=root)
        child2 = MockNode(provenance="enzymatic", feasibility_score=0.81, parent=child1)

        result = scaled_policy.rollout(child2, {})

        # Geometric mean of [target=1.0, child1=0.81, child2=0.81]
        # = (1.0 * 0.81 * 0.81)^(1/3) = (0.6561)^(1/3) ≈ 0.869
        expected_feas = (1.0 * 0.81 * 0.81) ** (1.0 / 3.0)
        assert result.metadata["pathway_feasibility"] == pytest.approx(expected_feas, rel=0.01)
        assert result.reward == pytest.approx(expected_feas, rel=0.01)

    def test_passes_context_to_base_policy(self):
        """Should pass context to base policy."""
        base_policy = MagicMock()
        base_policy.rollout.return_value = RolloutResult(reward=1.0, terminal=False)
        base_policy.name = "MockPolicy"

        scaled_policy = ThermodynamicScaledRolloutPolicy(base_policy=base_policy)
        node = MockNode(provenance="target")
        context = {"key": "value"}

        scaled_policy.rollout(node, context)

        base_policy.rollout.assert_called_once_with(node, context)


class TestThermodynamicScaledRewardPolicy:
    """Tests for reward policy wrapper."""

    def test_scales_terminal_reward(self):
        """Should scale terminal reward by pathway feasibility."""
        base_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)
        scaled_policy = ThermodynamicScaledRewardPolicy(
            base_policy=base_policy,
            feasibility_weight=1.0,
        )

        # Create sink compound node with known feasibility
        node = MockNode(
            provenance="synthetic",
            enthalpy_of_reaction=15.0,  # ΔH=15 → feas≈0.5
            is_sink_compound=True,
        )

        reward = scaled_policy.calculate_reward(node, {})

        # Base reward is 1.0, pathway_feas ≈ 0.5
        assert reward == pytest.approx(0.5, rel=0.1)

    def test_zero_base_reward_unchanged(self):
        """Zero base reward should remain zero."""
        base_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)
        scaled_policy = ThermodynamicScaledRewardPolicy(base_policy=base_policy)

        # Non-terminal node (base reward = 0)
        node = MockNode(provenance="synthetic", is_sink_compound=False)

        reward = scaled_policy.calculate_reward(node, {})
        assert reward == 0.0

    def test_zero_weight_returns_unchanged(self):
        """Weight=0 should return base reward unchanged."""
        base_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)
        scaled_policy = ThermodynamicScaledRewardPolicy(
            base_policy=base_policy,
            feasibility_weight=0.0,
        )

        node = MockNode(
            provenance="synthetic",
            enthalpy_of_reaction=50.0,  # Very unfavorable
            is_sink_compound=True,
        )

        reward = scaled_policy.calculate_reward(node, {})
        assert reward == pytest.approx(1.0, rel=0.01)

    def test_half_weight_blends(self):
        """Weight=0.5 should blend base reward with feasibility."""
        base_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)
        scaled_policy = ThermodynamicScaledRewardPolicy(
            base_policy=base_policy,
            feasibility_weight=0.5,
        )

        node = MockNode(
            provenance="enzymatic",
            feasibility_score=0.5,
            is_sink_compound=True,
        )

        reward = scaled_policy.calculate_reward(node, {})

        # scaling_factor = 0.5 + 0.5 * 0.5 = 0.75
        # reward = 1.0 * 0.75 = 0.75
        assert reward == pytest.approx(0.75, rel=0.01)

    def test_name_property(self):
        """Should return descriptive name."""
        base_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)
        scaled_policy = ThermodynamicScaledRewardPolicy(
            base_policy=base_policy,
            feasibility_weight=0.8,
        )

        name = scaled_policy.name
        assert "ThermodynamicScaled" in name
        assert "SparseTerminal" in name
        assert "0.8" in name

    def test_repr(self):
        """Should return informative repr."""
        base_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)
        scaled_policy = ThermodynamicScaledRewardPolicy(
            base_policy=base_policy,
            feasibility_weight=0.8,
            sigmoid_k=0.3,
            sigmoid_threshold=20.0,
            aggregation="product",
        )

        repr_str = repr(scaled_policy)
        assert "ThermodynamicScaledRewardPolicy" in repr_str
        assert "0.8" in repr_str
        assert "0.3" in repr_str
        assert "20.0" in repr_str
        assert "product" in repr_str

    def test_multi_node_pathway_scaling(self):
        """Should correctly scale using multi-node pathway feasibility."""
        base_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)
        scaled_policy = ThermodynamicScaledRewardPolicy(
            base_policy=base_policy,
            feasibility_weight=1.0,
        )

        # Create a 3-node pathway ending in sink compound
        root = MockNode(provenance="target")
        child1 = MockNode(provenance="enzymatic", feasibility_score=0.81, parent=root)
        child2 = MockNode(
            provenance="enzymatic",
            feasibility_score=0.81,
            parent=child1,
            is_sink_compound=True,
        )

        reward = scaled_policy.calculate_reward(child2, {})

        # Geometric mean of [target=1.0, child1=0.81, child2=0.81]
        # = (1.0 * 0.81 * 0.81)^(1/3) = (0.6561)^(1/3) ≈ 0.869
        expected_feas = (1.0 * 0.81 * 0.81) ** (1.0 / 3.0)
        assert reward == pytest.approx(expected_feas, rel=0.01)

    def test_passes_context_to_base_policy(self):
        """Should pass context to base policy."""
        base_policy = MagicMock()
        base_policy.calculate_reward.return_value = 1.0
        base_policy.name = "MockPolicy"

        scaled_policy = ThermodynamicScaledRewardPolicy(base_policy=base_policy)
        node = MockNode(provenance="target")
        context = {"key": "value"}

        scaled_policy.calculate_reward(node, context)

        base_policy.calculate_reward.assert_called_once_with(node, context)


class TestIntegrationBothPolicies:
    """Integration tests using both rollout and reward policies together."""

    def test_consistent_scaling_across_policies(self):
        """Both policies should compute the same pathway feasibility."""
        # Create a multi-node pathway
        root = MockNode(provenance="target")
        child1 = MockNode(provenance="enzymatic", feasibility_score=0.8, parent=root)
        child2 = MockNode(
            provenance="synthetic",
            enthalpy_of_reaction=0.0,
            parent=child1,
            is_sink_compound=True,
        )

        # Create base policies
        base_rollout = MagicMock()
        base_rollout.rollout.return_value = RolloutResult(reward=1.0, terminal=False)
        base_rollout.name = "MockRollout"

        base_reward = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)

        # Create scaled policies with same parameters
        scaled_rollout = ThermodynamicScaledRolloutPolicy(
            base_policy=base_rollout,
            feasibility_weight=1.0,
        )
        scaled_reward = ThermodynamicScaledRewardPolicy(
            base_policy=base_reward,
            feasibility_weight=1.0,
        )

        # Get results
        rollout_result = scaled_rollout.rollout(child2, {})
        reward_result = scaled_reward.calculate_reward(child2, {})

        # Both should use the same pathway feasibility
        assert rollout_result.metadata["pathway_feasibility"] == pytest.approx(
            reward_result / 1.0,  # reward_result = base_reward * pathway_feas
            rel=0.01,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
