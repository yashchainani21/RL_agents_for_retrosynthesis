"""Tests for DORAnet_agent.policies.thermodynamic — sigmoid, feasibility, and scaling."""

import math
from unittest.mock import MagicMock

import pytest
from rdkit import Chem

from DORAnet_agent.node import Node
from DORAnet_agent.policies.thermodynamic import (
    ThermodynamicScaledRewardPolicy,
    get_node_feasibility_score,
    get_pathway_feasibility,
    sigmoid_transform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(provenance, feasibility_score=None, enthalpy=None, parent=None):
    """Create a Node with controlled attributes for testing."""
    mol = Chem.MolFromSmiles("CCO")
    node = Node(fragment=mol, parent=parent, depth=0 if parent is None else parent.depth + 1,
                provenance=provenance)
    node.feasibility_score = feasibility_score
    node.enthalpy_of_reaction = enthalpy
    if parent is not None:
        parent.add_child(node)
    return node


# ---------------------------------------------------------------------------
# TestSigmoidTransform
# ---------------------------------------------------------------------------

class TestSigmoidTransform:
    """Tests for sigmoid_transform()."""

    def test_at_threshold_returns_half(self):
        assert sigmoid_transform(15.0) == pytest.approx(0.5)

    def test_very_favorable_returns_near_one(self):
        assert sigmoid_transform(-20.0) == pytest.approx(0.999, abs=0.001)

    def test_very_unfavorable_returns_near_zero(self):
        assert sigmoid_transform(50.0) == pytest.approx(0.001, abs=0.001)

    def test_custom_k_and_threshold(self):
        # At the custom threshold, sigmoid should return 0.5
        assert sigmoid_transform(10.0, k=0.5, threshold=10.0) == pytest.approx(0.5)

    def test_monotonically_decreasing(self):
        vals = [sigmoid_transform(x) for x in [-10, 0, 15, 30]]
        assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# TestGetNodeFeasibilityScore
# ---------------------------------------------------------------------------

class TestGetNodeFeasibilityScore:
    """Tests for get_node_feasibility_score()."""

    def test_target_provenance_returns_one(self):
        node = _make_node("target")
        assert get_node_feasibility_score(node) == 1.0

    def test_none_provenance_returns_one(self):
        node = _make_node(None)
        assert get_node_feasibility_score(node) == 1.0

    def test_enzymatic_with_feasibility_score(self):
        node = _make_node("enzymatic", feasibility_score=0.85)
        assert get_node_feasibility_score(node) == 0.85

    def test_enzymatic_fallback_to_enthalpy(self):
        node = _make_node("enzymatic", enthalpy=-5.0)
        expected = sigmoid_transform(-5.0)
        assert get_node_feasibility_score(node) == pytest.approx(expected)

    def test_enzymatic_both_none_returns_half(self):
        node = _make_node("enzymatic")
        assert get_node_feasibility_score(node) == 0.5

    def test_enzymatic_dora_xgb_disabled_uses_enthalpy(self):
        node = _make_node("enzymatic", feasibility_score=0.85, enthalpy=-5.0)
        expected = sigmoid_transform(-5.0)
        result = get_node_feasibility_score(node, use_dora_xgb_for_enzymatic=False)
        assert result == pytest.approx(expected)

    def test_synthetic_with_enthalpy(self):
        node = _make_node("synthetic", enthalpy=10.0)
        expected = sigmoid_transform(10.0)
        assert get_node_feasibility_score(node) == pytest.approx(expected)

    def test_synthetic_without_enthalpy_returns_half(self):
        node = _make_node("synthetic")
        assert get_node_feasibility_score(node) == 0.5


# ---------------------------------------------------------------------------
# TestGetPathwayFeasibility
# ---------------------------------------------------------------------------

class TestGetPathwayFeasibility:
    """Tests for get_pathway_feasibility()."""

    def test_single_target_node(self):
        node = _make_node("target")
        agg, scores = get_pathway_feasibility(node)
        assert agg == pytest.approx(1.0)
        assert scores == [1.0]

    def test_two_node_chain_geometric_mean(self):
        root = _make_node("target")
        child = _make_node("enzymatic", feasibility_score=0.81, parent=root)
        agg, scores = get_pathway_feasibility(child)
        # scores collected leaf→root: [0.81, 1.0]
        expected = (0.81 * 1.0) ** (1.0 / 2)
        assert agg == pytest.approx(expected)
        assert len(scores) == 2

    def test_product_aggregation(self):
        root = _make_node("target")
        child = _make_node("enzymatic", feasibility_score=0.81, parent=root)
        agg, _ = get_pathway_feasibility(child, aggregation="product")
        assert agg == pytest.approx(0.81 * 1.0)

    def test_minimum_aggregation(self):
        root = _make_node("target")
        mid = _make_node("enzymatic", feasibility_score=0.9, parent=root)
        leaf = _make_node("synthetic", enthalpy=25.0, parent=mid)
        agg, scores = get_pathway_feasibility(leaf, aggregation="minimum")
        assert agg == pytest.approx(min(scores))

    def test_arithmetic_mean_aggregation(self):
        root = _make_node("target")
        child = _make_node("enzymatic", feasibility_score=0.81, parent=root)
        agg, scores = get_pathway_feasibility(child, aggregation="arithmetic_mean")
        assert agg == pytest.approx(sum(scores) / len(scores))

    def test_unknown_aggregation_raises(self):
        node = _make_node("target")
        with pytest.raises(ValueError, match="Unknown aggregation"):
            get_pathway_feasibility(node, aggregation="invalid")


# ---------------------------------------------------------------------------
# TestThermodynamicScaledRewardPolicyCalculateReward
# ---------------------------------------------------------------------------

class TestThermodynamicScaledRewardPolicyCalculateReward:
    """Tests for ThermodynamicScaledRewardPolicy.calculate_reward()."""

    def _make_policy(self, base_return, feasibility_weight=1.0):
        base = MagicMock()
        base.calculate_reward.return_value = base_return
        base.name = "MockPolicy"
        return ThermodynamicScaledRewardPolicy(
            base_policy=base,
            feasibility_weight=feasibility_weight,
        )

    def test_base_reward_zero_returns_zero(self):
        policy = self._make_policy(base_return=0.0)
        node = _make_node("target")
        assert policy.calculate_reward(node, {}) == 0.0

    def test_feasibility_weight_zero_returns_base_unchanged(self):
        policy = self._make_policy(base_return=0.8, feasibility_weight=0.0)
        node = _make_node("target")
        # scaling = (1 - 0) + 0 * feas = 1.0
        assert policy.calculate_reward(node, {}) == pytest.approx(0.8)

    def test_feasibility_weight_one_target_node(self):
        policy = self._make_policy(base_return=0.8, feasibility_weight=1.0)
        node = _make_node("target")  # feasibility = 1.0
        # scaling = 0 + 1.0 * 1.0 = 1.0
        assert policy.calculate_reward(node, {}) == pytest.approx(0.8)

    def test_intermediate_weight_partial_feasibility(self):
        policy = self._make_policy(base_return=1.0, feasibility_weight=0.5)
        root = _make_node("target")
        # Enzymatic child with feasibility 0.5 → pathway geo mean = (1.0*0.5)^0.5 ≈ 0.707
        child = _make_node("enzymatic", feasibility_score=0.5, parent=root)
        result = policy.calculate_reward(child, {})
        pathway_feas = (1.0 * 0.5) ** 0.5
        expected = 1.0 * ((1.0 - 0.5) + 0.5 * pathway_feas)
        assert result == pytest.approx(expected)

    def test_name_includes_base_and_weight(self):
        policy = self._make_policy(base_return=0.0, feasibility_weight=0.75)
        assert "MockPolicy" in policy.name
        assert "0.75" in policy.name
