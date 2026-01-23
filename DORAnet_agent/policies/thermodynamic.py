"""
Thermodynamic-scaled rollout and reward policies for DORAnet MCTS.

This module provides wrapper policies that scale rewards by pathway thermodynamic
feasibility, allowing MCTS to bias exploration toward thermodynamically feasible
pathways while still permitting exploration of less favorable routes.

Key Design Decisions:
    - Soft biasing over hard pruning: Infeasible pathways receive reduced (but non-zero) rewards
    - Unified 0-1 scale: Sigmoid transform for ΔH, DORA-XGB scores used directly
    - Pathway-level assessment: Geometric mean of step feasibilities (length-normalized)
    - Composable wrappers: Can wrap any existing rollout/reward policy

Sigmoid Transformation:
    score = 1.0 / (1.0 + exp(k * (ΔH - threshold)))

    Where:
    - k = 0.2 (steepness)
    - threshold = 15.0 kcal/mol

    Example mappings:
    - ΔH = -20 → score ≈ 0.999
    - ΔH = 0   → score ≈ 0.953
    - ΔH = 15  → score = 0.500
    - ΔH = 30  → score ≈ 0.047
    - ΔH = 50  → score ≈ 0.001
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from .base import RewardPolicy, RolloutPolicy, RolloutResult

if TYPE_CHECKING:
    from ..node import Node


def sigmoid_transform(
    delta_h: float,
    k: float = 0.2,
    threshold: float = 15.0,
) -> float:
    """
    Transform enthalpy of reaction (ΔH) to a 0-1 feasibility score.

    Uses sigmoid function: 1 / (1 + exp(k * (ΔH - threshold)))

    Args:
        delta_h: Enthalpy of reaction in kcal/mol (lower = more favorable)
        k: Steepness of sigmoid (default 0.2)
        threshold: Center point of sigmoid in kcal/mol (default 15.0)

    Returns:
        Feasibility score in [0.0, 1.0] where higher = more feasible
    """
    return 1.0 / (1.0 + math.exp(k * (delta_h - threshold)))


def get_node_feasibility_score(
    node: "Node",
    sigmoid_k: float = 0.2,
    sigmoid_threshold: float = 15.0,
    use_dora_xgb_for_enzymatic: bool = True,
) -> float:
    """
    Get unified 0-1 feasibility score for a single node.

    For enzymatic reactions:
        - Uses DORA-XGB score if available and use_dora_xgb_for_enzymatic=True
        - Falls back to sigmoid-transformed ΔH if DORA-XGB unavailable
    For synthetic reactions:
        - Uses sigmoid-transformed ΔH
    For unknown/target nodes:
        - Returns 1.0 (assume feasible)

    Args:
        node: The node to score
        sigmoid_k: Steepness parameter for sigmoid transform
        sigmoid_threshold: Center point for sigmoid transform (kcal/mol)
        use_dora_xgb_for_enzymatic: Whether to prefer DORA-XGB for enzymatic reactions

    Returns:
        Feasibility score in [0.0, 1.0]
    """
    provenance = getattr(node, 'provenance', None)

    if provenance == "enzymatic":
        # Prefer DORA-XGB for enzymatic reactions
        if use_dora_xgb_for_enzymatic and node.feasibility_score is not None:
            return node.feasibility_score
        # Fall back to thermodynamic score
        if node.enthalpy_of_reaction is not None:
            return sigmoid_transform(node.enthalpy_of_reaction, sigmoid_k, sigmoid_threshold)
        return 1.0  # Unknown, assume feasible

    elif provenance == "synthetic":
        # Use thermodynamic score for synthetic reactions
        if node.enthalpy_of_reaction is not None:
            return sigmoid_transform(node.enthalpy_of_reaction, sigmoid_k, sigmoid_threshold)
        return 1.0  # Unknown, assume feasible

    else:
        # Target node or unknown provenance
        return 1.0


def get_pathway_feasibility(
    node: "Node",
    sigmoid_k: float = 0.2,
    sigmoid_threshold: float = 15.0,
    use_dora_xgb_for_enzymatic: bool = True,
    aggregation: str = "geometric_mean",
) -> Tuple[float, List[float]]:
    """
    Compute pathway feasibility from root to this node.

    Walks the parent chain and aggregates individual node feasibility scores.

    Args:
        node: Terminal node of the pathway
        sigmoid_k: Steepness parameter for sigmoid transform
        sigmoid_threshold: Center point for sigmoid transform (kcal/mol)
        use_dora_xgb_for_enzymatic: Whether to prefer DORA-XGB for enzymatic reactions
        aggregation: How to aggregate scores. Options:
            - "geometric_mean": (∏ scores)^(1/n) - normalized for path length
            - "product": ∏ scores - harsh, penalizes longer paths
            - "minimum": min(scores) - worst step determines pathway quality
            - "arithmetic_mean": average of scores

    Returns:
        Tuple of (aggregated_score, list_of_individual_scores)
    """
    # Collect scores along the pathway (from node to root)
    scores = []
    current = node
    while current is not None:
        score = get_node_feasibility_score(
            current, sigmoid_k, sigmoid_threshold, use_dora_xgb_for_enzymatic
        )
        scores.append(score)
        current = current.parent

    if not scores:
        return 1.0, []

    # Aggregate based on method
    if aggregation == "geometric_mean":
        # Geometric mean: (∏ scores)^(1/n)
        product = math.prod(scores)
        aggregated = product ** (1.0 / len(scores))
    elif aggregation == "product":
        aggregated = math.prod(scores)
    elif aggregation == "minimum":
        aggregated = min(scores)
    elif aggregation == "arithmetic_mean":
        aggregated = sum(scores) / len(scores)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return aggregated, scores


class ThermodynamicScaledRolloutPolicy(RolloutPolicy):
    """
    Wrapper that scales any rollout policy's rewards by pathway thermodynamic feasibility.

    This policy wraps an existing rollout policy and multiplies its returned reward
    by a pathway feasibility factor computed from thermodynamic information stored
    on nodes along the path from root to the current node.

    Feasibility Scoring:
        - Enzymatic reactions: Uses DORA-XGB score (0-1) if available
        - Synthetic reactions: Uses sigmoid-transformed ΔH from pathermo
        - Unknown: Defaults to 1.0 (assume feasible)

    Pathway Aggregation:
        - Default: Geometric mean of step feasibilities (path-length normalized)
        - Alternative: Product, minimum, or arithmetic mean

    Example:
        base_policy = SAScore_and_SpawnRetroTideOnDatabaseCheck()
        scaled_policy = ThermodynamicScaledRolloutPolicy(
            base_policy=base_policy,
            feasibility_weight=0.8,
        )
        agent = DORAnetMCTS(rollout_policy=scaled_policy, ...)
    """

    def __init__(
        self,
        base_policy: RolloutPolicy,
        feasibility_weight: float = 1.0,
        sigmoid_k: float = 0.2,
        sigmoid_threshold: float = 15.0,
        use_dora_xgb_for_enzymatic: bool = True,
        aggregation: str = "geometric_mean",
    ):
        """
        Args:
            base_policy: The rollout policy to wrap
            feasibility_weight: How much to weight feasibility (0.0-1.0).
                0.0 = ignore feasibility (returns base reward unchanged)
                1.0 = full scaling (reward × pathway_feasibility)
                0.5 = blend: reward × (0.5 + 0.5 × pathway_feasibility)
            sigmoid_k: Steepness of sigmoid for ΔH transformation (default 0.2)
            sigmoid_threshold: Center point of sigmoid in kcal/mol (default 15.0)
            use_dora_xgb_for_enzymatic: Use DORA-XGB scores for enzymatic reactions
                instead of thermodynamic scores (default True)
            aggregation: Method for aggregating pathway scores:
                "geometric_mean" (default), "product", "minimum", "arithmetic_mean"
        """
        self.base_policy = base_policy
        self.feasibility_weight = feasibility_weight
        self.sigmoid_k = sigmoid_k
        self.sigmoid_threshold = sigmoid_threshold
        self.use_dora_xgb_for_enzymatic = use_dora_xgb_for_enzymatic
        self.aggregation = aggregation

    def rollout(self, node: "Node", context: Dict[str, Any]) -> RolloutResult:
        """
        Execute base policy rollout and scale reward by pathway feasibility.
        """
        # Get base rollout result
        result = self.base_policy.rollout(node, context)

        # If base reward is 0, no point scaling
        if result.reward == 0.0:
            return result

        # Compute pathway feasibility
        pathway_feas, step_scores = get_pathway_feasibility(
            node=node,
            sigmoid_k=self.sigmoid_k,
            sigmoid_threshold=self.sigmoid_threshold,
            use_dora_xgb_for_enzymatic=self.use_dora_xgb_for_enzymatic,
            aggregation=self.aggregation,
        )

        # Apply feasibility weight
        # weight=0: ignore feasibility → multiplier = 1.0
        # weight=1: full scaling → multiplier = pathway_feas
        # weight=0.5: blend → multiplier = 0.5 + 0.5 × pathway_feas
        scaling_factor = (1.0 - self.feasibility_weight) + self.feasibility_weight * pathway_feas
        scaled_reward = result.reward * scaling_factor

        # Create new result with scaled reward and augmented metadata
        return RolloutResult(
            reward=scaled_reward,
            terminal=result.terminal,
            terminal_type=result.terminal_type,
            metadata={
                **result.metadata,
                "base_reward": result.reward,
                "pathway_feasibility": pathway_feas,
                "pathway_step_scores": step_scores,
                "feasibility_scaling_factor": scaling_factor,
            },
        )

    @property
    def name(self) -> str:
        return (f"ThermodynamicScaled({self.base_policy.name}, "
                f"weight={self.feasibility_weight})")

    def __repr__(self) -> str:
        return (f"ThermodynamicScaledRolloutPolicy("
                f"base_policy={self.base_policy!r}, "
                f"feasibility_weight={self.feasibility_weight}, "
                f"sigmoid_k={self.sigmoid_k}, "
                f"sigmoid_threshold={self.sigmoid_threshold}, "
                f"aggregation='{self.aggregation}')")


class ThermodynamicScaledRewardPolicy(RewardPolicy):
    """
    Wrapper that scales any reward policy's rewards by pathway thermodynamic feasibility.

    This policy wraps an existing reward policy and multiplies its returned reward
    by a pathway feasibility factor. Used for scaling terminal rewards (sink compounds,
    PKS terminals) based on how thermodynamically feasible the pathway to reach them is.

    Note: This is complementary to ThermodynamicScaledRolloutPolicy. For complete
    coverage, use both wrappers together:
        - ThermodynamicScaledRolloutPolicy: Scales dense rewards from rollouts (SA score, etc.)
        - ThermodynamicScaledRewardPolicy: Scales sparse terminal rewards (sink compounds)

    Example:
        base_reward = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)
        scaled_reward = ThermodynamicScaledRewardPolicy(
            base_policy=base_reward,
            feasibility_weight=0.8,
        )
        agent = DORAnetMCTS(reward_policy=scaled_reward, ...)
    """

    def __init__(
        self,
        base_policy: RewardPolicy,
        feasibility_weight: float = 1.0,
        sigmoid_k: float = 0.2,
        sigmoid_threshold: float = 15.0,
        use_dora_xgb_for_enzymatic: bool = True,
        aggregation: str = "geometric_mean",
    ):
        """
        Args:
            base_policy: The reward policy to wrap
            feasibility_weight: How much to weight feasibility (0.0-1.0).
                0.0 = ignore feasibility (returns base reward unchanged)
                1.0 = full scaling (reward × pathway_feasibility)
            sigmoid_k: Steepness of sigmoid for ΔH transformation (default 0.2)
            sigmoid_threshold: Center point of sigmoid in kcal/mol (default 15.0)
            use_dora_xgb_for_enzymatic: Use DORA-XGB scores for enzymatic reactions
            aggregation: Method for aggregating pathway scores
        """
        self.base_policy = base_policy
        self.feasibility_weight = feasibility_weight
        self.sigmoid_k = sigmoid_k
        self.sigmoid_threshold = sigmoid_threshold
        self.use_dora_xgb_for_enzymatic = use_dora_xgb_for_enzymatic
        self.aggregation = aggregation

    def calculate_reward(self, node: "Node", context: Dict[str, Any]) -> float:
        """
        Calculate base reward and scale by pathway feasibility.
        """
        # Get base reward
        base_reward = self.base_policy.calculate_reward(node, context)

        # If base reward is 0, no point scaling
        if base_reward == 0.0:
            return 0.0

        # Compute pathway feasibility
        pathway_feas, _ = get_pathway_feasibility(
            node=node,
            sigmoid_k=self.sigmoid_k,
            sigmoid_threshold=self.sigmoid_threshold,
            use_dora_xgb_for_enzymatic=self.use_dora_xgb_for_enzymatic,
            aggregation=self.aggregation,
        )

        # Apply feasibility weight
        scaling_factor = (1.0 - self.feasibility_weight) + self.feasibility_weight * pathway_feas
        return base_reward * scaling_factor

    @property
    def name(self) -> str:
        return (f"ThermodynamicScaled({self.base_policy.name}, "
                f"weight={self.feasibility_weight})")

    def __repr__(self) -> str:
        return (f"ThermodynamicScaledRewardPolicy("
                f"base_policy={self.base_policy!r}, "
                f"feasibility_weight={self.feasibility_weight}, "
                f"sigmoid_k={self.sigmoid_k}, "
                f"sigmoid_threshold={self.sigmoid_threshold}, "
                f"aggregation='{self.aggregation}')")
