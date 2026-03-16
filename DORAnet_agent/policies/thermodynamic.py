"""
Thermodynamic-scaled reward policies for DORAnet MCTS.

This module provides wrapper policies that scale rewards by pathway thermodynamic
feasibility, allowing MCTS to bias exploration toward thermodynamically feasible
pathways while still permitting exploration of less favorable routes.

Key Design Decisions:
    - Soft biasing over hard pruning: Infeasible pathways receive reduced (but non-zero) rewards
    - Unified 0-1 scale: Sigmoid transform for ΔH, DORA-XGB scores used directly
    - Pathway-level assessment: Geometric mean of step feasibilities (length-normalized)
    - Composable wrappers: Can wrap any existing reward policy

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

from .base import RewardPolicy

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
        - Returns 0.5 (neutral/borderline) if neither is available
    For synthetic reactions:
        - Uses sigmoid-transformed ΔH
        - Returns 0.5 (neutral/borderline) if ΔH unavailable
    For unknown/target nodes:
        - Returns 1.0 (not a reaction step, no penalty)

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
        return 0.5  # Unknown, assign neutral (borderline) score

    elif provenance == "synthetic":
        # Use thermodynamic score for synthetic reactions
        if node.enthalpy_of_reaction is not None:
            return sigmoid_transform(node.enthalpy_of_reaction, sigmoid_k, sigmoid_threshold)
        return 0.5  # Unknown, assign neutral (borderline) score

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


class ThermodynamicScaledRewardPolicy(RewardPolicy):
    """
    Wrapper that scales any reward policy's rewards by pathway thermodynamic feasibility.

    This policy wraps an existing reward policy and multiplies its returned reward
    by a pathway feasibility factor. Used for scaling terminal rewards (sink compounds,
    PKS terminals) based on how thermodynamically feasible the pathway to reach them is.

    Example:
        base_reward = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)
        scaled_reward = ThermodynamicScaledRewardPolicy(
            base_policy=base_reward,
            feasibility_weight=1.0,
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
