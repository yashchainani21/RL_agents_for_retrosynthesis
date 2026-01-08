"""
Reward policies for DORAnet MCTS.

This module provides concrete implementations of reward computation strategies.
Rewards are computed based on node properties without simulation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from rdkit import Chem

from .base import RewardPolicy

if TYPE_CHECKING:
    from ..node import Node


@lru_cache(maxsize=50000)
def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form, returning None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


class SinkCompoundRewardPolicy(RewardPolicy):
    """
    Reward policy that returns a reward for sink compounds (building blocks).

    Sink compounds are commercially available building blocks that don't need
    further synthesis. This policy checks the node's `is_sink_compound` flag.
    """

    def __init__(self, reward_value: float = 1.0):
        """
        Args:
            reward_value: Reward to return for sink compounds. Default 1.0.
        """
        self.reward_value = reward_value

    def calculate_reward(self, node: "Node", context: Dict[str, Any]) -> float:
        """Return reward_value if node is a sink compound, else 0.0."""
        if node.is_sink_compound:
            return self.reward_value
        return 0.0

    @property
    def name(self) -> str:
        return f"SinkCompound(reward={self.reward_value})"


class PKSLibraryRewardPolicy(RewardPolicy):
    """
    Reward policy based on PKS library membership.

    Returns a reward if the node's SMILES matches a known PKS product.
    The PKS library should be provided in the context or at initialization.
    """

    def __init__(
        self,
        pks_library: Optional[Set[str]] = None,
        reward_value: float = 1.0,
    ):
        """
        Args:
            pks_library: Set of canonical PKS product SMILES. If None, will be
                retrieved from context["pks_library"] at runtime.
            reward_value: Reward to return for PKS matches. Default 1.0.
        """
        self._pks_library = pks_library
        self.reward_value = reward_value

    def _get_pks_library(self, context: Dict[str, Any]) -> Set[str]:
        """Get PKS library from instance or context."""
        if self._pks_library is not None:
            return self._pks_library
        return context.get("pks_library", set())

    def calculate_reward(self, node: "Node", context: Dict[str, Any]) -> float:
        """Return reward_value if node matches PKS library, else 0.0."""
        pks_library = self._get_pks_library(context)
        if not pks_library:
            return 0.0

        smiles = node.smiles
        if smiles is None:
            return 0.0

        canonical = _canonicalize_smiles(smiles)
        if canonical and canonical in pks_library:
            return self.reward_value

        return 0.0

    @property
    def name(self) -> str:
        return f"PKSLibrary(reward={self.reward_value})"


class SparseTerminalRewardPolicy(RewardPolicy):
    """
    Sparse reward policy that combines sink compound and PKS terminal checks.

    This replicates the original DORAnetMCTS.calculate_reward() behavior:
    - Returns sink_terminal_reward for sink compounds
    - Returns 1.0 for PKS library matches
    - Returns 0.0 otherwise
    """

    def __init__(
        self,
        sink_terminal_reward: float = 1.0,
        pks_library: Optional[Set[str]] = None,
    ):
        """
        Args:
            sink_terminal_reward: Reward for sink compounds. Default 1.0.
            pks_library: Set of canonical PKS product SMILES. If None, will be
                retrieved from context["pks_library"] at runtime.
        """
        self.sink_terminal_reward = sink_terminal_reward
        self._pks_library = pks_library

    def _get_pks_library(self, context: Dict[str, Any]) -> Set[str]:
        """Get PKS library from instance or context."""
        if self._pks_library is not None:
            return self._pks_library
        return context.get("pks_library", set())

    def calculate_reward(self, node: "Node", context: Dict[str, Any]) -> float:
        """
        Calculate sparse terminal reward.

        Returns sink_terminal_reward for sink compounds, 1.0 for PKS matches,
        0.0 otherwise.
        """
        # Sink compounds are valuable terminal building blocks
        if node.is_sink_compound:
            return self.sink_terminal_reward

        # Check PKS terminal flag first (already determined)
        if node.is_pks_terminal:
            return 1.0

        # Check PKS library membership
        pks_library = self._get_pks_library(context)
        if not pks_library:
            return 0.0

        smiles = node.smiles
        if smiles is None:
            return 0.0

        canonical = _canonicalize_smiles(smiles)
        if canonical and canonical in pks_library:
            return 1.0

        return 0.0

    @property
    def name(self) -> str:
        return f"SparseTerminal(sink_reward={self.sink_terminal_reward})"


class ComposedRewardPolicy(RewardPolicy):
    """
    Compose multiple reward policies with weights.

    The final reward is the weighted sum of individual policy rewards:
        reward = sum(weight_i * policy_i.calculate_reward(node, context))

    This allows combining different reward signals (e.g., terminal rewards
    plus shaped intermediate rewards).
    """

    def __init__(self, policies: List[Tuple[RewardPolicy, float]]):
        """
        Args:
            policies: List of (policy, weight) tuples. Weights can be any float
                and are not required to sum to 1.0.
        """
        if not policies:
            raise ValueError("ComposedRewardPolicy requires at least one policy")
        self.policies = policies

    def calculate_reward(self, node: "Node", context: Dict[str, Any]) -> float:
        """Calculate weighted sum of all policy rewards."""
        total = 0.0
        for policy, weight in self.policies:
            total += weight * policy.calculate_reward(node, context)
        return total

    @property
    def name(self) -> str:
        policy_names = [f"{p.name}*{w}" for p, w in self.policies]
        return f"Composed({', '.join(policy_names)})"

    def __repr__(self) -> str:
        policy_strs = [f"({p!r}, {w})" for p, w in self.policies]
        return f"ComposedRewardPolicy([{', '.join(policy_strs)}])"
