"""
Base classes for MCTS policies.

This module defines the abstract interfaces that all rollout and reward
policies must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from rdkit import Chem
    from ..node import Node


@dataclass
class RolloutResult:
    """
    Result from a rollout policy execution.

    Attributes:
        reward: The reward value computed by the rollout.
        terminal: Whether the node should be marked as terminal (no further expansion).
        terminal_type: Type of terminal state (e.g., "pks_terminal", "retrotide_success").
        metadata: Policy-specific data (e.g., RetroTide results, simulation traces).
    """

    reward: float
    terminal: bool = False
    terminal_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        term_str = f", terminal={self.terminal_type}" if self.terminal else ""
        return f"RolloutResult(reward={self.reward:.3f}{term_str})"


class RolloutPolicy(ABC):
    """
    Abstract base class for rollout policies.

    A rollout policy defines how to simulate from a leaf node to estimate
    its value. This could be:
    - No-op (return immediately with no reward)
    - Random playouts
    - Heuristic-guided simulation
    - Model-based prediction
    - Spawning a sub-agent (e.g., RetroTide)

    The rollout is executed on child nodes after expansion, following
    classical MCTS:
        Selection → Expansion → Rollout (on children) → Backpropagation
    """

    @abstractmethod
    def rollout(self, node: "Node", context: Dict[str, Any]) -> RolloutResult:
        """
        Perform a rollout from the given node.

        Args:
            node: The node to perform rollout from (typically a newly expanded child).
            context: Dictionary containing MCTS state, which may include:
                - target_molecule: The synthesis target (RDKit Mol)
                - pks_library: Set of PKS product SMILES
                - sink_compounds: Set of building block SMILES
                - retrotide_kwargs: Parameters for RetroTide if applicable
                - agent: Reference to the DORAnetMCTS instance

        Returns:
            RolloutResult containing reward, terminal status, and metadata.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging and identification."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RewardPolicy(ABC):
    """
    Abstract base class for reward computation policies.

    A reward policy defines how to compute the immediate reward for a node
    based on its properties. This is separate from rollout—rewards can be
    computed without simulation (e.g., checking if a fragment matches a database).

    Multiple reward policies can be composed using ComposedRewardPolicy.
    """

    @abstractmethod
    def calculate_reward(self, node: "Node", context: Dict[str, Any]) -> float:
        """
        Calculate reward for a node.

        Args:
            node: The node to compute reward for.
            context: Dictionary containing MCTS state (same as rollout context).

        Returns:
            Reward value (typically 0.0 to 1.0, but not strictly bounded).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging and identification."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
