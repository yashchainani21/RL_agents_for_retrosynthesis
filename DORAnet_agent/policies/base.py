"""
Base classes for MCTS policies.

This module defines the abstract interfaces for:
- TerminalDetector: Post-expansion terminal detection (e.g., RetroTide verification)
- RewardPolicy: Reward computation for nodes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from rdkit import Chem
    from ..node import Node


@dataclass
class TerminalDetectionResult:
    """
    Result from a terminal detector.

    Unlike RolloutResult, this does NOT carry a reward — reward computation
    is exclusively the domain of RewardPolicy.

    Attributes:
        terminal: Whether the node should be marked as terminal (no further expansion).
        terminal_type: Type of terminal state (e.g., "pks_terminal").
        metadata: Detector-specific data (e.g., RetroTide agent, results).
    """

    terminal: bool = False
    terminal_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        if self.terminal:
            return f"TerminalDetectionResult(terminal=True, type={self.terminal_type})"
        return "TerminalDetectionResult(terminal=False)"


class TerminalDetector(ABC):
    """
    Abstract base class for post-expansion terminal detection.

    A terminal detector determines whether a newly expanded child node
    should be marked as terminal. This replaces the old RolloutPolicy
    abstraction, making explicit that no rollout/simulation is performed.

    Typical usage: checking if a fragment matches the PKS library and
    spawning RetroTide to verify that a PKS assembly line can produce it.
    """

    @abstractmethod
    def detect(self, node: "Node", context: Dict[str, Any]) -> TerminalDetectionResult:
        """
        Check whether a node should be marked as terminal.

        Args:
            node: The node to check (typically a newly expanded child).
            context: Dictionary containing MCTS state, which may include:
                - target_molecule: The synthesis target (RDKit Mol)
                - pks_library: Set of PKS product SMILES
                - sink_compounds: Set of building block SMILES
                - retrotide_kwargs: Parameters for RetroTide if applicable
                - agent: Reference to the DORAnetMCTS instance

        Returns:
            TerminalDetectionResult with terminal status and metadata.
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
    based on its properties. Rewards are computed without simulation
    (e.g., checking terminal status, computing SA score).
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
