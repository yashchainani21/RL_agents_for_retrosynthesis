
"""
Reward policies for DORAnet MCTS.

This module provides concrete implementations of reward computation strategies.
Rewards are computed based on node properties without simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

from rdkit import Chem

from .base import RewardPolicy
from .utils import (
    canonicalize_smiles as _canonicalize_smiles,
    calculate_sa_score as _calculate_sa_score,
    sa_score_to_reward as _sa_score_to_reward,
)

if TYPE_CHECKING:
    from ..node import Node


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


class SAScore_and_TerminalRewardPolicy(RewardPolicy):
    """
    Unified reward policy combining terminal rewards with dense non-terminal scoring.

    This is the production reward policy for DORAnet MCTS. It provides:
    - Full terminal rewards for sink compounds (chemical/biological building blocks)
    - Full terminal rewards for PKS-synthesizable compounds (RetroTide verified)
    - Dense rewards for all other compounds via a pluggable non-terminal scorer
      (defaults to SA score)

    Recommended setup::

        from DORAnet_agent.policies import (
            SAScore_and_TerminalRewardPolicy,    # Reward: terminals + dense scoring
            ThermodynamicScaledRewardPolicy,     # Path-dependent feasibility scaling
        )

        # Base reward policy: terminal rewards + SA score for non-terminals
        base_reward = SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=1.0,
            pks_terminal_reward=1.0,
        )

        # Wrap with thermodynamic scaling for path-dependent rewards
        reward_policy = ThermodynamicScaledRewardPolicy(
            base_policy=base_reward,
            feasibility_weight=1.0,
        )

    Pluggable non-terminal scorer::

        # Replace SA score with a custom scorer (e.g., future PKS classifier)
        reward = SAScore_and_TerminalRewardPolicy(
            non_terminal_scorer=lambda node: pks_classifier.predict_proba(node.smiles),
        )

    Priority order:
        1. Sink compound (is_sink_compound=True) → sink_terminal_reward
        2. PKS terminal (is_pks_terminal=True) → pks_terminal_reward
        3. PKS library match (optional check) → pks_terminal_reward
        4. Non-terminal → non_terminal_scorer(node) or SA score reward (dense signal)

    Args:
        sink_terminal_reward: Reward for chemical/biological sink compounds (default 1.0)
        pks_terminal_reward: Reward for PKS-synthesizable compounds (default 1.0)
        sa_max_reward: Cap on SA score reward (default 1.0, no effective cap)
        sa_fallback_reward: Reward when SA score cannot be computed (default 0.0)
        pks_library: Optional set of PKS SMILES for library matching
        non_terminal_scorer: Optional callable (Node → float in [0, 1]) to replace
            SA score for non-terminal nodes. If None, uses SA score. Useful for
            swapping in a PKS binary classifier or other domain-specific scorer.
    """

    def __init__(
        self,
        sink_terminal_reward: float = 1.0,
        pks_terminal_reward: float = 1.0,
        sa_max_reward: float = 1.0,
        sa_fallback_reward: float = 0.0,
        pks_library: Optional[Set[str]] = None,
        non_terminal_scorer: Optional["Callable[[Node], float]"] = None,
    ):
        """
        Args:
            sink_terminal_reward: Reward for sink compounds. Default 1.0.
            pks_terminal_reward: Reward for PKS terminals. Default 1.0.
            sa_max_reward: Cap on SA Score rewards. Default 1.0 (no effective cap
                since SA rewards max at 0.9).
            sa_fallback_reward: Fallback reward when SA score cannot be computed.
                Default 0.0.
            pks_library: Optional set of canonical PKS product SMILES. If provided,
                nodes matching the library will receive pks_terminal_reward even if
                not marked as is_pks_terminal.
            non_terminal_scorer: Optional callable that takes a Node and returns a
                float in [0, 1] as the reward for non-terminal nodes. If None
                (default), uses SA score via ``(10 - SA) / 10``. This allows
                swapping in alternative scorers (e.g., a PKS binary classifier
                that returns probabilities in [0, 1]).
        """
        self.sink_terminal_reward = sink_terminal_reward
        self.pks_terminal_reward = pks_terminal_reward
        self.sa_max_reward = sa_max_reward
        self.sa_fallback_reward = sa_fallback_reward
        self._pks_library = pks_library
        self._non_terminal_scorer = non_terminal_scorer

    def calculate_reward(self, node: "Node", context: Dict[str, Any]) -> float:
        """
        Calculate reward based on terminal status or SA score.

        Priority:
            1. Sink compounds → sink_terminal_reward
            2. PKS terminals → pks_terminal_reward
            3. PKS library match → pks_terminal_reward
            4. Non-terminals → SA score reward

        Args:
            node: The node to compute reward for
            context: Context dictionary, may contain "pks_library" set

        Returns:
            Reward value
        """
        # Priority 1: Sink compounds (chemical/biological building blocks)
        if getattr(node, 'is_sink_compound', False):
            return self.sink_terminal_reward

        # Priority 2: PKS terminals (RetroTide verified)
        if getattr(node, 'is_pks_terminal', False):
            return self.pks_terminal_reward

        # Priority 3: PKS library match (optional)
        pks_library = self._get_pks_library(context)
        if pks_library and self._is_pks_library_match(node, pks_library):
            return self.pks_terminal_reward

        # Default: non-terminal scorer or SA score
        return self._compute_non_terminal_reward(node)

    def _get_pks_library(self, context: Dict[str, Any]) -> Optional[Set[str]]:
        """Get PKS library from init or context."""
        if self._pks_library is not None:
            return self._pks_library
        return context.get("pks_library")

    def _is_pks_library_match(self, node: "Node", pks_library: Set[str]) -> bool:
        """Check if node's SMILES is in PKS library."""
        smiles = getattr(node, 'smiles', None)
        if smiles is None:
            return False
        canonical = _canonicalize_smiles(smiles)
        return canonical in pks_library if canonical else False

    def _compute_non_terminal_reward(self, node: "Node") -> float:
        """Compute reward for non-terminal node using custom scorer or SA score."""
        if self._non_terminal_scorer is not None:
            try:
                return self._non_terminal_scorer(node)
            except Exception:
                return self.sa_fallback_reward
        return self._compute_sa_reward(node)

    def _compute_sa_reward(self, node: "Node") -> float:
        """Compute SA score reward for non-terminal node."""
        mol = self._get_molecule(node)
        if mol is None:
            return self.sa_fallback_reward

        sa_score = _calculate_sa_score(mol)
        if sa_score is None:
            return self.sa_fallback_reward

        return _sa_score_to_reward(sa_score, self.sa_max_reward)

    def _get_molecule(self, node: "Node") -> Optional[Chem.Mol]:
        """Extract RDKit Mol from node."""
        if hasattr(node, 'fragment') and node.fragment is not None:
            return node.fragment
        if hasattr(node, 'smiles') and node.smiles is not None:
            return Chem.MolFromSmiles(node.smiles)
        return None

    @property
    def name(self) -> str:
        scorer_name = "custom" if self._non_terminal_scorer is not None else "SAScore"
        return f"{scorer_name}+Terminal(sink={self.sink_terminal_reward}, pks={self.pks_terminal_reward})"

    def __repr__(self) -> str:
        scorer_info = ", non_terminal_scorer=custom" if self._non_terminal_scorer is not None else ""
        return (
            f"SAScore_and_TerminalRewardPolicy("
            f"sink_terminal_reward={self.sink_terminal_reward}, "
            f"pks_terminal_reward={self.pks_terminal_reward}, "
            f"sa_max_reward={self.sa_max_reward}"
            f"{scorer_info})"
        )
