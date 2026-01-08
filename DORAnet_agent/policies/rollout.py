"""
Rollout policies for DORAnet MCTS.

This module provides concrete implementations of rollout strategies.
Rollouts are performed on child nodes after expansion, following classical MCTS:
    Selection → Expansion → Rollout (on children) → Backpropagation

A rollout simulates from a node to estimate its value, which can involve:
- No simulation (NoOpRolloutPolicy)
- Spawning a sub-agent like RetroTide (SpawnRetroTideOnDatabaseCheck)
- SA Score based rewards (SAScore_and_SpawnRetroTideOnDatabaseCheck)
- Future: Model-based predictions, random playouts, etc.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from rdkit import Chem
from rdkit.Chem import RDConfig

from .base import RolloutPolicy, RolloutResult

# SA Score import setup
_SA_SCORE_PATH = os.path.join(RDConfig.RDContribDir, 'SA_Score')
if _SA_SCORE_PATH not in sys.path:
    sys.path.insert(0, _SA_SCORE_PATH)

try:
    import sascorer  # type: ignore[import-not-found]
    _SA_SCORE_AVAILABLE = True
except ImportError:
    _SA_SCORE_AVAILABLE = False
    sascorer = None

if TYPE_CHECKING:
    from ..node import Node


@lru_cache(maxsize=50000)
def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form, returning None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _calculate_sa_score(mol: Chem.Mol) -> Optional[float]:
    """
    Calculate SA Score for a molecule.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        SA Score (1-10, 1=easy, 10=hard) or None if calculation fails
    """
    if not _SA_SCORE_AVAILABLE or sascorer is None:
        return None
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return None


def _sa_score_to_reward(sa_score: float, max_reward: float = 1.0) -> float:
    """
    Convert SA Score to reward.
    
    Formula: (10 - sa_score) / 10
    This gives range 0.0-0.9 for typical SA scores (1-10).
    
    Args:
        sa_score: SA Score from RDKit (1-10 scale)
        max_reward: Optional cap on the reward (default 1.0, no effective cap)
        
    Returns:
        Reward in range [0.0, min(0.9, max_reward)]
    """
    reward = (10.0 - sa_score) / 10.0
    return min(reward, max_reward)


class NoOpRolloutPolicy(RolloutPolicy):
    """
    No-op rollout policy that returns immediately with zero reward.

    Use this when you want to skip rollouts entirely and rely only on
    immediate reward computation (e.g., sparse terminal rewards).
    """

    def rollout(self, node: "Node", context: Dict[str, Any]) -> RolloutResult:
        """Return immediately with zero reward and no terminal marking."""
        return RolloutResult(reward=0.0, terminal=False)

    @property
    def name(self) -> str:
        return "NoOp"


class SpawnRetroTideOnDatabaseCheck(RolloutPolicy):
    """
    Rollout policy that spawns RetroTide when a node matches the PKS library.

    This implements the classical MCTS rollout phase by:
    1. Checking if the node's SMILES matches the PKS library (database check)
    2. If matched, spawning a RetroTide MCTS to simulate forward PKS design
    3. If RetroTide finds a valid design, marking the node as terminal

    The rollout is performed on each newly expanded child node. If RetroTide
    succeeds, the child is marked as `is_pks_terminal=True` (same as sink
    compounds, it will never be expanded again).

    Flow:
        Expansion creates child → PKS library check → RetroTide rollout →
        Mark terminal if successful → Backpropagate reward
    """

    def __init__(
        self,
        pks_library: Optional[Set[str]] = None,
        retrotide_kwargs: Optional[Dict[str, Any]] = None,
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
    ):
        """
        Args:
            pks_library: Set of canonical PKS product SMILES. If None, will be
                retrieved from context["pks_library"] at runtime.
            retrotide_kwargs: Parameters passed to RetroTide MCTS constructor.
                If None, will be retrieved from context["retrotide_kwargs"].
            success_reward: Reward when RetroTide finds a valid PKS design.
            failure_reward: Reward when RetroTide fails or node doesn't match PKS library.
        """
        self._pks_library = pks_library
        self._retrotide_kwargs = retrotide_kwargs or {}
        self.success_reward = success_reward
        self.failure_reward = failure_reward

        # Track RetroTide availability
        self._retrotide_available = self._check_retrotide_available()

    def _check_retrotide_available(self) -> bool:
        """Check if RetroTide is available for import."""
        try:
            from RetroTide_agent.mcts import MCTS as RetroTideMCTS
            from RetroTide_agent.node import Node as RetroTideNode
            return True
        except ImportError:
            return False

    def _get_pks_library(self, context: Dict[str, Any]) -> Set[str]:
        """Get PKS library from instance or context."""
        if self._pks_library is not None:
            return self._pks_library
        return context.get("pks_library", set())

    def _get_retrotide_kwargs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get RetroTide kwargs from instance or context."""
        kwargs = dict(self._retrotide_kwargs)
        context_kwargs = context.get("retrotide_kwargs", {})
        # Context kwargs override instance kwargs
        kwargs.update(context_kwargs)
        return kwargs

    def _is_pks_match(self, node: "Node", pks_library: Set[str]) -> bool:
        """Check if node's SMILES matches the PKS library."""
        if not pks_library:
            return False
        smiles = node.smiles
        if smiles is None:
            return False
        canonical = _canonicalize_smiles(smiles)
        return canonical is not None and canonical in pks_library

    def _run_retrotide(
        self,
        node: "Node",
        target_molecule: Chem.Mol,
        retrotide_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run RetroTide MCTS to attempt PKS synthesis.

        Args:
            node: The node containing the fragment to synthesize.
            target_molecule: The original target molecule (for RetroTide's bag of graphs).
            retrotide_kwargs: Parameters for RetroTide MCTS.

        Returns:
            Dictionary with RetroTide results:
                - successful: bool
                - num_successful_nodes: int
                - best_score: float
                - total_nodes: int
                - agent: RetroTideMCTS instance (for further inspection)
        """
        from RetroTide_agent.mcts import MCTS as RetroTideMCTS
        from RetroTide_agent.node import Node as RetroTideNode

        # Get the fragment molecule
        fragment_mol = node.fragment
        if fragment_mol is None:
            return {"successful": False, "num_successful_nodes": 0, "best_score": 0.0, "total_nodes": 0}

        fragment_smiles = Chem.MolToSmiles(fragment_mol)
        print(f"[RetroTide Rollout] Spawning RetroTide for: {fragment_smiles}")

        # Create RetroTide root and run MCTS
        root = RetroTideNode(PKS_product=None, PKS_design=None, parent=None, depth=0)
        agent = RetroTideMCTS(
            root=root,
            target_molecule=fragment_mol,  # RetroTide designs PKS to make this fragment
            **retrotide_kwargs,
        )
        agent.run()

        # Extract results
        successful_nodes = getattr(agent, "successful_nodes", set())
        simulated_successes = getattr(agent, "successful_simulated_designs", [])
        num_successful = len(successful_nodes) + len(simulated_successes)

        # Get best score
        best_score = 0.0
        if successful_nodes or simulated_successes:
            best_score = 1.0
        else:
            for n in getattr(agent, "nodes", []):
                if hasattr(n, "value") and n.visits > 0:
                    avg_value = n.value / n.visits
                    best_score = max(best_score, avg_value)

        return {
            "successful": num_successful > 0,
            "num_successful_nodes": num_successful,
            "best_score": best_score,
            "total_nodes": len(getattr(agent, "nodes", [])),
            "agent": agent,
            "target_smiles": fragment_smiles,
        }

    def rollout(self, node: "Node", context: Dict[str, Any]) -> RolloutResult:
        """
        Perform RetroTide rollout if node matches PKS library.

        Args:
            node: The node to perform rollout from (a newly expanded child).
            context: Dictionary containing:
                - pks_library: Set of PKS product SMILES
                - target_molecule: Original synthesis target (RDKit Mol)
                - retrotide_kwargs: Parameters for RetroTide MCTS

        Returns:
            RolloutResult with:
                - reward: success_reward if RetroTide succeeds, failure_reward otherwise
                - terminal: True if RetroTide succeeds (node becomes is_pks_terminal)
                - terminal_type: "pks_terminal" if successful
                - metadata: Contains RetroTide results for traceability
        """
        # Check RetroTide availability
        if not self._retrotide_available:
            print("[RetroTide Rollout] WARNING: RetroTide not available, skipping rollout")
            return RolloutResult(reward=self.failure_reward, terminal=False)

        # Skip if already attempted (prevents duplicate rollouts)
        if getattr(node, "retrotide_attempted", False):
            return RolloutResult(reward=self.failure_reward, terminal=False)

        # Get PKS library and check membership
        pks_library = self._get_pks_library(context)
        if not self._is_pks_match(node, pks_library):
            # Not a PKS match, no rollout needed
            return RolloutResult(reward=self.failure_reward, terminal=False)

        # Node matches PKS library - spawn RetroTide
        print(f"[RetroTide Rollout] Node {node.node_id} matches PKS library, spawning RetroTide")

        # Mark as attempted to prevent duplicate rollouts
        node.retrotide_attempted = True

        # Get target molecule and RetroTide kwargs from context
        target_molecule = context.get("target_molecule")
        if target_molecule is None:
            print("[RetroTide Rollout] WARNING: No target_molecule in context, skipping rollout")
            return RolloutResult(reward=self.failure_reward, terminal=False)

        retrotide_kwargs = self._get_retrotide_kwargs(context)

        # Run RetroTide
        result = self._run_retrotide(node, target_molecule, retrotide_kwargs)

        if result["successful"]:
            print(f"[RetroTide Rollout] SUCCESS! Found {result['num_successful_nodes']} valid PKS designs")
            return RolloutResult(
                reward=self.success_reward,
                terminal=True,
                terminal_type="pks_terminal",
                metadata={
                    "retrotide_successful": True,
                    "retrotide_num_successful_nodes": result["num_successful_nodes"],
                    "retrotide_best_score": result["best_score"],
                    "retrotide_total_nodes": result["total_nodes"],
                    "retrotide_target_smiles": result["target_smiles"],
                    "retrotide_agent": result["agent"],
                },
            )
        else:
            print(f"[RetroTide Rollout] No valid PKS design found (explored {result['total_nodes']} nodes)")
            return RolloutResult(
                reward=self.failure_reward,
                terminal=False,
                metadata={
                    "retrotide_successful": False,
                    "retrotide_total_nodes": result["total_nodes"],
                    "retrotide_target_smiles": result["target_smiles"],
                },
            )

    @property
    def name(self) -> str:
        return f"SpawnRetroTideOnDatabaseCheck(success={self.success_reward})"

    def __repr__(self) -> str:
        return (
            f"SpawnRetroTideOnDatabaseCheck("
            f"success_reward={self.success_reward}, "
            f"failure_reward={self.failure_reward}, "
            f"retrotide_available={self._retrotide_available})"
        )


class SAScore_and_SpawnRetroTideOnDatabaseCheck(RolloutPolicy):
    """
    Rollout policy combining SA Score rewards with RetroTide spawning.
    
    This policy provides dense intermediate rewards based on synthetic 
    accessibility while still spawning RetroTide for PKS library matches.
    
    Reward Logic:
        1. Terminal Nodes (sink compounds): success_reward (default 1.0)
        2. PKS Library Match + RetroTide Success: success_reward (default 1.0)
        3. PKS Library Match + RetroTide Failure: SA Score reward (0.0-0.9)
        4. Non-PKS Nodes: SA Score reward (0.0-0.9)
    
    SA Score Reward Formula:
        reward = (10 - sa_score) / 10
        
    This produces rewards in range [0.0, 0.9] for typical SA scores (1-10).
    Lower SA scores (easier synthesis) → higher rewards.
    
    Example SA Score rewards:
        - SA Score 1.0 (very easy) → reward 0.9
        - SA Score 3.0 (easy) → reward 0.7
        - SA Score 5.0 (moderate) → reward 0.5
        - SA Score 8.0 (hard) → reward 0.2
        - SA Score 10.0 (very hard) → reward 0.0
    
    This provides denser training signals compared to SpawnRetroTideOnDatabaseCheck
    which only rewards successful PKS designs (sparse rewards).
    """

    def __init__(
        self,
        pks_library: Optional[Set[str]] = None,
        retrotide_kwargs: Optional[Dict[str, Any]] = None,
        success_reward: float = 1.0,
        sa_max_reward: float = 1.0,
    ):
        """
        Args:
            pks_library: Set of canonical PKS product SMILES. If None, will be
                retrieved from context["pks_library"] at runtime.
            retrotide_kwargs: Parameters passed to RetroTide MCTS constructor.
                If None, will be retrieved from context["retrotide_kwargs"].
            success_reward: Reward for terminal nodes (sink) or successful 
                RetroTide designs. Default 1.0.
            sa_max_reward: Optional cap on SA Score rewards. Default 1.0 
                (no effective cap since SA rewards max at 0.9).
        """
        self._pks_library = pks_library
        self._retrotide_kwargs = retrotide_kwargs or {}
        self.success_reward = success_reward
        self.sa_max_reward = sa_max_reward

        # Track RetroTide availability
        self._retrotide_available = self._check_retrotide_available()
        
        # Track SA Score availability
        self._sa_score_available = _SA_SCORE_AVAILABLE

    def _check_retrotide_available(self) -> bool:
        """Check if RetroTide is available for import."""
        try:
            from RetroTide_agent.mcts import MCTS as RetroTideMCTS
            from RetroTide_agent.node import Node as RetroTideNode
            return True
        except ImportError:
            return False

    def _get_pks_library(self, context: Dict[str, Any]) -> Set[str]:
        """Get PKS library from instance or context."""
        if self._pks_library is not None:
            return self._pks_library
        return context.get("pks_library", set())

    def _get_retrotide_kwargs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get RetroTide kwargs from instance or context."""
        kwargs = dict(self._retrotide_kwargs)
        context_kwargs = context.get("retrotide_kwargs", {})
        # Context kwargs override instance kwargs
        kwargs.update(context_kwargs)
        return kwargs

    def _is_pks_match(self, node: "Node", pks_library: Set[str]) -> bool:
        """Check if node's SMILES matches the PKS library."""
        if not pks_library:
            return False
        smiles = node.smiles
        if smiles is None:
            return False
        canonical = _canonicalize_smiles(smiles)
        return canonical is not None and canonical in pks_library

    def _get_sa_reward(self, node: "Node") -> float:
        """
        Calculate SA Score reward for a node.
        
        Returns sa_max_reward if SA Score calculation fails.
        """
        if not self._sa_score_available:
            # Fallback if SA Score not available
            return 0.0
            
        # Get molecule from node
        mol = None
        if hasattr(node, 'fragment') and node.fragment is not None:
            mol = node.fragment
        elif hasattr(node, 'smiles') and node.smiles is not None:
            mol = Chem.MolFromSmiles(node.smiles)
            
        if mol is None:
            return 0.0
            
        sa_score = _calculate_sa_score(mol)
        if sa_score is None:
            return 0.0
            
        return _sa_score_to_reward(sa_score, self.sa_max_reward)

    def _run_retrotide(
        self,
        node: "Node",
        target_molecule: Chem.Mol,
        retrotide_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run RetroTide MCTS to attempt PKS synthesis.

        Args:
            node: The node containing the fragment to synthesize.
            target_molecule: The original target molecule (for RetroTide's bag of graphs).
            retrotide_kwargs: Parameters for RetroTide MCTS.

        Returns:
            Dictionary with RetroTide results:
                - successful: bool
                - num_successful_nodes: int
                - best_score: float
                - total_nodes: int
                - agent: RetroTideMCTS instance (for further inspection)
        """
        from RetroTide_agent.mcts import MCTS as RetroTideMCTS
        from RetroTide_agent.node import Node as RetroTideNode

        # Get the fragment molecule
        fragment_mol = node.fragment
        if fragment_mol is None:
            return {"successful": False, "num_successful_nodes": 0, "best_score": 0.0, "total_nodes": 0}

        fragment_smiles = Chem.MolToSmiles(fragment_mol)
        print(f"[RetroTide+SA Rollout] Spawning RetroTide for: {fragment_smiles}")

        # Create RetroTide root and run MCTS
        root = RetroTideNode(PKS_product=None, PKS_design=None, parent=None, depth=0)
        agent = RetroTideMCTS(
            root=root,
            target_molecule=fragment_mol,  # RetroTide designs PKS to make this fragment
            **retrotide_kwargs,
        )
        agent.run()

        # Extract results
        successful_nodes = getattr(agent, "successful_nodes", set())
        simulated_successes = getattr(agent, "successful_simulated_designs", [])
        num_successful = len(successful_nodes) + len(simulated_successes)

        # Get best score
        best_score = 0.0
        if successful_nodes or simulated_successes:
            best_score = 1.0
        else:
            for n in getattr(agent, "nodes", []):
                if hasattr(n, "value") and n.visits > 0:
                    avg_value = n.value / n.visits
                    best_score = max(best_score, avg_value)

        return {
            "successful": num_successful > 0,
            "num_successful_nodes": num_successful,
            "best_score": best_score,
            "total_nodes": len(getattr(agent, "nodes", [])),
            "agent": agent,
            "target_smiles": fragment_smiles,
        }

    def rollout(self, node: "Node", context: Dict[str, Any]) -> RolloutResult:
        """
        Perform rollout with SA Score rewards and RetroTide spawning.

        Logic:
            1. Check if node is terminal (sink) → return success_reward
            2. Compute SA Score reward for the node
            3. If PKS library match:
               a. Spawn RetroTide
               b. If RetroTide succeeds → return success_reward, mark terminal
               c. If RetroTide fails → return SA Score reward (dense signal)
            4. If not PKS match → return SA Score reward (dense signal)

        Args:
            node: The node to perform rollout from (a newly expanded child).
            context: Dictionary containing:
                - pks_library: Set of PKS product SMILES
                - target_molecule: Original synthesis target (RDKit Mol)
                - retrotide_kwargs: Parameters for RetroTide MCTS

        Returns:
            RolloutResult with:
                - reward: success_reward, or SA Score reward (0.0-0.9)
                - terminal: True if RetroTide succeeds
                - terminal_type: "pks_terminal" if successful
                - metadata: Contains SA Score, RetroTide results for traceability
        """
        # Compute SA Score reward upfront (will be used as fallback)
        sa_reward = self._get_sa_reward(node)
        
        # Get molecule for SA score logging
        sa_score = None
        mol = None
        if hasattr(node, 'fragment') and node.fragment is not None:
            mol = node.fragment
        elif hasattr(node, 'smiles') and node.smiles is not None:
            mol = Chem.MolFromSmiles(node.smiles)
        if mol is not None and self._sa_score_available:
            sa_score = _calculate_sa_score(mol)

        # Skip if already attempted (prevents duplicate rollouts)
        if getattr(node, "retrotide_attempted", False):
            return RolloutResult(
                reward=sa_reward,
                terminal=False,
                metadata={"sa_score": sa_score, "sa_reward": sa_reward}
            )

        # Get PKS library and check membership
        pks_library = self._get_pks_library(context)
        is_pks_match = self._is_pks_match(node, pks_library)
        
        if not is_pks_match:
            # Not a PKS match - return SA Score reward as dense signal
            return RolloutResult(
                reward=sa_reward,
                terminal=False,
                metadata={
                    "sa_score": sa_score,
                    "sa_reward": sa_reward,
                    "pks_match": False,
                }
            )

        # Node matches PKS library - attempt RetroTide
        if not self._retrotide_available:
            print("[RetroTide+SA Rollout] WARNING: RetroTide not available, using SA reward")
            return RolloutResult(
                reward=sa_reward,
                terminal=False,
                metadata={
                    "sa_score": sa_score,
                    "sa_reward": sa_reward,
                    "pks_match": True,
                    "retrotide_available": False,
                }
            )

        print(f"[RetroTide+SA Rollout] Node {node.node_id} matches PKS library, spawning RetroTide")

        # Mark as attempted to prevent duplicate rollouts
        node.retrotide_attempted = True

        # Get target molecule from context
        target_molecule = context.get("target_molecule")
        if target_molecule is None:
            print("[RetroTide+SA Rollout] WARNING: No target_molecule in context, using SA reward")
            return RolloutResult(
                reward=sa_reward,
                terminal=False,
                metadata={
                    "sa_score": sa_score,
                    "sa_reward": sa_reward,
                    "pks_match": True,
                    "retrotide_skipped": "no_target_molecule",
                }
            )

        retrotide_kwargs = self._get_retrotide_kwargs(context)

        # Run RetroTide
        result = self._run_retrotide(node, target_molecule, retrotide_kwargs)

        if result["successful"]:
            print(f"[RetroTide+SA Rollout] SUCCESS! Found {result['num_successful_nodes']} valid PKS designs")
            return RolloutResult(
                reward=self.success_reward,
                terminal=True,
                terminal_type="pks_terminal",
                metadata={
                    "sa_score": sa_score,
                    "sa_reward": sa_reward,
                    "pks_match": True,
                    "retrotide_successful": True,
                    "retrotide_num_successful_nodes": result["num_successful_nodes"],
                    "retrotide_best_score": result["best_score"],
                    "retrotide_total_nodes": result["total_nodes"],
                    "retrotide_target_smiles": result["target_smiles"],
                    "retrotide_agent": result["agent"],
                },
            )
        else:
            # RetroTide failed - return SA Score reward as dense signal
            print(f"[RetroTide+SA Rollout] No valid PKS design found, returning SA reward: {sa_reward:.3f}")
            return RolloutResult(
                reward=sa_reward,
                terminal=False,
                metadata={
                    "sa_score": sa_score,
                    "sa_reward": sa_reward,
                    "pks_match": True,
                    "retrotide_successful": False,
                    "retrotide_total_nodes": result["total_nodes"],
                    "retrotide_target_smiles": result["target_smiles"],
                },
            )

    @property
    def name(self) -> str:
        return f"SAScore_and_SpawnRetroTideOnDatabaseCheck(success={self.success_reward}, sa_max={self.sa_max_reward})"

    def __repr__(self) -> str:
        return (
            f"SAScore_and_SpawnRetroTideOnDatabaseCheck("
            f"success_reward={self.success_reward}, "
            f"sa_max_reward={self.sa_max_reward}, "
            f"retrotide_available={self._retrotide_available}, "
            f"sa_score_available={self._sa_score_available})"
        )
