
"""
Reward policies for DORAnet MCTS.

This module provides concrete implementations of reward computation strategies.
Rewards are computed based on node properties without simulation.
"""

from __future__ import annotations

import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem

from .base import RewardPolicy

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


def _generate_morgan_fingerprint(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[DataStructs.ExplicitBitVect]:
    """
    Generate Morgan (ECFP-like) fingerprint for a molecule.

    Args:
        mol: RDKit Mol object
        radius: Fingerprint radius (default 2, equivalent to ECFP4)
        n_bits: Number of bits in fingerprint (default 2048)

    Returns:
        Morgan fingerprint as ExplicitBitVect, or None if generation fails
    """
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def _calculate_tanimoto_similarity(
    query_fp: DataStructs.ExplicitBitVect,
    reference_fp: DataStructs.ExplicitBitVect,
) -> float:
    """Calculate Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(query_fp, reference_fp)


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


class PKSSimilarityRewardPolicy(RewardPolicy):
    """
    Reward policy based purely on PKS Tanimoto similarity.

    Uses Morgan fingerprint similarity to PKS building blocks as the reward
    signal for ALL nodes, including sink compounds. This guides MCTS toward
    PKS-compatible chemical space rather than generic purchasable building blocks.

    Key behavior:
    - Sink compounds are still marked terminal (stop expansion)
    - But their reward is PKS similarity, not a flat 1.0
    - Non-terminal nodes also get PKS similarity rewards (dense signal)
    - Optional exponential scaling: reward = similarity ^ exponent

    Args:
        pks_building_blocks_path: Path to PKS SMILES file (one per line)
        similarity_threshold: Early termination threshold (default 0.95)
        fingerprint_radius: Morgan fingerprint radius (default 2 = ECFP4)
        fingerprint_bits: Number of fingerprint bits (default 2048)
        similarity_exponent: Scaling exponent (default 2.0, squared rewards)
        project_root: Project root for resolving relative paths
    """

    # Default path relative to project root
    DEFAULT_PKS_BUILDING_BLOCKS_PATH = "data/processed/expanded_PKS_SMILES_V3.txt"

    def __init__(
        self,
        pks_building_blocks_path: Optional[str] = None,
        similarity_threshold: float = 0.95,
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        similarity_exponent: float = 2.0,
        project_root: Optional[str] = None,
    ):
        """
        Args:
            pks_building_blocks_path: Path to file containing PKS building block
                SMILES (one per line). If None, uses default path relative to
                project root. Default: "data/processed/expanded_PKS_SMILES_V3.txt"
            similarity_threshold: Early termination threshold. Stop computing
                similarities if we find a score >= this value. Default 0.95.
            fingerprint_radius: Radius for Morgan fingerprints. Default 2
                (equivalent to ECFP4).
            fingerprint_bits: Number of bits for Morgan fingerprints. Default 2048.
            similarity_exponent: Exponent for scaling similarity rewards.
                Applied as: reward = similarity ^ exponent. Values > 1.0 penalize
                low similarities more heavily, < 1.0 boost them. Default 2.0
                (squared). Examples with exponent=2.0: 0.9→0.81, 0.7→0.49, 0.5→0.25.
            project_root: Root directory of the project for resolving relative
                paths. If None, attempts to auto-detect from this file's location.
        """
        self.similarity_threshold = similarity_threshold
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.similarity_exponent = similarity_exponent

        # Determine project root
        if project_root is not None:
            self._project_root = Path(project_root)
        else:
            # Auto-detect: this file is in DORAnet_agent/policies/reward.py
            # Project root is 2 levels up
            self._project_root = Path(__file__).parent.parent.parent

        # Resolve PKS building blocks path
        if pks_building_blocks_path is not None:
            pks_path = Path(pks_building_blocks_path)
            if not pks_path.is_absolute():
                pks_path = self._project_root / pks_path
        else:
            pks_path = self._project_root / self.DEFAULT_PKS_BUILDING_BLOCKS_PATH

        self._pks_building_blocks_path = pks_path

        # Store pre-computed fingerprints: List[Tuple[str, ExplicitBitVect]]
        self._pks_building_blocks: List[Tuple[str, DataStructs.ExplicitBitVect]] = []
        self._load_pks_fingerprints()

        print(f"[PKSSimilarityRewardPolicy] Loaded {len(self._pks_building_blocks)} PKS building blocks")
        print(f"[PKSSimilarityRewardPolicy] Similarity threshold (early termination): {self.similarity_threshold}")
        print(f"[PKSSimilarityRewardPolicy] Similarity exponent: {self.similarity_exponent}")
        print(f"[PKSSimilarityRewardPolicy] Fingerprint: Morgan r={self.fingerprint_radius}, bits={self.fingerprint_bits}")

    def _get_cache_path(self) -> Path:
        """Get path to fingerprint cache file."""
        return self._pks_building_blocks_path.with_suffix('.fingerprints.pkl')

    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is newer than source SMILES file."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False
        return cache_path.stat().st_mtime > self._pks_building_blocks_path.stat().st_mtime

    def _load_fingerprints_from_cache(self) -> bool:
        """
        Load fingerprints from cache. Returns True if successful.

        Validates that cache format and fingerprint parameters match current settings.
        """
        try:
            with open(self._get_cache_path(), 'rb') as f:
                cached_data = pickle.load(f)
            # Validate cache format and fingerprint params match
            if cached_data.get('radius') != self.fingerprint_radius:
                return False
            if cached_data.get('bits') != self.fingerprint_bits:
                return False
            self._pks_building_blocks = cached_data['fingerprints']
            return True
        except Exception:
            return False

    def _save_fingerprints_to_cache(self) -> None:
        """Save fingerprints to cache file."""
        try:
            cached_data = {
                'radius': self.fingerprint_radius,
                'bits': self.fingerprint_bits,
                'fingerprints': self._pks_building_blocks,
            }
            with open(self._get_cache_path(), 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            print(f"[PKSSimilarityRewardPolicy] WARNING: Failed to save cache: {e}")

    def _load_pks_fingerprints(self) -> None:
        """Load PKS building blocks with pre-computed Morgan fingerprints."""
        if not self._pks_building_blocks_path.exists():
            print(f"[PKSSimilarityRewardPolicy] WARNING: PKS building blocks file not found: "
                  f"{self._pks_building_blocks_path}")
            return

        # Try to load from cache first
        if self._is_cache_valid():
            if self._load_fingerprints_from_cache():
                print(f"[PKSSimilarityRewardPolicy] Loaded fingerprints from cache: {self._get_cache_path()}")
                return

        # Load from SMILES file and compute fingerprints
        loaded = 0
        failed = 0

        with open(self._pks_building_blocks_path, 'r') as f:
            for line in f:
                smiles = line.strip()
                if not smiles:
                    continue

                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        fp = _generate_morgan_fingerprint(
                            mol,
                            radius=self.fingerprint_radius,
                            n_bits=self.fingerprint_bits,
                        )
                        if fp is not None:
                            self._pks_building_blocks.append((smiles, fp))
                            loaded += 1
                        else:
                            failed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

        if failed > 0:
            print(f"[PKSSimilarityRewardPolicy] WARNING: Failed to parse {failed} SMILES")

        # Save to cache for future runs
        self._save_fingerprints_to_cache()

    def _compute_pks_similarity(self, node: "Node") -> Tuple[float, Dict[str, Any]]:
        """
        Compute Tanimoto fingerprint similarity to PKS building blocks.

        Args:
            node: The node to compute similarity for

        Returns:
            Tuple of (similarity_score, metadata_dict) where:
                - similarity_score: Max Tanimoto similarity in [0.0, 1.0]
                - metadata_dict: Contains computation details for debugging
        """
        metadata: Dict[str, Any] = {
            "pks_building_blocks_checked": 0,
            "best_similarity": 0.0,
        }

        if not self._pks_building_blocks:
            return 0.0, metadata

        # Get molecule from node
        mol = None
        if hasattr(node, 'fragment') and node.fragment is not None:
            mol = node.fragment
        elif hasattr(node, 'smiles') and node.smiles is not None:
            mol = Chem.MolFromSmiles(node.smiles)

        if mol is None:
            return 0.0, metadata

        # Generate fingerprint for query molecule
        query_fp = _generate_morgan_fingerprint(
            mol,
            radius=self.fingerprint_radius,
            n_bits=self.fingerprint_bits,
        )

        if query_fp is None:
            return 0.0, metadata

        best_similarity = 0.0

        for pks_smiles, pks_fp in self._pks_building_blocks:
            metadata["pks_building_blocks_checked"] += 1

            # Calculate Tanimoto similarity (O(1) operation)
            similarity = _calculate_tanimoto_similarity(query_fp, pks_fp)

            if similarity > best_similarity:
                best_similarity = similarity

            # Early termination if we found a very good match
            if best_similarity >= self.similarity_threshold:
                break

        metadata["best_similarity"] = best_similarity
        return best_similarity, metadata

    def calculate_reward(self, node: "Node", context: Dict[str, Any]) -> float:
        """
        Calculate PKS similarity reward for any node.

        Returns similarity^exponent where similarity is max Tanimoto
        similarity to any PKS building block.

        Args:
            node: The node to compute reward for
            context: Context dictionary (currently unused)

        Returns:
            Reward value in [0.0, 1.0], computed as similarity^exponent
        """
        raw_similarity, metadata = self._compute_pks_similarity(node)

        # Apply exponential scaling
        if self.similarity_exponent != 1.0:
            return raw_similarity ** self.similarity_exponent

        return raw_similarity

    @property
    def name(self) -> str:
        return f"PKSSimilarity(exp={self.similarity_exponent})"

    def __repr__(self) -> str:
        return (
            f"PKSSimilarityRewardPolicy("
            f"exponent={self.similarity_exponent}, "
            f"threshold={self.similarity_threshold}, "
            f"radius={self.fingerprint_radius}, "
            f"bits={self.fingerprint_bits}, "
            f"pks_building_blocks={len(self._pks_building_blocks)})"
        )


class SAScore_and_TerminalRewardPolicy(RewardPolicy):
    """
    Unified reward policy combining terminal rewards with SA score for non-terminals.

    This policy provides:
    - Full terminal rewards for sink compounds (chemical/biological building blocks)
    - Full terminal rewards for PKS-synthesizable compounds (RetroTide verified)
    - SA score-based dense rewards for all other compounds

    This cleanly separates reward computation from rollout behavior. Use with
    SpawnRetroTideOnDatabaseCheck rollout policy for the recommended architecture:

    Recommended clean setup (rollout + reward separation)::

        from DORAnet_agent.policies import (
            SpawnRetroTideOnDatabaseCheck,       # Rollout: PKS matching + RetroTide
            SAScore_and_TerminalRewardPolicy,    # Reward: terminals + SA score
            ThermodynamicScaledRewardPolicy,     # Optional: thermodynamic scaling
        )

        # Rollout policy: handles PKS matching and RetroTide spawning only
        rollout_policy = SpawnRetroTideOnDatabaseCheck(
            success_reward=1.0,
            retrotide_kwargs={"max_depth": 6, "total_iterations": 100},
        )

        # Reward policy: terminal rewards + SA score for non-terminals
        base_reward = SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=1.0,
            pks_terminal_reward=1.0,
        )

        # Optional: wrap with thermodynamic scaling
        reward_policy = ThermodynamicScaledRewardPolicy(
            base_policy=base_reward,
            feasibility_weight=0.8,
        )

    Priority order:
        1. Sink compound (is_sink_compound=True) → sink_terminal_reward
        2. PKS terminal (is_pks_terminal=True) → pks_terminal_reward
        3. PKS library match (optional check) → pks_terminal_reward
        4. Non-terminal → SA score reward (dense signal)

    Args:
        sink_terminal_reward: Reward for chemical/biological sink compounds (default 1.0)
        pks_terminal_reward: Reward for PKS-synthesizable compounds (default 1.0)
        sa_max_reward: Cap on SA score reward (default 1.0, no effective cap)
        sa_fallback_reward: Reward when SA score cannot be computed (default 0.0)
        pks_library: Optional set of PKS SMILES for library matching
    """

    def __init__(
        self,
        sink_terminal_reward: float = 1.0,
        pks_terminal_reward: float = 1.0,
        sa_max_reward: float = 1.0,
        sa_fallback_reward: float = 0.0,
        pks_library: Optional[Set[str]] = None,
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
        """
        self.sink_terminal_reward = sink_terminal_reward
        self.pks_terminal_reward = pks_terminal_reward
        self.sa_max_reward = sa_max_reward
        self.sa_fallback_reward = sa_fallback_reward
        self._pks_library = pks_library

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

        # Default: SA score for non-terminals
        return self._compute_sa_reward(node)

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
        return f"SAScore+Terminal(sink={self.sink_terminal_reward}, pks={self.pks_terminal_reward})"

    def __repr__(self) -> str:
        return (
            f"SAScore_and_TerminalRewardPolicy("
            f"sink_terminal_reward={self.sink_terminal_reward}, "
            f"pks_terminal_reward={self.pks_terminal_reward}, "
            f"sa_max_reward={self.sa_max_reward})"
        )
