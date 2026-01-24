"""
Reward policies for DORAnet MCTS.

This module provides concrete implementations of reward computation strategies.
Rewards are computed based on node properties without simulation.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

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
