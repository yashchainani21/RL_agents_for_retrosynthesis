"""
Terminal detection policies for DORAnet MCTS.

This module provides concrete implementations of post-expansion terminal
detection. After MCTS expands a node and creates children, each child is
checked to determine whether it should be marked as terminal (i.e., no
further expansion is needed because a synthesis route has been verified).

Available detectors:
    - NoOpTerminalDetector: Never marks nodes as terminal (fastest, no RetroTide)
    - VerifyWithRetroTide: Spawns RetroTide for PKS library matches
    - SimilarityGuidedRetroTideDetector: Spawns RetroTide for exact matches
      or high-similarity fragments

These replace the legacy RolloutPolicy classes. Unlike rollout policies,
terminal detectors do NOT compute rewards — reward computation is exclusively
handled by RewardPolicy.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem, DataStructs

from .base import TerminalDetector, TerminalDetectionResult
from .utils import (
    canonicalize_smiles,
    generate_morgan_fingerprint,
    calculate_tanimoto_similarity,
    calculate_mcs_similarity_without_stereo,
)

if TYPE_CHECKING:
    from ..node import Node


class NoOpTerminalDetector(TerminalDetector):
    """
    No-op terminal detector that never marks nodes as terminal.

    Use this when you want to skip RetroTide verification entirely and
    rely only on sink compound matching for terminal detection.
    """

    def detect(self, node: "Node", context: Dict[str, Any]) -> TerminalDetectionResult:
        """Return immediately — node is not terminal."""
        return TerminalDetectionResult(terminal=False)

    @property
    def name(self) -> str:
        return "NoOp"


class VerifyWithRetroTide(TerminalDetector):
    """
    Terminal detector that spawns RetroTide when a node matches the PKS library.

    After expansion, each child node is checked against the PKS library.
    If a match is found, RetroTide MCTS is spawned to verify that a PKS
    assembly line can synthesize the fragment. If RetroTide succeeds, the
    node is marked as terminal (is_pks_terminal=True).

    This replaces the legacy SpawnRetroTideOnDatabaseCheck rollout policy.
    Unlike the rollout policy, this detector does NOT compute rewards —
    reward computation is handled entirely by the configured RewardPolicy.

    Flow:
        Expansion creates child → PKS library check → RetroTide verification →
        Mark terminal if successful → RewardPolicy computes reward → Backpropagate
    """

    def __init__(
        self,
        pks_library: Optional[Set[str]] = None,
        retrotide_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            pks_library: Set of canonical PKS product SMILES. If None, will be
                retrieved from context["pks_library"] at runtime.
            retrotide_kwargs: Parameters passed to RetroTide MCTS constructor.
                If None, will be retrieved from context["retrotide_kwargs"].
        """
        self._pks_library = pks_library
        self._retrotide_kwargs = retrotide_kwargs or {}

        # Track RetroTide availability
        self._retrotide_available = self._check_retrotide_available()

    def _check_retrotide_available(self) -> bool:
        """Check if RetroTide is available for import."""
        try:
            from RetroTide_agent.mcts import MCTS as RetroTideMCTS  # noqa: F401
            from RetroTide_agent.node import Node as RetroTideNode  # noqa: F401
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
        canonical = canonicalize_smiles(smiles)
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
                - target_smiles: str
        """
        from RetroTide_agent.mcts import MCTS as RetroTideMCTS
        from RetroTide_agent.node import Node as RetroTideNode

        # Get the fragment molecule
        fragment_mol = node.fragment
        if fragment_mol is None:
            return {
                "successful": False,
                "num_successful_nodes": 0,
                "best_score": 0.0,
                "total_nodes": 0,
            }

        fragment_smiles = Chem.MolToSmiles(fragment_mol)
        print(f"[VerifyWithRetroTide] Spawning RetroTide for: {fragment_smiles}")

        # Create RetroTide root and run MCTS
        root = RetroTideNode(PKS_product=None, PKS_design=None, parent=None, depth=0)
        agent = RetroTideMCTS(
            root=root,
            target_molecule=fragment_mol,
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

    def detect(self, node: "Node", context: Dict[str, Any]) -> TerminalDetectionResult:
        """
        Check if node matches PKS library and verify with RetroTide.

        Args:
            node: The node to check (a newly expanded child).
            context: Dictionary containing:
                - pks_library: Set of PKS product SMILES
                - target_molecule: Original synthesis target (RDKit Mol)
                - retrotide_kwargs: Parameters for RetroTide MCTS

        Returns:
            TerminalDetectionResult with:
                - terminal: True if RetroTide succeeds
                - terminal_type: "pks_terminal" if successful
                - metadata: Contains RetroTide results for traceability
        """
        # Check RetroTide availability
        if not self._retrotide_available:
            print("[VerifyWithRetroTide] WARNING: RetroTide not available, skipping")
            return TerminalDetectionResult(terminal=False)

        # Skip if already attempted (prevents duplicate verifications)
        if getattr(node, "retrotide_attempted", False):
            return TerminalDetectionResult(terminal=False)

        # Get PKS library and check membership
        pks_library = self._get_pks_library(context)
        if not self._is_pks_match(node, pks_library):
            # Not a PKS match, no verification needed
            return TerminalDetectionResult(terminal=False)

        # Node matches PKS library — spawn RetroTide
        print(f"[VerifyWithRetroTide] Node {node.node_id} matches PKS library, spawning RetroTide")

        # Mark as attempted to prevent duplicate verifications
        node.retrotide_attempted = True

        # Get target molecule and RetroTide kwargs from context
        target_molecule = context.get("target_molecule")
        if target_molecule is None:
            print("[VerifyWithRetroTide] WARNING: No target_molecule in context, skipping")
            return TerminalDetectionResult(terminal=False)

        retrotide_kwargs = self._get_retrotide_kwargs(context)

        # Run RetroTide
        result = self._run_retrotide(node, target_molecule, retrotide_kwargs)

        if result["successful"]:
            print(f"[VerifyWithRetroTide] SUCCESS! Found {result['num_successful_nodes']} valid PKS designs")
            return TerminalDetectionResult(
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
            print(f"[VerifyWithRetroTide] No valid PKS design found "
                  f"(explored {result['total_nodes']} nodes)")
            return TerminalDetectionResult(
                terminal=False,
                metadata={
                    "retrotide_successful": False,
                    "retrotide_total_nodes": result["total_nodes"],
                    "retrotide_target_smiles": result["target_smiles"],
                },
            )

    @property
    def name(self) -> str:
        return "VerifyWithRetroTide"

    def __repr__(self) -> str:
        return (
            f"VerifyWithRetroTide("
            f"retrotide_available={self._retrotide_available})"
        )


class SimilarityGuidedRetroTideDetector(TerminalDetector):
    """
    Terminal detector that spawns RetroTide for exact PKS matches or
    high-similarity fragments.

    Extends VerifyWithRetroTide with similarity-based triggering: RetroTide
    is spawned not only for exact PKS library matches but also for fragments
    with high structural similarity (Tanimoto or MCS) to known PKS building
    blocks.

    This replaces the legacy PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck
    rollout policy. Unlike the rollout policy, this detector does NOT compute
    rewards — reward computation (including similarity-based dense rewards)
    is handled by the configured RewardPolicy (e.g., PKSSimilarityRewardPolicy).

    Similarity Methods:
        - "tanimoto" (default): Fast Morgan fingerprint-based similarity
        - "mcs": Maximum Common Substructure without stereochemistry matching

    Usage:
        detector = SimilarityGuidedRetroTideDetector(
            retrotide_spawn_threshold=0.9,  # Spawn RetroTide at >= 0.9 similarity
            similarity_method="tanimoto",
        )
    """

    # Default path relative to project root
    DEFAULT_PKS_BUILDING_BLOCKS_PATH = "data/processed/expanded_PKS_SMILES_V3.txt"

    def __init__(
        self,
        pks_building_blocks_path: Optional[str] = None,
        pks_library: Optional[Set[str]] = None,
        retrotide_kwargs: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.9,
        similarity_method: str = "tanimoto",
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        mcs_timeout: float = 1.0,
        atom_count_tolerance: float = 0.5,
        project_root: Optional[str] = None,
        retrotide_spawn_threshold: float = 0.9,
    ):
        """
        Args:
            pks_building_blocks_path: Path to file containing PKS building block
                SMILES (one per line). If None, uses default path relative to
                project root.
            pks_library: Set of canonical PKS product SMILES for exact matching.
                If None, retrieved from context["pks_library"].
            retrotide_kwargs: Parameters passed to RetroTide MCTS constructor.
                If None, retrieved from context["retrotide_kwargs"].
            similarity_threshold: Early termination threshold for similarity
                computation. Default 0.9.
            similarity_method: Method for computing similarity ("tanimoto" or "mcs").
            fingerprint_radius: Radius for Morgan fingerprints (default 2, ECFP4).
            fingerprint_bits: Number of bits for Morgan fingerprints (default 2048).
            mcs_timeout: Timeout in seconds for each MCS calculation (default 1.0).
            atom_count_tolerance: Fraction tolerance for atom count pre-filtering
                in MCS method (default 0.5).
            project_root: Root directory for resolving relative paths.
            retrotide_spawn_threshold: Minimum similarity to trigger RetroTide.
                Default 0.9. Set to 1.0 for exact-match-only behavior.
        """
        if similarity_method not in ("tanimoto", "mcs"):
            raise ValueError(
                f"Invalid similarity_method: {similarity_method}. "
                "Must be 'tanimoto' or 'mcs'."
            )

        self._pks_library = pks_library
        self._retrotide_kwargs = retrotide_kwargs or {}
        self.similarity_threshold = similarity_threshold
        self.similarity_method = similarity_method
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.mcs_timeout = mcs_timeout
        self.atom_count_tolerance = atom_count_tolerance
        self.retrotide_spawn_threshold = retrotide_spawn_threshold

        # Track RetroTide availability
        self._retrotide_available = self._check_retrotide_available()

        # Determine project root
        if project_root is not None:
            self._project_root = Path(project_root)
        else:
            self._project_root = Path(__file__).parent.parent.parent

        # Resolve PKS building blocks path
        if pks_building_blocks_path is not None:
            pks_path = Path(pks_building_blocks_path)
            if not pks_path.is_absolute():
                pks_path = self._project_root / pks_path
        else:
            pks_path = self._project_root / self.DEFAULT_PKS_BUILDING_BLOCKS_PATH

        self._pks_building_blocks_path = pks_path

        # Load and pre-parse PKS building blocks
        self._pks_building_blocks: Union[
            List[Tuple[str, DataStructs.ExplicitBitVect]],
            List[Tuple[Chem.Mol, int]],
        ] = []
        self._load_pks_building_blocks()

        print(f"[SimilarityGuidedRetroTide] Loaded {len(self._pks_building_blocks)} PKS building blocks")
        print(f"[SimilarityGuidedRetroTide] Similarity method: {self.similarity_method}")
        print(f"[SimilarityGuidedRetroTide] RetroTide spawn threshold: {self.retrotide_spawn_threshold}")

    # --- PKS building block loading (same as legacy policy) ---

    def _load_pks_building_blocks(self) -> None:
        """Load PKS building blocks from file and pre-parse."""
        if not self._pks_building_blocks_path.exists():
            print(f"[SimilarityGuidedRetroTide] WARNING: PKS building blocks file not found: "
                  f"{self._pks_building_blocks_path}")
            return

        if self.similarity_method == "tanimoto":
            self._load_pks_building_blocks_tanimoto()
        else:
            self._load_pks_building_blocks_mcs()

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
        """Load fingerprints from cache. Returns True if successful."""
        try:
            with open(self._get_cache_path(), 'rb') as f:
                cached_data = pickle.load(f)
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
            print(f"[SimilarityGuidedRetroTide] WARNING: Failed to save cache: {e}")

    def _load_pks_building_blocks_tanimoto(self) -> None:
        """Load PKS building blocks with pre-computed Morgan fingerprints."""
        if self._is_cache_valid():
            if self._load_fingerprints_from_cache():
                print(f"[SimilarityGuidedRetroTide] Loaded fingerprints from cache: "
                      f"{self._get_cache_path()}")
                return

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
                        fp = generate_morgan_fingerprint(
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
            print(f"[SimilarityGuidedRetroTide] WARNING: Failed to parse {failed} SMILES")

        self._save_fingerprints_to_cache()

    def _load_pks_building_blocks_mcs(self) -> None:
        """Load PKS building blocks with atom counts for MCS comparison."""
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
                        num_atoms = mol.GetNumAtoms()
                        self._pks_building_blocks.append((mol, num_atoms))
                        loaded += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

        if failed > 0:
            print(f"[SimilarityGuidedRetroTide] WARNING: Failed to parse {failed} SMILES")

    # --- RetroTide helpers ---

    def _check_retrotide_available(self) -> bool:
        """Check if RetroTide is available for import."""
        try:
            from RetroTide_agent.mcts import MCTS as RetroTideMCTS  # noqa: F401
            from RetroTide_agent.node import Node as RetroTideNode  # noqa: F401
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
        kwargs.update(context_kwargs)
        return kwargs

    def _is_pks_match(self, node: "Node", pks_library: Set[str]) -> bool:
        """Check if node's SMILES matches the PKS library."""
        if not pks_library:
            return False
        smiles = node.smiles
        if smiles is None:
            return False
        canonical = canonicalize_smiles(smiles)
        return canonical is not None and canonical in pks_library

    def _get_best_pks_similarity(self, node: "Node") -> Tuple[float, Dict[str, Any]]:
        """
        Compute best PKS building block similarity for a node.

        Routes to the appropriate method based on self.similarity_method.

        Returns:
            Tuple of (best_similarity, metadata_dict)
        """
        if self.similarity_method == "tanimoto":
            return self._compute_tanimoto_similarity(node)
        else:
            return self._compute_mcs_similarity(node)

    def _compute_tanimoto_similarity(self, node: "Node") -> Tuple[float, Dict[str, Any]]:
        """Compute Tanimoto fingerprint similarity to PKS building blocks."""
        metadata: Dict[str, Any] = {
            "similarity_method": "tanimoto",
            "pks_building_blocks_checked": 0,
            "best_similarity": 0.0,
        }

        if not self._pks_building_blocks:
            return 0.0, metadata

        mol = None
        if hasattr(node, 'fragment') and node.fragment is not None:
            mol = node.fragment
        elif hasattr(node, 'smiles') and node.smiles is not None:
            mol = Chem.MolFromSmiles(node.smiles)

        if mol is None:
            return 0.0, metadata

        query_fp = generate_morgan_fingerprint(
            mol,
            radius=self.fingerprint_radius,
            n_bits=self.fingerprint_bits,
        )
        if query_fp is None:
            return 0.0, metadata

        best_similarity = 0.0
        for _pks_smiles, pks_fp in self._pks_building_blocks:
            metadata["pks_building_blocks_checked"] += 1
            similarity = calculate_tanimoto_similarity(query_fp, pks_fp)
            if similarity > best_similarity:
                best_similarity = similarity
            if best_similarity >= self.similarity_threshold:
                break

        metadata["best_similarity"] = best_similarity
        return best_similarity, metadata

    def _compute_mcs_similarity(self, node: "Node") -> Tuple[float, Dict[str, Any]]:
        """Compute MCS-based similarity to PKS building blocks."""
        metadata: Dict[str, Any] = {
            "similarity_method": "mcs",
            "pks_building_blocks_checked": 0,
            "pks_building_blocks_skipped_size": 0,
            "pks_building_blocks_skipped_timeout": 0,
            "best_similarity": 0.0,
        }

        if not self._pks_building_blocks:
            return 0.0, metadata

        mol = None
        if hasattr(node, 'fragment') and node.fragment is not None:
            mol = node.fragment
        elif hasattr(node, 'smiles') and node.smiles is not None:
            mol = Chem.MolFromSmiles(node.smiles)

        if mol is None:
            return 0.0, metadata

        query_atoms = mol.GetNumAtoms()
        min_atoms = int(query_atoms * (1.0 - self.atom_count_tolerance))
        max_atoms = int(query_atoms * (1.0 + self.atom_count_tolerance))

        best_similarity = 0.0
        for pks_mol, pks_atoms in self._pks_building_blocks:
            if pks_atoms < min_atoms or pks_atoms > max_atoms:
                metadata["pks_building_blocks_skipped_size"] += 1
                continue

            metadata["pks_building_blocks_checked"] += 1
            similarity = calculate_mcs_similarity_without_stereo(
                mol, pks_mol, timeout=self.mcs_timeout,
            )
            if similarity is None:
                metadata["pks_building_blocks_skipped_timeout"] += 1
                continue

            if similarity > best_similarity:
                best_similarity = similarity
            if best_similarity >= self.similarity_threshold:
                break

        metadata["best_similarity"] = best_similarity
        return best_similarity, metadata

    def _run_retrotide(
        self,
        node: "Node",
        target_molecule: Chem.Mol,
        retrotide_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run RetroTide MCTS to attempt PKS synthesis."""
        from RetroTide_agent.mcts import MCTS as RetroTideMCTS
        from RetroTide_agent.node import Node as RetroTideNode

        fragment_mol = node.fragment
        if fragment_mol is None:
            return {
                "successful": False,
                "num_successful_nodes": 0,
                "best_score": 0.0,
                "total_nodes": 0,
            }

        fragment_smiles = Chem.MolToSmiles(fragment_mol)
        print(f"[SimilarityGuidedRetroTide] Spawning RetroTide for: {fragment_smiles}")

        root = RetroTideNode(PKS_product=None, PKS_design=None, parent=None, depth=0)
        agent = RetroTideMCTS(
            root=root,
            target_molecule=fragment_mol,
            **retrotide_kwargs,
        )
        agent.run()

        successful_nodes = getattr(agent, "successful_nodes", set())
        simulated_successes = getattr(agent, "successful_simulated_designs", [])
        num_successful = len(successful_nodes) + len(simulated_successes)

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

    # --- Main detection logic ---

    def detect(self, node: "Node", context: Dict[str, Any]) -> TerminalDetectionResult:
        """
        Check if node qualifies for RetroTide verification via exact match or similarity.

        Logic:
            1. Check exact PKS library match
            2. If not exact match, compute PKS building block similarity
            3. If similarity >= retrotide_spawn_threshold OR exact match:
               → Spawn RetroTide to verify synthesis
            4. Otherwise → not terminal

        Args:
            node: The node to check (a newly expanded child).
            context: Dictionary with pks_library, target_molecule, retrotide_kwargs.

        Returns:
            TerminalDetectionResult with terminal status and metadata.
        """
        # Skip if already attempted
        if getattr(node, "retrotide_attempted", False):
            return TerminalDetectionResult(terminal=False)

        # Check exact PKS library match
        pks_library = self._get_pks_library(context)
        is_exact_pks_match = self._is_pks_match(node, pks_library)

        # Compute similarity to PKS building blocks
        best_similarity, sim_metadata = self._get_best_pks_similarity(node)
        is_high_similarity = best_similarity >= self.retrotide_spawn_threshold

        should_spawn = is_exact_pks_match or is_high_similarity

        if not should_spawn:
            return TerminalDetectionResult(
                terminal=False,
                metadata={
                    "pks_match": False,
                    "best_similarity": best_similarity,
                    "retrotide_spawn_threshold": self.retrotide_spawn_threshold,
                    **sim_metadata,
                },
            )

        # Qualifies for RetroTide verification
        match_type = "exact" if is_exact_pks_match else "high_similarity"

        if not self._retrotide_available:
            print("[SimilarityGuidedRetroTide] WARNING: RetroTide not available, skipping")
            return TerminalDetectionResult(
                terminal=False,
                metadata={
                    "pks_match": is_exact_pks_match,
                    "best_similarity": best_similarity,
                    "match_type": match_type,
                    "retrotide_available": False,
                    **sim_metadata,
                },
            )

        print(f"[SimilarityGuidedRetroTide] Node {node.node_id} qualifies for RetroTide "
              f"(match_type={match_type}, similarity={best_similarity:.3f})")

        # Mark as attempted
        node.retrotide_attempted = True

        target_molecule = context.get("target_molecule")
        if target_molecule is None:
            print("[SimilarityGuidedRetroTide] WARNING: No target_molecule in context, skipping")
            return TerminalDetectionResult(terminal=False)

        retrotide_kwargs = self._get_retrotide_kwargs(context)
        result = self._run_retrotide(node, target_molecule, retrotide_kwargs)

        if result["successful"]:
            print(f"[SimilarityGuidedRetroTide] SUCCESS! Found {result['num_successful_nodes']} "
                  f"valid PKS designs (match_type={match_type})")
            return TerminalDetectionResult(
                terminal=True,
                terminal_type="pks_terminal",
                metadata={
                    "pks_match": is_exact_pks_match,
                    "best_similarity": best_similarity,
                    "match_type": match_type,
                    "retrotide_spawn_threshold": self.retrotide_spawn_threshold,
                    "retrotide_successful": True,
                    "retrotide_num_successful_nodes": result["num_successful_nodes"],
                    "retrotide_best_score": result["best_score"],
                    "retrotide_total_nodes": result["total_nodes"],
                    "retrotide_target_smiles": result["target_smiles"],
                    "retrotide_agent": result["agent"],
                    **sim_metadata,
                },
            )
        else:
            print(f"[SimilarityGuidedRetroTide] No valid PKS design found "
                  f"(match_type={match_type})")
            return TerminalDetectionResult(
                terminal=False,
                metadata={
                    "pks_match": is_exact_pks_match,
                    "best_similarity": best_similarity,
                    "match_type": match_type,
                    "retrotide_spawn_threshold": self.retrotide_spawn_threshold,
                    "retrotide_successful": False,
                    "retrotide_total_nodes": result["total_nodes"],
                    "retrotide_target_smiles": result["target_smiles"],
                    **sim_metadata,
                },
            )

    @property
    def name(self) -> str:
        return (f"SimilarityGuidedRetroTide("
                f"method={self.similarity_method}, "
                f"spawn_thresh={self.retrotide_spawn_threshold})")

    def __repr__(self) -> str:
        return (
            f"SimilarityGuidedRetroTideDetector("
            f"similarity_method='{self.similarity_method}', "
            f"retrotide_spawn_threshold={self.retrotide_spawn_threshold}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"pks_building_blocks={len(self._pks_building_blocks)}, "
            f"retrotide_available={self._retrotide_available})"
        )
