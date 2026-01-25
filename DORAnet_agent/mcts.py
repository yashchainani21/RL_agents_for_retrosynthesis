"""
Monte Carlo Tree Search agent that explores DORAnet retro-biosynthetic
and retro-chemical transformations.

This implementation supports modular rollout and reward policies:
- Rollout policies: Define how to simulate from expanded nodes (e.g., RetroTide spawning)
- Reward policies: Define how to compute rewards for terminal states

The default behavior uses sparse rewards for sink compounds (building blocks)
and no rollouts for non-terminal nodes. To enable RetroTide rollouts, either:
- Set spawn_retrotide=True (backward compatible alias)
- Pass rollout_policy=SpawnRetroTideOnDatabaseCheck(...)
"""

from __future__ import annotations

import csv
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

import doranet.modules.enzymatic as enzymatic
import doranet.modules.synthetic as synthetic

from .node import Node
from .policies import (
    RolloutPolicy,
    RewardPolicy,
    RolloutResult,
    SpawnRetroTideOnDatabaseCheck,
    SparseTerminalRewardPolicy,
    NoOpRolloutPolicy,
)

# Optional RetroTide imports - may not be available in all environments
try:
    from RetroTide_agent.mcts import MCTS as RetroTideMCTS
    from RetroTide_agent.node import Node as RetroTideNode
    RETROTIDE_AVAILABLE = True
except ImportError:
    RetroTideMCTS = None
    RetroTideNode = None
    RETROTIDE_AVAILABLE = False

# Optional DORA-XGB imports for enzymatic reaction feasibility scoring
try:
    from DORA_XGB import DORA_XGB
    DORA_XGB_AVAILABLE = True
except ImportError:
    DORA_XGB = None
    DORA_XGB_AVAILABLE = False

# Optional pathermo imports for synthetic reaction thermodynamic scoring
try:
    from pathermo.properties import Hf as pathermo_Hf
    PATHERMO_AVAILABLE = True
except ImportError:
    pathermo_Hf = None
    PATHERMO_AVAILABLE = False

# Silence RDKit logs during network builds.
RDLogger.DisableLog("rdApp.*")


@dataclass
class RetroTideResult:
    """Stores results from a RetroTide MCTS run with traceability to DORAnet."""

    # DORAnet context
    doranet_node_id: int
    doranet_node_smiles: str
    doranet_node_depth: int
    doranet_node_provenance: str

    # DORAnet reaction info (how this fragment was created)
    doranet_reaction_name: Optional[str] = None
    doranet_reaction_smarts: Optional[str] = None
    doranet_reactants_smiles: List[str] = field(default_factory=list)
    doranet_products_smiles: List[str] = field(default_factory=list)

    # RetroTide results
    retrotide_target_smiles: str = ""
    retrotide_successful: bool = False
    retrotide_num_successful_nodes: int = 0
    retrotide_best_score: float = 0.0
    retrotide_total_nodes: int = 0

    # Optional: the actual RetroTide agent (for deeper inspection)
    retrotide_agent: Any = field(default=None, repr=False)


@lru_cache(maxsize=50000)
def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form, returning None on failure.

    Results are cached using LRU cache to avoid redundant RDKit calls
    for the same SMILES strings during the search.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def clear_smiles_cache() -> None:
    """Clear the SMILES canonicalization cache.

    Useful for freeing memory between independent runs or for benchmarking.
    """
    _canonicalize_smiles.cache_clear()


def preprocess_target_molecule(mol: Chem.Mol) -> Tuple[Chem.Mol, str]:
    """
    Preprocess a target molecule for MCTS search.

    Performs the following steps:
    1. Remove stereochemistry (chiral centers, E/Z bonds)
    2. Sanitize the molecule (kekulize, set aromaticity, etc.)
    3. Convert to canonical SMILES and re-parse to ensure canonical form

    This ensures consistent representation regardless of input SMILES format
    and removes stereochemistry that DORAnet operators may not preserve.

    Args:
        mol: RDKit Mol object to preprocess

    Returns:
        Tuple of (preprocessed_mol, canonical_smiles)

    Raises:
        ValueError: If the molecule cannot be sanitized or canonicalized
    """
    if mol is None:
        raise ValueError("Cannot preprocess None molecule")

    # Work on a copy to avoid modifying the original
    mol = Chem.RWMol(mol)

    # Step 1: Remove stereochemistry
    Chem.RemoveStereochemistry(mol)

    # Step 2: Sanitize the molecule
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError(f"Failed to sanitize molecule: {e}")

    # Step 3: Convert to canonical SMILES and re-parse
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    if not canonical_smiles:
        raise ValueError("Failed to generate canonical SMILES")

    # Re-parse to get a fresh, canonical molecule
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    if canonical_mol is None:
        raise ValueError(f"Failed to re-parse canonical SMILES: {canonical_smiles}")

    return canonical_mol, canonical_smiles


def get_smiles_cache_info():
    """Get statistics about the SMILES canonicalization cache.

    Returns:
        A named tuple with hits, misses, maxsize, and currsize fields.
    """
    return _canonicalize_smiles.cache_info()


def _reverse_reaction_string(rxn_str: str) -> str:
    """Reverse reaction direction: A.B>>C.D becomes C.D>>A.B.

    Used to convert reactions from retro direction (as stored by DORAnet)
    to forward direction (as expected by DORA-XGB).

    Args:
        rxn_str: Reaction string in format "reactants>>products"

    Returns:
        Reversed reaction string with products and reactants swapped.
    """
    if ">>" not in rxn_str:
        return rxn_str
    reactants, products = rxn_str.split(">>", 1)
    return f"{products}>>{reactants}"


class FeasibilityScorer:
    """Scores enzymatic reaction feasibility using DORA-XGB.

    This class provides lazy initialization of the DORA-XGB model and
    scoring of enzymatic reactions. Synthetic reactions are skipped.

    The scorer expects reactions in the DORAnet storage format (retro direction)
    and automatically reverses them to forward direction for DORA-XGB.
    """

    def __init__(self):
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of DORA-XGB model."""
        if self._initialized:
            return
        self._initialized = True
        if DORA_XGB_AVAILABLE:
            try:
                self._model = DORA_XGB.feasibility_classifier(
                    cofactor_positioning='by_descending_MW'
                )
            except Exception as e:
                print(f"[FeasibilityScorer] Failed to initialize DORA-XGB: {e}")
                self._model = None

    def score_reaction(
        self,
        reactants_smiles: List[str],
        products_smiles: List[str],
        provenance: str
    ) -> Tuple[Optional[float], Optional[int]]:
        """Score a reaction's feasibility.

        Args:
            reactants_smiles: List of reactant SMILES (as stored by DORAnet)
            products_smiles: List of product SMILES (as stored by DORAnet)
            provenance: "enzymatic" or "synthetic"

        Returns:
            (score, label) tuple where:
            - score: Probability of feasibility (0.0-1.0), or None if not scored
            - label: Binary label (0=infeasible, 1=feasible), or None if not scored
        """
        # Only score enzymatic reactions
        if provenance != "enzymatic":
            return None, None

        # Check for missing data
        if not reactants_smiles or not products_smiles:
            return None, None

        self._ensure_initialized()

        if self._model is None:
            return None, None

        try:
            # DORAnet stores reactions in RETRO direction for visualization
            # DORA-XGB expects FORWARD direction
            # Build the stored reaction string and reverse it
            reactants_str = ".".join(reactants_smiles)
            products_str = ".".join(products_smiles)
            stored_rxn = f"{reactants_str}>>{products_str}"
            forward_rxn = _reverse_reaction_string(stored_rxn)

            score = self._model.predict_proba(forward_rxn)
            label = self._model.predict_label(forward_rxn)
            return float(score), int(label)
        except Exception as e:
            print(f"[FeasibilityScorer] Prediction failed: {e}")
            return None, None


class ThermodynamicScorer:
    """Scores reaction thermodynamic feasibility using pathermo.

    This class computes enthalpy of reaction (ΔH) using group contribution
    methods via pathermo's Hf function. Reactions with ΔH < 15 kcal/mol
    are considered thermodynamically feasible.

    Applies to both enzymatic and synthetic reactions. While gas-phase
    enthalpies are only a proxy for solution-phase free energies, this
    approximation provides useful thermodynamic guidance.

    The scorer expects reactions in the DORAnet storage format (retro direction)
    and automatically reverses them to forward direction for thermodynamic calculation.
    """

    # Feasibility threshold in kcal/mol
    FEASIBILITY_THRESHOLD = 15.0

    def __init__(self):
        self._initialized = False

    def _ensure_initialized(self):
        """Check that pathermo is available."""
        if self._initialized:
            return
        self._initialized = True
        if not PATHERMO_AVAILABLE:
            print("[ThermodynamicScorer] pathermo not available - thermodynamic scoring disabled")

    def _calculate_enthalpy_of_formation(self, smiles: str) -> Optional[float]:
        """Calculate enthalpy of formation for a single molecule.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Enthalpy of formation in kcal/mol, or None if calculation fails.
        """
        if not PATHERMO_AVAILABLE or pathermo_Hf is None:
            return None
        try:
            hf = pathermo_Hf(smiles)
            return float(hf) if hf is not None else None
        except Exception:
            return None

    def score_reaction(
        self,
        reactants_smiles: List[str],
        products_smiles: List[str],
        provenance: str
    ) -> Tuple[Optional[float], Optional[int]]:
        """Score a reaction's thermodynamic feasibility.

        Args:
            reactants_smiles: List of reactant SMILES (as stored by DORAnet, retro direction)
            products_smiles: List of product SMILES (as stored by DORAnet, retro direction)
            provenance: "enzymatic" or "synthetic"

        Returns:
            (enthalpy_of_reaction, label) tuple where:
            - enthalpy_of_reaction: ΔH in kcal/mol, or None if not scored
            - label: 1 if ΔH < 15 kcal/mol (feasible), 0 otherwise, or None if not scored
        """
        # Check for missing data
        if not reactants_smiles or not products_smiles:
            return None, None

        self._ensure_initialized()

        if not PATHERMO_AVAILABLE:
            return None, None

        try:
            # DORAnet stores reactions in RETRO direction
            # For forward reaction: products_smiles (stored) become reactants
            #                       reactants_smiles (stored) become products
            # ΔH°rxn = Σ(ΔH°f products) - Σ(ΔH°f reactants)
            # In forward direction: ΔH = Σ(Hf of reactants_smiles) - Σ(Hf of products_smiles)

            # Calculate Hf for all "products" in forward direction (reactants_smiles in storage)
            products_hf = []
            for smiles in reactants_smiles:
                hf = self._calculate_enthalpy_of_formation(smiles)
                if hf is None:
                    return None, None  # Can't compute if any molecule fails
                products_hf.append(hf)

            # Calculate Hf for all "reactants" in forward direction (products_smiles in storage)
            reactants_hf = []
            for smiles in products_smiles:
                hf = self._calculate_enthalpy_of_formation(smiles)
                if hf is None:
                    return None, None  # Can't compute if any molecule fails
                reactants_hf.append(hf)

            # ΔH°rxn = Σ(products) - Σ(reactants)
            delta_h = sum(products_hf) - sum(reactants_hf)

            # Determine feasibility label
            label = 1 if delta_h < self.FEASIBILITY_THRESHOLD else 0

            return delta_h, label

        except Exception as e:
            print(f"[ThermodynamicScorer] Prediction failed: {e}")
            return None, None


def _load_enzymatic_rule_labels() -> List[str]:
    """
    Load enzymatic reaction rule labels from DORAnet.

    Returns:
        List of rule names indexed by operator position.
    """
    try:
        import doranet.modules.enzymatic as enzymatic
        import os
        import pandas as pd

        # Load rule labels from TSV file
        enzymatic_path = os.path.dirname(enzymatic.__file__)
        rules_file = os.path.join(enzymatic_path, 'JN3604IMT_rules.tsv')
        df = pd.read_csv(rules_file, sep='\t')

        # Create list of names indexed by position (same order as in file)
        labels = []
        for _, row in df.iterrows():
            name = row['Name']
            if pd.notna(name):
                labels.append(name)
            else:
                labels.append(None)

        return labels

    except Exception as e:
        print(f"[WARN] Could not load enzymatic rule labels: {e}")
        return []


def _load_synthetic_reaction_labels() -> List[str]:
    """
    Load synthetic reaction labels from DORAnet.

    Returns:
        List of reaction names indexed by operator position.
    """
    try:
        import doranet.modules.synthetic.Reaction_Smarts_Retro as retro_smarts

        # Create list of names indexed by position
        labels = []
        for op_def in retro_smarts.op_retro_smarts:
            if hasattr(op_def, 'name'):
                labels.append(op_def.name)
            else:
                labels.append(None)

        return labels

    except Exception as e:
        print(f"[WARN] Could not load synthetic reaction labels: {e}")
        return []


# Small molecules to exclude (common byproducts, not useful fragments)
DEFAULT_EXCLUDED_FRAGMENTS = (
    "O",           # water
    "O=O",         # O2
    "[H][H]",      # H2
    "O=C=O",       # CO2
    "C=O",         # formaldehyde
    "[C-]#[O+]",   # CO
    "N#N",         # N2
    "N",           # ammonia
    "S",           # sulfur
)


def _load_cofactors_from_csv(csv_path: str) -> Set[str]:
    """
    Load cofactor SMILES from a CSV file and return canonical SMILES set.

    Args:
        csv_path: Path to CSV file with a 'SMILES' column.

    Returns:
        Set of canonical SMILES strings for cofactors.
    """
    cofactors: Set[str] = set()
    path = Path(csv_path)

    if not path.exists():
        print(f"[WARN] Cofactors file not found: {csv_path}")
        return cofactors

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get("SMILES", "").strip()
            if smiles:
                # Skip SMILES with wildcards (e.g., "*c1c...")
                if "*" in smiles:
                    continue
                canonical = _canonicalize_smiles(smiles)
                if canonical:
                    cofactors.add(canonical)

    print(f"[DORAnet] Loaded {len(cofactors)} cofactors from {path.name}")
    return cofactors


def _load_pks_library(
    pks_file: str,
    use_cache: bool = True,
    show_progress: bool = True
) -> Set[str]:
    """
    Load PKS product SMILES from a text file (one SMILES per line).

    For large files (>10k entries), this function will:
    - Show progress during loading
    - Cache the canonicalized SMILES to a pickle file for faster subsequent loads

    If the specified file is not found, attempts to load from a fallback path
    on the Northwestern Quest supercomputing cluster. Raises an error if the
    PKS library cannot be found at either location.

    Args:
        pks_file: Path to text file with PKS SMILES.
        use_cache: If True, use cached canonical SMILES if available.
        show_progress: If True, show progress for large files.

    Returns:
        Set of canonical SMILES strings for PKS products.

    Raises:
        FileNotFoundError: If PKS library file cannot be found at the specified
            path or the fallback cluster path.
    """
    import pickle
    import hashlib

    # Fallback path for Northwestern Quest supercomputing cluster
    CLUSTER_FALLBACK_PATH = Path(
        "/projects/p30041/YashChainani/RL_agents_for_retrosynthesis/data/processed/expanded_PKS_SMILES_V3.txt"
    )

    pks_smiles: Set[str] = set()
    path = Path(pks_file)

    if not path.exists():
        print(f"[WARN] PKS library file not found at primary path: {pks_file}")
        print(f"[DORAnet] Attempting fallback path (Quest cluster): {CLUSTER_FALLBACK_PATH}")

        if CLUSTER_FALLBACK_PATH.exists():
            print(f"[DORAnet] Found PKS library at fallback cluster path")
            path = CLUSTER_FALLBACK_PATH
        else:
            raise FileNotFoundError(
                f"PKS library file not found at either location:\n"
                f"  Primary path: {pks_file}\n"
                f"  Cluster fallback: {CLUSTER_FALLBACK_PATH}\n"
                f"Please ensure the PKS library file exists at one of these locations."
            )

    # Check for cached version
    cache_dir = path.parent / ".cache"
    # Create cache filename based on file content hash (first 1000 chars + file size + mtime)
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(1000)
    file_stat = path.stat()
    cache_key = hashlib.md5(f"{sample}{file_stat.st_size}{file_stat.st_mtime}".encode()).hexdigest()[:12]
    cache_file = cache_dir / f"{path.stem}_{cache_key}.pkl"

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                pks_smiles = pickle.load(f)
            print(f"[DORAnet] Loaded {len(pks_smiles):,} PKS products from cache ({path.name})")
            return pks_smiles
        except Exception as e:
            print(f"[WARN] Could not load PKS cache, will regenerate: {e}")

    # Count lines for progress reporting
    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Determine if we should show progress (for large files)
    large_file = total_lines > 10000
    progress_interval = max(total_lines // 20, 1000) if large_file else total_lines + 1

    if large_file:
        print(f"[DORAnet] Loading {total_lines:,} PKS products from {path.name}...")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            smiles = line.strip()
            if smiles and not smiles.startswith("#"):
                canonical = _canonicalize_smiles(smiles)
                if canonical:
                    pks_smiles.add(canonical)

            # Show progress for large files
            if show_progress and large_file and (i + 1) % progress_interval == 0:
                pct = (i + 1) / total_lines * 100
                print(f"[DORAnet]   Progress: {i + 1:,}/{total_lines:,} ({pct:.0f}%)")

    print(f"[DORAnet] Loaded {len(pks_smiles):,} PKS products from {path.name}")

    # Cache the result for large files
    if use_cache and large_file:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(pks_smiles, f)
            print(f"[DORAnet] Cached canonicalized PKS SMILES for faster future loads")
        except Exception as e:
            print(f"[WARN] Could not save PKS cache: {e}")

    return pks_smiles


def _load_prohibited_chemicals(prohibited_file: str) -> Set[str]:
    """
    Load prohibited chemical SMILES from a text file (one SMILES per line).

    Prohibited chemicals are hazardous or controlled substances that should
    never appear as targets or intermediates in synthesis pathways.

    Args:
        prohibited_file: Path to text file with prohibited SMILES.

    Returns:
        Set of canonical SMILES strings for prohibited chemicals.
    """
    prohibited_smiles: Set[str] = set()
    path = Path(prohibited_file)

    if not path.exists():
        print(f"[WARN] Prohibited chemicals file not found: {prohibited_file}")
        return prohibited_smiles

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            smiles = line.strip()
            if smiles and not smiles.startswith("#"):
                canonical = _canonicalize_smiles(smiles)
                if canonical:
                    prohibited_smiles.add(canonical)

    print(f"[DORAnet] Loaded {len(prohibited_smiles)} prohibited chemicals from {path.name}")
    return prohibited_smiles


def _load_sink_compounds(
    sink_file: str,
    use_cache: bool = True,
    show_progress: bool = True
) -> Set[str]:
    """
    Load sink compound SMILES from a text file (one SMILES per line).

    Sink compounds are commercially available building blocks that don't
    need further retrosynthetic expansion.

    For large files (>10k entries), this function will:
    - Show progress during loading
    - Cache the canonicalized SMILES to a pickle file for faster subsequent loads

    Args:
        sink_file: Path to text file with sink compound SMILES.
        use_cache: If True, use cached canonical SMILES if available.
        show_progress: If True, show progress for large files.

    Returns:
        Set of canonical SMILES strings for sink compounds.
    """
    import pickle
    import hashlib

    sink_smiles: Set[str] = set()
    path = Path(sink_file)

    if not path.exists():
        print(f"[WARN] Sink compounds file not found: {sink_file}")
        return sink_smiles

    # Check for cached version
    cache_dir = path.parent / ".cache"
    # Create cache filename based on file content hash (first 1000 chars + file size)
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(1000)
    file_stat = path.stat()
    cache_key = hashlib.md5(f"{sample}{file_stat.st_size}{file_stat.st_mtime}".encode()).hexdigest()[:12]
    cache_file = cache_dir / f"{path.stem}_{cache_key}.pkl"

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                sink_smiles = pickle.load(f)
            print(f"[DORAnet] Loaded {len(sink_smiles)} sink compounds from cache ({path.name})")
            return sink_smiles
        except Exception as e:
            print(f"[WARN] Could not load cache, will regenerate: {e}")

    # Count lines for progress reporting
    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Determine if we should show progress (for large files)
    large_file = total_lines > 10000
    progress_interval = max(total_lines // 20, 1000) if large_file else total_lines + 1

    if large_file:
        print(f"[DORAnet] Loading {total_lines:,} sink compounds from {path.name}...")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            smiles = line.strip()
            if smiles and not smiles.startswith("#"):
                canonical = _canonicalize_smiles(smiles)
                if canonical:
                    sink_smiles.add(canonical)

            # Show progress for large files
            if show_progress and large_file and (i + 1) % progress_interval == 0:
                pct = (i + 1) / total_lines * 100
                print(f"[DORAnet]   Progress: {i + 1:,}/{total_lines:,} ({pct:.0f}%)")

    print(f"[DORAnet] Loaded {len(sink_smiles):,} sink compounds from {path.name}")

    # Cache the result for large files
    if use_cache and large_file:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(sink_smiles, f)
            print(f"[DORAnet] Cached canonicalized SMILES for faster future loads")
        except Exception as e:
            print(f"[WARN] Could not save cache: {e}")

    return sink_smiles


class DORAnetMCTS:
    """
    MCTS driver for DORAnet retro-fragmentation with modular policies.

    Supports pluggable rollout and reward policies for flexible experimentation:
    - Rollout policies define how to simulate from expanded nodes
    - Reward policies define how to compute rewards for nodes

    For backward compatibility, spawn_retrotide=True creates a
    SpawnRetroTideOnDatabaseCheck rollout policy automatically.
    """

    def __init__(
        self,
        root: Node,
        target_molecule: Chem.Mol,
        total_iterations: int = 100,
        max_depth: int = 3,
        use_enzymatic: bool = True,
        use_synthetic: bool = True,
        generations_per_expand: int = 1,
        max_children_per_expand: int = 10,
        child_downselection_strategy: str = "first_N",
        excluded_fragments: Optional[Iterable[str]] = None,
        cofactors_file: Optional[str] = None,
        cofactors_files: Optional[List[str]] = None,
        pks_library_file: Optional[str] = None,
        sink_compounds_file: Optional[str] = None,
        sink_compounds_files: Optional[List[str]] = None,
        prohibited_chemicals_file: Optional[str] = None,
        MW_multiple_to_exclude: float = 1.5,
        spawn_retrotide: bool = False,
        retrotide_kwargs: Optional[Dict] = None,
        sink_terminal_reward: float = 1.0,
        selection_policy: str = "depth_biased",
        depth_bonus_coefficient: float = 2.0,
        enable_visualization: bool = False,
        enable_interactive_viz: bool = False,
        enable_iteration_visualizations: bool = False,
        auto_open_viz: bool = False,
        auto_open_iteration_viz: bool = False,
        visualization_output_dir: Optional[str] = None,
        iteration_viz_interval: int = 1,
        fragment_cache_dir: Optional[str] = None,
        # New policy parameters
        rollout_policy: Optional[RolloutPolicy] = None,
        reward_policy: Optional[RewardPolicy] = None,
    ) -> None:
        """
        Args:
            root: Starting node containing the target molecule.
            target_molecule: The molecule to fragment.
            total_iterations: Number of MCTS iterations to run.
            max_depth: Maximum tree depth for fragmentation.
            use_enzymatic: Whether to use enzymatic retro-transformations.
            use_synthetic: Whether to use synthetic retro-transformations.
            generations_per_expand: DORAnet generations per expansion step.
            max_children_per_expand: Max fragments to retain per expansion.
            child_downselection_strategy: Strategy for selecting which fragments to keep
                when more than max_children_per_expand are generated. Options:
                - "first_N": Keep the first N fragments in DORAnet's order (default, fastest)
                - "hybrid": Prioritize sink compounds first, PKS matches second, then smaller MW
                - "most_thermo_feasible": Prioritize by thermodynamic feasibility score
                  (DORA-XGB for enzymatic, sigmoid-transformed ΔH for synthetic), with priority
                  bonuses for sink compounds (+1000) and PKS library matches (+500)
            excluded_fragments: SMILES of fragments to ignore (small byproducts).
            cofactors_file: Path to CSV file with cofactor SMILES to exclude (deprecated, use cofactors_files).
            cofactors_files: List of paths to CSV files with cofactor SMILES to exclude.
            pks_library_file: Path to text file with PKS product SMILES for reward calculation.
            sink_compounds_file: Path to text file with sink compound SMILES (deprecated, use sink_compounds_files).
            sink_compounds_files: List of paths to text files with sink compound SMILES.
                Sink compounds are commercially available building blocks that don't need
                further expansion. Supports both biological and chemical building blocks.
            prohibited_chemicals_file: Path to text file with prohibited chemical SMILES.
                If the target molecule matches a prohibited chemical, a ValueError is raised.
                Intermediate fragments matching prohibited chemicals are filtered out.
            MW_multiple_to_exclude: Maximum molecular weight ratio for fragments relative to target.
                Fragments with MW > target_MW * MW_multiple_to_exclude are filtered out.
                This prevents unrealistic dimerization products from enzymatic operators.
                Default is 1.5 (fragments up to 1.5x the target MW are allowed).
            spawn_retrotide: Whether to spawn RetroTide searches for PKS-matching fragments.
                This is a backward-compatible alias that creates a SpawnRetroTideOnDatabaseCheck
                rollout policy. If rollout_policy is explicitly provided, this is ignored.
            retrotide_kwargs: Parameters passed to RetroTide MCTS agents.
            selection_policy: Node selection policy for MCTS. Options:
                - "UCB1": Standard UCB1 (breadth-first tendency, explores all nodes at each level)
                - "depth_biased": Depth-biased UCB1 (depth-first tendency, reaches max_depth faster)
                Default is "depth_biased" for faster deep exploration.
            depth_bonus_coefficient: Coefficient for depth-biased selection in UCB1.
                Higher values encourage deeper exploration before exhaustive breadth search.
                The selection score becomes: UCB1 + depth_bonus_coefficient * depth.
                Default is 2.0. Set to 0.0 for standard UCB1 (breadth-first tendency).
            enable_visualization: Whether to automatically generate static visualizations (PNG).
            enable_interactive_viz: Whether to generate interactive HTML visualization.
            enable_iteration_visualizations: If True, generate visualizations after each iteration.
            auto_open_viz: If True, automatically open final interactive visualization in browser.
            auto_open_iteration_viz: If True, automatically open iteration visualizations in browser.
            visualization_output_dir: Directory to save visualizations (default: current directory).
            iteration_viz_interval: Generate iteration visualizations every N iterations (default: 1).
            rollout_policy: Policy for simulating from expanded nodes to estimate value.
                If None and spawn_retrotide=True, uses SpawnRetroTideOnDatabaseCheck.
                If None and spawn_retrotide=False, uses NoOpRolloutPolicy (no rollouts).
            reward_policy: Policy for computing rewards for nodes.
                If None, uses SparseTerminalRewardPolicy with sink_terminal_reward.
        """
        # Preprocess target molecule: remove stereochemistry, sanitize, canonicalize
        original_smiles = Chem.MolToSmiles(target_molecule) if target_molecule else "None"
        preprocessed_mol, canonical_smiles = preprocess_target_molecule(target_molecule)
        print(f"[DORAnet] Preprocessed target molecule:")
        print(f"[DORAnet]   Original SMILES: {original_smiles}")
        print(f"[DORAnet]   Canonical SMILES (no stereo): {canonical_smiles}")

        # Update root node's fragment to use the preprocessed molecule
        root.fragment = preprocessed_mol
        root._smiles = canonical_smiles  # Update cached SMILES if present

        self.root = root
        self.target_molecule = preprocessed_mol
        self.total_iterations = total_iterations
        self.max_depth = max_depth
        self.use_enzymatic = use_enzymatic
        self.use_synthetic = use_synthetic
        self.generations_per_expand = generations_per_expand
        self.max_children_per_expand = max_children_per_expand

        # Child downselection strategy configuration
        valid_downselection_strategies = ["first_N", "hybrid", "most_thermo_feasible"]
        if child_downselection_strategy not in valid_downselection_strategies:
            raise ValueError(f"Invalid child_downselection_strategy '{child_downselection_strategy}'. "
                           f"Must be one of: {valid_downselection_strategies}")
        self.child_downselection_strategy = child_downselection_strategy

        self.spawn_retrotide = spawn_retrotide and RETROTIDE_AVAILABLE
        self.retrotide_kwargs = retrotide_kwargs or {}
        self.sink_terminal_reward = sink_terminal_reward
        self.MW_multiple_to_exclude = MW_multiple_to_exclude

        # Initialize rollout and reward policies
        # Policy initialization is deferred until after pks_library is loaded (see below)
        self._rollout_policy_arg = rollout_policy
        self._reward_policy_arg = reward_policy

        # Selection policy configuration
        valid_policies = ["UCB1", "depth_biased"]
        if selection_policy not in valid_policies:
            raise ValueError(f"Invalid selection_policy '{selection_policy}'. Must be one of: {valid_policies}")
        self.selection_policy = selection_policy
        self.depth_bonus_coefficient = depth_bonus_coefficient if selection_policy == "depth_biased" else 0.0

        # Calculate target molecule MW for fragment size filtering
        self.target_MW = Descriptors.MolWt(target_molecule)
        self.max_fragment_MW = self.target_MW * MW_multiple_to_exclude
        print(f"[DORAnet] Target MW: {self.target_MW:.2f}, Max fragment MW: {self.max_fragment_MW:.2f} "
              f"(excluding fragments > {MW_multiple_to_exclude}x target)")
        # Log the selection policy
        if self.selection_policy == "depth_biased":
            print(f"[DORAnet] Selection policy: depth_biased (coefficient={self.depth_bonus_coefficient}) - "
                  f"favoring deeper exploration")
        else:
            print(f"[DORAnet] Selection policy: UCB1 (standard breadth-first tendency)")

        # Log the child downselection strategy
        if self.child_downselection_strategy == "hybrid":
            print(f"[DORAnet] Child downselection: hybrid (sink > PKS > smaller MW)")
        elif self.child_downselection_strategy == "most_thermo_feasible":
            print(f"[DORAnet] Child downselection: most_thermo_feasible (sink > PKS > thermodynamic feasibility)")
        else:
            print(f"[DORAnet] Child downselection: first_N (DORAnet order)")

        self.enable_visualization = enable_visualization
        self.enable_interactive_viz = enable_interactive_viz
        self.enable_iteration_visualizations = enable_iteration_visualizations
        self.auto_open_viz = auto_open_viz
        self.auto_open_iteration_viz = auto_open_iteration_viz
        self.visualization_output_dir = visualization_output_dir or "."
        self.iteration_viz_interval = iteration_viz_interval
        if fragment_cache_dir:
            self.fragment_cache_dir = Path(fragment_cache_dir)
        else:
            self.fragment_cache_dir = Path(self.visualization_output_dir) / ".cache" / "doranet_fragments"

        if spawn_retrotide and not RETROTIDE_AVAILABLE:
            print("[WARN] RetroTide not available - spawning disabled")

        # Tree tracking
        self.nodes: List[Node] = [root]
        self.edges: List[Tuple[int, int]] = []

        # Track all RetroTide runs with full traceability
        self.retrotide_results: List[RetroTideResult] = []

        # Build excluded fragments set from default small molecules
        excluded_iterable = (
            excluded_fragments if excluded_fragments is not None
            else DEFAULT_EXCLUDED_FRAGMENTS
        )
        self.excluded_fragments: Set[str] = {
            _canonicalize_smiles(smi)
            for smi in excluded_iterable
            if _canonicalize_smiles(smi)
        }

        # Add cofactors from CSV files if provided
        # Support both single file (deprecated) and list of files
        cofactor_files_to_load = []
        if cofactors_file:
            cofactor_files_to_load.append(cofactors_file)
        if cofactors_files:
            cofactor_files_to_load.extend(cofactors_files)

        # Track chemistry helpers separately for DORAnet synthetic network generation
        self.chemistry_helpers: Set[str] = set()

        for cofactor_file_path in cofactor_files_to_load:
            cofactor_smiles = _load_cofactors_from_csv(cofactor_file_path)
            self.excluded_fragments.update(cofactor_smiles)

            # If this is the chemistry_helpers file, also store for DORAnet
            if "chemistry_helpers" in str(cofactor_file_path).lower():
                self.chemistry_helpers.update(cofactor_smiles)

        # Load PKS product library for reward calculation
        self.pks_library: Set[str] = set()
        if pks_library_file:
            self.pks_library = _load_pks_library(pks_library_file)

        # Load sink compounds (commercially available building blocks)
        # Support both single file (deprecated) and list of files
        # Track biological vs chemical separately for reporting
        self.sink_compounds: Set[str] = set()
        self.biological_sink_compounds: Set[str] = set()
        self.chemical_sink_compounds: Set[str] = set()

        sink_files_to_load = []
        if sink_compounds_file:
            sink_files_to_load.append(sink_compounds_file)
        if sink_compounds_files:
            sink_files_to_load.extend(sink_compounds_files)

        for sink_file_path in sink_files_to_load:
            sink_smiles = _load_sink_compounds(sink_file_path)
            self.sink_compounds.update(sink_smiles)

            # Categorize by file name
            file_name_lower = str(sink_file_path).lower()
            if "biological" in file_name_lower:
                self.biological_sink_compounds.update(sink_smiles)
            elif "chemical" in file_name_lower:
                self.chemical_sink_compounds.update(sink_smiles)

        if sink_files_to_load:
            print(f"[DORAnet] Total sink compounds loaded: {len(self.sink_compounds):,} "
                  f"(biological: {len(self.biological_sink_compounds):,}, "
                  f"chemical: {len(self.chemical_sink_compounds):,})")

        # Load prohibited chemicals (hazardous/controlled substances to avoid)
        self.prohibited_chemicals: Set[str] = set()
        if prohibited_chemicals_file:
            self.prohibited_chemicals = _load_prohibited_chemicals(prohibited_chemicals_file)

            # Check if the target molecule is a prohibited chemical
            target_smiles = Chem.MolToSmiles(target_molecule)
            target_canonical = _canonicalize_smiles(target_smiles)
            if target_canonical and target_canonical in self.prohibited_chemicals:
                raise ValueError(
                    f"Target molecule is a prohibited chemical and cannot be synthesized.\n"
                    f"  Target SMILES: {target_smiles}\n"
                    f"  Canonical SMILES: {target_canonical}\n"
                    f"Please choose a different target molecule."
                )

        # Load reaction label mappings for human-readable names
        self._enzymatic_labels = _load_enzymatic_rule_labels()
        self._synthetic_labels = _load_synthetic_reaction_labels()
        print(f"[DORAnet] Loaded {len(self._enzymatic_labels)} enzymatic rule labels")
        print(f"[DORAnet] Loaded {len(self._synthetic_labels)} synthetic reaction labels")

        # Report chemistry helpers for synthetic networks
        if self.chemistry_helpers:
            print(f"[DORAnet] Using {len(self.chemistry_helpers)} chemistry helpers for synthetic network generation")

        # Initialize rollout and reward policies (now that pks_library is loaded)
        self._initialize_policies()

    def _initialize_policies(self) -> None:
        """
        Initialize rollout and reward policies based on constructor arguments.

        Called at the end of __init__ after all data files are loaded.
        Handles backward compatibility with spawn_retrotide parameter.
        """
        # Initialize reward policy
        if self._reward_policy_arg is not None:
            self.reward_policy = self._reward_policy_arg
        else:
            # Default: sparse terminal reward policy
            self.reward_policy = SparseTerminalRewardPolicy(
                sink_terminal_reward=self.sink_terminal_reward,
                pks_library=self.pks_library,
            )

        # Initialize rollout policy
        if self._rollout_policy_arg is not None:
            # Explicit rollout policy provided - use it
            self.rollout_policy = self._rollout_policy_arg
        elif self.spawn_retrotide:
            # Backward compatibility: spawn_retrotide=True creates RetroTide rollout policy
            self.rollout_policy = SpawnRetroTideOnDatabaseCheck(
                pks_library=self.pks_library,
                retrotide_kwargs=self.retrotide_kwargs,
                success_reward=1.0,
                failure_reward=0.0,
            )
        else:
            # Default: no rollouts (sparse rewards only)
            self.rollout_policy = NoOpRolloutPolicy()

        # Log the policies being used
        print(f"[DORAnet] Using rollout policy: {self.rollout_policy.name}")
        print(f"[DORAnet] Using reward policy: {self.reward_policy.name}")

        # Initialize feasibility scorer for enzymatic reactions
        self.feasibility_scorer = FeasibilityScorer()
        if DORA_XGB_AVAILABLE:
            print("[DORAnet] DORA-XGB available for enzymatic feasibility scoring")
        else:
            print("[DORAnet] DORA-XGB not available - enzymatic feasibility scoring disabled")

        # Initialize thermodynamic scorer for synthetic reactions
        self.thermodynamic_scorer = ThermodynamicScorer()
        if PATHERMO_AVAILABLE:
            print("[DORAnet] pathermo available for synthetic thermodynamic scoring")
        else:
            print("[DORAnet] pathermo not available - synthetic thermodynamic scoring disabled")

    def _build_rollout_context(self) -> Dict[str, Any]:
        """
        Build context dictionary for rollout and reward policies.

        Returns:
            Dictionary containing MCTS state for policy execution.
        """
        return {
            "target_molecule": self.target_molecule,
            "pks_library": self.pks_library,
            "sink_compounds": self.sink_compounds,
            "biological_building_blocks": self.biological_sink_compounds,
            "chemical_building_blocks": self.chemical_sink_compounds,
            "retrotide_kwargs": self.retrotide_kwargs,
            "agent": self,
        }

    @dataclass
    class FragmentInfo:
        """Information about a generated fragment and the reaction that created it."""
        molecule: Chem.Mol
        smiles: str
        reaction_smarts: Optional[str] = None
        reaction_name: Optional[str] = None
        reactants_smiles: List[str] = field(default_factory=list)
        products_smiles: List[str] = field(default_factory=list)
        # Pre-computed thermodynamic scoring fields (for most_thermo_feasible strategy)
        feasibility_score: Optional[float] = None
        dora_xgb_score: Optional[float] = None
        dora_xgb_label: Optional[int] = None
        enthalpy_of_reaction: Optional[float] = None
        thermodynamic_label: Optional[int] = None
        provenance: Optional[str] = None

    def _compute_fragment_feasibility_score(
        self,
        fragment_info: "DORAnetMCTS.FragmentInfo",
        provenance: str,
    ) -> "DORAnetMCTS.FragmentInfo":
        """
        Compute feasibility score for a fragment before node creation.

        For enzymatic reactions: Uses DORA-XGB probability if available,
        otherwise falls back to sigmoid-transformed ΔH.

        For synthetic reactions: Uses sigmoid-transformed ΔH.

        The sigmoid transform maps ΔH to (0, 1) where:
        - score = 1.0 / (1.0 + exp(0.2 * (ΔH - 15.0)))
        - ΔH < 15 kcal/mol → score > 0.5 (feasible)
        - ΔH > 15 kcal/mol → score < 0.5 (infeasible)

        Args:
            fragment_info: FragmentInfo with reaction data populated.
            provenance: "enzymatic" or "synthetic".

        Returns:
            FragmentInfo with feasibility fields populated.
        """
        fragment_info.provenance = provenance

        # Score DORA-XGB for enzymatic reactions
        if provenance == "enzymatic":
            score, label = self.feasibility_scorer.score_reaction(
                reactants_smiles=fragment_info.reactants_smiles,
                products_smiles=fragment_info.products_smiles,
                provenance=provenance
            )
            fragment_info.dora_xgb_score = score
            fragment_info.dora_xgb_label = label

        # Score thermodynamic feasibility (both enzymatic and synthetic)
        delta_h, thermo_label = self.thermodynamic_scorer.score_reaction(
            reactants_smiles=fragment_info.reactants_smiles,
            products_smiles=fragment_info.products_smiles,
            provenance=provenance
        )
        fragment_info.enthalpy_of_reaction = delta_h
        fragment_info.thermodynamic_label = thermo_label

        # Compute unified feasibility score (0-1 scale)
        if provenance == "enzymatic" and fragment_info.dora_xgb_score is not None:
            # Use DORA-XGB probability for enzymatic reactions
            fragment_info.feasibility_score = fragment_info.dora_xgb_score
        elif delta_h is not None:
            # Use sigmoid-transformed ΔH
            fragment_info.feasibility_score = 1.0 / (1.0 + math.exp(0.2 * (delta_h - 15.0)))
        else:
            # Unknown, assume feasible
            fragment_info.feasibility_score = 1.0

        return fragment_info

    def _generate_doranet_fragments(
        self,
        molecule: Chem.Mol,
        mode: str,
    ) -> List["DORAnetMCTS.FragmentInfo"]:
        """
        Run DORAnet in retro mode to generate fragment molecules.

        Args:
            molecule: The molecule to fragment.
            mode: Either "enzymatic" or "synthetic".

        Returns:
            List of FragmentInfo objects with molecule and reaction details.
        """
        if molecule is None:
            return []

        import hashlib
        import pickle

        starter_smiles = Chem.MolToSmiles(molecule)
        job_name = f"doranet_{mode}_retro_{uuid.uuid4().hex[:8]}"

        # Cache key based on input, mode, and key generation settings.
        excluded_hash = hashlib.md5(
            "\n".join(sorted(self.excluded_fragments)).encode()
        ).hexdigest()[:12]
        helpers_hash = hashlib.md5(
            "\n".join(sorted(self.chemistry_helpers)).encode()
        ).hexdigest()[:12]
        cache_key = hashlib.md5(
            f"{starter_smiles}|{mode}|gen={self.generations_per_expand}"
            f"|max_children={self.max_children_per_expand}"
            f"|strategy={self.child_downselection_strategy}"
            f"|excluded={excluded_hash}|helpers={helpers_hash}".encode()
        ).hexdigest()[:16]
        cache_file = self.fragment_cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                fragments: List[DORAnetMCTS.FragmentInfo] = []
                for item in cached:
                    rd_mol = Chem.MolFromSmiles(item["smiles"])
                    if rd_mol is None:
                        continue
                    fragments.append(DORAnetMCTS.FragmentInfo(
                        molecule=rd_mol,
                        smiles=item["smiles"],
                        reaction_smarts=item.get("reaction_smarts"),
                        reaction_name=item.get("reaction_name"),
                        reactants_smiles=item.get("reactants_smiles", []),
                        products_smiles=item.get("products_smiles", []),
                        # Restore pre-computed scores if available
                        feasibility_score=item.get("feasibility_score"),
                        dora_xgb_score=item.get("dora_xgb_score"),
                        dora_xgb_label=item.get("dora_xgb_label"),
                        enthalpy_of_reaction=item.get("enthalpy_of_reaction"),
                        thermodynamic_label=item.get("thermodynamic_label"),
                        provenance=item.get("provenance"),
                    ))
                return fragments
            except Exception:
                pass

        module = enzymatic if mode == "enzymatic" else synthetic
        try:
            # Build network generation kwargs
            network_kwargs = {
                "job_name": job_name,
                "starters": {starter_smiles},
                "gen": self.generations_per_expand,
                "direction": "retro",
            }

            # Add helpers for synthetic network generation
            if mode == "synthetic" and self.chemistry_helpers:
                network_kwargs["helpers"] = self.chemistry_helpers

            network = module.generate_network(**network_kwargs)
        except Exception as exc:
            print(f"[WARN] DORAnet {mode} failed for {starter_smiles}: {exc}")
            return []

        # Build a map from product SMILES to reaction info
        mol_to_reaction: Dict[str, Dict] = {}

        # Convert network mols and ops to lists for indexing
        mols_list = list(network.mols)
        ops_list = list(network.ops)

        for rxn in getattr(network, 'rxns', []):
            # Extract reaction info using indices
            try:
                # Get the operator using its index
                op_idx = rxn.operator
                op = ops_list[op_idx] if op_idx < len(ops_list) else None

                # Extract SMARTS - use uid attribute if available, otherwise convert to string
                if op and hasattr(op, 'uid'):
                    rxn_smarts = op.uid
                else:
                    rxn_smarts = str(op) if op else None

                # Get human-readable reaction label using operator index
                rxn_label = None
                if op_idx is not None:
                    # Look up label by operator index
                    if mode == "enzymatic":
                        if op_idx < len(self._enzymatic_labels):
                            rxn_label = self._enzymatic_labels[op_idx]
                    else:  # synthetic
                        if op_idx < len(self._synthetic_labels):
                            rxn_label = self._synthetic_labels[op_idx]

                # Fallback to truncated SMARTS if no label found
                if not rxn_label and rxn_smarts:
                    rxn_label = rxn_smarts[:60] + "..." if len(rxn_smarts) > 60 else rxn_smarts

                # Get reactants and products using their indices
                reactant_idxs = rxn.reactants
                product_idxs = rxn.products

                reactant_smiles = [
                    str(getattr(mols_list[i], 'uid', mols_list[i]))
                    for i in reactant_idxs if i < len(mols_list)
                ]
                product_smiles = [
                    str(getattr(mols_list[i], 'uid', mols_list[i]))
                    for i in product_idxs if i < len(mols_list)
                ]

                # Map each product to this reaction
                for prod in product_smiles:
                    canonical_prod = _canonicalize_smiles(prod)
                    if canonical_prod and canonical_prod not in mol_to_reaction:
                        mol_to_reaction[canonical_prod] = {
                            'smarts': rxn_smarts,
                            'label': rxn_label,  # Human-readable label
                            'reactants': reactant_smiles,
                            'products': product_smiles,
                        }
            except Exception:
                continue

        fragments: List[DORAnetMCTS.FragmentInfo] = []
        seen: Set[str] = set()

        for mol in network.mols:
            mol_smiles = getattr(mol, "uid", None)
            if mol_smiles is None:
                continue

            # Skip SMILES with wildcards (template patterns, not real molecules)
            mol_smiles_str = str(mol_smiles)
            if "*" in mol_smiles_str:
                continue

            canonical = _canonicalize_smiles(mol_smiles_str)
            if canonical is None:
                continue
            if canonical == _canonicalize_smiles(starter_smiles):
                continue  # Skip the input molecule itself
            if canonical in self.excluded_fragments:
                continue  # Skip cofactors and small byproducts
            if canonical in seen:
                continue  # Skip duplicates

            rd_mol = Chem.MolFromSmiles(canonical)
            if rd_mol is None:
                continue

            seen.add(canonical)

            # Get reaction info for this fragment
            rxn_info = mol_to_reaction.get(canonical, {})

            frag_info = DORAnetMCTS.FragmentInfo(
                molecule=rd_mol,
                smiles=canonical,
                reaction_smarts=rxn_info.get('smarts'),
                reaction_name=rxn_info.get('label'),  # Use the human-readable label
                reactants_smiles=rxn_info.get('reactants', []),
                products_smiles=rxn_info.get('products', []),
            )

            # Compute feasibility scores for most_thermo_feasible strategy
            if self.child_downselection_strategy == "most_thermo_feasible":
                frag_info = self._compute_fragment_feasibility_score(frag_info, mode)

            fragments.append(frag_info)

            # For first_N strategy, stop early once we have enough fragments
            if self.child_downselection_strategy == "first_N":
                if len(fragments) >= self.max_children_per_expand:
                    break

        # Apply downselection strategy if we have more fragments than the limit
        if len(fragments) > self.max_children_per_expand:
            fragments = self._downselect_fragments(fragments)

        # Cache filtered fragments for future reuse.
        try:
            self.fragment_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_payload = []
            for frag in fragments:
                item = {
                    "smiles": frag.smiles,
                    "reaction_smarts": frag.reaction_smarts,
                    "reaction_name": frag.reaction_name,
                    "reactants_smiles": frag.reactants_smiles,
                    "products_smiles": frag.products_smiles,
                }
                # Include pre-computed scores if available (for most_thermo_feasible)
                if frag.feasibility_score is not None:
                    item["feasibility_score"] = frag.feasibility_score
                    item["dora_xgb_score"] = frag.dora_xgb_score
                    item["dora_xgb_label"] = frag.dora_xgb_label
                    item["enthalpy_of_reaction"] = frag.enthalpy_of_reaction
                    item["thermodynamic_label"] = frag.thermodynamic_label
                    item["provenance"] = frag.provenance
                cache_payload.append(item)
            with open(cache_file, "wb") as f:
                pickle.dump(cache_payload, f)
        except Exception:
            pass

        return fragments

    def _downselect_fragments(
        self, fragments: List["DORAnetMCTS.FragmentInfo"]
    ) -> List["DORAnetMCTS.FragmentInfo"]:
        """
        Downselect fragments to max_children_per_expand using the configured strategy.

        Strategies:
        - "first_N": Simple truncation to max_children_per_expand.
        - "hybrid": Score by sink compounds, PKS library matches, and MW.
        - "most_thermo_feasible": Score by thermodynamic feasibility with priority
          bonuses for known terminals.

        For "hybrid" strategy, fragments are scored and sorted by priority:
        1. Sink compounds (highest priority) - commercially available building blocks
        2. PKS library matches (medium priority) - known PKS-synthesizable molecules
        3. Smaller molecular weight (base priority) - simpler precursors preferred

        For "most_thermo_feasible" strategy, fragments are scored by:
        1. Sink compounds: feasibility_score + 1000 (highest priority)
        2. PKS library matches: feasibility_score + 500 (medium priority)
        3. Other fragments: raw feasibility_score (0-1 scale)

        The feasibility_score is computed as:
        - Enzymatic: DORA-XGB probability (if available), else sigmoid(ΔH)
        - Synthetic: sigmoid-transformed ΔH
        - Unknown: 1.0 (assume feasible)

        Args:
            fragments: List of all valid fragments from DORAnet expansion.

        Returns:
            List of top-scoring fragments, limited to max_children_per_expand.
        """
        if self.child_downselection_strategy == "first_N":
            # Simple truncation (shouldn't reach here, but just in case)
            return fragments[:self.max_children_per_expand]

        elif self.child_downselection_strategy == "hybrid":
            # Score each fragment based on priority criteria
            scored_fragments: List[Tuple[float, int, "DORAnetMCTS.FragmentInfo"]] = []

            for idx, frag in enumerate(fragments):
                score = 0.0

                # Priority 1: Sink compounds get highest score (1000 points)
                if self._is_sink_compound(frag.smiles):
                    score += 1000.0

                # Priority 2: PKS library matches get medium score (500 points)
                elif self._is_in_pks_library(frag.smiles):
                    score += 500.0

                # Priority 3: Smaller MW gets higher base score
                # Normalize MW to 0-100 range (smaller = higher score)
                if frag.molecule is not None:
                    frag_mw = Descriptors.MolWt(frag.molecule)
                    # Score inversely proportional to MW, capped at target MW
                    # Fragments at 0 MW get 100 points, fragments at target_MW get 0 points
                    mw_score = max(0, 100 * (1 - frag_mw / max(self.target_MW, 1)))
                    score += mw_score

                # Use negative index as tiebreaker to maintain original order for equal scores
                scored_fragments.append((score, -idx, frag))

            # Sort by score descending (higher score = higher priority)
            scored_fragments.sort(key=lambda x: (x[0], x[1]), reverse=True)

            # Return top N fragments
            return [frag for _, _, frag in scored_fragments[:self.max_children_per_expand]]

        elif self.child_downselection_strategy == "most_thermo_feasible":
            # Score fragments by thermodynamic feasibility with priority bonuses
            # for known terminals (sink compounds, PKS library matches)
            scored_fragments: List[Tuple[float, int, "DORAnetMCTS.FragmentInfo"]] = []

            for idx, frag in enumerate(fragments):
                # Use pre-computed feasibility score, default to 1.0 if not available
                feas_score = frag.feasibility_score if frag.feasibility_score is not None else 1.0

                # Priority bonuses (same as hybrid) to preserve terminal prioritization
                if self._is_sink_compound(frag.smiles):
                    score = feas_score + 1000.0
                elif self._is_in_pks_library(frag.smiles):
                    score = feas_score + 500.0
                else:
                    score = feas_score

                # Use negative index as tiebreaker to maintain original order for equal scores
                scored_fragments.append((score, -idx, frag))

            # Sort by score descending (higher score = higher priority)
            scored_fragments.sort(key=lambda x: (x[0], x[1]), reverse=True)

            # Return top N fragments
            return [frag for _, _, frag in scored_fragments[:self.max_children_per_expand]]

        else:
            # Unknown strategy, fall back to first_N
            return fragments[:self.max_children_per_expand]

    def select(self, node: Node) -> Optional[Node]:
        """
        Traverse tree to find a leaf node to expand using the configured selection policy.

        Selection policies:
        - "UCB1": Standard UCB1 with infinite score for unvisited nodes (breadth-first tendency)
        - "depth_biased": UCB1 + depth_bonus_coefficient * depth (depth-first tendency)

        Returns the selected leaf node, or None if no valid node found.
        Sink compounds and PKS terminal nodes are skipped as they are terminal nodes.
        """
        # Large base score for unvisited nodes in depth_biased mode
        # (not infinite, so depth bonus can differentiate between unvisited nodes)
        UNVISITED_BASE_SCORE = 1000.0

        while node.children:
            best_node: Optional[Node] = None
            best_score = -math.inf
            log_parent_visits = math.log(max(node.visits, 1))

            for child in node.children:
                # Skip terminal nodes - they don't need further expansion
                # (sink compounds are commercially available, PKS terminals can be synthesized by PKS)
                if child.is_sink_compound or child.is_pks_terminal:
                    continue

                if self.selection_policy == "UCB1":
                    # Standard UCB1: unvisited nodes get infinite score (pure breadth-first for unvisited)
                    if child.visits == 0:
                        score = math.inf
                    else:
                        exploit = child.value / child.visits
                        explore = math.sqrt(2 * log_parent_visits / child.visits)
                        score = exploit + explore
                else:
                    # Depth-biased UCB1: add depth bonus to encourage deeper exploration
                    depth_bonus = self.depth_bonus_coefficient * child.depth

                    if child.visits == 0:
                        # Unvisited nodes get high base score + depth bonus
                        # Deeper unvisited nodes are preferred over shallower ones
                        score = UNVISITED_BASE_SCORE + depth_bonus
                    else:
                        # Standard UCB1 + depth bonus
                        exploit = child.value / child.visits
                        explore = math.sqrt(2 * log_parent_visits / child.visits)
                        score = exploit + explore + depth_bonus

                child.selection_score = score

                if score > best_score:
                    best_score = score
                    best_node = child

            if best_node is None:
                return None

            node = best_node

        # Don't return terminal nodes as they can't be expanded
        if node.is_sink_compound or node.is_pks_terminal:
            return None

        return node

    def _is_in_pks_library(self, smiles: str) -> bool:
        """Check if a SMILES string is in the PKS library."""
        if not self.pks_library:
            return False
        canonical = _canonicalize_smiles(smiles)
        return canonical is not None and canonical in self.pks_library

    def _is_sink_compound(self, smiles: str) -> bool:
        """Check if a SMILES string is a sink compound (commercially available building block)."""
        if not self.sink_compounds:
            return False
        canonical = _canonicalize_smiles(smiles)
        return canonical is not None and canonical in self.sink_compounds

    def _get_sink_compound_type(self, smiles: str) -> Optional[str]:
        """
        Get the type of sink compound for a SMILES string.

        Returns:
            "biological" if in biological building blocks,
            "chemical" if in chemical building blocks,
            None if not a sink compound.
        """
        canonical = _canonicalize_smiles(smiles)
        if canonical is None:
            return None

        # Check biological first (smaller set, likely faster)
        if canonical in self.biological_sink_compounds:
            return "biological"
        if canonical in self.chemical_sink_compounds:
            return "chemical"

        return None

    def _is_prohibited_chemical(self, smiles: str) -> bool:
        """
        Check if a SMILES string is a prohibited chemical.

        Prohibited chemicals are hazardous or controlled substances that should
        never appear as intermediates in synthesis pathways.

        Args:
            smiles: SMILES string to check.

        Returns:
            True if the molecule is prohibited, False otherwise.
        """
        if not self.prohibited_chemicals:
            return False
        canonical = _canonicalize_smiles(smiles)
        return canonical is not None and canonical in self.prohibited_chemicals

    def _exceeds_MW_threshold(self, molecule: Chem.Mol) -> bool:
        """
        Check if a molecule's MW exceeds the allowed threshold.

        Fragments larger than MW_multiple_to_exclude times the target MW are
        filtered out. This prevents unrealistic dimerization products from
        enzymatic operators that cause molecules to grow rather than fragment.

        Args:
            molecule: RDKit Mol object to check.

        Returns:
            True if the molecule exceeds the MW threshold, False otherwise.
        """
        if molecule is None:
            return True  # Filter out invalid molecules
        fragment_MW = Descriptors.MolWt(molecule)
        return fragment_MW > self.max_fragment_MW

    def expand(self, node: Node) -> List[Node]:
        """
        Expand a node by applying DORAnet retro-transformations.

        Creates child nodes for each fragment generated. Sink compounds
        (building blocks) are marked as terminal during expansion since
        they are known terminals with known value.

        Rollouts for non-sink children (e.g., PKS matching and RetroTide
        spawning) are handled separately in run() via the rollout policy.

        Returns:
            List of newly created child nodes.
        """
        fragment_infos: List[Tuple[DORAnetMCTS.FragmentInfo, str]] = []

        if self.use_enzymatic:
            for frag_info in self._generate_doranet_fragments(node.fragment, "enzymatic"):
                fragment_infos.append((frag_info, "enzymatic"))

        if self.use_synthetic:
            for frag_info in self._generate_doranet_fragments(node.fragment, "synthetic"):
                fragment_infos.append((frag_info, "synthetic"))

        new_children: List[Node] = []

        for frag_info, provenance in fragment_infos:
            # Check if this fragment is a prohibited chemical - skip it entirely
            if self._is_prohibited_chemical(frag_info.smiles):
                print(f"[DORAnet] Fragment {frag_info.smiles} is a PROHIBITED CHEMICAL - skipping")
                continue

            # Check if this fragment exceeds the MW threshold (unrealistic dimerization)
            if self._exceeds_MW_threshold(frag_info.molecule):
                frag_MW = Descriptors.MolWt(frag_info.molecule) if frag_info.molecule else 0
                print(f"[DORAnet] Fragment {frag_info.smiles} exceeds MW threshold "
                      f"({frag_MW:.1f} > {self.max_fragment_MW:.1f}) - skipping")
                continue

            # Create child node with reaction information
            child = Node(
                fragment=frag_info.molecule,
                parent=node,
                provenance=provenance,
                reaction_smarts=frag_info.reaction_smarts,
                reaction_name=frag_info.reaction_name,
                reactants_smiles=frag_info.reactants_smiles,
                products_smiles=frag_info.products_smiles,
            )

            # Score feasibility for enzymatic reactions using DORA-XGB
            if provenance == "enzymatic":
                score, label = self.feasibility_scorer.score_reaction(
                    reactants_smiles=frag_info.reactants_smiles,
                    products_smiles=frag_info.products_smiles,
                    provenance=provenance
                )
                child.feasibility_score = score
                child.feasibility_label = label

            # Score thermodynamic feasibility using pathermo (both enzymatic and synthetic)
            delta_h, thermo_label = self.thermodynamic_scorer.score_reaction(
                reactants_smiles=frag_info.reactants_smiles,
                products_smiles=frag_info.products_smiles,
                provenance=provenance
            )
            child.enthalpy_of_reaction = delta_h
            child.thermodynamic_label = thermo_label

            node.add_child(child)
            self.nodes.append(child)
            self.edges.append((node.node_id, child.node_id))
            new_children.append(child)

            # Check if this fragment is a sink compound (commercially available building block)
            # This is a cheap check for known terminals - no rollout needed
            sink_type = self._get_sink_compound_type(frag_info.smiles)
            if sink_type:
                child.is_sink_compound = True
                child.sink_compound_type = sink_type
                # Mark as expanded so we don't try to expand it further
                child.expanded = True
                type_label = "BIOLOGICAL" if sink_type == "biological" else "CHEMICAL"
                print(f"[DORAnet] Fragment {frag_info.smiles} is a {type_label} BUILDING BLOCK")
                # Note: Rollout and reward are handled in run() for consistency

        node.expanded = True
        return new_children

    def _launch_retrotide_agent(self, target: Chem.Mol, source_node: Node) -> RetroTideResult:
        """
        Spawn a RetroTide MCTS search to synthesize the given fragment.

        Args:
            target: The fragment molecule to synthesize.
            source_node: The DORAnet node that produced this fragment.
        """
        target_smiles = Chem.MolToSmiles(target)
        print(f"[DORAnet] Spawning RetroTide search for: {target_smiles}")

        source_node.retrotide_attempted = True
        root = RetroTideNode(PKS_product=None, PKS_design=None, parent=None, depth=0)
        agent = RetroTideMCTS(
            root=root,
            target_molecule=target,
            **self.retrotide_kwargs,
        )
        agent.run()

        # Extract results from the RetroTide agent
        successful_nodes = getattr(agent, 'successful_nodes', set())
        simulated_successes = getattr(agent, 'successful_simulated_designs', [])
        num_successful = len(successful_nodes)
        num_sim_success = len(simulated_successes)

        # Get best score from successful nodes or from all nodes
        best_score = 0.0
        if successful_nodes:
            best_score = 1.0  # Target was reached
        else:
            # Get best score from any node
            for node in getattr(agent, 'nodes', []):
                if hasattr(node, 'value') and node.visits > 0:
                    avg_value = node.value / node.visits
                    best_score = max(best_score, avg_value)

        # Create result record with full traceability
        result = RetroTideResult(
            doranet_node_id=source_node.node_id,
            doranet_node_smiles=source_node.smiles or "",
            doranet_node_depth=source_node.depth,
            doranet_node_provenance=source_node.provenance or "unknown",
            doranet_reaction_name=source_node.reaction_name,
            doranet_reaction_smarts=source_node.reaction_smarts,
            doranet_reactants_smiles=source_node.reactants_smiles or [],
            doranet_products_smiles=source_node.products_smiles or [],
            retrotide_target_smiles=target_smiles,
            retrotide_successful=(num_successful > 0 or num_sim_success > 0),
            retrotide_num_successful_nodes=num_successful + num_sim_success,
            retrotide_best_score=best_score,
            retrotide_total_nodes=len(getattr(agent, 'nodes', [])),
            retrotide_agent=agent,
        )
        self.retrotide_results.append(result)
        return result

    def calculate_reward(self, node: Node) -> float:
        """
        Calculate reward for a node using the reward policy.

        This method delegates to the configured reward_policy. If no policy
        is configured, falls back to the original sparse reward logic.

        Returns:
            Reward value computed by the reward policy.
        """
        if hasattr(self, 'reward_policy') and self.reward_policy is not None:
            context = self._build_rollout_context()
            return self.reward_policy.calculate_reward(node, context)

        # Fallback: original sparse reward logic (for backward compatibility)
        if node.is_sink_compound:
            return self.sink_terminal_reward

        if node.is_pks_terminal:
            return 1.0

        if not self.pks_library:
            return 0.0

        smiles = node.smiles
        if smiles is None:
            return 0.0

        canonical = _canonicalize_smiles(smiles)
        if canonical and canonical in self.pks_library:
            return 1.0

        return 0.0

    def backpropagate(self, node: Node, reward: float) -> None:
        """
        Backpropagate reward from a node up to the root.

        Args:
            node: The node to start backpropagation from.
            reward: The reward value to propagate.
        """
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def run(self) -> None:
        """
        Execute the MCTS loop: Selection → Expansion → Rollout → Backpropagation.

        For each expanded child:
        - Sink compounds: Use reward policy (known terminal, no rollout needed)
        - Non-sink compounds: Use rollout policy to simulate and get reward
        """
        print(f"[DORAnet] Starting MCTS with {self.total_iterations} iterations, "
              f"max_depth={self.max_depth}")

        # Build context for policies
        context = self._build_rollout_context()

        # Track current iteration for visualization
        self.current_iteration = 0

        for iteration in range(self.total_iterations):
            self.current_iteration = iteration

            # Selection: find a leaf node using UCB1
            leaf = self.select(self.root)
            if leaf is None:
                print(f"[DORAnet] No valid leaf found at iteration {iteration}, stopping.")
                break

            # Track when this node was selected
            leaf.selected_at_iterations.append(iteration)

            # Check depth limit
            if leaf.depth >= self.max_depth:
                # Backpropagate with current reward and continue
                reward = self.calculate_reward(leaf)
                self.backpropagate(leaf, reward)
                continue

            # Expansion: generate fragments
            if not leaf.expanded:
                leaf.expanded_at_iteration = iteration
                new_children = self.expand(leaf)
                
                terminals_found = 0
                rollouts_performed = 0

                # Process each child: PKS matches get rollout, pure sinks get reward
                for child in new_children:
                    child.created_at_iteration = iteration

                    # Check if this node matches PKS library (potential for RetroTide verification)
                    is_pks_library_match = self._is_in_pks_library(child.smiles or "")

                    if is_pks_library_match:
                        # PKS library matches: always run rollout (even if sink compound)
                        # This enables RetroTide spawning for PKS verification
                        print(f"[DORAnet] Fragment {child.smiles} is PKS library match - "
                              f"attempting RetroTide (sink={child.is_sink_compound})")

                        result = self.rollout_policy.rollout(child, context)

                        if result.terminal:
                            child.is_pks_terminal = True
                            child.expanded = True
                            terminals_found += 1

                            if "retrotide_agent" in result.metadata:
                                self._store_retrotide_result_from_rollout(child, result)

                            reward = result.reward
                        else:
                            # RetroTide failed - fall back to sink compound reward if applicable
                            if child.is_sink_compound:
                                reward = self.reward_policy.calculate_reward(child, context)
                                if reward > 0:
                                    terminals_found += 1
                            else:
                                reward = result.reward

                        if not isinstance(self.rollout_policy, NoOpRolloutPolicy):
                            rollouts_performed += 1

                    elif child.is_sink_compound:
                        # Pure sink compound (not in PKS library) - use reward policy directly
                        reward = self.reward_policy.calculate_reward(child, context)
                        if reward > 0:
                            terminals_found += 1
                    else:
                        # Non-sink, non-PKS: standard rollout
                        result = self.rollout_policy.rollout(child, context)

                        if result.terminal:
                            child.is_pks_terminal = True
                            child.expanded = True
                            terminals_found += 1

                            if "retrotide_agent" in result.metadata:
                                self._store_retrotide_result_from_rollout(child, result)

                        if not isinstance(self.rollout_policy, NoOpRolloutPolicy):
                            rollouts_performed += 1

                        reward = result.reward

                    self.backpropagate(child, reward)

                # Log iteration progress
                log_parts = [
                    f"[DORAnet] Iteration {iteration}: expanded node {leaf.node_id} (depth={leaf.depth})",
                    f"created {len(new_children)} children",
                    f"{terminals_found} terminals",
                ]
                if rollouts_performed > 0:
                    log_parts.append(f"{rollouts_performed} rollouts")
                print(", ".join(log_parts))

            else:
                # Node already expanded, just backpropagate
                reward = self.calculate_reward(leaf)
                self.backpropagate(leaf, reward)

            # Generate iteration visualization if enabled
            if self.enable_iteration_visualizations:
                if (iteration + 1) % self.iteration_viz_interval == 0:
                    self._generate_iteration_visualization(iteration)

        # Summary statistics
        sink_count = len(self.get_sink_compounds())
        pks_terminal_count = len(self.get_pks_terminal_nodes())
        print(f"[DORAnet] MCTS complete. Total nodes: {len(self.nodes)}, "
              f"PKS terminals: {pks_terminal_count}, Sink compounds: {sink_count}, "
              f"RetroTide results: {len(self.retrotide_results)}")

        # Log SMILES canonicalization cache statistics
        cache_info = _canonicalize_smiles.cache_info()
        hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses) * 100 if (cache_info.hits + cache_info.misses) > 0 else 0
        print(f"[DORAnet] SMILES cache: {cache_info.hits} hits, {cache_info.misses} misses "
              f"({hit_rate:.1f}% hit rate), {cache_info.currsize}/{cache_info.maxsize} cached")

    def _store_retrotide_result_from_rollout(
        self, node: Node, rollout_result: RolloutResult
    ) -> None:
        """
        Store RetroTide results from a rollout policy for traceability.

        Args:
            node: The node that was rolled out.
            rollout_result: The result from the rollout policy.
        """
        metadata = rollout_result.metadata
        result = RetroTideResult(
            doranet_node_id=node.node_id,
            doranet_node_smiles=node.smiles or "",
            doranet_node_depth=node.depth,
            doranet_node_provenance=node.provenance or "unknown",
            doranet_reaction_name=node.reaction_name,
            doranet_reaction_smarts=node.reaction_smarts,
            doranet_reactants_smiles=node.reactants_smiles or [],
            doranet_products_smiles=node.products_smiles or [],
            retrotide_target_smiles=metadata.get("retrotide_target_smiles", ""),
            retrotide_successful=metadata.get("retrotide_successful", False),
            retrotide_num_successful_nodes=metadata.get("retrotide_num_successful_nodes", 0),
            retrotide_best_score=metadata.get("retrotide_best_score", 0.0),
            retrotide_total_nodes=metadata.get("retrotide_total_nodes", 0),
            retrotide_agent=metadata.get("retrotide_agent"),
        )
        self.retrotide_results.append(result)

    def get_tree_summary(self, include_iteration_info: bool = False) -> str:
        """
        Return a summary of the search tree.

        Args:
            include_iteration_info: If True, include detailed iteration diagnostics
                (created_at, selected_at, expanded_at) for each node.
        """
        lines = ["DORAnet MCTS Tree Summary:", "=" * 40]
        for node in self.nodes:
            indent = "  " * node.depth
            # Indicate node type: sink compound (with type), PKS terminal, or neither
            if node.is_sink_compound:
                sink_type = node.sink_compound_type or "unknown"
                marker = f"■SINK({sink_type})"
            elif node.is_pks_terminal:
                marker = "✓PKS_TERMINAL"
            elif self._is_in_pks_library(node.smiles or ""):
                marker = "✓PKS"
            else:
                marker = ""
            avg_value = f"{node.value / node.visits:.2f}" if node.visits > 0 else "N/A"
            lines.append(f"{indent}Node {node.node_id}: {node.smiles} "
                        f"(depth={node.depth}, visits={node.visits}, value={avg_value}, "
                        f"via={node.provenance}) {marker}")

            # Add iteration diagnostics if requested
            if include_iteration_info:
                created = node.created_at_iteration
                expanded = node.expanded_at_iteration
                selected = node.selected_at_iterations

                created_str = f"created@iter={created}" if created is not None else "created@iter=0"
                expanded_str = f"expanded@iter={expanded}" if expanded is not None else "NOT_EXPANDED"
                if selected:
                    selected_str = f"selected@iters=[{','.join(map(str, selected))}]"
                else:
                    selected_str = "NEVER_SELECTED"

                lines.append(f"{indent}  └─ {created_str}, {expanded_str}, {selected_str}")

        return "\n".join(lines)

    def get_pks_matches(self) -> List[Node]:
        """Return nodes whose fragments match the PKS library."""
        return [n for n in self.nodes if self.calculate_reward(n) > 0]

    def get_sink_compounds(self) -> List[Node]:
        """Return nodes that are sink compounds (commercially available building blocks)."""
        return [n for n in self.nodes if n.is_sink_compound]

    def get_biological_sink_compounds(self) -> List[Node]:
        """Return nodes that are biological building blocks."""
        return [n for n in self.nodes if n.is_sink_compound and n.sink_compound_type == "biological"]

    def get_chemical_sink_compounds(self) -> List[Node]:
        """Return nodes that are chemical building blocks."""
        return [n for n in self.nodes if n.is_sink_compound and n.sink_compound_type == "chemical"]

    def get_pks_terminal_nodes(self) -> List[Node]:
        """Return nodes that are PKS terminals (can be synthesized by polyketide synthases)."""
        return [n for n in self.nodes if n.is_pks_terminal]

    def get_pathway_to_node(self, node: Node) -> List[Node]:
        """
        Trace the pathway from root to a given node.

        Returns:
            List of nodes from root to the given node (inclusive).
        """
        pathway = []
        current = node
        while current is not None:
            pathway.append(current)
            current = current.parent
        return list(reversed(pathway))

    def format_pathway(self, node: Node) -> str:
        """
        Format the pathway from root to node as a readable string.

        Returns:
            Multi-line string showing each step in the pathway.
        """
        pathway = self.get_pathway_to_node(node)
        if len(pathway) <= 1:
            return f"  Target molecule (no transformations needed)"

        lines = []
        for i, step_node in enumerate(pathway):
            if i == 0:
                lines.append(f"  Step 0 (Target): {step_node.smiles}")
            else:
                rxn_info = ""
                if step_node.reaction_name:
                    # Extract a readable portion of the reaction
                    rxn_short = step_node.reaction_name[:60] + "..." if len(step_node.reaction_name) > 60 else step_node.reaction_name
                    rxn_info = f"\n           Reaction: {rxn_short}"
                lines.append(f"  Step {i} ({step_node.provenance}): {step_node.smiles}{rxn_info}")

        return "\n".join(lines)

    def format_reaction_pathway(self, node: Node) -> str:
        """
        Format the pathway from root to node using reaction equations on edges.

        Returns:
            Multi-line string showing each reaction step as
            "reactants>>products" plus branched product annotations.
        """
        pathway = self.get_pathway_to_node(node)
        if len(pathway) <= 1:
            return "  Target molecule (no transformations needed)"

        lines = []
        for i in range(1, len(pathway)):
            step_node = pathway[i]
            rxn_label = step_node.reaction_name or "Unknown reaction"
            reactants = step_node.reactants_smiles or []
            products = step_node.products_smiles or []

            if reactants or products:
                rxn_equation = f"{'.'.join(reactants)}>>{'.'.join(products)}"
            else:
                rxn_equation = "N/A"

            lines.append(f"  Step {i} ({step_node.provenance}): {rxn_label}")
            lines.append(f"           Reaction: {rxn_equation}")

            # Add feasibility info for enzymatic reactions (DORA-XGB)
            if step_node.provenance == "enzymatic":
                if step_node.feasibility_score is not None:
                    label_str = "feasible" if step_node.feasibility_label == 1 else "infeasible"
                    lines.append(f"           Feasibility: Score={step_node.feasibility_score:.3f}, Label={step_node.feasibility_label} ({label_str})")
                else:
                    lines.append(f"           Feasibility: N/A (not scored)")

            # Add thermodynamic info for all reactions (pathermo)
            if step_node.enthalpy_of_reaction is not None:
                label_str = "feasible" if step_node.thermodynamic_label == 1 else "infeasible"
                lines.append(f"           Thermodynamics: ΔH={step_node.enthalpy_of_reaction:.2f} kcal/mol, Label={step_node.thermodynamic_label} ({label_str})")
            else:
                lines.append(f"           Thermodynamics: N/A (not scored)")

            lines.append(f"           Node: {step_node.node_id} | Fragment: {step_node.smiles}")

            # Annotate branched products with sink compound type if available.
            if products:
                primary_smiles = step_node.smiles or ""
                primary_canonical = _canonicalize_smiles(primary_smiles) or primary_smiles
                lines.append("           Products:")
                for prod in products:
                    prod_canonical = _canonicalize_smiles(prod) or prod
                    is_primary = prod_canonical == primary_canonical
                    role = "primary" if is_primary else "branch"
                    sink_type = self._get_sink_compound_type(prod)
                    sink_label = f"sink={sink_type}" if sink_type else "sink=No"
                    lines.append(f"             - {prod} [{role}, {sink_label}]")

        return "\n".join(lines)

    @staticmethod
    def format_pks_module(module, module_num: int) -> str:
        """
        Format a PKS module with detailed domain information.

        Args:
            module: A PKS module object from bcs.Cluster
            module_num: The module number (1-indexed)

        Returns:
            Formatted string describing the module domains.
        """
        lines = [f"      Module {module_num}:"]

        # Try to extract domain information from module
        mod_str = str(module)

        # Parse the module string to extract domain info
        # Typical format: ["AT{'substrate': 'xxx'}", "KR{'type': 'xxx'}", "DH{}", "ER{}", "loading: False"]

        # AT domain (Acyltransferase) - determines substrate
        if "AT{" in mod_str:
            import re
            at_match = re.search(r"AT\{'substrate':\s*'([^']+)'", mod_str)
            if at_match:
                substrate = at_match.group(1)
                # Map substrate codes to readable names
                substrate_names = {
                    'Malonyl-CoA': 'Malonyl-CoA (C2, no branch)',
                    'Methylmalonyl-CoA': 'Methylmalonyl-CoA (C2, methyl branch)',
                    'mxmal': 'Methoxymalonyl-ACP (C2, methoxy branch)',
                    'emal': 'Ethylmalonyl-CoA (C2, ethyl branch)',
                    'allylmal': 'Allylmalonyl-CoA (C2, allyl branch)',
                    'isobutmal': 'Isobutyrylmalonyl-CoA (C2, isobutyl branch)',
                    'D-isobutmal': 'D-Isobutyrylmalonyl-CoA (C2, D-isobutyl branch)',
                    'butmal': 'Butyrylmalonyl-CoA (C2, butyl branch)',
                    'hexmal': 'Hexylmalonyl-CoA (C2, hexyl branch)',
                    'hmal': 'Hydroxymalonyl-ACP (C2, hydroxy branch)',
                    'DCP': 'Dichloropropionyl-ACP',
                }
                substrate_desc = substrate_names.get(substrate, substrate)
                lines.append(f"        AT (Acyltransferase): {substrate_desc}")

        # KR domain (Ketoreductase) - determines stereochemistry
        if "KR{" in mod_str:
            import re
            kr_match = re.search(r"KR\{'type':\s*'([^']+)'", mod_str)
            if kr_match:
                kr_type = kr_match.group(1)
                kr_desc = {
                    'A1': 'A1 (L-hydroxyl, syn methyl)',
                    'A2': 'A2 (L-hydroxyl, anti methyl)',
                    'B1': 'B1 (D-hydroxyl, syn methyl)',
                    'B2': 'B2 (D-hydroxyl, anti methyl)',
                    'B': 'B (D-hydroxyl)',
                    'C1': 'C1 (inactive, syn methyl)',
                    'C2': 'C2 (inactive, anti methyl)',
                }.get(kr_type, kr_type)
                lines.append(f"        KR (Ketoreductase): {kr_desc}")
            else:
                lines.append(f"        KR (Ketoreductase): present")

        # DH domain (Dehydratase) - removes water to form double bond
        if "DH{}" in mod_str:
            lines.append(f"        DH (Dehydratase): active (forms C=C double bond)")

        # ER domain (Enoylreductase) - reduces double bond
        if "ER{}" in mod_str:
            lines.append(f"        ER (Enoylreductase): active (reduces C=C to C-C)")

        # Loading module
        if "loading: True" in mod_str:
            lines.append(f"        Loading Module: Yes (starter unit)")
        elif "loading: False" in mod_str:
            lines.append(f"        Loading Module: No (extension module)")

        return "\n".join(lines)

    @staticmethod
    def format_pks_cluster(cluster, score: float) -> str:
        """
        Format a full PKS cluster with all module details.

        Args:
            cluster: A bcs.Cluster object
            score: The design score

        Returns:
            Formatted string describing the full PKS design.
        """
        lines = [
            f"    Design Score: {score:.4f}",
        ]

        if hasattr(cluster, 'modules'):
            num_modules = len(cluster.modules)
            lines.append(f"    Number of PKS Modules: {num_modules}")
            lines.append(f"    PKS Assembly Line:")

            for k, mod in enumerate(cluster.modules):
                lines.append(DORAnetMCTS.format_pks_module(mod, k + 1))

        return "\n".join(lines)

    @property
    def retrotide_runs(self) -> List:
        """Backwards compatibility: return list of RetroTide agents."""
        return [r.retrotide_agent for r in self.retrotide_results if r.retrotide_agent]

    def get_successful_results(self) -> List[RetroTideResult]:
        """Return only the RetroTide results that found successful PKS designs."""
        return [r for r in self.retrotide_results if r.retrotide_successful]

    def get_results_summary(self) -> str:
        """Return a detailed summary of all RetroTide results."""
        lines = [
            "",
            "=" * 70,
            "RetroTide Results Summary",
            "=" * 70,
            "",
            f"Total RetroTide searches: {len(self.retrotide_results)}",
            f"Successful searches: {len(self.get_successful_results())}",
            "",
        ]

        # Group by success
        successful = self.get_successful_results()
        unsuccessful = [r for r in self.retrotide_results if not r.retrotide_successful]

        if successful:
            lines.append("✅ SUCCESSFUL PKS DESIGNS:")
            lines.append("-" * 70)
            for r in successful:
                lines.append(f"  DORAnet Node {r.doranet_node_id}:")
                lines.append(f"    Fragment SMILES: {r.doranet_node_smiles}")
                lines.append(f"    Depth: {r.doranet_node_depth}, Provenance: {r.doranet_node_provenance}")
                lines.append(f"    RetroTide Target: {r.retrotide_target_smiles}")
                lines.append(f"    Successful Nodes: {r.retrotide_num_successful_nodes}")
                lines.append(f"    Best Score: {r.retrotide_best_score:.3f}")
                lines.append(f"    Total RetroTide Nodes: {r.retrotide_total_nodes}")
                lines.append("")

        if unsuccessful:
            lines.append("❌ UNSUCCESSFUL SEARCHES (top 5 by score):")
            lines.append("-" * 70)
            # Sort by best score descending
            sorted_unsuccessful = sorted(unsuccessful, key=lambda r: r.retrotide_best_score, reverse=True)
            for r in sorted_unsuccessful[:5]:
                lines.append(f"  DORAnet Node {r.doranet_node_id}: {r.doranet_node_smiles[:40]}...")
                lines.append(f"    Provenance: {r.doranet_node_provenance}, Best Score: {r.retrotide_best_score:.3f}")
            lines.append("")

        return "\n".join(lines)

    def save_results(self, output_path: str) -> None:
        """
        Save detailed results to a log file.

        Args:
            output_path: Path to the output file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # Header
            f.write("=" * 70 + "\n")
            f.write("DORAnet + RetroTide Multi-Agent MCTS Results\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n\n")

            # Target info
            f.write("TARGET MOLECULE\n")
            f.write("-" * 70 + "\n")
            f.write(f"SMILES: {Chem.MolToSmiles(self.target_molecule)}\n\n")

            # DORAnet configuration
            f.write("DORANET CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total iterations: {self.total_iterations}\n")
            f.write(f"Max depth: {self.max_depth}\n")
            f.write(f"Use enzymatic: {self.use_enzymatic}\n")
            f.write(f"Use synthetic: {self.use_synthetic}\n")
            f.write(f"Max children per expand: {self.max_children_per_expand}\n")
            f.write(f"Child downselection strategy: {self.child_downselection_strategy}\n")
            f.write(f"Spawn RetroTide: {self.spawn_retrotide}\n")
            f.write(f"Sink compounds library size: {len(self.sink_compounds)}\n\n")

            # DORAnet tree with iteration diagnostics
            f.write("DORANET SEARCH TREE (with iteration diagnostics)\n")
            f.write("-" * 70 + "\n")
            f.write(self.get_tree_summary(include_iteration_info=True) + "\n\n")

            # Selection diagnostics summary
            f.write("=" * 70 + "\n")
            f.write("NODE SELECTION DIAGNOSTICS\n")
            f.write("=" * 70 + "\n\n")

            # Count nodes by selection status
            never_selected = [n for n in self.nodes if not n.selected_at_iterations]
            selected_once = [n for n in self.nodes if len(n.selected_at_iterations) == 1]
            selected_multiple = [n for n in self.nodes if len(n.selected_at_iterations) > 1]
            expanded_nodes = [n for n in self.nodes if n.expanded_at_iteration is not None]
            not_expanded = [n for n in self.nodes if n.expanded_at_iteration is None and not n.is_sink_compound and not n.is_pks_terminal]

            f.write("SELECTION SUMMARY:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total nodes: {len(self.nodes)}\n")
            f.write(f"Nodes never selected: {len(never_selected)} ({100*len(never_selected)/max(len(self.nodes),1):.1f}%)\n")
            f.write(f"Nodes selected once: {len(selected_once)} ({100*len(selected_once)/max(len(self.nodes),1):.1f}%)\n")
            f.write(f"Nodes selected multiple times: {len(selected_multiple)} ({100*len(selected_multiple)/max(len(self.nodes),1):.1f}%)\n")
            f.write(f"Nodes expanded: {len(expanded_nodes)}\n")
            f.write(f"Nodes NOT expanded (non-terminal): {len(not_expanded)}\n\n")

            # Nodes by depth that were never selected
            f.write("NODES NEVER SELECTED (by depth):\n")
            f.write("-" * 70 + "\n")
            max_depth_found = max((n.depth for n in self.nodes), default=0)
            for d in range(max_depth_found + 1):
                nodes_at_depth = [n for n in self.nodes if n.depth == d]
                never_sel_at_depth = [n for n in never_selected if n.depth == d]
                f.write(f"  Depth {d}: {len(never_sel_at_depth)}/{len(nodes_at_depth)} never selected\n")
            f.write("\n")

            # Nodes that could have been expanded but weren't
            if not_expanded:
                f.write("NODES NOT EXPANDED (could have deeper exploration):\n")
                f.write("-" * 70 + "\n")
                # Group by depth
                for d in range(max_depth_found + 1):
                    not_exp_at_depth = [n for n in not_expanded if n.depth == d]
                    if not_exp_at_depth:
                        f.write(f"  Depth {d}: {len(not_exp_at_depth)} nodes not expanded\n")
                        # Show first few examples
                        for n in not_exp_at_depth[:5]:
                            smiles_short = n.smiles[:40] + "..." if n.smiles and len(n.smiles) > 40 else n.smiles
                            f.write(f"    Node {n.node_id}: {smiles_short} (visits={n.visits})\n")
                        if len(not_exp_at_depth) > 5:
                            f.write(f"    ... and {len(not_exp_at_depth) - 5} more\n")
                f.write("\n")

            # Most selected nodes (hotspots)
            if selected_multiple:
                f.write("MOST FREQUENTLY SELECTED NODES (hotspots):\n")
                f.write("-" * 70 + "\n")
                sorted_by_selections = sorted(selected_multiple, key=lambda n: len(n.selected_at_iterations), reverse=True)
                for n in sorted_by_selections[:10]:
                    smiles_short = n.smiles[:40] + "..." if n.smiles and len(n.smiles) > 40 else n.smiles
                    f.write(f"  Node {n.node_id} (depth={n.depth}): selected {len(n.selected_at_iterations)} times\n")
                    f.write(f"    SMILES: {smiles_short}\n")
                    f.write(f"    Selected at iterations: {n.selected_at_iterations[:20]}")
                    if len(n.selected_at_iterations) > 20:
                        f.write(f"... (+{len(n.selected_at_iterations) - 20} more)")
                    f.write("\n")
                f.write("\n")

            # PKS Library match summary for all DORAnet-generated precursors
            f.write("=" * 70 + "\n")
            f.write("DORANET PRECURSORS - PKS LIBRARY ANALYSIS\n")
            f.write("=" * 70 + "\n\n")

            if self.pks_library:
                f.write(f"PKS Library Size: {len(self.pks_library)} molecules\n\n")

                # Properly categorize nodes using actual PKS library membership check
                # (not calculate_reward which includes sink compounds)
                pks_library_matches = []  # Nodes whose SMILES are in PKS library
                pks_terminals = []        # Nodes verified by RetroTide
                non_pks_nodes = []        # Nodes not in PKS library (excluding sink compounds)

                for n in self.nodes:
                    # Check actual PKS library membership
                    is_in_pks = self._is_in_pks_library(n.smiles or "")

                    if getattr(n, 'is_pks_terminal', False):
                        # Verified by RetroTide
                        pks_terminals.append(n)
                    elif is_in_pks:
                        # In PKS library but not yet verified
                        pks_library_matches.append(n)
                    elif not n.is_sink_compound:
                        # Not in PKS library and not a sink compound
                        non_pks_nodes.append(n)
                    # Sink compounds are reported separately below

                f.write(f"Total DORAnet nodes: {len(self.nodes)}\n")
                f.write(f"PKS terminals (RetroTide verified): {len(pks_terminals)}\n")
                f.write(f"PKS library matches (unverified): {len(pks_library_matches)}\n")
                f.write(f"Non-PKS nodes (excluding sinks): {len(non_pks_nodes)}\n\n")

                # List PKS terminals (RetroTide verified)
                if pks_terminals:
                    f.write("✅ PKS TERMINALS (RETROTIDE VERIFIED):\n")
                    f.write("-" * 70 + "\n")
                    for node in pks_terminals:
                        f.write(f"  Node {node.node_id} (depth={node.depth}, {node.provenance}): {node.smiles}\n")
                    f.write("\n")

                    # Detailed pathways for PKS terminals
                    f.write("=" * 70 + "\n")
                    f.write("DORANET PATHWAYS TO PKS-VERIFIED FRAGMENTS\n")
                    f.write("=" * 70 + "\n\n")

                    for i, node in enumerate(pks_terminals):
                        f.write(f"PATHWAY #{i + 1}: Node {node.node_id}\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Final Fragment: {node.smiles}\n")
                        f.write(f"Depth: {node.depth}, Provenance: {node.provenance}\n\n")
                        f.write("Retrosynthetic Route:\n")
                        f.write(self.format_pathway(node) + "\n\n")

                # List PKS library matches (not yet verified by RetroTide)
                if pks_library_matches:
                    f.write("⏳ PKS LIBRARY MATCHES (AWAITING RETROTIDE VERIFICATION):\n")
                    f.write("-" * 70 + "\n")
                    display_limit = min(50, len(pks_library_matches))
                    for node in pks_library_matches[:display_limit]:
                        f.write(f"  Node {node.node_id} (depth={node.depth}, {node.provenance}): {node.smiles}\n")
                    if len(pks_library_matches) > display_limit:
                        f.write(f"  ... and {len(pks_library_matches) - display_limit} more\n")
                    f.write("\n")

                # List non-PKS nodes (only first 20 to avoid clutter)
                if non_pks_nodes:
                    f.write("❌ NON-PKS NODES (not in PKS library, not sink compounds):\n")
                    f.write("-" * 70 + "\n")
                    display_limit = min(20, len(non_pks_nodes))
                    for node in non_pks_nodes[:display_limit]:
                        smiles_display = node.smiles[:50] + "..." if node.smiles and len(node.smiles) > 50 else node.smiles
                        f.write(f"  Node {node.node_id} (depth={node.depth}, {node.provenance}): {smiles_display}\n")
                    if len(non_pks_nodes) > display_limit:
                        f.write(f"  ... and {len(non_pks_nodes) - display_limit} more\n")
                    f.write("\n")
            else:
                f.write("No PKS library loaded - skipping PKS match analysis.\n\n")

            # Sink compounds section
            f.write("=" * 70 + "\n")
            f.write("SINK COMPOUNDS (COMMERCIALLY AVAILABLE BUILDING BLOCKS)\n")
            f.write("=" * 70 + "\n\n")

            sink_nodes = self.get_sink_compounds()
            bio_sinks = self.get_biological_sink_compounds()
            chem_sinks = self.get_chemical_sink_compounds()

            f.write(f"Sink compounds library size: {len(self.sink_compounds):,}\n")
            f.write(f"  - Biological building blocks: {len(self.biological_sink_compounds):,}\n")
            f.write(f"  - Chemical building blocks: {len(self.chemical_sink_compounds):,}\n\n")
            f.write(f"Sink compounds found in tree: {len(sink_nodes)}\n")
            f.write(f"  - Biological: {len(bio_sinks)}\n")
            f.write(f"  - Chemical: {len(chem_sinks)}\n\n")

            if bio_sinks:
                f.write("■ BIOLOGICAL BUILDING BLOCKS DISCOVERED:\n")
                f.write("-" * 70 + "\n")
                for node in bio_sinks:
                    f.write(f"  Node {node.node_id} (depth={node.depth}, {node.provenance}): {node.smiles}\n")
                f.write("\n")

            if chem_sinks:
                f.write("■ CHEMICAL BUILDING BLOCKS DISCOVERED:\n")
                f.write("-" * 70 + "\n")
                for node in chem_sinks:
                    f.write(f"  Node {node.node_id} (depth={node.depth}, {node.provenance}): {node.smiles}\n")
                f.write("\n")

            if sink_nodes:
                # Detailed pathways for each sink compound
                f.write("RETROSYNTHETIC PATHWAYS TO SINK COMPOUNDS:\n")
                f.write("-" * 70 + "\n\n")

                for i, node in enumerate(sink_nodes):
                    sink_type = node.sink_compound_type or "unknown"
                    f.write(f"PATHWAY #{i + 1}: Node {node.node_id} [{sink_type.upper()}]\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Sink Compound: {node.smiles}\n")
                    f.write(f"Type: {sink_type.capitalize()} building block\n")
                    f.write(f"Depth: {node.depth}, Provenance: {node.provenance}\n\n")
                    f.write("Retrosynthetic Route:\n")
                    f.write(self.format_pathway(node) + "\n\n")
            else:
                f.write("No sink compounds discovered in the search tree.\n\n")

            # RetroTide results summary
            f.write(self.get_results_summary() + "\n")

            # Detailed successful results
            successful = self.get_successful_results()
            if successful:
                f.write("\n" + "=" * 70 + "\n")
                f.write("DETAILED SUCCESSFUL RETROTIDE RESULTS\n")
                f.write("=" * 70 + "\n\n")

                for i, r in enumerate(successful):
                    f.write(f"SUCCESS #{i + 1}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"DORAnet Node ID: {r.doranet_node_id}\n")
                    f.write(f"Fragment SMILES: {r.doranet_node_smiles}\n")
                    f.write(f"Fragment Depth: {r.doranet_node_depth}\n")
                    f.write(f"Fragment Provenance: {r.doranet_node_provenance}\n")

                    # Find the DORAnet node and show the pathway
                    doranet_node = next((n for n in self.nodes if n.node_id == r.doranet_node_id), None)
                    if doranet_node:
                        f.write("\n--- DORAnet Pathway (Target → Fragment) ---\n")
                        f.write(self.format_pathway(doranet_node) + "\n")

                    # DORAnet reaction that created this fragment
                    f.write("\n--- DORAnet Reaction (how fragment was created) ---\n")
                    if r.doranet_reaction_name:
                        f.write(f"Reaction Name: {r.doranet_reaction_name}\n")
                    if r.doranet_reaction_smarts:
                        # Truncate very long SMARTS
                        smarts_display = r.doranet_reaction_smarts[:100] + "..." if len(r.doranet_reaction_smarts) > 100 else r.doranet_reaction_smarts
                        f.write(f"Reaction SMARTS: {smarts_display}\n")
                    if r.doranet_reactants_smiles:
                        f.write(f"Reactants: {' + '.join(r.doranet_reactants_smiles[:3])}\n")
                    if r.doranet_products_smiles:
                        f.write(f"Products: {' + '.join(r.doranet_products_smiles[:3])}\n")

                    f.write("\n--- RetroTide Forward Synthesis Results ---\n")
                    f.write(f"RetroTide Target: {r.retrotide_target_smiles}\n")
                    f.write(f"Successful PKS Nodes: {r.retrotide_num_successful_nodes}\n")
                    f.write(f"Best Score: {r.retrotide_best_score:.4f}\n")
                    f.write(f"Total RetroTide Nodes Explored: {r.retrotide_total_nodes}\n")

                    # If we have the agent, try to get PKS design details
                    if r.retrotide_agent:
                        agent = r.retrotide_agent
                        succ_nodes = getattr(agent, 'successful_nodes', set())
                        if succ_nodes:
                            f.write("\n--- Successful PKS Designs (Exact Matches) ---\n")
                            f.write(f"Found {len(succ_nodes)} design(s) that exactly synthesize the target.\n")
                            for j, pks_node in enumerate(list(succ_nodes)[:5]):  # Limit to 5
                                f.write(f"\n  PKS Design #{j + 1}:\n")
                                if hasattr(pks_node, 'PKS_product') and pks_node.PKS_product:
                                    prod_smi = Chem.MolToSmiles(pks_node.PKS_product)
                                    f.write(f"    Product SMILES: {prod_smi}\n")
                                if hasattr(pks_node, 'PKS_design') and pks_node.PKS_design:
                                    f.write(f"    Design Depth (# modules): {pks_node.depth}\n")
                                    # Use the new formatting function
                                    try:
                                        cluster, score, _ = pks_node.PKS_design
                                        f.write(self.format_pks_cluster(cluster, score) + "\n")
                                    except Exception as e:
                                        f.write(f"    (Could not extract module details: {e})\n")

                        # Also show simulated successful designs with detailed formatting
                        sim_designs = getattr(agent, 'successful_simulated_designs', [])
                        if sim_designs:
                            f.write("\n--- Successful Simulated PKS Designs ---\n")
                            f.write(f"Found {len(sim_designs)} design(s) that reached target in simulation.\n")
                            for j, design in enumerate(sim_designs[:5]):  # Limit to 5
                                f.write(f"\n  Simulated Design #{j + 1}:\n")
                                # Format each module in the design
                                if isinstance(design, (list, tuple)):
                                    f.write(f"    Number of Modules: {len(design)}\n")
                                    for k, mod in enumerate(design):
                                        f.write(self.format_pks_module(mod, k + 1) + "\n")
                                else:
                                    f.write(f"    Modules: {design}\n")

                        # Show best designs even if not successful (for debugging)
                        if not succ_nodes and not sim_designs:
                            f.write("\n--- Best PKS Designs (No Exact Match) ---\n")
                            f.write(f"No exact matches found. Showing best scoring designs:\n")
                            all_nodes = getattr(agent, 'nodes', [])
                            # Sort by value/visits ratio
                            scored_nodes = [(n, n.value / max(n.visits, 1)) for n in all_nodes if hasattr(n, 'PKS_design') and n.PKS_design]
                            scored_nodes.sort(key=lambda x: x[1], reverse=True)
                            for j, (pks_node, avg_score) in enumerate(scored_nodes[:3]):
                                f.write(f"\n  Best Design #{j + 1} (avg score: {avg_score:.4f}):\n")
                                if hasattr(pks_node, 'PKS_product') and pks_node.PKS_product:
                                    prod_smi = Chem.MolToSmiles(pks_node.PKS_product)
                                    f.write(f"    Product SMILES: {prod_smi}\n")
                                if hasattr(pks_node, 'PKS_design') and pks_node.PKS_design:
                                    try:
                                        cluster, score, _ = pks_node.PKS_design
                                        f.write(self.format_pks_cluster(cluster, score) + "\n")
                                    except Exception as e:
                                        f.write(f"    (Could not extract details: {e})\n")

                    f.write("\n")

            # All results table
            f.write("\n" + "=" * 70 + "\n")
            f.write("ALL RETROTIDE RESULTS TABLE\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'Node ID':<10} {'Provenance':<12} {'Success':<10} {'Score':<10} {'Fragment SMILES'}\n")
            f.write("-" * 70 + "\n")

            for r in self.retrotide_results:
                success_str = "✅ YES" if r.retrotide_successful else "❌ NO"
                smiles_short = r.doranet_node_smiles[:35] + "..." if len(r.doranet_node_smiles) > 35 else r.doranet_node_smiles
                f.write(f"{r.doranet_node_id:<10} {r.doranet_node_provenance:<12} {success_str:<10} {r.retrotide_best_score:<10.3f} {smiles_short}\n")

            # Detailed PKS designs for ALL RetroTide searches (including unsuccessful)
            if self.retrotide_results:
                f.write("\n" + "=" * 70 + "\n")
                f.write("DETAILED PKS MODULE INFORMATION FOR ALL SEARCHES\n")
                f.write("=" * 70 + "\n\n")

                for r in self.retrotide_results:
                    f.write(f"Fragment: {r.doranet_node_smiles}\n")
                    f.write(f"DORAnet Node: {r.doranet_node_id} ({r.doranet_node_provenance})\n")
                    f.write(f"Best Score: {r.retrotide_best_score:.4f}\n")
                    f.write(f"Exact Match Found: {'Yes' if r.retrotide_successful else 'No'}\n")

                    # Show DORAnet pathway
                    doranet_node = next((n for n in self.nodes if n.node_id == r.doranet_node_id), None)
                    if doranet_node:
                        f.write(f"\nDORAnet Pathway:\n")
                        f.write(self.format_pathway(doranet_node) + "\n")

                    # Show PKS designs from RetroTide agent
                    if r.retrotide_agent:
                        agent = r.retrotide_agent

                        # Show successful exact match designs first
                        succ_nodes = getattr(agent, 'successful_nodes', set())
                        if succ_nodes:
                            f.write(f"\n✅ EXACT MATCH PKS DESIGNS:\n")
                            for j, pks_node in enumerate(list(succ_nodes)[:3]):
                                f.write(f"\n  Design #{j + 1}:\n")
                                if hasattr(pks_node, 'PKS_product') and pks_node.PKS_product:
                                    f.write(f"    Product: {Chem.MolToSmiles(pks_node.PKS_product)}\n")
                                if hasattr(pks_node, 'PKS_design') and pks_node.PKS_design:
                                    try:
                                        cluster, score, _ = pks_node.PKS_design
                                        f.write(self.format_pks_cluster(cluster, score) + "\n")
                                    except Exception:
                                        pass

                        # Show simulated successful designs
                        sim_designs = getattr(agent, 'successful_simulated_designs', [])
                        if sim_designs:
                            f.write(f"\n✅ SIMULATED SUCCESSFUL PKS DESIGNS:\n")
                            for j, design in enumerate(sim_designs[:3]):
                                f.write(f"\n  Simulated Design #{j + 1}:\n")
                                if isinstance(design, (list, tuple)):
                                    f.write(f"    Number of Modules: {len(design)}\n")
                                    for k, mod in enumerate(design):
                                        f.write(self.format_pks_module(mod, k + 1) + "\n")
                                else:
                                    f.write(f"    Modules: {design}\n")

                        # Show best designs even if no exact match
                        if not succ_nodes:
                            f.write(f"\n📊 BEST SCORING PKS DESIGNS (closest to target):\n")
                            all_nodes = getattr(agent, 'nodes', [])
                            scored_nodes = [
                                (n, n.value / max(n.visits, 1))
                                for n in all_nodes
                                if hasattr(n, 'PKS_design') and n.PKS_design
                            ]
                            scored_nodes.sort(key=lambda x: x[1], reverse=True)

                            for j, (pks_node, avg_score) in enumerate(scored_nodes[:3]):
                                f.write(f"\n  Design #{j + 1} (score: {avg_score:.4f}):\n")
                                if hasattr(pks_node, 'PKS_product') and pks_node.PKS_product:
                                    f.write(f"    Product: {Chem.MolToSmiles(pks_node.PKS_product)}\n")
                                if hasattr(pks_node, 'PKS_design') and pks_node.PKS_design:
                                    try:
                                        cluster, score, _ = pks_node.PKS_design
                                        f.write(self.format_pks_cluster(cluster, score) + "\n")
                                    except Exception as e:
                                        f.write(f"    (Could not extract details)\n")

                    f.write("\n" + "-" * 70 + "\n\n")

        print(f"[DORAnet] Results saved to: {path}")

        # Generate visualizations if enabled
        if self.enable_visualization:
            self._generate_visualizations(output_path)

    def save_finalized_pathways(self, output_path: str, total_runtime_seconds: Optional[float] = None) -> None:
        """
        Save reaction-based pathways to a separate file.

        This follows reactions stored on edges rather than only listing nodes.

        Args:
            output_path: Path to the output file.
            total_runtime_seconds: Optional total runtime in seconds for the run.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer sink compounds and PKS matches as terminal nodes
        terminal_nodes = [
            n for n in self.nodes
            if n.is_sink_compound or self.calculate_reward(n) > 0
        ]

        # Fallback to leaf nodes if no terminal nodes exist
        if not terminal_nodes:
            terminal_nodes = [n for n in self.nodes if not n.children]

        with open(path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("FINALIZED REACTION-BASED PATHWAYS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total pathways: {len(terminal_nodes)}\n\n")
            if total_runtime_seconds is not None:
                f.write(f"Total runtime (seconds): {total_runtime_seconds:.2f}\n\n")

            for i, node in enumerate(terminal_nodes):
                self._write_pathway_block(f, i + 1, node)

        print(f"[DORAnet] Finalized pathways saved to: {path}")

    def _categorize_pathway(self, node: Node) -> str:
        """
        Categorize a pathway based on the synthesis modalities used.

        Categories:
        - "purely_synthetic": All steps are synthetic chemistry, no PKS involvement
        - "purely_enzymatic": All steps are enzymatic, no PKS involvement
        - "synthetic_enzymatic": Mix of synthetic and enzymatic, no PKS involvement
        - "synthetic_pks": All steps are synthetic chemistry, has PKS terminal or byproduct
        - "enzymatic_pks": All steps are enzymatic, has PKS terminal or byproduct
        - "synthetic_enzymatic_pks": Mix of synthetic and enzymatic, has PKS terminal or byproduct
        - "direct_pks": No chemistry steps (target directly matches PKS)

        A pathway is considered PKS-involved if either:
        1. The terminal node is PKS-synthesizable, OR
        2. Any byproduct along the pathway is PKS-synthesizable

        Args:
            node: Terminal node of the pathway

        Returns:
            Category string
        """
        pathway = self.get_pathway_to_node(node)

        # Check if terminal is PKS-synthesizable
        is_pks = getattr(node, 'is_pks_terminal', False)
        if not is_pks:
            # Also check retrotide results for this node
            for r in self.retrotide_results:
                if r.doranet_node_id == node.node_id and r.retrotide_successful:
                    is_pks = True
                    break

        # Collect provenance types from pathway steps (excluding target)
        has_synthetic = False
        has_enzymatic = False

        for step_node in pathway[1:]:  # Skip target (first node)
            provenance = getattr(step_node, 'provenance', None)
            if provenance == "synthetic":
                has_synthetic = True
            elif provenance == "enzymatic":
                has_enzymatic = True

        # Check if any byproduct requires PKS synthesis
        if not is_pks:
            pks_byproducts = self._collect_pks_byproducts_for_pathway(node)
            if pks_byproducts:
                is_pks = True

        # Categorize based on combination
        if is_pks:
            if not has_synthetic and not has_enzymatic:
                return "direct_pks"
            elif has_synthetic and has_enzymatic:
                return "synthetic_enzymatic_pks"
            elif has_synthetic:
                return "synthetic_pks"
            else:  # has_enzymatic
                return "enzymatic_pks"
        else:
            if has_synthetic and has_enzymatic:
                return "synthetic_enzymatic"
            elif has_synthetic:
                return "purely_synthetic"
            elif has_enzymatic:
                return "purely_enzymatic"
            else:
                # Edge case: no steps (shouldn't happen for successful pathways)
                return "unknown"

    def _get_pathway_type_counts(
        self, nodes: List[Node]
    ) -> Tuple[Dict[str, int], Dict[str, List[int]], Dict[str, Tuple[int, int]]]:
        """
        Count pathways by category and track which pathways belong to each category.

        Args:
            nodes: List of terminal nodes

        Returns:
            Tuple of:
            - Dictionary mapping category names to pathway counts
            - Dictionary mapping category names to list of pathway numbers (1-indexed)
            - Dictionary mapping category names to (exact_designs, simulated_designs) counts
        """
        counts: Dict[str, int] = {
            "purely_synthetic": 0,
            "purely_enzymatic": 0,
            "synthetic_enzymatic": 0,
            "synthetic_pks": 0,
            "enzymatic_pks": 0,
            "synthetic_enzymatic_pks": 0,
            "direct_pks": 0,
            "unknown": 0,
        }
        pathway_numbers: Dict[str, List[int]] = {key: [] for key in counts}
        design_counts: Dict[str, Tuple[int, int]] = {key: (0, 0) for key in counts}

        for i, node in enumerate(nodes):
            pathway_num = i + 1  # 1-indexed pathway number
            category = self._categorize_pathway(node)
            if category in counts:
                counts[category] = counts.get(category, 0) + 1
                pathway_numbers[category].append(pathway_num)
                # Count RetroTide designs for PKS categories
                if category in ("synthetic_pks", "enzymatic_pks", "synthetic_enzymatic_pks", "direct_pks"):
                    exact, simulated = self._count_retrotide_designs_for_pathway(node)
                    prev_exact, prev_sim = design_counts[category]
                    design_counts[category] = (prev_exact + exact, prev_sim + simulated)
            else:
                counts["unknown"] = counts.get("unknown", 0) + 1
                pathway_numbers["unknown"].append(pathway_num)

        return counts, pathway_numbers, design_counts

    def save_successful_pathways(self, output_path: str) -> None:
        """
        Save pathways where all products are PKS-synthesizable or sink compounds.

        Args:
            output_path: Path to the output file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pks_success_smiles: Set[str] = set()
        for r in self.retrotide_results:
            if r.retrotide_successful:
                smi = _canonicalize_smiles(r.doranet_node_smiles or "")
                if smi:
                    pks_success_smiles.add(smi)

        def is_product_covered(smiles: str) -> bool:
            """
            Check if a product is 'covered' (available as a building block).

            A product is covered if:
            1. It's a sink compound (commercially available)
            2. It's in the PKS success list (can be synthesized by PKS)
            3. It's an excluded fragment (common small molecule like H2, H2O, CO2)
            """
            # Check if it's a sink compound
            if self._get_sink_compound_type(smiles):
                return True

            canonical = _canonicalize_smiles(smiles)
            if canonical is None:
                return False

            # Check if it's a PKS-synthesizable molecule
            if canonical in pks_success_smiles:
                return True

            # Check if it's an excluded fragment (common reagent/byproduct)
            # These are small molecules like H2, H2O, CO2, formaldehyde, etc.
            # that are readily available and don't need to be synthesized
            if canonical in self.excluded_fragments:
                return True

            return False

        # Use same terminal set as finalized pathways
        terminal_nodes = [
            n for n in self.nodes
            if n.is_sink_compound or self.calculate_reward(n) > 0
        ]
        if not terminal_nodes:
            terminal_nodes = [n for n in self.nodes if not n.children]

        successful_nodes: List[Node] = []
        for node in terminal_nodes:
            pathway = self.get_pathway_to_node(node)
            all_covered = True

            for step_node in pathway[1:]:
                # The primary product is the step_node's own fragment - it continues along the pathway
                # Only byproducts (other products) need to be checked for coverage
                primary_smiles = _canonicalize_smiles(step_node.smiles or "")

                for prod in step_node.products_smiles or []:
                    prod_canonical = _canonicalize_smiles(prod)

                    # Skip the primary product - it continues along the pathway
                    # and will be checked at the terminal node
                    if prod_canonical == primary_smiles:
                        continue

                    # Check if this byproduct is covered (sink, PKS, or common reagent)
                    if not is_product_covered(prod):
                        all_covered = False
                        break

                if not all_covered:
                    break

            if all_covered:
                successful_nodes.append(node)

        # Get pathway type counts and pathway numbers
        pathway_counts, pathway_numbers, design_counts = self._get_pathway_type_counts(successful_nodes)

        def format_pathway_list(nums: List[int]) -> str:
            """Format pathway numbers as comma-separated list with # prefix."""
            if not nums:
                return ""
            return "(" + ", ".join(f"#{n}" for n in nums) + ")"

        # Calculate total design counts for PKS categories
        pks_categories = ["direct_pks", "synthetic_pks", "enzymatic_pks", "synthetic_enzymatic_pks"]
        total_exact = sum(design_counts[cat][0] for cat in pks_categories)
        total_simulated = sum(design_counts[cat][1] for cat in pks_categories)
        total_pks_pathways = sum(pathway_counts[cat] for cat in pks_categories)
        total_non_pks = (pathway_counts['purely_synthetic'] +
                         pathway_counts['purely_enzymatic'] +
                         pathway_counts['synthetic_enzymatic'])

        with open(path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("SUCCESSFUL PATHWAYS (PKS OR SINK PRODUCTS ONLY)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total pathways: {len(successful_nodes)}\n\n")

            # Write pathway type breakdown
            f.write("PATHWAY TYPE BREAKDOWN\n")
            f.write("-" * 40 + "\n")

            # Sink compound pathways (no PKS)
            f.write("Sink Compound Pathways:\n")
            f.write(f"  Purely synthetic:        {pathway_counts['purely_synthetic']}  {format_pathway_list(pathway_numbers['purely_synthetic'])}\n")
            f.write(f"  Purely enzymatic:        {pathway_counts['purely_enzymatic']}  {format_pathway_list(pathway_numbers['purely_enzymatic'])}\n")
            f.write(f"  Synthetic + enzymatic:   {pathway_counts['synthetic_enzymatic']}  {format_pathway_list(pathway_numbers['synthetic_enzymatic'])}\n")

            # PKS pathways with design counts
            f.write("\nPKS-Synthesizable Pathways:\n")
            for cat, display_name in [
                ("direct_pks", "Direct PKS match"),
                ("synthetic_pks", "Synthetic + PKS"),
                ("enzymatic_pks", "Enzymatic + PKS"),
                ("synthetic_enzymatic_pks", "Synthetic + enz + PKS"),
            ]:
                exact, sim = design_counts[cat]
                count = pathway_counts[cat]
                pathway_list = format_pathway_list(pathway_numbers[cat])
                if count > 0 and (exact > 0 or sim > 0):
                    f.write(f"  {display_name:22} {count:3} pathways -> {exact:4} exact, {sim:4} simulated designs  {pathway_list}\n")
                else:
                    f.write(f"  {display_name:22} {count:3}  {pathway_list}\n")

            if pathway_counts['unknown'] > 0:
                f.write(f"\nUnknown/Other:             {pathway_counts['unknown']}  {format_pathway_list(pathway_numbers['unknown'])}\n")

            # Write design-based proportions summary
            if total_pks_pathways > 0:
                f.write("\nRETROTIDE DESIGN SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total PKS pathways:        {total_pks_pathways}\n")
                f.write(f"Total exact match designs: {total_exact}\n")
                f.write(f"Total simulated designs:   {total_simulated}\n")
                f.write(f"Total all designs:         {total_exact + total_simulated}\n")
                f.write(f"Avg designs per PKS path:  {(total_exact + total_simulated) / total_pks_pathways:.1f}\n")
                f.write("\n")

                # Proportions counting each design as a route
                f.write("Proportions (counting each design as a synthesis route):\n")
                total_with_exact = total_non_pks + total_exact
                pct_exact = 100 * total_exact / total_with_exact if total_with_exact > 0 else 0
                f.write(f"  Exact match designs:     {total_exact:4} PKS + {total_non_pks} non-PKS = {total_with_exact} routes ({pct_exact:.1f}% PKS)\n")

                total_all = total_non_pks + total_exact + total_simulated
                pct_all = 100 * (total_exact + total_simulated) / total_all if total_all > 0 else 0
                f.write(f"  All designs:             {total_exact + total_simulated:4} PKS + {total_non_pks} non-PKS = {total_all} routes ({pct_all:.1f}% PKS)\n")

            f.write("\n" + "=" * 70 + "\n\n")

            for i, node in enumerate(successful_nodes):
                self._write_pathway_block(f, i + 1, node)

        print(f"[DORAnet] Successful pathways saved to: {path}")

    def _get_retrotide_result_by_smiles(self, smiles: str) -> Optional["RetroTideResult"]:
        """
        Look up a successful RetroTide result by canonical SMILES.
        
        Args:
            smiles: SMILES string to look up
            
        Returns:
            RetroTideResult if found and successful, None otherwise
        """
        canonical = _canonicalize_smiles(smiles)
        if canonical is None:
            return None
        
        for r in self.retrotide_results:
            if r.retrotide_successful:
                result_canonical = _canonicalize_smiles(r.doranet_node_smiles or "")
                if result_canonical == canonical:
                    return r
        return None

    def _write_pks_design_details(self, f, result: "RetroTideResult", indent: str = "") -> None:
        """
        Write PKS design details for a RetroTide result.
        
        Args:
            f: File handle to write to
            result: RetroTideResult containing the PKS designs
            indent: Optional indentation prefix for formatting
        """
        f.write(f"{indent}Target: {result.retrotide_target_smiles}\n")
        f.write(f"{indent}Success: {'Yes' if result.retrotide_successful else 'No'}\n")
        f.write(f"{indent}Best Score: {result.retrotide_best_score:.4f}\n")
        f.write(f"{indent}Total Nodes: {result.retrotide_total_nodes}\n")

        agent = result.retrotide_agent
        if agent:
            succ_nodes = getattr(agent, 'successful_nodes', set())
            if succ_nodes:
                f.write(f"{indent}Exact Match Designs:\n")
                for j, pks_node in enumerate(list(succ_nodes)[:3]):
                    f.write(f"{indent}  Design #{j + 1}:\n")
                    if hasattr(pks_node, 'PKS_product') and pks_node.PKS_product:
                        f.write(f"{indent}    Product: {Chem.MolToSmiles(pks_node.PKS_product)}\n")
                    if hasattr(pks_node, 'PKS_design') and pks_node.PKS_design:
                        try:
                            cluster, score, _ = pks_node.PKS_design
                            # Add indent to each line of the cluster format
                            cluster_text = self.format_pks_cluster(cluster, score)
                            indented_cluster = "\n".join(
                                f"{indent}    {line}" if line.strip() else line 
                                for line in cluster_text.split("\n")
                            )
                            f.write(indented_cluster + "\n")
                        except Exception:
                            f.write(f"{indent}    (Could not extract module details)\n")

            sim_designs = getattr(agent, 'successful_simulated_designs', [])
            if sim_designs:
                f.write(f"{indent}Simulated Successful Designs:\n")
                for j, design in enumerate(sim_designs[:3]):
                    f.write(f"{indent}  Simulated Design #{j + 1}:\n")
                    if isinstance(design, (list, tuple)):
                        f.write(f"{indent}    Number of Modules: {len(design)}\n")
                        for k, mod in enumerate(design):
                            # Add indent to each line of the module format
                            mod_text = self.format_pks_module(mod, k + 1)
                            indented_mod = "\n".join(
                                f"{indent}    {line}" if line.strip() else line
                                for line in mod_text.split("\n")
                            )
                            f.write(indented_mod + "\n")
                    else:
                        f.write(f"{indent}    Modules: {design}\n")

    def _collect_pks_byproducts_for_pathway(self, node: Node) -> List[Tuple[int, str, "RetroTideResult"]]:
        """
        Collect PKS-synthesizable byproducts along a pathway.
        
        Only includes byproducts that are NOT sink compounds and NOT excluded fragments,
        i.e., byproducts that required PKS coverage to make the pathway "successful".
        
        Args:
            node: Terminal node of the pathway
            
        Returns:
            List of tuples: (step_number, byproduct_smiles, retrotide_result)
            Ordered by step number in the pathway.
        """
        pks_byproducts: List[Tuple[int, str, "RetroTideResult"]] = []
        pathway = self.get_pathway_to_node(node)
        
        for step_idx, step_node in enumerate(pathway[1:], start=1):  # Skip root (index 0)
            primary_smiles = _canonicalize_smiles(step_node.smiles or "")
            
            for prod in step_node.products_smiles or []:
                prod_canonical = _canonicalize_smiles(prod)
                
                # Skip the primary product (it continues along the pathway)
                if prod_canonical == primary_smiles:
                    continue
                
                # Skip sink compounds (they don't need PKS coverage)
                if self._get_sink_compound_type(prod):
                    continue
                
                # Skip excluded fragments (common reagents like H2, H2O, etc.)
                if prod_canonical and prod_canonical in self.excluded_fragments:
                    continue
                
                # Check if this byproduct has a successful PKS design
                result = self._get_retrotide_result_by_smiles(prod)
                if result:
                    pks_byproducts.append((step_idx, prod, result))
        
        return pks_byproducts

    def _count_retrotide_designs(self, result: "RetroTideResult") -> Tuple[int, int]:
        """
        Count the number of RetroTide designs in a result.

        Args:
            result: RetroTideResult containing the PKS designs

        Returns:
            Tuple of (exact_match_count, simulated_count)
        """
        exact_count = 0
        simulated_count = 0

        agent = result.retrotide_agent
        if agent:
            succ_nodes = getattr(agent, 'successful_nodes', set())
            exact_count = len(succ_nodes) if succ_nodes else 0

            sim_designs = getattr(agent, 'successful_simulated_designs', [])
            simulated_count = len(sim_designs) if sim_designs else 0

        return exact_count, simulated_count

    def _count_retrotide_designs_for_pathway(self, node: Node) -> Tuple[int, int]:
        """
        Count all RetroTide designs for a pathway (terminal + byproducts).

        Args:
            node: Terminal node of the pathway

        Returns:
            Tuple of (total_exact_match_count, total_simulated_count)
        """
        total_exact = 0
        total_simulated = 0

        # Check terminal node's RetroTide result
        is_pks_terminal = getattr(node, 'is_pks_terminal', False)
        if is_pks_terminal:
            # Find the RetroTide result for this terminal
            for r in self.retrotide_results:
                if r.doranet_node_id == node.node_id and r.retrotide_successful:
                    exact, sim = self._count_retrotide_designs(r)
                    total_exact += exact
                    total_simulated += sim
                    break
        else:
            # Check retrotide_results for terminal (may have been found via SMILES match)
            for r in self.retrotide_results:
                if r.doranet_node_id == node.node_id and r.retrotide_successful:
                    exact, sim = self._count_retrotide_designs(r)
                    total_exact += exact
                    total_simulated += sim
                    break

        # Count designs from PKS byproducts
        pks_byproducts = self._collect_pks_byproducts_for_pathway(node)
        for _, _, result in pks_byproducts:
            exact, sim = self._count_retrotide_designs(result)
            total_exact += exact
            total_simulated += sim

        return total_exact, total_simulated

    def _write_pathway_block(self, f, index: int, node: Node, include_pks_byproducts: bool = True) -> None:
        """Write a single pathway block with reaction and RetroTide details.
        
        Args:
            f: File handle to write to
            index: Pathway index number
            node: Terminal node of the pathway
            include_pks_byproducts: Whether to include PKS designs for byproducts
        """
        f.write(f"PATHWAY #{index}: Node {node.node_id}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Terminal Fragment: {node.smiles}\n")
        f.write(f"Depth: {node.depth}, Provenance: {node.provenance}\n")
        if node.is_sink_compound:
            sink_type = node.sink_compound_type or "unknown"
            f.write(f"Sink Compound: Yes ({sink_type})\n")
        elif self.calculate_reward(node) > 0:
            f.write("PKS Match: Yes\n")
        else:
            f.write("Terminal: Leaf\n")
        f.write("\nReaction Pathway:\n")
        f.write(self.format_reaction_pathway(node) + "\n\n")

        # Attach RetroTide PKS designs if available for the terminal node
        if self.calculate_reward(node) > 0:
            results = [r for r in self.retrotide_results if r.doranet_node_id == node.node_id]
            if results:
                f.write("RetroTide PKS Designs:\n")
                f.write("-" * 40 + "\n")
                for r in results:
                    self._write_pks_design_details(f, r)
                f.write("\n")

        # Attach PKS designs for byproducts that required PKS coverage
        if include_pks_byproducts:
            pks_byproducts = self._collect_pks_byproducts_for_pathway(node)
            if pks_byproducts:
                f.write("PKS-Synthesizable Byproducts:\n")
                f.write("-" * 40 + "\n")
                for step_num, byproduct_smiles, result in pks_byproducts:
                    f.write(f"  Step {step_num} Byproduct: {byproduct_smiles}\n")
                    self._write_pks_design_details(f, result, indent="    ")
                    f.write("\n")

    def _generate_visualizations(self, results_path: str) -> None:
        """
        Generate tree and pathway visualizations.

        Args:
            results_path: Path to the results text file (used to derive viz filenames).
        """
        try:
            from .visualize import (
                visualize_doranet_tree,
                visualize_pks_pathways,
                create_enhanced_interactive_html,
                create_pathways_interactive_html
            )

            print("\n[DORAnet] Generating visualizations...")

            # Derive visualization paths from results path
            results_path_obj = Path(results_path)
            base_name = results_path_obj.stem  # filename without extension
            viz_dir = Path(self.visualization_output_dir)
            viz_dir.mkdir(parents=True, exist_ok=True)

            # Full tree visualization
            tree_viz_path = viz_dir / f"{base_name}_tree.png"
            visualize_doranet_tree(self, output_path=str(tree_viz_path))
            print(f"[DORAnet] Tree visualization saved to: {tree_viz_path}")

            # PKS pathways visualization
            pks_viz_path = viz_dir / f"{base_name}_pks_pathways.png"
            visualize_pks_pathways(self, output_path=str(pks_viz_path))
            print(f"[DORAnet] PKS pathways visualization saved to: {pks_viz_path}")

            # Interactive HTML visualizations (if enabled)
            if self.enable_interactive_viz:
                # Full tree interactive visualization
                interactive_path = viz_dir / f"{base_name}_interactive.html"
                create_enhanced_interactive_html(
                    self,
                    output_path=str(interactive_path),
                    auto_open=self.auto_open_viz
                )
                print(f"[DORAnet] Interactive HTML saved to: {interactive_path}")

                # Pathways-only interactive visualization (filtered to PKS matches and sink compounds)
                pathways_path = viz_dir / f"{base_name}_pathways.html"
                create_pathways_interactive_html(
                    self,
                    output_path=str(pathways_path),
                    auto_open=self.auto_open_viz
                )
                print(f"[DORAnet] Pathways interactive HTML saved to: {pathways_path}")

        except ImportError as e:
            print(f"[DORAnet] Could not generate visualizations: {e}")
            print("[DORAnet] Install required packages: pip install networkx matplotlib bokeh")
        except Exception as e:
            print(f"[DORAnet] Error generating visualizations: {e}")

    def _generate_iteration_visualization(self, iteration: int) -> None:
        """
        Generate tree visualizations for a specific iteration to show MCTS dynamics.

        This creates iteration-specific visualizations showing the current state of
        the tree, which nodes have been selected, and how the tree is growing.

        Args:
            iteration: The current iteration number (0-indexed).
        """
        if not self.visualization_output_dir:
            return

        try:
            from .visualize import (
                visualize_doranet_tree,
                create_enhanced_interactive_html
            )

            # Create iterations subdirectory
            viz_dir = Path(self.visualization_output_dir)
            iterations_dir = viz_dir / "iterations"
            iterations_dir.mkdir(parents=True, exist_ok=True)

            # Zero-padded iteration number for proper sorting
            iter_str = f"{iteration:05d}"

            # Generate static tree visualization
            tree_viz_path = iterations_dir / f"iteration_{iter_str}_tree.png"
            visualize_doranet_tree(
                self,
                output_path=str(tree_viz_path),
                title=f"DORAnet Tree - Iteration {iteration + 1}/{self.total_iterations}"
            )

            # Generate interactive HTML visualization if enabled
            if self.enable_interactive_viz:
                interactive_path = iterations_dir / f"iteration_{iter_str}_interactive.html"
                create_enhanced_interactive_html(
                    self,
                    output_path=str(interactive_path),
                    auto_open=self.auto_open_iteration_viz,  # Auto-open iteration visualizations
                    title=f"DORAnet Tree - Iteration {iteration + 1}/{self.total_iterations}"
                )

            # Print status every 10 iterations or at specified intervals
            if (iteration + 1) % max(10, self.iteration_viz_interval) == 0:
                print(f"[DORAnet] Generated iteration {iteration + 1} visualizations")

        except ImportError as e:
            # Only print warning once
            if not hasattr(self, '_viz_warning_shown'):
                print(f"[DORAnet] Could not generate iteration visualizations: {e}")
                print("[DORAnet] Install required packages: pip install networkx matplotlib bokeh")
                self._viz_warning_shown = True
        except Exception as e:
            print(f"[DORAnet] Error generating iteration {iteration} visualization: {e}")
