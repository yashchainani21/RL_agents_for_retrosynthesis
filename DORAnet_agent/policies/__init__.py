"""
Modular policies for DORAnet MCTS.

This module provides abstract base classes and concrete implementations for:
- Terminal detectors: Post-expansion terminal detection (e.g., RetroTide verification)
- Reward policies: Reward computation strategies (e.g., dense SA score, sparse terminal)
- Thermodynamic scaling: Wrappers that scale rewards by pathway feasibility
"""

# Base abstractions
from .base import TerminalDetector, TerminalDetectionResult, RewardPolicy

# Terminal detection implementations
from .terminal_detection import (
    NoOpTerminalDetector,
    VerifyWithRetroTide,
    SimilarityGuidedRetroTideDetector,
)

# Reward policies
from .reward import (
    SparseTerminalRewardPolicy,
    SAScore_and_TerminalRewardPolicy,
)

# Thermodynamic scaling
from .thermodynamic import (
    sigmoid_transform,
    get_node_feasibility_score,
    get_pathway_feasibility,
    ThermodynamicScaledRewardPolicy,
)

# Shared utilities
from .utils import (
    canonicalize_smiles,
    calculate_sa_score,
    sa_score_to_reward,
    generate_morgan_fingerprint,
    calculate_tanimoto_similarity,
    calculate_mcs_similarity_without_stereo,
)

__all__ = [
    # Base classes
    "TerminalDetector",
    "TerminalDetectionResult",
    "RewardPolicy",
    # Terminal detectors
    "NoOpTerminalDetector",
    "VerifyWithRetroTide",
    "SimilarityGuidedRetroTideDetector",
    # Reward policies
    "SparseTerminalRewardPolicy",
    "SAScore_and_TerminalRewardPolicy",
    # Thermodynamic scaling
    "sigmoid_transform",
    "get_node_feasibility_score",
    "get_pathway_feasibility",
    "ThermodynamicScaledRewardPolicy",
    # Utilities
    "canonicalize_smiles",
    "calculate_sa_score",
    "sa_score_to_reward",
    "generate_morgan_fingerprint",
    "calculate_tanimoto_similarity",
    "calculate_mcs_similarity_without_stereo",
]
