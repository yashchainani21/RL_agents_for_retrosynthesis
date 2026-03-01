"""
Modular policies for DORAnet MCTS.

This module provides abstract base classes and concrete implementations for:
- Terminal detectors: Post-expansion terminal detection (e.g., RetroTide verification)
- Reward policies: Reward computation strategies (e.g., sparse, shaped)
- Thermodynamic scaling: Wrappers that scale rewards by pathway feasibility

Legacy rollout policies are still exported for backward compatibility but are
deprecated and will be removed in a future release.

Selection policies remain in mcts.py for now and will be migrated in a future PR.
"""

# New abstractions
from .base import TerminalDetector, TerminalDetectionResult, RewardPolicy

# New terminal detection implementations
from .terminal_detection import (
    NoOpTerminalDetector,
    VerifyWithRetroTide,
    SimilarityGuidedRetroTideDetector,
)

# Legacy rollout abstractions (deprecated — use TerminalDetector instead)
from .base import RolloutPolicy, RolloutResult
from .rollout import (
    NoOpRolloutPolicy,
    SpawnRetroTideOnDatabaseCheck,
    SAScore_and_SpawnRetroTideOnDatabaseCheck,
    PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
)
from .reward import (
    SparseTerminalRewardPolicy,
    SinkCompoundRewardPolicy,
    PKSLibraryRewardPolicy,
    ComposedRewardPolicy,
    PKSSimilarityRewardPolicy,
    SAScore_and_TerminalRewardPolicy,
)
from .thermodynamic import (
    sigmoid_transform,
    get_node_feasibility_score,
    get_pathway_feasibility,
    ThermodynamicScaledRolloutPolicy,
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
    # New base classes
    "TerminalDetector",
    "TerminalDetectionResult",
    "RewardPolicy",
    # New terminal detectors
    "NoOpTerminalDetector",
    "VerifyWithRetroTide",
    "SimilarityGuidedRetroTideDetector",
    # Legacy base classes (deprecated)
    "RolloutPolicy",
    "RolloutResult",
    # Legacy rollout policies (deprecated)
    "NoOpRolloutPolicy",
    "SpawnRetroTideOnDatabaseCheck",
    "SAScore_and_SpawnRetroTideOnDatabaseCheck",
    "PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck",
    # Reward policies
    "SparseTerminalRewardPolicy",
    "SinkCompoundRewardPolicy",
    "PKSLibraryRewardPolicy",
    "ComposedRewardPolicy",
    "PKSSimilarityRewardPolicy",
    "SAScore_and_TerminalRewardPolicy",
    # Thermodynamic scaling
    "sigmoid_transform",
    "get_node_feasibility_score",
    "get_pathway_feasibility",
    "ThermodynamicScaledRolloutPolicy",  # deprecated
    "ThermodynamicScaledRewardPolicy",
    # Utilities
    "canonicalize_smiles",
    "calculate_sa_score",
    "sa_score_to_reward",
    "generate_morgan_fingerprint",
    "calculate_tanimoto_similarity",
    "calculate_mcs_similarity_without_stereo",
]
