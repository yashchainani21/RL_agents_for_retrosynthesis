"""
Modular policies for DORAnet MCTS.

This module provides abstract base classes and concrete implementations for:
- Rollout policies: Simulation strategies from leaf nodes (e.g., RetroTide spawning)
- Reward policies: Reward computation strategies (e.g., sparse, shaped)

Selection policies remain in mcts.py for now and will be migrated in a future PR.
"""

from .base import RolloutPolicy, RewardPolicy, RolloutResult
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
)

__all__ = [
    # Base classes
    "RolloutPolicy",
    "RewardPolicy",
    "RolloutResult",
    # Rollout policies
    "NoOpRolloutPolicy",
    "SpawnRetroTideOnDatabaseCheck",
    "SAScore_and_SpawnRetroTideOnDatabaseCheck",
    "PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck",
    # Reward policies
    "SparseTerminalRewardPolicy",
    "SinkCompoundRewardPolicy",
    "PKSLibraryRewardPolicy",
    "ComposedRewardPolicy",
]
