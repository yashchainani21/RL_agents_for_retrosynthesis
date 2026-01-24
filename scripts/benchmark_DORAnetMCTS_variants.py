"""
Benchmark sequential vs async-expansion DORAnet MCTS variants.

This script compares the performance of:
- DORAnetMCTS (sequential expansion)
- AsyncExpansionDORAnetMCTS (parallel multiprocessing expansion)

Policy System:
By default, this benchmark uses SAScore_and_SpawnRetroTideOnDatabaseCheck for
dense reward signals. Alternative policies:
- spawn_retrotide=True → SpawnRetroTideOnDatabaseCheck (sparse rewards)
- spawn_retrotide=False → NoOpRolloutPolicy (no rewards, fastest)
- rollout_policy=SpawnRetroTideOnDatabaseCheck(...) → explicit sparse policy
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from rdkit import Chem
from rdkit import RDLogger

# Ensure the repository root is discoverable when running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent import DORAnetMCTS, AsyncExpansionDORAnetMCTS, Node
from DORAnet_agent.policies import (
    NoOpRolloutPolicy,
    SpawnRetroTideOnDatabaseCheck,
    SAScore_and_SpawnRetroTideOnDatabaseCheck,
    SparseTerminalRewardPolicy,
)

RDLogger.DisableLog("rdApp.*")


def _build_common_kwargs(target_molecule: Chem.Mol, total_iterations: int) -> dict:
    """Build common kwargs shared by sequential and async agents."""
    cofactors_files = [
        REPO_ROOT / "data" / "raw" / "all_cofactors.csv",
        REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv",
    ]
    pks_library_file = REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V3.txt"
    sink_compounds_files = [
        REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt",
        REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt",
    ]
    prohibited_chemicals_file = REPO_ROOT / "data" / "processed" / "prohibited_chemical_SMILES.txt"

    return dict(
        target_molecule=target_molecule,
        total_iterations=total_iterations,
        max_depth=3,
        use_enzymatic=True,
        use_synthetic=True,
        generations_per_expand=1,
        max_children_per_expand=30,
        child_downselection_strategy="first_N",
        cofactors_files=[str(f) for f in cofactors_files],
        pks_library_file=str(pks_library_file),
        sink_compounds_files=[str(f) for f in sink_compounds_files],
        prohibited_chemicals_file=str(prohibited_chemicals_file),
        MW_multiple_to_exclude=1.5,
        
        # ---- Policy Configuration ----
        # Option 1: Dense rewards - SA Score + RetroTide (RECOMMENDED)
        rollout_policy=SAScore_and_SpawnRetroTideOnDatabaseCheck(
            success_reward=1.0,
            sa_max_reward=1.0,
        ),
        reward_policy=SparseTerminalRewardPolicy(sink_terminal_reward=1.0),
        
        # Option 2: Sparse rewards - SpawnRetroTideOnDatabaseCheck
        # rollout_policy=SpawnRetroTideOnDatabaseCheck(success_reward=1.0, failure_reward=0.0),
        # reward_policy=SparseTerminalRewardPolicy(sink_terminal_reward=1.0),
        
        # Option 3: No rollout (fastest, for pure timing benchmarks)
        # spawn_retrotide=False,
        
        # RetroTide kwargs (used by SA Score and SpawnRetroTide policies)
        retrotide_kwargs={
            "max_depth": 6,
            "total_iterations": 100,
            "maxPKSDesignsRetroTide": 25,
        },
        
        # ---- Selection & Reward ----
        sink_terminal_reward=1.0,
        selection_policy="UCB1",
        depth_bonus_coefficient=4.0,
        
        # ---- Visualization (disabled for benchmarking) ----
        enable_visualization=False,
        enable_interactive_viz=False,
        enable_iteration_visualizations=False,
        visualization_output_dir=str(REPO_ROOT / "results"),
    )


def _run_agent(agent) -> float:
    start = time.perf_counter()
    agent.run()
    return time.perf_counter() - start


def main() -> None:
    # ---- Benchmark configuration ----
    target_smiles = "C1C=CC(=O)OC1C=CCC(CC(C=CC2=CC=CC=C2)O)O"
    total_iterations = 100
    num_workers = None  # None = max available
    max_inflight_expansions = None  # None = same as num_workers

    target_molecule = Chem.MolFromSmiles(target_smiles)
    if target_molecule is None:
        raise ValueError(f"Could not parse target SMILES: {target_smiles}")

    common_kwargs = _build_common_kwargs(target_molecule, total_iterations)

    # Sequential agent
    root_seq = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")
    seq_agent = DORAnetMCTS(root=root_seq, **common_kwargs)
    seq_time = _run_agent(seq_agent)

    # Async-expansion agent
    root_async = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")
    async_agent = AsyncExpansionDORAnetMCTS(
        root=root_async,
        num_workers=num_workers,
        max_inflight_expansions=max_inflight_expansions,
        **common_kwargs,
    )
    async_time = _run_agent(async_agent)

    speedup = seq_time / async_time if async_time > 0 else float("inf")
    delta = seq_time - async_time

    results_lines = [
        "Benchmark Results",
        "=" * 60,
        f"Target SMILES: {target_smiles}",
        f"Total iterations: {total_iterations}",
        f"Sequential time: {seq_time:.2f}s",
        f"Async time:      {async_time:.2f}s",
        f"Delta:           {delta:.2f}s (positive = async faster)",
        f"Speedup:         {speedup:.2f}x",
    ]

    print("\n" + "\n".join(results_lines))

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_smiles = target_smiles.replace("/", "_").replace("\\", "_")[:20]
    output_path = results_dir / f"benchmark_DORAnetMCTS_variants_{safe_smiles}_{total_iterations}x_iterations.txt"
    output_path.write_text("\n".join(results_lines) + "\n")
    print(f"\nSaved benchmark to: {output_path}")


if __name__ == "__main__":
    main()
