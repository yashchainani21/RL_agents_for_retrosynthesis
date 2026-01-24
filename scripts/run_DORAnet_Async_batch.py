"""
Runner for AsyncExpansionDORAnetMCTS (multiprocessing expansion).

This script runs DORAnet MCTS with parallel expansion using multiprocessing,
significantly speeding up search on multi-core systems.

Policy System:
- rollout_policy: Controls what happens after expansion (default: NoOpRolloutPolicy)
  - NoOpRolloutPolicy: No additional work after expansion (just returns 0 reward)
  - SpawnRetroTideOnDatabaseCheck: Spawns RetroTide for PKS library matches (sparse rewards)
  - SAScore_and_SpawnRetroTideOnDatabaseCheck: SA Score rewards + RetroTide spawning (dense rewards)
- reward_policy: Controls how terminal rewards are calculated (default: SparseTerminalRewardPolicy)
  - SparseTerminalRewardPolicy: 1.0 for sink compounds, 1.0 for PKS matches, 0.0 otherwise
  - SinkCompoundRewardPolicy: Only rewards sink compounds
  - ComposedRewardPolicy: Combine multiple reward policies with weights

Backward Compatibility:
- spawn_retrotide=True creates SpawnRetroTideOnDatabaseCheck automatically
- Explicit rollout_policy/reward_policy override spawn_retrotide
- reward_fn parameter is deprecated (use reward_policy instead)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from rdkit import Chem
from rdkit import RDLogger

# Ensure the repository root is discoverable when running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Optional

from DORAnet_agent import AsyncExpansionDORAnetMCTS, Node
from DORAnet_agent.visualize import create_enhanced_interactive_html, create_pathways_interactive_html
from DORAnet_agent.policies import (
    RolloutPolicy,
    RewardPolicy,
    NoOpRolloutPolicy,
    SpawnRetroTideOnDatabaseCheck,
    PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
    SAScore_and_SpawnRetroTideOnDatabaseCheck,
    SparseTerminalRewardPolicy,
    SinkCompoundRewardPolicy,
    PKSLibraryRewardPolicy,
    ComposedRewardPolicy,
    # Thermodynamic scaling wrappers
    ThermodynamicScaledRolloutPolicy,
    ThermodynamicScaledRewardPolicy,
)

RDLogger.DisableLog("rdApp.*")


### ---- Molecules ---- 

# cryptofolione # O=C1C=CCC(C=CCC(O)CC(O)C=Cc2ccccc2)O1

## Kavalactones ##

# kavain # COC1=CC(OC(C=CC2=CC=CC=C2)C1)=O
# Yangonin # COC1=CC=C(C=CC2=CC(OC)=CC(O2)=O)C=C1
# 10-methoxyyangonin # COC1=CC(OC)=C(C=CC2=CC(OC)=CC(O2)=O)C=C1
# methysticin # COC1=CC(OC(C=CC2=CC3=C(OCO3)C=C2)C1)=O
# 11-methoxyyangonin # COC1=C(OC)C=C(C=CC2=CC(OC)=CC(O2)=O)C=C1
# 5,6-dihydroyangoin # COC1=CC(OC(C=CC2=CC=C(OC)C=C2)C1)=O
# 5,6,7,8-tetrahydroyangonin # COC1=CC(OC(CCC2=CC=C(OC)C=C2)C1)=O
# desmethoxyyangonin # COC1=CC(OC(C=CC2=CC=CC=C2)=C1)=O
# 11_methoxy_12_hydroxydehydrokavain # COC1=CC=C(C=CC2=CC(OC)=CC(O2)=O)C=C1O
# 7,8-dihydroyangonin # COC1=CC=C(CCC2=CC(OC)=CC(O2)=O)C=C1
# 5-hydroxykavain # COC1=CC(OC(C=CC2=CC=CC=C2)C1O)=O
# 7,8-dihydrokavain # COC1=CC(OC(CCC2=CC=CC=C2)C1)=O
# 5,6-dehydromethysticin # COC1=CC(OC(C=CC2=CC3=C(OCO3)C=C2)=C1)=O
# methysticin # COC1=CC(OC(C=CC2=CC3=C(OCO3)C=C2)C1)=O
# 7,8-dihydromethysticin # COC1=CC(OC(CCC2=CC3=C(OCO3)C=C2)C1)=O

def main(
    target_smiles: str,
    molecule_name: str,
    rollout_policy: Optional[RolloutPolicy] = None,
    reward_policy: Optional[RewardPolicy] = None,
    MW_multiple_to_exclude: float = 1.5,
    child_downselection_strategy: str = "most_thermo_feasible",
) -> None:
    """
    Run the async DORAnet MCTS agent for batch processing.

    Args:
        target_smiles: SMILES string of the target molecule
        molecule_name: Human-readable name for the molecule (used in filenames)
        rollout_policy: Policy controlling what happens after node expansion.
            Options include:
            - NoOpRolloutPolicy(): No additional work (returns 0 reward)
            - SpawnRetroTideOnDatabaseCheck(): Spawns RetroTide for PKS matches (sparse)
            - SAScore_and_SpawnRetroTideOnDatabaseCheck(): SA Score + RetroTide (dense)
            - PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(): PKS similarity + RetroTide
            - ThermodynamicScaledRolloutPolicy(base_policy): Wrapper that scales rewards
              by pathway thermodynamic feasibility
            If None, defaults to PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck().
        reward_policy: Policy controlling how terminal rewards are calculated.
            Options include:
            - SparseTerminalRewardPolicy(): 1.0 for terminals, 0.0 otherwise
            - SinkCompoundRewardPolicy(): Only rewards sink compounds
            - ComposedRewardPolicy(): Combine multiple policies with weights
            - ThermodynamicScaledRewardPolicy(base_policy): Wrapper that scales rewards
              by pathway thermodynamic feasibility
            If None, defaults to SparseTerminalRewardPolicy().
        MW_multiple_to_exclude: Exclude fragments with MW > target_MW * this value.
                               Default 1.5 (exclude fragments >150% of target MW).
        child_downselection_strategy: Strategy for selecting which fragments to keep
            when more than max_children_per_expand are generated. Options:
            - "first_N": Keep first N fragments in DORAnet's order (fastest)
            - "hybrid": Prioritize sink compounds > PKS matches > smaller MW
            - "most_thermo_feasible": Prioritize by thermodynamic feasibility
              (DORA-XGB for enzymatic, sigmoid-transformed ΔH for synthetic),
              with bonuses for sink compounds (+1000) and PKS matches (+500)
            Default is "most_thermo_feasible".
    """
    # ---- Runner configuration ----
    create_interactive_visualization = False
    enable_iteration_viz = False
    iteration_interval = 1
    auto_open_iteration_viz = False
    auto_cleanup_pgnet_files = True
    num_workers = None  # None means "max available"
    max_inflight_expansions = None  # None means "same as num_workers"

    target_molecule = Chem.MolFromSmiles(target_smiles)

    if target_molecule is None:
        raise ValueError(f"Could not parse target SMILES: {target_smiles}")

    if molecule_name:
        print(f"Target molecule: {molecule_name} ({target_smiles})")
    else:
        print(f"Target molecule: {target_smiles}")

    # Paths to cofactor files (metabolites and chemistry helpers to exclude from the network)
    cofactors_files = [
        REPO_ROOT / "data" / "raw" / "all_cofactors.csv",
        REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv",
    ]

    # Path to PKS library file for reward calculation
    pks_library_file = REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V2.txt"

    # Paths to sink compounds files (commercially available building blocks)
    sink_compounds_files = [
        REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt",
        REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt",
    ]

    # Path to prohibited chemicals file (hazardous/controlled substances to avoid)
    prohibited_chemicals_file = REPO_ROOT / "data" / "processed" / "prohibited_chemical_SMILES.txt"

    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    # ---- Policy Configuration ----
    # Use provided policies or create defaults
    if rollout_policy is None:
        # Default: PKS similarity + RetroTide
        rollout_policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
            pks_building_blocks_path=pks_library_file
        )
    if reward_policy is None:
        reward_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)

    agent = AsyncExpansionDORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=200,
        max_depth=3,
        use_enzymatic=True,
        use_synthetic=True,
        generations_per_expand=1,
        max_children_per_expand=70,
        child_downselection_strategy=child_downselection_strategy,
        cofactors_files=[str(f) for f in cofactors_files],
        pks_library_file=str(pks_library_file),
        sink_compounds_files=[str(f) for f in sink_compounds_files],
        prohibited_chemicals_file=str(prohibited_chemicals_file),
        MW_multiple_to_exclude=MW_multiple_to_exclude,

        # Policies passed as explicit arguments
        rollout_policy=rollout_policy,
        reward_policy=reward_policy,

        # Enable RetroTide spawning for PKS library matches
        spawn_retrotide=True,

        # RetroTide configuration (used when rollout policy spawns RetroTide)
        retrotide_kwargs={
            "max_depth": 5,
            "total_iterations": 50,
            "maxPKSDesignsRetroTide": 500,
        },
        
        # ---- Selection & Reward Configuration ----
        sink_terminal_reward=1.0,
        selection_policy="UCB1",
        depth_bonus_coefficient=4.0,
        
        # ---- Visualization Configuration ----
        enable_visualization=False,
        enable_interactive_viz=False,
        enable_iteration_visualizations=enable_iteration_viz,
        iteration_viz_interval=iteration_interval,
        auto_open_iteration_viz=auto_open_iteration_viz,
        visualization_output_dir=str(REPO_ROOT / "results"),
        
        # ---- Async Configuration ----
        num_workers=num_workers,
        max_inflight_expansions=max_inflight_expansions,
    )

    start_time = time.time()
    agent.run()
    total_runtime = time.time() - start_time

    print("\n" + agent.get_tree_summary())

    results_dir = REPO_ROOT / "results"
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")

    if molecule_name:
        safe_name = molecule_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
        filename_base = f"{safe_name}_async"
    else:
        safe_smiles = target_smiles.replace("/", "_").replace("\\", "_")[:20]
        filename_base = f"{safe_smiles}_async"

    results_path = results_dir / f"doranet_results_{filename_base}_{timestamp}.txt"
    agent.save_results(str(results_path))
    finalized_pathways_path = results_dir / f"finalized_pathways_{filename_base}_{timestamp}.txt"
    agent.save_finalized_pathways(str(finalized_pathways_path), total_runtime_seconds=total_runtime)
    successful_pathways_path = results_dir / f"successful_pathways_{filename_base}_{timestamp}.txt"
    agent.save_successful_pathways(str(successful_pathways_path))

    if create_interactive_visualization:
        results_dir.mkdir(parents=True, exist_ok=True)
        interactive_path = results_dir / f"doranet_interactive_{filename_base}_{timestamp}.html"
        pathways_path = results_dir / f"doranet_pathways_{filename_base}_{timestamp}.html"

        create_enhanced_interactive_html(
            agent=agent,
            output_path=str(interactive_path),
            molecule_img_size=(250, 250),
            auto_open=True,
        )

        create_pathways_interactive_html(
            agent=agent,
            output_path=str(pathways_path),
            molecule_img_size=(250, 250),
            auto_open=True,
        )

        print("\n" + "=" * 70)
        print("✓ Interactive visualizations complete!")
        print("=" * 70)

    if auto_cleanup_pgnet_files:
        cleanup_script = REPO_ROOT / "scripts" / "cleanup_pgnet_files.py"
        try:
            subprocess.run([sys.executable, str(cleanup_script), "-y"], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[Runner] Warning: .pgnet cleanup failed ({exc}).")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run async DORAnet for a single molecule.")
    parser.add_argument("--name", required=True, help="Molecule name for output filenames.")
    parser.add_argument("--smiles", required=True, help="Target molecule SMILES.")
    parser.add_argument(
        "--child-downselection-strategy",
        default="most_thermo_feasible",
        choices=["first_N", "hybrid", "most_thermo_feasible"],
        help="Strategy for selecting fragments when more than max_children are generated. "
             "Options: first_N (fastest), hybrid (sink > PKS > smaller MW), "
             "most_thermo_feasible (thermodynamic feasibility). Default: most_thermo_feasible."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ---- Configure Policies (shared across batch runs) ----
    # Option 1: PKS similarity + RetroTide (DEFAULT, uses Tanimoto fingerprint similarity)
    selected_rollout_policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck()

    # Option 2: Dense rewards - SA Score + RetroTide
    # selected_rollout_policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
    #     success_reward=1.0,
    #     sa_max_reward=1.0,
    # )

    # Option 3: Sparse rewards - RetroTide only for PKS library matches
    # selected_rollout_policy = SpawnRetroTideOnDatabaseCheck(
    #     success_reward=1.0,
    #     failure_reward=0.0,
    # )

    # Option 4: No rollout (just expand, no RetroTide spawning)
    # selected_rollout_policy = NoOpRolloutPolicy()

    # Option 5: Thermodynamic-scaled rollout (wrap any base policy)
    # This scales rewards by pathway thermodynamic feasibility using DORA-XGB
    # for enzymatic reactions and sigmoid-transformed ΔH for synthetic reactions.
    # selected_rollout_policy = ThermodynamicScaledRolloutPolicy(
    #     base_policy=PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(),
    #     feasibility_weight=0.8,      # 0.0=ignore feasibility, 1.0=full scaling
    #     sigmoid_k=0.2,               # Steepness of sigmoid for ΔH
    #     sigmoid_threshold=15.0,      # Center point in kcal/mol
    #     use_dora_xgb_for_enzymatic=True,  # Use DORA-XGB for enzymatic reactions
    #     aggregation="geometric_mean",     # How to aggregate pathway scores
    # )

    selected_reward_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)

    # Alternative: Thermodynamic-scaled reward policy (wrap any base policy)
    # This scales terminal rewards by pathway thermodynamic feasibility.
    # selected_reward_policy = ThermodynamicScaledRewardPolicy(
    #     base_policy=SparseTerminalRewardPolicy(sink_terminal_reward=1.0),
    #     feasibility_weight=0.8,
    #     sigmoid_k=0.2,
    #     sigmoid_threshold=15.0,
    #     use_dora_xgb_for_enzymatic=True,
    #     aggregation="geometric_mean",
    # )

    main(
        target_smiles=args.smiles,
        molecule_name=args.name,
        rollout_policy=selected_rollout_policy,
        reward_policy=selected_reward_policy,
        MW_multiple_to_exclude=1.5,
        child_downselection_strategy=args.child_downselection_strategy.replace("-", "_"),
    )
