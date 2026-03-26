"""
Runner for AsyncExpansionDORAnetMCTS (multiprocessing expansion).

This script runs DORAnet MCTS with parallel expansion using multiprocessing,
significantly speeding up search on multi-core systems.

Policy System (Clean Architecture):
The MCTS loop uses two separate policies with distinct responsibilities:

- terminal_detector: Determines TERMINAL STATUS (RetroTide verification for PKS fragments)
  - VerifyWithRetroTide: Spawns RetroTide for PKS library matches (RECOMMENDED)
  - SimilarityGuidedRetroTideDetector: PKS similarity gating + RetroTide
  - NoOpTerminalDetector: No verification — just expand (fastest, no RetroTide)

- reward_policy: Computes REWARD for ALL nodes (dense or sparse)
  - SAScore_and_TerminalRewardPolicy: SA score for non-terminals + terminal rewards (RECOMMENDED)
    - Provides dense signals via SA score for ALL nodes
    - Full terminal reward for sink compounds and PKS terminals
    - Supports custom non_terminal_scorer for alternative non-terminal signals (e.g., PKS classifier)
  - SparseTerminalRewardPolicy: 1.0 for terminals, 0.0 otherwise (sparse)
  - ThermodynamicScaledRewardPolicy: Wrapper that scales any base policy by thermodynamic feasibility

KEY ARCHITECTURE POINT:
  The reward_policy.calculate_reward() is ALWAYS called for every node to compute
  the backpropagation reward. This ensures dense rewards (like SA scores) are applied
  to ALL nodes, not just terminals. The terminal_detector only determines whether a
  node becomes terminal (via RetroTide verification).

Example: Recommended clean setup (terminal detection + reward separation)
    from DORAnet_agent.policies import (
        VerifyWithRetroTide,                 # Terminal detection: PKS matching + RetroTide
        SAScore_and_TerminalRewardPolicy,    # Reward: terminals + SA score
        ThermodynamicScaledRewardPolicy,     # Optional: thermodynamic scaling
    )

    # Terminal detector: handles PKS matching and RetroTide verification only
    terminal_detector = VerifyWithRetroTide(
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
        feasibility_weight=1.0,
    )

    main(
        target_smiles="...",
        molecule_name="...",
        terminal_detector=terminal_detector,
        reward_policy=reward_policy,
        ...
    )
"""

from __future__ import annotations

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
    TerminalDetector,
    RewardPolicy,
    NoOpTerminalDetector,
    VerifyWithRetroTide,
    SimilarityGuidedRetroTideDetector,
    SparseTerminalRewardPolicy,
    SAScore_and_TerminalRewardPolicy,
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
# 5,6-dihydroyangonin # COC1=CC(OC(C=CC2=CC=C(OC)C=C2)C1)=O
# 5,6,7,8-tetrahydroyangonin # COC1=CC(OC(CCC2=CC=C(OC)C=C2)C1)=O
# desmethoxyyangonin # COC1=CC(OC(C=CC2=CC=CC=C2)=C1)=O
# 11_methoxy_12_hydroxydehydrokavain # COC1=CC=C(C=CC2=CC(OC)=CC(O2)=O)C=C1O
# 7,8-dihydroyangonin # COC1=CC=C(CCC2=CC(OC)=CC(O2)=O)C=C1
# 5-hydroxykavain # COC1=CC(OC(C=CC2=CC=CC=C2)C1O)=O
# 7,8-dihydrokavain # COC1=CC(OC(CCC2=CC=CC=C2)C1)=O
# 5,6-dehydromethysticin # COC1=CC(OC(C=CC2=CC3=C(OCO3)C=C2)=C1)=O
# methysticin # COC1=CC(OC(C=CC2=CC3=C(OCO3)C=C2)C1)=O
# 7,8-dihydromethysticin # COC1=CC(OC(CCC2=CC3=C(OCO3)C=C2)C1)=O

def main(target_smiles: str,
         molecule_name: str,
         total_iterations: int,
         max_depth: int,
         max_children_per_expand: int,
         terminal_detector: Optional[TerminalDetector] = None,
         reward_policy: Optional[RewardPolicy] = None,
         results_subfolder: str = None,
         MW_multiple_to_exclude: float = 1.2,
         child_downselection_strategy: str = "most_thermo_feasible",
         use_enzymatic: bool = True,
         use_synthetic: bool = True,
         use_chem_building_blocksDB: bool = True,
         use_bio_building_blocksDB: bool = True,
         use_PKS_building_blocksDB: bool = True,
         stop_on_first_pathway: bool = False,
         selection_policy: str = "depth_biased",
         depth_bonus_coefficient: float = 10.0,
         enable_frontier_fallback: bool = True,
         search_strategy: str = "mcts") -> None:
    """
    Run the async DORAnet MCTS agent with multiprocessing expansion.

    Args:
        target_smiles: SMILES string of the target molecule
        molecule_name: Human-readable name for the molecule (used in filenames)
        total_iterations: Number of MCTS iterations to run
        max_depth: Maximum depth of the retrosynthetic search tree
        max_children_per_expand: Maximum number of children to generate per expansion
        terminal_detector: Policy controlling terminal detection after node expansion
            (determines when RetroTide verification should be attempted).
            Options include:
            - VerifyWithRetroTide(): Spawns RetroTide for PKS matches (RECOMMENDED)
            - SimilarityGuidedRetroTideDetector(): PKS similarity gating + RetroTide
            - NoOpTerminalDetector(): No verification (fastest, no RetroTide)
            If None, defaults to VerifyWithRetroTide().
        reward_policy: Policy controlling how terminal rewards are calculated.
            Options include:
            - SAScore_and_TerminalRewardPolicy(): Terminal rewards + SA score (RECOMMENDED)
            - SparseTerminalRewardPolicy(): 1.0 for terminals, 0.0 otherwise
            - ThermodynamicScaledRewardPolicy(base_policy): Wrapper that scales rewards
              by pathway thermodynamic feasibility
            If None, defaults to SAScore_and_TerminalRewardPolicy().
        results_subfolder: Optional subfolder within results/ to save outputs.
                          If None, saves directly to results/. Useful for batch runs.
        MW_multiple_to_exclude: Exclude fragments with MW > target_MW * this value.
                               Default 1.2 (exclude fragments >120% of target MW).
        child_downselection_strategy: Strategy for selecting which fragments to keep
            when more than max_children_per_expand are generated. Options:
            - "first_N": Keep first N fragments in DORAnet's order (fastest)
            - "hybrid": Prioritize sink compounds > PKS matches > smaller MW
            - "most_thermo_feasible": Prioritize by thermodynamic feasibility
              (DORA-XGB for enzymatic, sigmoid-transformed ΔH for synthetic),
              with bonuses for sink compounds (+1000) and PKS matches (+500)
            Default is "most_thermo_feasible".
        use_enzymatic: Whether to use enzymatic retro-transformations. Default True.
        use_synthetic: Whether to use synthetic retro-transformations. Default True.
        use_chem_building_blocksDB: Whether to load chemical building blocks as sink
            compounds. Default True. Set False for ablation studies.
        use_bio_building_blocksDB: Whether to load biological building blocks as sink
            compounds. Default True. Set False for ablation studies.
        use_PKS_building_blocksDB: Whether to load PKS library for reward calculation.
            Default True. Set False for ablation studies.
        stop_on_first_pathway: If True, stop MCTS as soon as a complete pathway is found.
            Useful for benchmarking time-to-first-solution. Default False.
        selection_policy: Selection policy for MCTS. Options:
            - "UCB1": Standard UCB1 (breadth-first tendency, explores all nodes at each level)
            - "depth_biased": Depth-biased UCB1 (depth-first tendency, reaches max_depth faster)
            Default is "depth_biased".
        depth_bonus_coefficient: Coefficient for depth bonus in depth_biased selection.
            Higher values encourage deeper exploration. Default 4.0. Ignored if selection_policy="UCB1".
        enable_frontier_fallback: If True, maintain a frontier of unexpanded non-terminal
            nodes and fall back to selecting from this frontier when standard tree traversal
            returns None (hits an all-terminal branch). This enables deeper exploration by
            ensuring iterations are not wasted. Default True.
        search_strategy: Search strategy to use. Must be "mcts" for the async runner.
            BFS mode is not supported with AsyncExpansionDORAnetMCTS because BFS is
            single-core only. Use run_DORAnet_single_agent.py for BFS experiments.
    """
    # ---- Validate search strategy ----
    if search_strategy != "mcts":
        raise ValueError(
            f"search_strategy='{search_strategy}' is not supported with AsyncExpansionDORAnetMCTS. "
            f"BFS mode is single-core only. Use run_DORAnet_single_agent.py for BFS, "
            f"or use search_strategy='mcts' for async."
        )
    # ---- Runner configuration ----
    create_interactive_visualization = True
    enable_iteration_viz = True
    iteration_interval = 1
    auto_open_iteration_viz = True
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
    pks_library_file = REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V3.txt"

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
    # Default: Clean architecture with separate terminal detection and reward policies
    if terminal_detector is None:
        # Terminal detector handles PKS matching + RetroTide verification only
        terminal_detector = VerifyWithRetroTide()
    if reward_policy is None:
        # Reward handles terminal rewards + SA score for non-terminals
        reward_policy = SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=1.0,
            pks_terminal_reward=1.0,
        )

    agent = AsyncExpansionDORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=total_iterations,
        max_depth=max_depth,
        use_enzymatic=use_enzymatic,
        use_synthetic=use_synthetic,
        generations_per_expand=1,
        max_children_per_expand=max_children_per_expand,
        child_downselection_strategy=child_downselection_strategy,
        cofactors_files=[str(f) for f in cofactors_files],
        pks_library_file=str(pks_library_file),
        sink_compounds_files=[str(f) for f in sink_compounds_files],
        prohibited_chemicals_file=str(prohibited_chemicals_file),
        use_chem_building_blocksDB=use_chem_building_blocksDB,
        use_bio_building_blocksDB=use_bio_building_blocksDB,
        use_PKS_building_blocksDB=use_PKS_building_blocksDB,
        MW_multiple_to_exclude=MW_multiple_to_exclude,

        # Policies passed as explicit arguments
        terminal_detector=terminal_detector,
        reward_policy=reward_policy,

        # RetroTide configuration (used by VerifyWithRetroTide terminal detector)
        retrotide_kwargs={
            "max_depth": 6,
            "total_iterations": 100,
            "maxPKSDesignsRetroTide": 500,
        },
        
        # ---- Selection & Reward Configuration ----
        sink_terminal_reward=1.0,
        selection_policy=selection_policy,
        depth_bonus_coefficient=depth_bonus_coefficient,
        
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

        # ---- Early Stopping Configuration ----
        stop_on_first_pathway=stop_on_first_pathway,

        # ---- Frontier Fallback Configuration ----
        enable_frontier_fallback=enable_frontier_fallback,
    )

    start_time = time.time()
    agent.run()
    total_runtime = time.time() - start_time

    print("\n" + agent.get_tree_summary())

    results_dir = REPO_ROOT / "results"
    if results_subfolder:
        results_dir = results_dir / results_subfolder
    results_dir.mkdir(parents=True, exist_ok=True)
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

if __name__ == "__main__":

    # ---- Configure Policies ----
    # RECOMMENDED: Clean architecture with separate terminal detection and reward policies
    # Terminal detector handles PKS matching + RetroTide verification only
    selected_terminal_detector = VerifyWithRetroTide()

    # ---- Non-terminal scoring ----
    # Toggle: "sa_score" (default) or "gnn_pks" (GNN polyketide classifier)
    non_terminal_scoring = "sa_score"

    non_terminal_scorer = None  # None = SA score (default)
    if non_terminal_scoring == "gnn_pks":
        from DORAnet_agent.policies import GNNPolyketideScorer
        non_terminal_scorer = GNNPolyketideScorer(
            checkpoint_path="models/gnn_pks_classifier/best_model.pt",
        )

    # Reward handles terminal rewards + non-terminal scoring, scaled by thermodynamic feasibility
    selected_reward_policy = ThermodynamicScaledRewardPolicy(
        base_policy=SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=1.0, pks_terminal_reward=1.0,
            non_terminal_scorer=non_terminal_scorer,
        ),
        feasibility_weight=1.0,
        sigmoid_k=0.2,
        sigmoid_threshold=15.0,
        use_dora_xgb_for_enzymatic=True,
        aggregation="geometric_mean",
    )

    # Alternative: No terminal detection (just expand, no RetroTide)
    # selected_terminal_detector = NoOpTerminalDetector()

    main(
        target_smiles="C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",
        molecule_name="tryptophan",
        total_iterations=2000,
        max_depth=11,
        max_children_per_expand=20,
        terminal_detector=selected_terminal_detector,
        reward_policy=selected_reward_policy,
        results_subfolder="tryptophan_biosynthesis",
        MW_multiple_to_exclude=2.5,
        child_downselection_strategy="most_thermo_feasible",
        use_enzymatic=True,
        use_synthetic=False,
        use_chem_building_blocksDB=False,
        use_bio_building_blocksDB=True,
        use_PKS_building_blocksDB=False,
        stop_on_first_pathway=False,
        selection_policy="depth_biased",
        enable_frontier_fallback=True,  # Enable frontier fallback for deep exploration
        # search_strategy="mcts",  # Only "mcts" is supported for async runner.
        # For BFS baseline experiments, use run_DORAnet_single_agent.py with search_strategy="bfs".
        # NOTE: In BFS mode, total_iterations controls max depth levels to expand,
        # NOT the number of MCTS iterations.
    )
