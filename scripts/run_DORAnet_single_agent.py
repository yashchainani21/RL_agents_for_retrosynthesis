"""
Example runner for the simplified DORAnet MCTS agent.

This script launches a DORAnet tree search that fragments a target molecule
using retro-enzymatic and retro-synthetic transformations. For each fragment
discovered, a RetroTide forward MCTS search is spawned to attempt synthesis
from PKS building blocks.

Policy System:
- rollout_policy: Controls what happens after expansion (default: SpawnRetroTideOnDatabaseCheck)
  - NoOpRolloutPolicy: No additional work after expansion (just returns 0 reward)
  - SpawnRetroTideOnDatabaseCheck: Spawns RetroTide for PKS library matches (sparse rewards)
  - SAScore_and_SpawnRetroTideOnDatabaseCheck: SA Score rewards + RetroTide spawning (legacy)
  - PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck: PKS similarity + RetroTide (dense, PKS-focused)
- reward_policy: Controls how terminal rewards are calculated (default: SAScore_and_TerminalRewardPolicy)
  - SAScore_and_TerminalRewardPolicy: Terminal rewards + SA score for non-terminals (RECOMMENDED)
    - Provides dense signals via SA score for synthetic accessibility
    - Full terminal reward for sink compounds and PKS terminals
    - Cleanly separates reward from rollout concerns
  - SparseTerminalRewardPolicy: 1.0 for sink compounds, 1.0 for PKS matches, 0.0 otherwise
  - SinkCompoundRewardPolicy: Only rewards sink compounds
  - ComposedRewardPolicy: Combine multiple reward policies with weights
  - PKSSimilarityRewardPolicy: PKS Tanimoto similarity as sole reward signal

Example: Recommended clean setup (rollout + reward separation)
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

Backward Compatibility:
- spawn_retrotide=True creates SpawnRetroTideOnDatabaseCheck automatically
- Explicit rollout_policy/reward_policy override spawn_retrotide
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

from DORAnet_agent import DORAnetMCTS, Node
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
    PKSSimilarityRewardPolicy,
    SAScore_and_TerminalRewardPolicy,
    # Thermodynamic scaling wrappers
    ThermodynamicScaledRolloutPolicy,
    ThermodynamicScaledRewardPolicy,
)
RDLogger.DisableLog("rdApp.*")

### ---- Molecules ---- 

## commodity chemicals ##

# target_smiles = "CCCCCCCCC(=O)O"  # nonanoic acid (known PKS product)
# target_smiles = "CCCC(C)=O"  # 3-pentanone (simple ketone)
# target_smiles = "OCCCC(=O)O"  # 4-hydroxybutyric acid (gamma-hydroxybutyric acid)
# target_smiles = "OCCCCO"  # 1,4-butanediol
# target_smiles = "CCCCC(=O)O"  # pentanoic acid (valeric acid)
# target_smiles = "CC1CCCCC(CC1)C" # DMCO

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

## some pharmaceuticals ##

# cryptofolione # O=C1C=CCC(C=CCC(O)CC(O)C=Cc2ccccc2)O1
# dronabinol # CCCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O
# arformoterol # CC(CC1=CC=C(C=C1)OC)NCC(C2=CC(=C(C=C2)O)NC=O)O
# patchoul # OC23CCC(C1CC(CCC12C)C3(C)C)C" 

def main(target_smiles: str,
         molecule_name: str,
         total_iterations: int,
         max_depth: int,
         max_children_per_expand: int,
         rollout_policy: Optional[RolloutPolicy] = None,
         reward_policy: Optional[RewardPolicy] = None,
         results_subfolder: str = None,
         MW_multiple_to_exclude: float = 1.5,
         child_downselection_strategy: str = "most_thermo_feasible",
         use_enzymatic: bool = True,
         use_synthetic: bool = True,
         use_chem_building_blocksDB: bool = True,
         use_bio_building_blocksDB: bool = True,
         use_PKS_building_blocksDB: bool = True,
         stop_on_first_pathway: bool = False) -> None:
    """
    Run the DORAnet MCTS agent.

    Args:
        target_smiles: SMILES string of the target molecule
        molecule_name: Human-readable name for the molecule (used in filenames)
        total_iterations: Number of MCTS iterations to run
        max_depth: Maximum depth of the retrosynthetic search tree
        max_children_per_expand: Maximum number of children to generate per expansion
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
        results_subfolder: Optional subfolder within results/ to save outputs.
                          If None, saves directly to results/. Useful for batch runs.
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
    """
    create_interactive_visualization = False
    enable_iteration_viz = False
    iteration_interval = 1
    auto_open_iteration_viz = False
    auto_cleanup_pgnet_files = True

    target_molecule = Chem.MolFromSmiles(target_smiles)
    
    if target_molecule is None:
        raise ValueError(f"Could not parse target SMILES: {target_smiles}")

    if molecule_name:
        print(f"Target molecule: {molecule_name} ({target_smiles})")
    else:
        print(f"Target molecule: {target_smiles}")

    # specify paths to cofactor files (metabolites and chemistry helpers to exclude from the network)
    cofactors_files = [
        REPO_ROOT / "data" / "raw" / "all_cofactors.csv",
        REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv",
    ]

    # specify path to PKS library file for reward calculation
    pks_library_file = REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V3.txt"

    # specify paths to sink compounds files (commercially available building blocks)
    # both biological and chemical building blocks are loaded as sink compounds
    sink_compounds_files = [
        REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt",
        REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt",
    ]

    # specify path to prohibited chemicals file (hazardous/controlled substances to avoid)
    prohibited_chemicals_file = REPO_ROOT / "data" / "processed" / "prohibited_chemical_SMILES.txt"

    # create root node with target
    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    # ---- Policy Configuration ----
    # Use provided policies or create defaults
    # Default: Clean architecture with separate rollout and reward policies
    if rollout_policy is None:
        # Rollout handles PKS matching + RetroTide spawning only
        rollout_policy = SpawnRetroTideOnDatabaseCheck(
            success_reward=1.0,
            failure_reward=0.0,
        )
    if reward_policy is None:
        # Reward handles terminal rewards + SA score for non-terminals
        reward_policy = SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=1.0,
            pks_terminal_reward=1.0,
        )

    # Common configuration for both sequential and parallel agents
    agent_kwargs = dict(
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
        sink_terminal_reward=1.0,  # bias selection toward terminal sink compounds
        selection_policy="UCB1",  # "UCB1" for standard or "depth_biased" for depth-first
        depth_bonus_coefficient=4.0,  # only used with depth_biased policy (higher = more depth-first)
        
        # ---- Visualization Configuration ----
        enable_visualization=False,
        enable_interactive_viz=False,  # enable interactive HTML visualizations
        enable_iteration_visualizations=enable_iteration_viz,  # generate visualizations per iteration
        iteration_viz_interval=iteration_interval,   # how often to generate iteration visualizations
        auto_open_iteration_viz=auto_open_iteration_viz,  # auto-open iteration visualizations in browser
        visualization_output_dir=str(REPO_ROOT / "results"),

        # ---- Early Stopping Configuration ----
        stop_on_first_pathway=stop_on_first_pathway,
    )

    print("[Runner] Using sequential DORAnetMCTS")
    agent = DORAnetMCTS(**agent_kwargs)

    # Run the search
    start_time = time.time()
    agent.run()
    total_runtime = time.time() - start_time

    # Print tree summary
    print("\n" + agent.get_tree_summary())

    # Check if we're in PKS library mode or RetroTide mode
    if agent.pks_library:
        # PKS library mode - show matches
        pks_matches = agent.get_pks_matches()
        sink_compounds = agent.get_sink_compounds()
        print(f"\n" + "=" * 70)
        print("PKS Library Match & Sink Compound Results")
        print("=" * 70)
        print(f"\nTotal nodes explored: {len(agent.nodes)}")
        print(f"PKS library matches: {len([n for n in pks_matches if not n.is_sink_compound])}")
        print(f"Sink compounds (building blocks): {len(sink_compounds)}")

        if sink_compounds:
            print(f"\n■ SINK COMPOUNDS (commercially available building blocks):")
            print("-" * 70)
            for node in sink_compounds:
                avg_value = node.value / node.visits if node.visits > 0 else 0
                print(f"  Node {node.node_id} ({node.provenance}): {node.smiles}")
                print(f"    Depth: {node.depth}, Visits: {node.visits}, Avg Value: {avg_value:.2f}")
                if node.reaction_name:
                    rxn_display = node.reaction_name[:60] + "..." if len(node.reaction_name) > 60 else node.reaction_name
                    print(f"    Reaction: {rxn_display}")

        # Show non-sink PKS matches
        non_sink_pks = [n for n in pks_matches if not n.is_sink_compound]
        if non_sink_pks:
            print(f"\n✅ FRAGMENTS MATCHING PKS LIBRARY (non-sink):")
            print("-" * 70)
            for node in non_sink_pks:
                avg_value = node.value / node.visits if node.visits > 0 else 0
                print(f"  Node {node.node_id} ({node.provenance}): {node.smiles}")
                print(f"    Depth: {node.depth}, Visits: {node.visits}, Avg Value: {avg_value:.2f}")
                if node.reaction_name:
                    rxn_display = node.reaction_name[:60] + "..." if len(node.reaction_name) > 60 else node.reaction_name
                    print(f"    Reaction: {rxn_display}")
    else:
        # RetroTide mode - show detailed results
        print(agent.get_results_summary())

    # Save detailed results to file (and generate visualizations if enabled)
    results_dir = REPO_ROOT / "results"
    if results_subfolder:
        results_dir = results_dir / results_subfolder
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use molecule name for filename if provided, otherwise use SMILES
    if molecule_name:
        # Sanitize molecule name for filename
        safe_name = molecule_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
        filename_base = f"{safe_name}_sequential"
    else:
        safe_smiles = target_smiles.replace("/", "_").replace("\\", "_")[:20]
        filename_base = f"{safe_smiles}_sequential"

    results_path = results_dir / f"doranet_results_{filename_base}_{timestamp}.txt"
    agent.save_results(str(results_path))
    finalized_pathways_path = results_dir / f"finalized_pathways_{filename_base}_{timestamp}.txt"
    agent.save_finalized_pathways(str(finalized_pathways_path), total_runtime_seconds=total_runtime)
    successful_pathways_path = results_dir / f"successful_pathways_{filename_base}_{timestamp}.txt"
    agent.save_successful_pathways(str(successful_pathways_path))

    # Print summary
    if agent.pks_library:
        pks_matches = agent.get_pks_matches()
        if pks_matches:
            print(f"\n🎉 Found {len(pks_matches)} PKS-synthesizable fragments!")
    else:
        successful = agent.get_successful_results()
        if successful:
            print(f"\n🎉 Found {len(successful)} successful PKS designs!")
            for r in successful:
                print(f"   Node {r.doranet_node_id} ({r.doranet_node_provenance}): {r.doranet_node_smiles}")

    if create_interactive_visualization:
        # results_dir already set above (with subfolder if specified)
        # Use same filename base for interactive visualization
        interactive_path = results_dir / f"doranet_interactive_{filename_base}_{timestamp}.html"
        pathways_path = results_dir / f"doranet_pathways_{filename_base}_{timestamp}.html"

        # Create full tree interactive visualization
        create_enhanced_interactive_html(
            agent=agent,
            output_path=str(interactive_path),
            molecule_img_size=(250, 250),  # size of molecule images in pixels
            auto_open=True,  # automatically open in browser!
            )

        # Create pathways-only interactive visualization (filtered to PKS matches and sink compounds)
        create_pathways_interactive_html(
            agent=agent,
            output_path=str(pathways_path),
            molecule_img_size=(250, 250),
            auto_open=True,  # automatically open in browser!
            )

        print("\n" + "=" * 70)
        print("✓ Interactive visualizations complete!")
        print("=" * 70)
        print(f"\nTwo browser tabs opened:")
        print(f"  1. Full tree: file://{interactive_path}")
        print(f"  2. Pathways only: file://{pathways_path}")
        print("\nFeatures:")
        print("  • Hover over nodes to see molecule structures")
        print("  • View metadata: provenance, PKS match, visits, value")
        print("  • Hover over edges to see reaction SMARTS")
        print("  • Use mouse wheel to zoom")
        print("  • Drag to pan around the tree")
        print("  • Click reset button to restore original view")
        print("\nColor scheme:")
        print("  🟠 Orange = Target molecule")
        print("  🔵 Blue = Enzymatic pathway")
        print("  🟣 Purple = Synthetic pathway")
        print("  🟢 Green = PKS library match ✓")
        print("  ■ Square = Sink compound (building block)")
        print("\nPathways view shows ONLY paths that lead to PKS matches or sink compounds.")

    if auto_cleanup_pgnet_files:
        cleanup_script = REPO_ROOT / "scripts" / "cleanup_pgnet_files.py"
        try:
            subprocess.run([sys.executable, str(cleanup_script), "-y"], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[Runner] Warning: .pgnet cleanup failed ({exc}).")

if __name__ == "__main__":
    # ---- Configure Policies ----
    # RECOMMENDED: Clean architecture with separate rollout and reward policies
    # Rollout handles PKS matching + RetroTide spawning only
    selected_rollout_policy = SpawnRetroTideOnDatabaseCheck(
        success_reward=1.0,
        failure_reward=0.0,
    )
    # Reward handles terminal rewards + SA score for non-terminals
    # selected_reward_policy = SAScore_and_TerminalRewardPolicy(
    #     sink_terminal_reward=1.0,
    #     pks_terminal_reward=1.0,
    # )

    # Alternative: PKS similarity + RetroTide (uses Tanimoto fingerprint similarity)
    # selected_rollout_policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck()
    # selected_reward_policy = PKSSimilarityRewardPolicy(similarity_exponent=2.0)

    # Alternative: Dense rewards - SA Score + RetroTide (legacy, conflates concerns)
    # selected_rollout_policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
    #     success_reward=1.0,
    #     sa_max_reward=1.0,
    # )
    # selected_reward_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)

    # Alternative: No rollout (just expand, no RetroTide spawning)
    # selected_rollout_policy = NoOpRolloutPolicy()

    # Alternative: Thermodynamic-scaled policies (wrap any base policy)
    # This scales rewards by pathway thermodynamic feasibility using DORA-XGB
    # for enzymatic reactions and sigmoid-transformed ΔH for synthetic reactions.
    # selected_rollout_policy = ThermodynamicScaledRolloutPolicy(
    #     base_policy=SpawnRetroTideOnDatabaseCheck(success_reward=1.0),
    #     feasibility_weight=0.8,      # 0.0=ignore feasibility, 1.0=full scaling
    #     sigmoid_k=0.2,               # Steepness of sigmoid for ΔH
    #     sigmoid_threshold=15.0,      # Center point in kcal/mol
    #     use_dora_xgb_for_enzymatic=True,  # Use DORA-XGB for enzymatic reactions
    #     aggregation="geometric_mean",     # How to aggregate pathway scores
    # )
    selected_reward_policy = ThermodynamicScaledRewardPolicy(
        base_policy=SAScore_and_TerminalRewardPolicy(sink_terminal_reward=1.0, pks_terminal_reward=1.0),
        feasibility_weight=0.8,
        sigmoid_k=0.2,
        sigmoid_threshold=15.0,
        use_dora_xgb_for_enzymatic=True,
        aggregation="geometric_mean")

    main(
        target_smiles="CCCCC(=O)O",
        molecule_name="pentanoic_acid",
        total_iterations=100,
        max_depth=4,
        max_children_per_expand=30,
        rollout_policy=selected_rollout_policy,
        reward_policy=selected_reward_policy,
        results_subfolder=None,
        MW_multiple_to_exclude=1.5,
        child_downselection_strategy="most_thermo_feasible",
        use_enzymatic=True,
        use_synthetic=True,
        use_chem_building_blocksDB=True,
        use_bio_building_blocksDB=True,
        use_PKS_building_blocksDB=True,
        stop_on_first_pathway=False,  # Set to True to enable early stopping
    )
