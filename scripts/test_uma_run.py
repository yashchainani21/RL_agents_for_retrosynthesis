"""Quick test run: 5,6-dihydroyangonin with UMA thermodynamic scorer."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent import AsyncExpansionDORAnetMCTS, Node
from DORAnet_agent.policies import (
    VerifyWithRetroTide,
    SAScore_and_TerminalRewardPolicy,
    ThermodynamicScaledRewardPolicy,
)

RDLogger.DisableLog("rdApp.*")


def main():
    # HuggingFace auth for UMA model download
    from dotenv import load_dotenv
    from huggingface_hub import login
    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    # 5,6-dihydroyangonin
    TARGET_SMILES = "COC1=CC(OC(C=CC2=CC=C(OC)C=C2)C1)=O"
    MOLECULE_NAME = "5_6_dihydroyangonin"

    target_molecule = Chem.MolFromSmiles(TARGET_SMILES)
    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    # Policies: thermodynamic-scaled reward with SA score base
    terminal_detector = VerifyWithRetroTide()
    reward_policy = ThermodynamicScaledRewardPolicy(
        base_policy=SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=1.0, pks_terminal_reward=1.0
        ),
        feasibility_weight=1.0,
        sigmoid_k=0.2,
        sigmoid_threshold=15.0,
        use_dora_xgb_for_enzymatic=True,
        aggregation="geometric_mean",
    )

    agent = AsyncExpansionDORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=50,       # small test run
        max_depth=3,
        use_enzymatic=True,
        use_synthetic=True,
        generations_per_expand=1,
        max_children_per_expand=1,
        child_downselection_strategy=None,
        cofactors_files=[
            str(REPO_ROOT / "data" / "raw" / "all_cofactors.csv"),
            str(REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv"),
        ],
        pks_library_file=str(REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V3.txt"),
        sink_compounds_files=[
            str(REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt"),
            str(REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt"),
        ],
        prohibited_chemicals_file=str(REPO_ROOT / "data" / "processed" / "prohibited_chemical_SMILES.txt"),
        MW_multiple_to_exclude=1.5,
        terminal_detector=terminal_detector,
        reward_policy=reward_policy,
        retrotide_kwargs={"max_depth": 6, "total_iterations": 100, "maxPKSDesignsRetroTide": 500},
        sink_terminal_reward=1.0,
        selection_policy="UCB1",
        enable_visualization=False,
        enable_interactive_viz=False,
        enable_iteration_visualizations=False,
        stop_on_first_pathway=False,
        enable_frontier_fallback=False,
        # --- UMA scorer ---
        synthetic_thermo_scorer="uma",
    )

    print(f"\nScorer type: {type(agent.thermodynamic_scorer).__name__}")
    print(f"Starting MCTS run for {MOLECULE_NAME} with UMA scorer...")
    start = time.time()
    agent.run()
    elapsed = time.time() - start

    print("\n" + agent.get_tree_summary())
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
