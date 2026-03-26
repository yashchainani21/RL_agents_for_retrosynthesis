"""Command-line entry point for RetroTide MCTS forward PKS synthesis."""
from __future__ import annotations

import argparse

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run RetroTide MCTS forward PKS synthesis search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target_smiles", help="SMILES string of the target molecule")
    parser.add_argument("--max-depth", type=int, default=15, help="Maximum PKS module depth")
    parser.add_argument("--iterations", type=int, default=150, help="Number of MCTS iterations")
    parser.add_argument("--max-designs", type=int, default=1000, help="Max PKS designs to enumerate")
    parser.add_argument("--selection", choices=["UCB1"], default="UCB1", help="Selection policy")
    parser.add_argument("--save-logs", action="store_true", help="Save detailed search logs")
    args = parser.parse_args(argv)

    target_molecule = Chem.MolFromSmiles(args.target_smiles)
    if target_molecule is None:
        parser.error(f"Could not parse SMILES: {args.target_smiles}")

    print(f"Target molecule: {args.target_smiles}")

    # Late imports so --help is fast
    from RetroTide_agent.node import Node
    from RetroTide_agent.mcts import MCTS

    root = Node(PKS_product=None, PKS_design=None, parent=None, depth=0)

    mcts = MCTS(
        root=root,
        target_molecule=target_molecule,
        max_depth=args.max_depth,
        total_iterations=args.iterations,
        maxPKSDesignsRetroTide=args.max_designs,
        selection_policy=args.selection,
        save_logs=args.save_logs,
    )

    mcts.run()
    mcts.save_results()

    print("\nSuccessful nodes reached by RetroTide MCTS:\n")
    for node in mcts.successful_nodes:
        print(node)
        print()

    print("\nSuccessful PKS designs reached in simulation:\n")
    for design in mcts.successful_simulated_designs:
        print(design)
        print()


if __name__ == "__main__":
    main()
