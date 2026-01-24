"""
Extract reactions from DORAnet network expansions and compile to CSV.

This script runs DORAnet enzymatic and/or synthetic expansions on input molecules
and extracts all reactions with their operators into a CSV file.

Output CSV format:
- Column 1: reaction_string (A.B>>C.D format)
- Column 2: operator (rule name or SMARTS)
- Additional columns: mode, reactants, products

Usage:
    python scripts/extract_doranet_reactions.py --smiles "CCCCC(=O)O" --output reactions.csv
    python scripts/extract_doranet_reactions.py --smiles-file molecules.txt --output reactions.csv
    python scripts/extract_doranet_reactions.py --run-mcts --iterations 20 --output reactions.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit import RDLogger

# Ensure repository root is discoverable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Silence RDKit logs
RDLogger.DisableLog("rdApp.*")


@dataclass
class ReactionRecord:
    """A single extracted reaction."""
    reaction_string: str  # A.B>>C.D format
    operator: str  # Rule name or SMARTS
    mode: str  # "enzymatic" or "synthetic"
    reactants: List[str]
    products: List[str]
    operator_index: Optional[int] = None
    smarts: Optional[str] = None


def load_rule_labels() -> Tuple[List[str], List[str]]:
    """Load human-readable labels for enzymatic and synthetic rules."""
    enzymatic_labels = []
    synthetic_labels = []

    try:
        import doranet.modules.enzymatic as enzymatic
        rules_file = Path(enzymatic.__file__).parent / "JN3604IMT_rules.tsv"
        if rules_file.exists():
            with open(rules_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        enzymatic_labels.append(parts[1])
    except Exception as e:
        print(f"[WARN] Could not load enzymatic labels: {e}")

    try:
        import doranet.modules.synthetic.Reaction_Smarts_Retro as synthetic_module
        for name in dir(synthetic_module):
            if not name.startswith("_"):
                obj = getattr(synthetic_module, name)
                if isinstance(obj, dict) and "smarts" in obj:
                    synthetic_labels.append(name)
    except Exception as e:
        print(f"[WARN] Could not load synthetic labels: {e}")

    return enzymatic_labels, synthetic_labels


def extract_reactions_from_network(
    network,
    mode: str,
    enzymatic_labels: List[str],
    synthetic_labels: List[str],
) -> List[ReactionRecord]:
    """
    Extract all reactions from a DORAnet network.

    Args:
        network: DORAnet ChemNetwork object
        mode: "enzymatic" or "synthetic"
        enzymatic_labels: List of enzymatic rule labels
        synthetic_labels: List of synthetic rule labels

    Returns:
        List of ReactionRecord objects
    """
    records = []

    mols_list = list(network.mols)
    ops_list = list(network.ops)

    for rxn in getattr(network, 'rxns', []):
        try:
            # Get operator index and info
            op_idx = rxn.operator
            op = ops_list[op_idx] if op_idx < len(ops_list) else None

            # Extract SMARTS
            if op and hasattr(op, 'uid'):
                smarts = str(op.uid)
            else:
                smarts = str(op) if op else None

            # Get human-readable label
            operator_label = None
            if op_idx is not None:
                if mode == "enzymatic":
                    if op_idx < len(enzymatic_labels):
                        operator_label = enzymatic_labels[op_idx]
                else:  # synthetic
                    if op_idx < len(synthetic_labels):
                        operator_label = synthetic_labels[op_idx]

            # Fallback to SMARTS or index if no label
            if not operator_label:
                if smarts:
                    operator_label = smarts[:60] + "..." if len(smarts) > 60 else smarts
                else:
                    operator_label = f"rule{op_idx:05d}"

            # Get reactants and products
            reactant_idxs = rxn.reactants
            product_idxs = rxn.products

            reactant_smiles = []
            for i in reactant_idxs:
                if i < len(mols_list):
                    mol = mols_list[i]
                    smiles = str(getattr(mol, 'uid', mol))
                    reactant_smiles.append(smiles)

            product_smiles = []
            for i in product_idxs:
                if i < len(mols_list):
                    mol = mols_list[i]
                    smiles = str(getattr(mol, 'uid', mol))
                    product_smiles.append(smiles)

            # Skip if no valid reactants/products
            if not reactant_smiles or not product_smiles:
                continue

            # Create reaction string (A.B>>C.D format)
            reaction_string = ".".join(reactant_smiles) + ">>" + ".".join(product_smiles)

            record = ReactionRecord(
                reaction_string=reaction_string,
                operator=operator_label,
                mode=mode,
                reactants=reactant_smiles,
                products=product_smiles,
                operator_index=op_idx,
                smarts=smarts,
            )
            records.append(record)

        except Exception as e:
            continue

    return records


def run_doranet_expansion(
    smiles: str,
    mode: str,
    generations: int = 1,
    helpers: Optional[List[Chem.Mol]] = None,
) -> Optional[object]:
    """
    Run DORAnet expansion on a molecule.

    Args:
        smiles: Input molecule SMILES
        mode: "enzymatic" or "synthetic"
        generations: Number of expansion generations
        helpers: List of helper molecules (for synthetic mode)

    Returns:
        DORAnet network object or None if failed
    """
    # Validate SMILES first
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[WARN] Could not parse SMILES: {smiles}")
        return None

    # Canonicalize SMILES for consistency
    canonical_smiles = Chem.MolToSmiles(mol)

    try:
        if mode == "enzymatic":
            import doranet.modules.enzymatic as module
            network = module.generate_network(
                starters={canonical_smiles},  # Set of SMILES strings
                gen=generations,
                direction="retro",
            )
        else:  # synthetic
            import doranet.modules.synthetic as module
            network_kwargs = {
                "starters": {canonical_smiles},  # Set of SMILES strings
                "gen": generations,
                "direction": "retro",
            }
            if helpers:
                network_kwargs["helpers"] = helpers
            network = module.generate_network(**network_kwargs)
        return network
    except Exception as e:
        print(f"[WARN] DORAnet {mode} failed for {smiles}: {e}")
        return None


def load_cofactors(cofactors_files: List[str]) -> List[str]:
    """Load cofactor SMILES from files."""
    cofactors = []
    for filepath in cofactors_files:
        if not Path(filepath).exists():
            continue
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            if 'SMILES' in df.columns:
                cofactors.extend(df['SMILES'].dropna().tolist())
            elif 'smiles' in df.columns:
                cofactors.extend(df['smiles'].dropna().tolist())
        except Exception:
            # Try simple text file
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        cofactors.append(line)
    return cofactors


def load_helpers(helpers_file: str) -> List[Chem.Mol]:
    """Load helper molecules from file."""
    helpers = []
    if not Path(helpers_file).exists():
        return helpers
    try:
        import pandas as pd
        df = pd.read_csv(helpers_file)
        smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
        if smiles_col in df.columns:
            for smiles in df[smiles_col].dropna():
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    helpers.append(mol)
    except Exception:
        pass
    return helpers


def extract_reactions_from_mcts_run(
    target_smiles: str,
    iterations: int = 20,
    max_depth: int = 2,
    use_enzymatic: bool = True,
    use_synthetic: bool = True,
) -> List[ReactionRecord]:
    """
    Run a DORAnet MCTS search and extract all reactions from nodes.

    This uses the reaction information stored in MCTS tree nodes.
    """
    from DORAnet_agent import DORAnetMCTS, Node

    target_molecule = Chem.MolFromSmiles(target_smiles)
    if target_molecule is None:
        print(f"[ERROR] Could not parse target SMILES: {target_smiles}")
        return []

    # Paths to data files
    cofactors_files = [
        str(REPO_ROOT / "data" / "raw" / "all_cofactors.csv"),
        str(REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv"),
    ]
    pks_library_file = str(REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V3.txt")
    sink_compounds_files = [
        str(REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt"),
        str(REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt"),
    ]

    root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

    agent = DORAnetMCTS(
        root=root,
        target_molecule=target_molecule,
        total_iterations=iterations,
        max_depth=max_depth,
        use_enzymatic=use_enzymatic,
        use_synthetic=use_synthetic,
        spawn_retrotide=False,
        cofactors_files=cofactors_files,
        pks_library_file=pks_library_file,
        sink_compounds_files=sink_compounds_files,
        enable_visualization=False,
        enable_interactive_viz=False,
    )

    print(f"[INFO] Running MCTS on {target_smiles}...")
    agent.run()

    # Extract reactions from nodes
    records = []
    seen_reactions = set()

    for node in agent.nodes:
        if node.reaction_smarts and node.reactants_smiles and node.products_smiles:
            # Create reaction string
            reaction_string = ".".join(node.reactants_smiles) + ">>" + ".".join(node.products_smiles)

            # Skip duplicates
            if reaction_string in seen_reactions:
                continue
            seen_reactions.add(reaction_string)

            record = ReactionRecord(
                reaction_string=reaction_string,
                operator=node.reaction_name or node.reaction_smarts[:60],
                mode=node.provenance or "unknown",
                reactants=node.reactants_smiles,
                products=node.products_smiles,
                smarts=node.reaction_smarts,
            )
            records.append(record)

    return records


def save_reactions_to_csv(records: List[ReactionRecord], output_path: str) -> None:
    """Save reaction records to CSV file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'reaction_string',
            'operator',
            'mode',
            'reactants',
            'products',
            'smarts',
        ])

        # Write records
        for record in records:
            writer.writerow([
                record.reaction_string,
                record.operator,
                record.mode,
                ";".join(record.reactants),
                ";".join(record.products),
                record.smarts or "",
            ])

    print(f"[INFO] Saved {len(records)} reactions to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract reactions from DORAnet expansions to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MCTS and extract reactions from tree nodes
  python scripts/extract_doranet_reactions.py --run-mcts --smiles "CCCCC(=O)O" -o reactions.csv

  # Direct DORAnet expansion on single molecule
  python scripts/extract_doranet_reactions.py --smiles "CCCCC(=O)O" -o reactions.csv

  # Process multiple molecules from file
  python scripts/extract_doranet_reactions.py --smiles-file molecules.txt -o reactions.csv

  # Enzymatic only
  python scripts/extract_doranet_reactions.py --smiles "CCO" --enzymatic-only -o reactions.csv
        """
    )

    parser.add_argument(
        "--smiles", "-s",
        type=str,
        help="Single SMILES string to process"
    )
    parser.add_argument(
        "--smiles-file", "-f",
        type=str,
        help="File containing SMILES strings (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="reactions.csv",
        help="Output CSV file path (default: reactions.csv)"
    )
    parser.add_argument(
        "--run-mcts",
        action="store_true",
        help="Run full MCTS search and extract reactions from tree nodes"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=20,
        help="MCTS iterations (default: 20, only used with --run-mcts)"
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=2,
        help="Maximum expansion depth (default: 2)"
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=1,
        help="DORAnet generations per expansion (default: 1)"
    )
    parser.add_argument(
        "--enzymatic-only",
        action="store_true",
        help="Only run enzymatic expansions"
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Only run synthetic expansions"
    )

    args = parser.parse_args()

    # Determine which modes to use
    use_enzymatic = not args.synthetic_only
    use_synthetic = not args.enzymatic_only

    # Collect input SMILES
    input_smiles = []
    if args.smiles:
        input_smiles.append(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    input_smiles.append(line)

    if not input_smiles:
        parser.print_help()
        print("\n[ERROR] No input SMILES provided. Use --smiles or --smiles-file")
        sys.exit(1)

    print(f"[INFO] Processing {len(input_smiles)} molecule(s)")
    print(f"[INFO] Modes: enzymatic={use_enzymatic}, synthetic={use_synthetic}")

    all_records = []

    if args.run_mcts:
        # Run MCTS and extract from tree nodes
        for smiles in input_smiles:
            records = extract_reactions_from_mcts_run(
                target_smiles=smiles,
                iterations=args.iterations,
                max_depth=args.max_depth,
                use_enzymatic=use_enzymatic,
                use_synthetic=use_synthetic,
            )
            all_records.extend(records)
            print(f"[INFO] Extracted {len(records)} reactions from MCTS on {smiles[:30]}...")
    else:
        # Direct DORAnet expansion
        enzymatic_labels, synthetic_labels = load_rule_labels()
        print(f"[INFO] Loaded {len(enzymatic_labels)} enzymatic labels, {len(synthetic_labels)} synthetic labels")

        # Load helpers for synthetic mode
        helpers_file = str(REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv")
        helpers = load_helpers(helpers_file)
        print(f"[INFO] Loaded {len(helpers)} helpers for synthetic mode")

        for smiles in input_smiles:
            print(f"[INFO] Processing: {smiles[:50]}...")

            if use_enzymatic:
                network = run_doranet_expansion(
                    smiles=smiles,
                    mode="enzymatic",
                    generations=args.generations,
                )
                if network:
                    records = extract_reactions_from_network(
                        network, "enzymatic", enzymatic_labels, synthetic_labels
                    )
                    all_records.extend(records)
                    print(f"  - Enzymatic: {len(records)} reactions")

            if use_synthetic:
                network = run_doranet_expansion(
                    smiles=smiles,
                    mode="synthetic",
                    generations=args.generations,
                    helpers=helpers,
                )
                if network:
                    records = extract_reactions_from_network(
                        network, "synthetic", enzymatic_labels, synthetic_labels
                    )
                    all_records.extend(records)
                    print(f"  - Synthetic: {len(records)} reactions")

    # Remove duplicates
    seen = set()
    unique_records = []
    for record in all_records:
        key = (record.reaction_string, record.operator)
        if key not in seen:
            seen.add(key)
            unique_records.append(record)

    print(f"\n[INFO] Total unique reactions: {len(unique_records)}")

    # Save to CSV
    save_reactions_to_csv(unique_records, args.output)


if __name__ == "__main__":
    main()
