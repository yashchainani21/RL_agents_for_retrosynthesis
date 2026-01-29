"""
Modular runtime benchmarking script for DORAnet synthesis programs.

Supports benchmarking ONE of three synthesis modes at a time:
1. DORAnet Standalone - Raw library expansion (no MCTS)
2. DORAnetMCTS Sequential - Single-threaded MCTS
3. AsyncExpansionDORAnetMCTS - Multiprocessing MCTS

Output: Human-readable .txt files with `benchmark_runtime_` prefix.

Usage:
    # Configure toggles at bottom of script, then:
    python scripts/benchmark_runtimes.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rdkit import Chem
from rdkit import RDLogger

# Ensure the repository root is discoverable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent import DORAnetMCTS, AsyncExpansionDORAnetMCTS, Node, clear_smiles_cache

# Silence RDKit logs
RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Configuration and Result Dataclasses
# =============================================================================

@dataclass
class RuntimeBenchmarkConfig:
    """Configuration container for runtime benchmarks."""
    # Benchmark mode: "standalone", "sequential", or "async"
    mode: str

    # Expansion mode toggles
    use_enzymatic: bool = True
    use_synthetic: bool = True

    # MCTS parameters
    total_iterations: int = 50
    max_depth: int = 3
    max_children_per_expand: Optional[int] = None  # None = no limit
    child_downselection_strategy: Optional[str] = None  # None = no filtering

    # Standalone parameters
    generations_per_expand: int = 1

    # Async parameters
    num_workers: Optional[int] = None  # None = auto-detect

    # Data paths (populated in __post_init__)
    cofactors_files: List[str] = field(default_factory=list)
    pks_library_file: Optional[str] = None
    sink_compounds_files: List[str] = field(default_factory=list)
    prohibited_chemicals_file: Optional[str] = None

    def __post_init__(self):
        """Set default data paths if not provided."""
        if not self.cofactors_files:
            self.cofactors_files = [
                str(REPO_ROOT / "data" / "raw" / "all_cofactors.csv"),
                str(REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv"),
            ]
        if not self.pks_library_file:
            self.pks_library_file = str(
                REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V3.txt"
            )
        if not self.sink_compounds_files:
            self.sink_compounds_files = [
                str(REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt"),
                str(REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt"),
            ]
        if not self.prohibited_chemicals_file:
            self.prohibited_chemicals_file = str(
                REPO_ROOT / "data" / "processed" / "prohibited_chemical_SMILES.txt"
            )


@dataclass
class RuntimeBenchmarkResult:
    """Results container for runtime benchmarks."""
    # Identification
    molecule_name: str
    molecule_smiles: str
    mode: str
    timestamp: str

    # Expansion configuration
    use_enzymatic: bool
    use_synthetic: bool

    # Runtime
    runtime_seconds: float

    # Standalone metrics (only populated for standalone mode)
    num_molecules_in_network: Optional[int] = None
    num_reactions_in_network: Optional[int] = None

    # MCTS metrics (only populated for sequential/async modes)
    total_nodes: Optional[int] = None
    unique_smiles: Optional[int] = None  # Number of unique molecules explored
    iterations_completed: Optional[int] = None
    terminal_nodes: Optional[int] = None
    sink_compounds: Optional[int] = None
    pks_matches: Optional[int] = None

    # Async-specific metrics
    num_workers: Optional[int] = None

    # Error tracking
    error: Optional[str] = None


# =============================================================================
# Runner Functions
# =============================================================================

def run_doranet_standalone(
    target_smiles: str,
    molecule_name: str,
    config: RuntimeBenchmarkConfig,
) -> RuntimeBenchmarkResult:
    """
    Run raw DORAnet library expansion (no MCTS).

    Calls enzymatic.generate_network() and/or synthetic.generate_network() directly.
    """
    import doranet.modules.enzymatic as enzymatic
    import doranet.modules.synthetic as synthetic

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="standalone",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            runtime_seconds=0.0,
            error=f"Could not parse SMILES: {target_smiles}",
        )

    # Load chemistry helpers for synthetic expansion
    chemistry_helpers = set()
    for path in config.cofactors_files:
        if Path(path).exists():
            import csv
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smiles = row.get("smiles") or row.get("SMILES")
                    if smiles:
                        chemistry_helpers.add(smiles)

    total_mols = 0
    total_rxns = 0

    try:
        start_time = time.perf_counter()

        if config.use_enzymatic:
            enz_network = enzymatic.generate_network(
                job_name="benchmark_enzymatic",
                starters={target_smiles},
                gen=config.generations_per_expand,
                direction="retro",
            )
            total_mols += len(list(enz_network.mols))
            total_rxns += len(list(getattr(enz_network, 'rxns', [])))

        if config.use_synthetic:
            syn_network = synthetic.generate_network(
                job_name="benchmark_synthetic",
                starters={target_smiles},
                gen=config.generations_per_expand,
                direction="retro",
                helpers=chemistry_helpers if chemistry_helpers else None,
            )
            total_mols += len(list(syn_network.mols))
            total_rxns += len(list(getattr(syn_network, 'rxns', [])))

        runtime = time.perf_counter() - start_time

        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="standalone",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            runtime_seconds=runtime,
            num_molecules_in_network=total_mols,
            num_reactions_in_network=total_rxns,
        )

    except Exception as e:
        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="standalone",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            runtime_seconds=0.0,
            error=str(e),
        )


def run_doranet_mcts_sequential(
    target_smiles: str,
    molecule_name: str,
    config: RuntimeBenchmarkConfig,
) -> RuntimeBenchmarkResult:
    """
    Run DORAnetMCTS (single-threaded sequential MCTS).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="sequential",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            runtime_seconds=0.0,
            error=f"Could not parse SMILES: {target_smiles}",
        )

    # Clear caches for fair comparison
    clear_smiles_cache()
    Node.node_counter = 0

    root = Node(fragment=mol, parent=None, depth=0, provenance="target")

    try:
        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=config.total_iterations,
            max_depth=config.max_depth,
            max_children_per_expand=config.max_children_per_expand,
            child_downselection_strategy=config.child_downselection_strategy,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            generations_per_expand=config.generations_per_expand,
            cofactors_files=config.cofactors_files,
            pks_library_file=config.pks_library_file,
            sink_compounds_files=config.sink_compounds_files,
            prohibited_chemicals_file=config.prohibited_chemicals_file,
            spawn_retrotide=False,  # Disable for faster benchmarks
            enable_visualization=False,
            enable_interactive_viz=False,
            stop_on_first_pathway=True,  # Stop when first pathway found
        )

        start_time = time.perf_counter()
        agent.run()
        runtime = time.perf_counter() - start_time

        # Collect metrics
        sink_compounds = agent.get_sink_compounds()
        pks_matches = agent.get_pks_matches()
        terminal_nodes = len(sink_compounds) + len(pks_matches)

        # Count unique SMILES across all nodes
        unique_smiles_set = set()
        for node in agent.nodes:
            if node.fragment is not None:
                smiles = Chem.MolToSmiles(node.fragment, canonical=True)
                unique_smiles_set.add(smiles)

        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="sequential",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            runtime_seconds=runtime,
            total_nodes=len(agent.nodes),
            unique_smiles=len(unique_smiles_set),
            iterations_completed=config.total_iterations,
            terminal_nodes=terminal_nodes,
            sink_compounds=len(sink_compounds),
            pks_matches=len(pks_matches),
        )

    except Exception as e:
        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="sequential",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            runtime_seconds=0.0,
            error=str(e),
        )


def run_doranet_mcts_async(
    target_smiles: str,
    molecule_name: str,
    config: RuntimeBenchmarkConfig,
) -> RuntimeBenchmarkResult:
    """
    Run AsyncExpansionDORAnetMCTS (multiprocessing MCTS).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="async",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            runtime_seconds=0.0,
            error=f"Could not parse SMILES: {target_smiles}",
        )

    # Clear caches for fair comparison
    clear_smiles_cache()
    Node.node_counter = 0

    root = Node(fragment=mol, parent=None, depth=0, provenance="target")

    try:
        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=config.total_iterations,
            max_depth=config.max_depth,
            max_children_per_expand=config.max_children_per_expand,
            child_downselection_strategy=config.child_downselection_strategy,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            generations_per_expand=config.generations_per_expand,
            cofactors_files=config.cofactors_files,
            pks_library_file=config.pks_library_file,
            sink_compounds_files=config.sink_compounds_files,
            prohibited_chemicals_file=config.prohibited_chemicals_file,
            num_workers=config.num_workers,
            spawn_retrotide=False,  # Disable for faster benchmarks
            enable_visualization=False,
            enable_interactive_viz=False,
            stop_on_first_pathway=True,  # Stop when first pathway found
        )

        start_time = time.perf_counter()
        agent.run()
        runtime = time.perf_counter() - start_time

        # Collect metrics
        sink_compounds = agent.get_sink_compounds()
        pks_matches = agent.get_pks_matches()
        terminal_nodes = len(sink_compounds) + len(pks_matches)

        # Count unique SMILES across all nodes
        unique_smiles_set = set()
        for node in agent.nodes:
            if node.fragment is not None:
                smiles = Chem.MolToSmiles(node.fragment, canonical=True)
                unique_smiles_set.add(smiles)

        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="async",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            runtime_seconds=runtime,
            total_nodes=len(agent.nodes),
            unique_smiles=len(unique_smiles_set),
            iterations_completed=config.total_iterations,
            terminal_nodes=terminal_nodes,
            sink_compounds=len(sink_compounds),
            pks_matches=len(pks_matches),
            num_workers=agent.num_workers,
        )

    except Exception as e:
        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="async",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            runtime_seconds=0.0,
            error=str(e),
        )


# =============================================================================
# Output Function
# =============================================================================

def save_results_txt(
    result: RuntimeBenchmarkResult,
    config: RuntimeBenchmarkConfig,
    output_dir: Path,
) -> Path:
    """
    Save benchmark results to a human-readable .txt file.

    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"benchmark_runtime_{result.mode}_{result.molecule_name}_{result.timestamp}.txt"
    filepath = output_dir / filename

    lines = [
        "=" * 70,
        "RUNTIME BENCHMARK RESULTS",
        "=" * 70,
        "",
        "RUN INFORMATION",
        "-" * 70,
        f"Mode:                      {result.mode}",
        f"Molecule:                  {result.molecule_name}",
        f"SMILES:                    {result.molecule_smiles}",
        f"Timestamp:                 {result.timestamp}",
        f"Expansion modes:           enzymatic={result.use_enzymatic}, synthetic={result.use_synthetic}",
        "",
        "CONFIGURATION",
        "-" * 70,
    ]

    if result.mode == "standalone":
        lines.append(f"Generations per expand:    {config.generations_per_expand}")
    else:
        lines.append(f"Total iterations:          {config.total_iterations}")
        lines.append(f"Max depth:                 {config.max_depth}")
        max_children_str = str(config.max_children_per_expand) if config.max_children_per_expand else "None (unlimited)"
        lines.append(f"Max children per expand:   {max_children_str}")
        downselection_str = config.child_downselection_strategy if config.child_downselection_strategy else "None (no filtering)"
        lines.append(f"Downselection strategy:    {downselection_str}")

    if result.mode == "async":
        lines.append(f"Num workers:               {result.num_workers}")

    lines.extend([
        "",
        "RUNTIME METRICS",
        "-" * 70,
        f"Total runtime:             {result.runtime_seconds:.4f} seconds",
    ])

    if result.mode == "standalone":
        lines.extend([
            "",
            "NETWORK METRICS",
            "-" * 70,
            f"Molecules in network:      {result.num_molecules_in_network}",
            f"Reactions in network:      {result.num_reactions_in_network}",
        ])
    else:
        lines.extend([
            "",
            "TREE METRICS (MCTS)",
            "-" * 70,
            f"Total nodes:               {result.total_nodes}",
            f"Unique SMILES:             {result.unique_smiles}",
            f"Terminal nodes:            {result.terminal_nodes}",
            f"  - Sink compounds:        {result.sink_compounds}",
            f"  - PKS matches:           {result.pks_matches}",
            f"Iterations completed:      {result.iterations_completed}",
        ])

    lines.extend([
        "",
        "STATUS",
        "-" * 70,
        f"Error:                     {result.error if result.error else 'None'}",
        "",
        "=" * 70,
    ])

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # ========== BENCHMARK MODE SELECTION (enable exactly ONE) ==========
    RUN_DORANET_STANDALONE = False      # Raw DORAnet library expansion
    RUN_DORANET_MCTS_SEQUENTIAL = True  # DORAnetMCTS (single-threaded)
    RUN_DORANET_MCTS_ASYNC = False      # AsyncExpansionDORAnetMCTS

    # ========== EXPANSION MODE TOGGLES ==========
    use_enzymatic = True
    use_synthetic = True

    # ========== TARGET MOLECULE ==========
    target_smiles = "COC1=CC(OC(C=CC2=CC=C(OC)C=C2)C1)=O"
    molecule_name = "5_6_dihydroyangonin"

    # ========== MCTS PARAMETERS ==========
    total_iterations = 50
    max_depth = 3
    max_children_per_expand = None  # None = no limit (keep all fragments)
    child_downselection_strategy = None  # None = no filtering

    # ========== STANDALONE PARAMETERS ==========
    generations_per_expand = 1

    # ========== ASYNC PARAMETERS ==========
    num_workers = None  # None = auto-detect CPU count

    # ========== VALIDATION ==========
    mode_flags = [RUN_DORANET_STANDALONE, RUN_DORANET_MCTS_SEQUENTIAL, RUN_DORANET_MCTS_ASYNC]
    enabled_count = sum(mode_flags)
    if enabled_count != 1:
        print("ERROR: Exactly ONE benchmark mode must be enabled.")
        print(f"  RUN_DORANET_STANDALONE:      {RUN_DORANET_STANDALONE}")
        print(f"  RUN_DORANET_MCTS_SEQUENTIAL: {RUN_DORANET_MCTS_SEQUENTIAL}")
        print(f"  RUN_DORANET_MCTS_ASYNC:      {RUN_DORANET_MCTS_ASYNC}")
        sys.exit(1)

    # Determine mode string
    if RUN_DORANET_STANDALONE:
        mode = "standalone"
    elif RUN_DORANET_MCTS_SEQUENTIAL:
        mode = "sequential"
    else:
        mode = "async"

    # Build configuration
    config = RuntimeBenchmarkConfig(
        mode=mode,
        use_enzymatic=use_enzymatic,
        use_synthetic=use_synthetic,
        total_iterations=total_iterations,
        max_depth=max_depth,
        max_children_per_expand=max_children_per_expand,
        child_downselection_strategy=child_downselection_strategy,
        generations_per_expand=generations_per_expand,
        num_workers=num_workers,
    )

    # Print banner
    print("=" * 70)
    print("RUNTIME BENCHMARK")
    print("=" * 70)
    print(f"Mode:            {mode}")
    print(f"Molecule:        {molecule_name}")
    print(f"SMILES:          {target_smiles}")
    print(f"Expansion:       enzymatic={use_enzymatic}, synthetic={use_synthetic}")
    if mode != "standalone":
        print(f"Iterations:      {total_iterations}")
        print(f"Max depth:       {max_depth}")
    print("=" * 70)
    print()

    # Run appropriate benchmark
    if RUN_DORANET_STANDALONE:
        print(f"Running DORAnet standalone expansion...")
        result = run_doranet_standalone(target_smiles, molecule_name, config)
    elif RUN_DORANET_MCTS_SEQUENTIAL:
        print(f"Running DORAnetMCTS (sequential)...")
        result = run_doranet_mcts_sequential(target_smiles, molecule_name, config)
    else:
        print(f"Running AsyncExpansionDORAnetMCTS...")
        result = run_doranet_mcts_async(target_smiles, molecule_name, config)

    # Save results
    output_dir = REPO_ROOT / "results" / "benchmarks"
    filepath = save_results_txt(result, config, output_dir)

    # Print summary
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Runtime:         {result.runtime_seconds:.4f} seconds")
    if result.mode == "standalone":
        print(f"Molecules:       {result.num_molecules_in_network}")
        print(f"Reactions:       {result.num_reactions_in_network}")
    else:
        print(f"Total nodes:     {result.total_nodes}")
        print(f"Sink compounds:  {result.sink_compounds}")
        print(f"PKS matches:     {result.pks_matches}")
    if result.error:
        print(f"Error:           {result.error}")
    print(f"Results saved:   {filepath}")
    print("=" * 70)
