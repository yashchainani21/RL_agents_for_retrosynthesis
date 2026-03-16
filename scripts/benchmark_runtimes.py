"""
Modular runtime benchmarking script for DORAnet synthesis programs.

Supports benchmarking in several modes:
1. DORAnet Standalone - Raw library expansion (no MCTS)
2. DORAnetMCTS Sequential - Single-threaded MCTS
3. AsyncExpansionDORAnetMCTS - Multiprocessing MCTS
4. MCTS vs BFS Comparison - Head-to-head wall-clock comparison of MCTS and
   exhaustive breadth-first search on the same molecule/configuration.

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
from DORAnet_agent.policies import NoOpTerminalDetector

# Silence RDKit logs
RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Configuration and Result Dataclasses
# =============================================================================

@dataclass
class RuntimeBenchmarkConfig:
    """Configuration container for runtime benchmarks."""
    # Benchmark mode: "standalone", "sequential", "async", or "mcts_vs_bfs"
    mode: str

    # Expansion mode toggles
    use_enzymatic: bool = True
    use_synthetic: bool = True

    # MCTS parameters
    total_iterations: int = 50
    max_depth: int = 3
    max_children_per_expand: Optional[int] = None  # None = no limit
    child_downselection_strategy: Optional[str] = None  # None = no filtering

    # Search strategy: "mcts" (default) or "bfs" (exhaustive breadth-first)
    # In BFS mode, total_iterations controls the number of depth levels to expand.
    search_strategy: str = "mcts"

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

    # Search strategy used ("mcts" or "bfs")
    search_strategy: str = "mcts"

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

    # Pathway metrics (populated when stop_on_first_pathway=True)
    pathway_found: Optional[bool] = None
    pathway_time_seconds: Optional[float] = None
    pathway_iteration: Optional[int] = None
    pathway_terminal_smiles: Optional[str] = None
    pathway_terminal_depth: Optional[int] = None

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
            search_strategy=config.search_strategy,
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
            terminal_detector=NoOpTerminalDetector(),  # Disable RetroTide for faster benchmarks
            enable_visualization=False,
            enable_interactive_viz=False,
            stop_on_first_pathway=True,  # Stop when first pathway found
            search_strategy=config.search_strategy,
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

        # Extract pathway metrics
        pathway_found = agent.first_pathway_found
        pathway_time_seconds = agent.first_pathway_time
        pathway_iteration = agent.first_pathway_iteration
        pathway_terminal_smiles = None
        pathway_terminal_depth = None
        if agent.first_pathway_node is not None:
            if agent.first_pathway_node.fragment is not None:
                pathway_terminal_smiles = Chem.MolToSmiles(
                    agent.first_pathway_node.fragment, canonical=True
                )
            pathway_terminal_depth = agent.first_pathway_node.depth

        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="sequential",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            search_strategy=config.search_strategy,
            runtime_seconds=runtime,
            total_nodes=len(agent.nodes),
            unique_smiles=len(unique_smiles_set),
            iterations_completed=agent.current_iteration + 1,
            terminal_nodes=terminal_nodes,
            sink_compounds=len(sink_compounds),
            pks_matches=len(pks_matches),
            pathway_found=pathway_found,
            pathway_time_seconds=pathway_time_seconds,
            pathway_iteration=pathway_iteration,
            pathway_terminal_smiles=pathway_terminal_smiles,
            pathway_terminal_depth=pathway_terminal_depth,
        )

    except Exception as e:
        return RuntimeBenchmarkResult(
            molecule_name=molecule_name,
            molecule_smiles=target_smiles,
            mode="sequential",
            timestamp=timestamp,
            use_enzymatic=config.use_enzymatic,
            use_synthetic=config.use_synthetic,
            search_strategy=config.search_strategy,
            runtime_seconds=0.0,
            pathway_found=False,
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
            terminal_detector=NoOpTerminalDetector(),  # Disable RetroTide for faster benchmarks
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
        lines.append(f"Search strategy:           {result.search_strategy}")
        if result.search_strategy == "bfs":
            lines.append(f"  (In BFS mode, iterations = depth levels expanded)")

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
        strategy_label = result.search_strategy.upper()
        lines.extend([
            "",
            f"TREE METRICS ({strategy_label})",
            "-" * 70,
            f"Total nodes:               {result.total_nodes}",
            f"Unique SMILES:             {result.unique_smiles}",
            f"Terminal nodes:            {result.terminal_nodes}",
            f"  - Sink compounds:        {result.sink_compounds}",
            f"  - PKS matches:           {result.pks_matches}",
            f"Iterations completed:      {result.iterations_completed}",
        ])

    # Pathway metrics (when available)
    if result.pathway_found is not None:
        lines.extend([
            "",
            "PATHWAY METRICS",
            "-" * 70,
            f"Pathway found:             {'Yes' if result.pathway_found else 'No'}",
        ])
        if result.pathway_found:
            pt = f"{result.pathway_time_seconds:.4f}" if result.pathway_time_seconds is not None else "N/A"
            lines.extend([
                f"Time to first pathway (s): {pt}",
                f"Iteration/level:           {result.pathway_iteration}",
                f"Terminal SMILES:            {result.pathway_terminal_smiles or 'N/A'}",
                f"Terminal depth:             {result.pathway_terminal_depth}",
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
# MCTS vs BFS Comparison
# =============================================================================

def run_mcts_vs_bfs_comparison(
    target_smiles: str,
    molecule_name: str,
    config: RuntimeBenchmarkConfig,
) -> tuple:
    """
    Run both MCTS and BFS on the same molecule/configuration and return both results.

    The comparison uses the sequential DORAnetMCTS runner for both strategies,
    differing only in search_strategy ("mcts" vs "bfs").

    For a fair comparison:
    - Both runs use the same max_depth, expansion settings, and building block databases.
    - MCTS uses total_iterations as the number of MCTS iterations.
    - BFS uses total_iterations = max_depth so that it expands ALL depth levels up to
      max_depth (exhaustive breadth-first search to the same depth ceiling).
    - Caches are cleared between runs for fairness.

    Returns:
        Tuple of (mcts_result, bfs_result) RuntimeBenchmarkResult objects.
    """
    import copy

    # --- Run MCTS ---
    print("\n" + "=" * 70)
    print("PHASE 1: Running MCTS")
    print("=" * 70)
    mcts_config = copy.copy(config)
    mcts_config.search_strategy = "mcts"
    mcts_result = run_doranet_mcts_sequential(target_smiles, molecule_name, mcts_config)

    # --- Run BFS ---
    print("\n" + "=" * 70)
    print("PHASE 2: Running BFS")
    print("=" * 70)
    bfs_config = copy.copy(config)
    bfs_config.search_strategy = "bfs"
    # BFS total_iterations = max_depth so it expands every level up to the depth ceiling
    bfs_config.total_iterations = config.max_depth
    bfs_result = run_doranet_mcts_sequential(target_smiles, molecule_name, bfs_config)

    return mcts_result, bfs_result


def save_comparison_report(
    mcts_result: RuntimeBenchmarkResult,
    bfs_result: RuntimeBenchmarkResult,
    config: RuntimeBenchmarkConfig,
    output_dir: Path,
) -> Path:
    """
    Save a head-to-head MCTS vs BFS comparison report.

    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_mcts_vs_bfs_{mcts_result.molecule_name}_{timestamp}.txt"
    filepath = output_dir / filename

    # Compute comparison metrics
    if mcts_result.runtime_seconds > 0 and bfs_result.runtime_seconds > 0:
        speedup = bfs_result.runtime_seconds / mcts_result.runtime_seconds
        speedup_str = f"{speedup:.2f}x"
        if speedup > 1:
            faster_strategy = "MCTS"
        elif speedup < 1:
            faster_strategy = "BFS"
        else:
            faster_strategy = "TIE"
    else:
        speedup_str = "N/A"
        faster_strategy = "N/A"

    mcts_nodes = mcts_result.total_nodes or 0
    bfs_nodes = bfs_result.total_nodes or 0
    if bfs_nodes > 0:
        node_ratio = f"{bfs_nodes / max(mcts_nodes, 1):.2f}x"
    else:
        node_ratio = "N/A"

    lines = [
        "=" * 70,
        "MCTS vs BFS COMPARISON BENCHMARK",
        "=" * 70,
        "",
        "EXPERIMENT INFORMATION",
        "-" * 70,
        f"Molecule:                  {mcts_result.molecule_name}",
        f"SMILES:                    {mcts_result.molecule_smiles}",
        f"Timestamp:                 {timestamp}",
        f"Expansion modes:           enzymatic={config.use_enzymatic}, synthetic={config.use_synthetic}",
        "",
        "SHARED CONFIGURATION",
        "-" * 70,
        f"Max depth:                 {config.max_depth}",
        f"Max children per expand:   {config.max_children_per_expand or 'None (unlimited)'}",
        f"Downselection strategy:    {config.child_downselection_strategy or 'None (no filtering)'}",
        f"MCTS total_iterations:     {config.total_iterations}",
        f"BFS total_iterations:      {config.max_depth} (= max_depth, expands all levels)",
        "",
        "=" * 70,
        "HEAD-TO-HEAD RESULTS",
        "=" * 70,
        "",
        f"{'Metric':<30s} {'MCTS':>15s} {'BFS':>15s}",
        "-" * 62,
        f"{'Runtime (seconds)':<30s} {mcts_result.runtime_seconds:>15.4f} {bfs_result.runtime_seconds:>15.4f}",
        f"{'Total nodes':<30s} {mcts_nodes:>15,d} {bfs_nodes:>15,d}",
        f"{'Unique SMILES':<30s} {(mcts_result.unique_smiles or 0):>15,d} {(bfs_result.unique_smiles or 0):>15,d}",
        f"{'Terminal nodes':<30s} {(mcts_result.terminal_nodes or 0):>15,d} {(bfs_result.terminal_nodes or 0):>15,d}",
        f"{'  Sink compounds':<30s} {(mcts_result.sink_compounds or 0):>15,d} {(bfs_result.sink_compounds or 0):>15,d}",
        f"{'  PKS matches':<30s} {(mcts_result.pks_matches or 0):>15,d} {(bfs_result.pks_matches or 0):>15,d}",
        "",
        "COMPARISON SUMMARY",
        "-" * 70,
        f"Faster strategy:           {faster_strategy}",
        f"BFS/MCTS runtime ratio:    {speedup_str}",
        f"BFS/MCTS node ratio:       {node_ratio}",
        "",
    ]

    # Add interpretation
    lines.extend([
        "INTERPRETATION",
        "-" * 70,
    ])
    if faster_strategy == "MCTS":
        lines.append(
            f"MCTS was {speedup_str} faster than exhaustive BFS while exploring"
        )
        lines.append(
            f"  {mcts_nodes:,d} nodes (vs {bfs_nodes:,d} for BFS)."
        )
        lines.append(
            "  This demonstrates MCTS's ability to find solutions efficiently"
        )
        lines.append(
            "  through intelligent selection rather than exhaustive enumeration."
        )
    elif faster_strategy == "BFS":
        lines.append(
            f"BFS was faster in this configuration. This may occur when the search"
        )
        lines.append(
            f"  tree is small enough that exhaustive expansion is cheaper than"
        )
        lines.append(
            f"  MCTS overhead (selection scoring, backpropagation)."
        )
    else:
        lines.append("Both strategies had comparable runtime.")

    # Error summary
    mcts_err = mcts_result.error or "None"
    bfs_err = bfs_result.error or "None"
    lines.extend([
        "",
        "STATUS",
        "-" * 70,
        f"MCTS error:                {mcts_err}",
        f"BFS error:                 {bfs_err}",
        "",
        "=" * 70,
    ])

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


# =============================================================================
# Fair MCTS vs BFS Comparison (pathway-focused)
# =============================================================================

def run_fair_mcts_vs_bfs_comparison(
    target_smiles: str,
    molecule_name: str,
    config: RuntimeBenchmarkConfig,
) -> tuple:
    """
    Run both BFS and MCTS on the same molecule with production-realistic parameters.

    The primary metric is wall-clock time to first complete pathway.

    BFS runs first so that the DORAnet disk cache (.cache/) is warm for MCTS,
    making the comparison conservative for MCTS (it gets the cache advantage).

    Both strategies use stop_on_first_pathway=True and identical expansion settings.
    - BFS: total_iterations = max_depth (expands all levels exhaustively)
    - MCTS: total_iterations from config (e.g. 1000)

    Returns:
        Tuple of (mcts_result, bfs_result) RuntimeBenchmarkResult objects.
    """
    import copy

    # --- Run BFS first (so MCTS benefits from warmed disk cache) ---
    print("\n" + "=" * 70)
    print("PHASE 1: Running BFS (exhaustive breadth-first search)")
    print("=" * 70)
    bfs_config = copy.copy(config)
    bfs_config.search_strategy = "bfs"
    bfs_config.total_iterations = config.max_depth
    print(f"  Search strategy:           bfs")
    print(f"  Depth levels to expand:    {bfs_config.total_iterations}")
    print(f"  Max children per expand:   {config.max_children_per_expand}")
    print(f"  Downselection strategy:    {config.child_downselection_strategy}")
    print(f"  stop_on_first_pathway:     True")
    bfs_result = run_doranet_mcts_sequential(target_smiles, molecule_name, bfs_config)
    print(f"  BFS completed in {bfs_result.runtime_seconds:.2f}s "
          f"({bfs_result.total_nodes or 0} nodes, "
          f"pathway_found={bfs_result.pathway_found})")

    # --- Run MCTS second (benefits from cached expansions) ---
    print("\n" + "=" * 70)
    print("PHASE 2: Running MCTS (Monte Carlo Tree Search)")
    print("=" * 70)
    mcts_config = copy.copy(config)
    mcts_config.search_strategy = "mcts"
    print(f"  Search strategy:           mcts")
    print(f"  Total iterations:          {mcts_config.total_iterations}")
    print(f"  Max children per expand:   {config.max_children_per_expand}")
    print(f"  Downselection strategy:    {config.child_downselection_strategy}")
    print(f"  stop_on_first_pathway:     True")
    mcts_result = run_doranet_mcts_sequential(target_smiles, molecule_name, mcts_config)
    print(f"  MCTS completed in {mcts_result.runtime_seconds:.2f}s "
          f"({mcts_result.total_nodes or 0} nodes, "
          f"pathway_found={mcts_result.pathway_found})")

    return mcts_result, bfs_result


def save_fair_comparison_report(
    mcts_result: RuntimeBenchmarkResult,
    bfs_result: RuntimeBenchmarkResult,
    config: RuntimeBenchmarkConfig,
    output_dir: Path,
) -> Path:
    """
    Save a pathway-focused MCTS vs BFS comparison report.

    The primary metric is time to first complete pathway.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_fair_mcts_vs_bfs_{mcts_result.molecule_name}_{timestamp}.txt"
    filepath = output_dir / filename

    lines = [
        "=" * 70,
        "FAIR MCTS vs BFS COMPARISON — TIME TO FIRST PATHWAY",
        "=" * 70,
        "",
        "EXPERIMENT INFORMATION",
        "-" * 70,
        f"Molecule:                  {mcts_result.molecule_name}",
        f"SMILES:                    {mcts_result.molecule_smiles}",
        f"Timestamp:                 {timestamp}",
        f"Expansion modes:           enzymatic={config.use_enzymatic}, synthetic={config.use_synthetic}",
        "",
        "SHARED CONFIGURATION",
        "-" * 70,
        f"Max depth:                 {config.max_depth}",
        f"Max children per expand:   {config.max_children_per_expand or 'None (unlimited)'}",
        f"Downselection strategy:    {config.child_downselection_strategy or 'None (no filtering)'}",
        f"stop_on_first_pathway:     True",
        f"MCTS total_iterations:     {config.total_iterations}",
        f"BFS total_iterations:      {config.max_depth} (= max_depth, expands all levels)",
        "",
        "CACHE NOTE: BFS ran first; MCTS benefits from warmed disk cache.",
        "  This makes the comparison conservative for MCTS.",
        "",
    ]

    # --- Primary metric: pathway ---
    lines.extend([
        "=" * 70,
        "PRIMARY METRIC: TIME TO FIRST PATHWAY",
        "=" * 70,
        "",
        f"{'Metric':<30s} {'MCTS':>15s} {'BFS':>15s}",
        "-" * 62,
    ])

    def _fmt_bool(v):
        return "Yes" if v else "No"

    def _fmt_opt(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None else "N/A"

    def _fmt_str(v):
        return v if v is not None else "N/A"

    def _fmt_int(v):
        return str(v) if v is not None else "N/A"

    mcts_pf = _fmt_bool(mcts_result.pathway_found) if mcts_result.pathway_found is not None else "N/A"
    bfs_pf = _fmt_bool(bfs_result.pathway_found) if bfs_result.pathway_found is not None else "N/A"

    lines.append(f"{'Pathway found?':<30s} {mcts_pf:>15s} {bfs_pf:>15s}")
    lines.append(f"{'Time to first pathway (s)':<30s} {_fmt_opt(mcts_result.pathway_time_seconds):>15s} {_fmt_opt(bfs_result.pathway_time_seconds):>15s}")
    lines.append(f"{'Iteration/level at pathway':<30s} {_fmt_int(mcts_result.pathway_iteration):>15s} {_fmt_int(bfs_result.pathway_iteration):>15s}")
    lines.append(f"{'Terminal node SMILES':<30s} {_fmt_str(mcts_result.pathway_terminal_smiles):>15s} {_fmt_str(bfs_result.pathway_terminal_smiles):>15s}")
    lines.append(f"{'Terminal node depth':<30s} {_fmt_int(mcts_result.pathway_terminal_depth):>15s} {_fmt_int(bfs_result.pathway_terminal_depth):>15s}")

    # --- Secondary metrics ---
    mcts_nodes = mcts_result.total_nodes or 0
    bfs_nodes = bfs_result.total_nodes or 0

    lines.extend([
        "",
        "=" * 70,
        "SECONDARY METRICS (at time of stopping)",
        "=" * 70,
        "",
        f"{'Metric':<30s} {'MCTS':>15s} {'BFS':>15s}",
        "-" * 62,
        f"{'Total runtime (s)':<30s} {mcts_result.runtime_seconds:>15.4f} {bfs_result.runtime_seconds:>15.4f}",
        f"{'Total nodes':<30s} {mcts_nodes:>15,d} {bfs_nodes:>15,d}",
        f"{'Unique SMILES':<30s} {(mcts_result.unique_smiles or 0):>15,d} {(bfs_result.unique_smiles or 0):>15,d}",
        f"{'Iterations completed':<30s} {(mcts_result.iterations_completed or 0):>15,d} {(bfs_result.iterations_completed or 0):>15,d}",
        f"{'Terminal nodes':<30s} {(mcts_result.terminal_nodes or 0):>15,d} {(bfs_result.terminal_nodes or 0):>15,d}",
        f"{'  Sink compounds':<30s} {(mcts_result.sink_compounds or 0):>15,d} {(bfs_result.sink_compounds or 0):>15,d}",
        f"{'  PKS matches':<30s} {(mcts_result.pks_matches or 0):>15,d} {(bfs_result.pks_matches or 0):>15,d}",
    ])

    # --- Efficiency ---
    lines.extend([
        "",
        "=" * 70,
        "EFFICIENCY",
        "=" * 70,
        "",
    ])
    if bfs_nodes > 0 and mcts_nodes > 0:
        node_ratio = bfs_nodes / mcts_nodes
        lines.append(f"BFS/MCTS node ratio:       {node_ratio:.2f}x (BFS explored {node_ratio:.1f}x as many nodes)")
    else:
        lines.append("BFS/MCTS node ratio:       N/A")

    # --- Winner determination ---
    lines.extend([
        "",
        "=" * 70,
        "WINNER",
        "=" * 70,
        "",
    ])

    mcts_found = mcts_result.pathway_found or False
    bfs_found = bfs_result.pathway_found or False

    if mcts_found and bfs_found:
        # Both found — compare times
        mcts_t = mcts_result.pathway_time_seconds
        bfs_t = bfs_result.pathway_time_seconds
        if mcts_t is not None and bfs_t is not None:
            if mcts_t < bfs_t:
                speedup = bfs_t / mcts_t
                winner = f"MCTS — {speedup:.2f}x faster to first pathway ({mcts_t:.2f}s vs {bfs_t:.2f}s)"
            elif bfs_t < mcts_t:
                speedup = mcts_t / bfs_t
                winner = f"BFS — {speedup:.2f}x faster to first pathway ({bfs_t:.2f}s vs {mcts_t:.2f}s)"
            else:
                winner = "TIE — both found a pathway at the same time"
        else:
            winner = "Both found a pathway (times unavailable for comparison)"
    elif mcts_found and not bfs_found:
        winner = "MCTS — found a pathway; BFS did not"
    elif bfs_found and not mcts_found:
        winner = "BFS — found a pathway; MCTS did not"
    else:
        winner = "NEITHER — no pathway found by either strategy"

    lines.append(f"Result: {winner}")

    # Error summary
    mcts_err = mcts_result.error or "None"
    bfs_err = bfs_result.error or "None"
    lines.extend([
        "",
        "STATUS",
        "-" * 70,
        f"MCTS error:                {mcts_err}",
        f"BFS error:                 {bfs_err}",
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
    RUN_DORANET_MCTS_SEQUENTIAL = False # DORAnetMCTS (single-threaded)
    RUN_DORANET_MCTS_ASYNC = False      # AsyncExpansionDORAnetMCTS
    RUN_MCTS_VS_BFS_COMPARISON = False  # Head-to-head MCTS vs BFS comparison (old, unfair)
    RUN_FAIR_MCTS_VS_BFS = True         # Fair pathway-focused MCTS vs BFS comparison

    # ========== EXPANSION MODE TOGGLES ==========
    use_enzymatic = True
    use_synthetic = True

    # ========== TARGET MOLECULE ==========
    target_smiles = "COC1=CC(OC(CCC2=CC=C(OC)C=C2)C1)=O"
    molecule_name = "5678_tetrahydroyangonin"

    # ========== MCTS PARAMETERS ==========
    # For MCTS vs BFS comparison:
    #   - MCTS uses total_iterations as the number of MCTS iterations
    #   - BFS automatically uses total_iterations = max_depth (expands all depth levels)
    total_iterations = 1000
    max_depth = 4
    max_children_per_expand = 30
    child_downselection_strategy = "most_thermo_feasible"

    # ========== STANDALONE PARAMETERS ==========
    generations_per_expand = 1

    # ========== ASYNC PARAMETERS ==========
    num_workers = None  # None = auto-detect CPU count

    # ========== VALIDATION ==========
    mode_flags = [
        RUN_DORANET_STANDALONE,
        RUN_DORANET_MCTS_SEQUENTIAL,
        RUN_DORANET_MCTS_ASYNC,
        RUN_MCTS_VS_BFS_COMPARISON,
        RUN_FAIR_MCTS_VS_BFS,
    ]
    enabled_count = sum(mode_flags)
    if enabled_count != 1:
        print("ERROR: Exactly ONE benchmark mode must be enabled.")
        print(f"  RUN_DORANET_STANDALONE:      {RUN_DORANET_STANDALONE}")
        print(f"  RUN_DORANET_MCTS_SEQUENTIAL: {RUN_DORANET_MCTS_SEQUENTIAL}")
        print(f"  RUN_DORANET_MCTS_ASYNC:      {RUN_DORANET_MCTS_ASYNC}")
        print(f"  RUN_MCTS_VS_BFS_COMPARISON:  {RUN_MCTS_VS_BFS_COMPARISON}")
        print(f"  RUN_FAIR_MCTS_VS_BFS:        {RUN_FAIR_MCTS_VS_BFS}")
        sys.exit(1)

    # Determine mode string
    if RUN_DORANET_STANDALONE:
        mode = "standalone"
    elif RUN_DORANET_MCTS_SEQUENTIAL:
        mode = "sequential"
    elif RUN_DORANET_MCTS_ASYNC:
        mode = "async"
    elif RUN_MCTS_VS_BFS_COMPARISON:
        mode = "mcts_vs_bfs"
    else:
        mode = "fair_mcts_vs_bfs"

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
    if mode in ("mcts_vs_bfs", "fair_mcts_vs_bfs"):
        print(f"MCTS iterations: {total_iterations}")
        print(f"BFS iterations:  {max_depth} (= max_depth, exhaustive to depth ceiling)")
        print(f"Max depth:       {max_depth}")
        print(f"Max children:    {max_children_per_expand}")
        print(f"Downselection:   {child_downselection_strategy}")
    elif mode != "standalone":
        print(f"Iterations:      {total_iterations}")
        print(f"Max depth:       {max_depth}")
    print("=" * 70)
    print()

    output_dir = REPO_ROOT / "results" / "benchmarks"

    # ========== Fair MCTS vs BFS Comparison Mode ==========
    if RUN_FAIR_MCTS_VS_BFS:
        print("Running fair MCTS vs BFS comparison (pathway-focused)...")
        mcts_result, bfs_result = run_fair_mcts_vs_bfs_comparison(
            target_smiles, molecule_name, config,
        )

        # Save individual results
        import copy
        mcts_config_for_save = copy.copy(config)
        mcts_config_for_save.search_strategy = "mcts"
        mcts_filepath = save_results_txt(mcts_result, mcts_config_for_save, output_dir)

        bfs_config_for_save = copy.copy(config)
        bfs_config_for_save.search_strategy = "bfs"
        bfs_config_for_save.total_iterations = config.max_depth
        bfs_filepath = save_results_txt(bfs_result, bfs_config_for_save, output_dir)

        # Save fair comparison report
        comparison_filepath = save_fair_comparison_report(
            mcts_result, bfs_result, config, output_dir,
        )

        # Print console summary
        print()
        print("=" * 70)
        print("FAIR MCTS vs BFS COMPARISON COMPLETE")
        print("=" * 70)
        print()
        print("PRIMARY METRIC: TIME TO FIRST PATHWAY")
        print("-" * 51)
        print(f"{'Metric':<25s} {'MCTS':>12s} {'BFS':>12s}")
        print("-" * 51)

        def _console_fmt(v, fmt=".4f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        mcts_pf = "Yes" if mcts_result.pathway_found else "No"
        bfs_pf = "Yes" if bfs_result.pathway_found else "No"
        print(f"{'Pathway found':<25s} {mcts_pf:>12s} {bfs_pf:>12s}")
        print(f"{'Pathway time (s)':<25s} {_console_fmt(mcts_result.pathway_time_seconds):>12s} {_console_fmt(bfs_result.pathway_time_seconds):>12s}")
        print(f"{'Pathway iteration':<25s} {_console_fmt(mcts_result.pathway_iteration, 'd'):>12s} {_console_fmt(bfs_result.pathway_iteration, 'd'):>12s}")
        print()
        print("SECONDARY METRICS")
        print("-" * 51)
        print(f"{'Runtime (s)':<25s} {mcts_result.runtime_seconds:>12.4f} {bfs_result.runtime_seconds:>12.4f}")
        print(f"{'Total nodes':<25s} {(mcts_result.total_nodes or 0):>12,d} {(bfs_result.total_nodes or 0):>12,d}")
        print(f"{'Unique SMILES':<25s} {(mcts_result.unique_smiles or 0):>12,d} {(bfs_result.unique_smiles or 0):>12,d}")
        print(f"{'Sink compounds':<25s} {(mcts_result.sink_compounds or 0):>12,d} {(bfs_result.sink_compounds or 0):>12,d}")
        print(f"{'PKS matches':<25s} {(mcts_result.pks_matches or 0):>12,d} {(bfs_result.pks_matches or 0):>12,d}")
        print("-" * 51)

        # Winner
        mcts_found = mcts_result.pathway_found or False
        bfs_found = bfs_result.pathway_found or False
        if mcts_found and bfs_found:
            mt = mcts_result.pathway_time_seconds
            bt = bfs_result.pathway_time_seconds
            if mt is not None and bt is not None:
                if mt < bt:
                    print(f"WINNER: MCTS — {bt/mt:.2f}x faster to first pathway")
                elif bt < mt:
                    print(f"WINNER: BFS — {mt/bt:.2f}x faster to first pathway")
                else:
                    print("WINNER: TIE")
            else:
                print("Both found a pathway (times unavailable)")
        elif mcts_found:
            print("WINNER: MCTS — found a pathway; BFS did not")
        elif bfs_found:
            print("WINNER: BFS — found a pathway; MCTS did not")
        else:
            print("NEITHER strategy found a pathway")

        if mcts_result.error:
            print(f"MCTS error:      {mcts_result.error}")
        if bfs_result.error:
            print(f"BFS error:       {bfs_result.error}")
        print()
        print(f"MCTS results:    {mcts_filepath}")
        print(f"BFS results:     {bfs_filepath}")
        print(f"Comparison:      {comparison_filepath}")
        print("=" * 70)

    # ========== MCTS vs BFS Comparison Mode (legacy) ==========
    elif RUN_MCTS_VS_BFS_COMPARISON:
        print("Running MCTS vs BFS head-to-head comparison...")
        mcts_result, bfs_result = run_mcts_vs_bfs_comparison(
            target_smiles, molecule_name, config,
        )

        # Save individual results
        mcts_filepath = save_results_txt(mcts_result, config, output_dir)
        # Use a modified config for BFS file to reflect its actual total_iterations
        import copy
        bfs_config_for_save = copy.copy(config)
        bfs_config_for_save.search_strategy = "bfs"
        bfs_config_for_save.total_iterations = config.max_depth
        bfs_filepath = save_results_txt(bfs_result, bfs_config_for_save, output_dir)

        # Save comparison report
        comparison_filepath = save_comparison_report(
            mcts_result, bfs_result, config, output_dir,
        )

        # Print summary
        print()
        print("=" * 70)
        print("MCTS vs BFS COMPARISON COMPLETE")
        print("=" * 70)
        print(f"{'Metric':<25s} {'MCTS':>12s} {'BFS':>12s}")
        print("-" * 51)
        print(f"{'Runtime (s)':<25s} {mcts_result.runtime_seconds:>12.4f} {bfs_result.runtime_seconds:>12.4f}")
        print(f"{'Total nodes':<25s} {(mcts_result.total_nodes or 0):>12,d} {(bfs_result.total_nodes or 0):>12,d}")
        print(f"{'Unique SMILES':<25s} {(mcts_result.unique_smiles or 0):>12,d} {(bfs_result.unique_smiles or 0):>12,d}")
        print(f"{'Sink compounds':<25s} {(mcts_result.sink_compounds or 0):>12,d} {(bfs_result.sink_compounds or 0):>12,d}")
        print(f"{'PKS matches':<25s} {(mcts_result.pks_matches or 0):>12,d} {(bfs_result.pks_matches or 0):>12,d}")
        print("-" * 51)
        if mcts_result.runtime_seconds > 0 and bfs_result.runtime_seconds > 0:
            ratio = bfs_result.runtime_seconds / mcts_result.runtime_seconds
            if ratio > 1:
                print(f"MCTS was {ratio:.2f}x faster than BFS")
            elif ratio < 1:
                print(f"BFS was {1/ratio:.2f}x faster than MCTS")
            else:
                print(f"Both strategies had comparable runtime")
        if mcts_result.error:
            print(f"MCTS error:      {mcts_result.error}")
        if bfs_result.error:
            print(f"BFS error:       {bfs_result.error}")
        print()
        print(f"MCTS results:    {mcts_filepath}")
        print(f"BFS results:     {bfs_filepath}")
        print(f"Comparison:      {comparison_filepath}")
        print("=" * 70)

    # ========== Single-Mode Benchmarks ==========
    else:
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
