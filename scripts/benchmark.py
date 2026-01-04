"""
Benchmark script comparing sequential vs parallel MCTS implementations.

This script runs both DORAnetMCTS and ParallelDORAnetMCTS against a test set
of molecules to measure runtime improvements and verify result consistency.

Usage:
    python scripts/benchmark.py                    # Run full benchmark
    python scripts/benchmark.py --quick            # Quick benchmark (fewer iterations)
    python scripts/benchmark.py --molecules 3      # Test first N molecules only
    python scripts/benchmark.py --workers 2 4 8    # Test specific worker counts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit import RDLogger

# Ensure the repository root is discoverable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent import DORAnetMCTS, ParallelDORAnetMCTS, Node, clear_smiles_cache

# Silence RDKit logs
RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Test Molecules
# =============================================================================

TEST_MOLECULES = [
    # Simple molecules (fast to process)
    {
        "name": "3-pentanone",
        "smiles": "CCCC(C)=O",
        "complexity": "simple",
    },
    {
        "name": "pentanoic_acid",
        "smiles": "CCCCC(=O)O",
        "complexity": "simple",
    },
    {
        "name": "4-hydroxybutyric_acid",
        "smiles": "OCCCC(=O)O",
        "complexity": "simple",
    },
    # Medium complexity molecules
    {
        "name": "nonanoic_acid",
        "smiles": "CCCCCCCCC(=O)O",
        "complexity": "medium",
    },
    {
        "name": "1,4-butanediol",
        "smiles": "OCCCCO",
        "complexity": "medium",
    },
    {
        "name": "kavain",
        "smiles": "COC1=CC(OC(/C=C/C2=CC=CC=C2)C1)=O",
        "complexity": "medium",
    },
    # Complex molecules (slower to process)
    {
        "name": "cryptofolione",
        "smiles": "C1C=CC(=O)OC1C=CCC(CC(C=CC2=CC=CC=C2)O)O",
        "complexity": "complex",
    },
    {
        "name": "triacetic_acid_lactone",
        "smiles": "CC(=O)CC1=CC(=O)OC=C1",
        "complexity": "medium",
    },
]


# =============================================================================
# Benchmark Result Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    molecule_name: str
    molecule_smiles: str
    mode: str  # "sequential" or "parallel"
    num_workers: int
    runtime_seconds: float
    total_nodes: int
    pks_matches: int
    sink_compounds: int
    iterations_completed: int
    iterations_failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary comparing sequential vs parallel runs."""
    molecule_name: str
    molecule_smiles: str
    sequential_runtime: float
    parallel_runtime: float
    num_workers: int
    speedup: float
    sequential_nodes: int
    parallel_nodes: int
    sequential_pks_matches: int
    parallel_pks_matches: int
    sequential_sinks: int
    parallel_sinks: int


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # MCTS parameters
    total_iterations: int = 20
    max_depth: int = 2
    max_children_per_expand: int = 5

    # Parallel parameters
    worker_counts: List[int] = field(default_factory=lambda: [4])
    virtual_loss: float = 1.0

    # Benchmark control
    warmup_runs: int = 0
    repetitions: int = 1

    # Data paths
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
            self.pks_library_file = str(REPO_ROOT / "data" / "processed" / "PKS_smiles.txt")
        if not self.sink_compounds_files:
            self.sink_compounds_files = [
                str(REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt"),
                str(REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt"),
            ]
        if not self.prohibited_chemicals_file:
            self.prohibited_chemicals_file = str(
                REPO_ROOT / "data" / "processed" / "prohibited_chemical_SMILES.txt"
            )


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Runs benchmarks comparing sequential and parallel MCTS."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.summaries: List[BenchmarkSummary] = []

    def _create_agent_kwargs(self) -> Dict:
        """Create common kwargs for both agent types."""
        return {
            "total_iterations": self.config.total_iterations,
            "max_depth": self.config.max_depth,
            "max_children_per_expand": self.config.max_children_per_expand,
            "use_enzymatic": True,
            "use_synthetic": True,
            "generations_per_expand": 1,
            "cofactors_files": self.config.cofactors_files,
            "pks_library_file": self.config.pks_library_file,
            "sink_compounds_files": self.config.sink_compounds_files,
            "prohibited_chemicals_file": self.config.prohibited_chemicals_file,
            "MW_multiple_to_exclude": 2.0,
            "spawn_retrotide": False,  # Disable RetroTide for faster benchmarks
            "sink_terminal_reward": 2.0,
            "selection_policy": "depth_biased",
            "depth_bonus_coefficient": 2.0,
            "enable_visualization": False,
            "enable_interactive_viz": False,
        }

    def run_sequential(
        self,
        molecule_name: str,
        molecule: Chem.Mol,
    ) -> BenchmarkResult:
        """Run sequential MCTS and return results."""
        # Clear cache for fair comparison
        clear_smiles_cache()

        # Reset node counter
        Node.node_counter = 0

        root = Node(fragment=molecule, parent=None, depth=0, provenance="target")
        agent_kwargs = self._create_agent_kwargs()

        try:
            agent = DORAnetMCTS(root=root, target_molecule=molecule, **agent_kwargs)

            start_time = time.perf_counter()
            agent.run()
            runtime = time.perf_counter() - start_time

            # Get cache stats
            from DORAnet_agent.mcts import _canonicalize_smiles
            cache_info = _canonicalize_smiles.cache_info()

            return BenchmarkResult(
                molecule_name=molecule_name,
                molecule_smiles=Chem.MolToSmiles(molecule),
                mode="sequential",
                num_workers=1,
                runtime_seconds=runtime,
                total_nodes=len(agent.nodes),
                pks_matches=len(agent.get_pks_matches()),
                sink_compounds=len(agent.get_sink_compounds()),
                iterations_completed=self.config.total_iterations,
                cache_hits=cache_info.hits,
                cache_misses=cache_info.misses,
            )

        except Exception as e:
            return BenchmarkResult(
                molecule_name=molecule_name,
                molecule_smiles=Chem.MolToSmiles(molecule),
                mode="sequential",
                num_workers=1,
                runtime_seconds=0,
                total_nodes=0,
                pks_matches=0,
                sink_compounds=0,
                iterations_completed=0,
                error=str(e),
            )

    def run_parallel(
        self,
        molecule_name: str,
        molecule: Chem.Mol,
        num_workers: int,
    ) -> BenchmarkResult:
        """Run parallel MCTS and return results."""
        # Clear cache for fair comparison
        clear_smiles_cache()

        # Reset node counter
        Node.node_counter = 0

        root = Node(fragment=molecule, parent=None, depth=0, provenance="target")
        agent_kwargs = self._create_agent_kwargs()

        try:
            agent = ParallelDORAnetMCTS(
                root=root,
                target_molecule=molecule,
                num_workers=num_workers,
                virtual_loss=self.config.virtual_loss,
                **agent_kwargs,
            )

            start_time = time.perf_counter()
            agent.run()
            runtime = time.perf_counter() - start_time

            # Get cache stats
            from DORAnet_agent.mcts import _canonicalize_smiles
            cache_info = _canonicalize_smiles.cache_info()

            # Get parallel stats
            parallel_stats = agent.get_parallel_stats()

            return BenchmarkResult(
                molecule_name=molecule_name,
                molecule_smiles=Chem.MolToSmiles(molecule),
                mode="parallel",
                num_workers=num_workers,
                runtime_seconds=runtime,
                total_nodes=len(agent.nodes),
                pks_matches=len(agent.get_pks_matches()),
                sink_compounds=len(agent.get_sink_compounds()),
                iterations_completed=parallel_stats["completed_iterations"],
                iterations_failed=parallel_stats["failed_iterations"],
                cache_hits=cache_info.hits,
                cache_misses=cache_info.misses,
            )

        except Exception as e:
            return BenchmarkResult(
                molecule_name=molecule_name,
                molecule_smiles=Chem.MolToSmiles(molecule),
                mode="parallel",
                num_workers=num_workers,
                runtime_seconds=0,
                total_nodes=0,
                pks_matches=0,
                sink_compounds=0,
                iterations_completed=0,
                error=str(e),
            )

    def benchmark_molecule(
        self,
        molecule_info: Dict,
    ) -> Tuple[BenchmarkResult, List[BenchmarkResult]]:
        """
        Benchmark a single molecule with sequential and parallel runs.

        Returns:
            Tuple of (sequential_result, list_of_parallel_results)
        """
        name = molecule_info["name"]
        smiles = molecule_info["smiles"]
        complexity = molecule_info.get("complexity", "unknown")

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            print(f"  [ERROR] Could not parse SMILES: {smiles}")
            error_result = BenchmarkResult(
                molecule_name=name,
                molecule_smiles=smiles,
                mode="error",
                num_workers=0,
                runtime_seconds=0,
                total_nodes=0,
                pks_matches=0,
                sink_compounds=0,
                iterations_completed=0,
                error="Could not parse SMILES",
            )
            return error_result, []

        print(f"\n{'='*60}")
        print(f"Benchmarking: {name} ({complexity})")
        print(f"SMILES: {smiles}")
        print(f"{'='*60}")

        # Run sequential
        print(f"\n  [Sequential] Running...")
        seq_result = self.run_sequential(name, molecule)
        if seq_result.error:
            print(f"  [Sequential] ERROR: {seq_result.error}")
        else:
            print(f"  [Sequential] {seq_result.runtime_seconds:.2f}s, "
                  f"{seq_result.total_nodes} nodes, "
                  f"{seq_result.pks_matches} PKS, {seq_result.sink_compounds} sinks")
        self.results.append(seq_result)

        # Run parallel with different worker counts
        parallel_results = []
        for num_workers in self.config.worker_counts:
            print(f"\n  [Parallel {num_workers}w] Running...")
            par_result = self.run_parallel(name, molecule, num_workers)
            if par_result.error:
                print(f"  [Parallel {num_workers}w] ERROR: {par_result.error}")
            else:
                speedup = seq_result.runtime_seconds / par_result.runtime_seconds if par_result.runtime_seconds > 0 else 0
                print(f"  [Parallel {num_workers}w] {par_result.runtime_seconds:.2f}s, "
                      f"{par_result.total_nodes} nodes, "
                      f"speedup={speedup:.2f}x")
            self.results.append(par_result)
            parallel_results.append(par_result)

            # Create summary for this comparison
            if not seq_result.error and not par_result.error:
                summary = BenchmarkSummary(
                    molecule_name=name,
                    molecule_smiles=smiles,
                    sequential_runtime=seq_result.runtime_seconds,
                    parallel_runtime=par_result.runtime_seconds,
                    num_workers=num_workers,
                    speedup=speedup,
                    sequential_nodes=seq_result.total_nodes,
                    parallel_nodes=par_result.total_nodes,
                    sequential_pks_matches=seq_result.pks_matches,
                    parallel_pks_matches=par_result.pks_matches,
                    sequential_sinks=seq_result.sink_compounds,
                    parallel_sinks=par_result.sink_compounds,
                )
                self.summaries.append(summary)

        return seq_result, parallel_results

    def run_benchmarks(self, molecules: List[Dict]) -> None:
        """Run benchmarks on all provided molecules."""
        print("\n" + "=" * 70)
        print("MCTS Benchmark: Sequential vs Parallel")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Iterations: {self.config.total_iterations}")
        print(f"  Max depth: {self.config.max_depth}")
        print(f"  Max children: {self.config.max_children_per_expand}")
        print(f"  Worker counts: {self.config.worker_counts}")
        print(f"  Virtual loss: {self.config.virtual_loss}")
        print(f"  Molecules to test: {len(molecules)}")

        for mol_info in molecules:
            self.benchmark_molecule(mol_info)

        self.print_summary()

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        if not self.summaries:
            print("\nNo successful comparisons to summarize.")
            return

        # Group by worker count
        by_workers: Dict[int, List[BenchmarkSummary]] = {}
        for s in self.summaries:
            if s.num_workers not in by_workers:
                by_workers[s.num_workers] = []
            by_workers[s.num_workers].append(s)

        for num_workers, summaries in sorted(by_workers.items()):
            print(f"\n{num_workers} Workers:")
            print("-" * 70)
            print(f"{'Molecule':<25} {'Seq (s)':<10} {'Par (s)':<10} {'Speedup':<10} {'Nodes':<15}")
            print("-" * 70)

            total_speedup = 0
            for s in summaries:
                node_diff = s.parallel_nodes - s.sequential_nodes
                node_str = f"{s.sequential_nodes}/{s.parallel_nodes}"
                if node_diff != 0:
                    node_str += f" ({node_diff:+d})"

                print(f"{s.molecule_name:<25} {s.sequential_runtime:<10.2f} "
                      f"{s.parallel_runtime:<10.2f} {s.speedup:<10.2f}x {node_str}")
                total_speedup += s.speedup

            avg_speedup = total_speedup / len(summaries) if summaries else 0
            print("-" * 70)
            print(f"{'Average':<25} {'':<10} {'':<10} {avg_speedup:<10.2f}x")

        # Overall statistics
        print("\n" + "=" * 70)
        print("OVERALL STATISTICS")
        print("=" * 70)

        all_speedups = [s.speedup for s in self.summaries]
        if all_speedups:
            avg_speedup = sum(all_speedups) / len(all_speedups)
            min_speedup = min(all_speedups)
            max_speedup = max(all_speedups)

            print(f"\nSpeedup Statistics:")
            print(f"  Average: {avg_speedup:.2f}x")
            print(f"  Min: {min_speedup:.2f}x")
            print(f"  Max: {max_speedup:.2f}x")

    def save_results(self, output_dir: Path) -> None:
        """Save benchmark results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as CSV
        results_file = output_dir / f"benchmark_results_{timestamp}.csv"
        with open(results_file, "w", newline="") as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=list(self.results[0].__dict__.keys()))
                writer.writeheader()
                for r in self.results:
                    writer.writerow(r.__dict__)
        print(f"\nDetailed results saved to: {results_file}")

        # Save summaries as CSV
        if self.summaries:
            summary_file = output_dir / f"benchmark_summary_{timestamp}.csv"
            with open(summary_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.summaries[0].__dict__.keys()))
                writer.writeheader()
                for s in self.summaries:
                    writer.writerow(s.__dict__)
            print(f"Summary saved to: {summary_file}")

        # Save as JSON for programmatic access
        json_file = output_dir / f"benchmark_{timestamp}.json"
        data = {
            "config": {
                "total_iterations": self.config.total_iterations,
                "max_depth": self.config.max_depth,
                "max_children_per_expand": self.config.max_children_per_expand,
                "worker_counts": self.config.worker_counts,
                "virtual_loss": self.config.virtual_loss,
            },
            "results": [r.__dict__ for r in self.results],
            "summaries": [s.__dict__ for s in self.summaries],
        }
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"JSON data saved to: {json_file}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sequential vs parallel MCTS implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark.py                     # Full benchmark
  python scripts/benchmark.py --quick             # Quick benchmark (fewer iterations)
  python scripts/benchmark.py --molecules 3       # Test first 3 molecules
  python scripts/benchmark.py --workers 2 4 8     # Test specific worker counts
  python scripts/benchmark.py --iterations 50     # Custom iteration count
        """
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick benchmark with fewer iterations (10 instead of 20)"
    )
    parser.add_argument(
        "--molecules", "-m",
        type=int,
        default=None,
        help="Number of molecules to test (default: all)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        nargs="+",
        default=[4],
        help="Worker counts to test (default: 4)"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=20,
        help="Number of MCTS iterations (default: 20)"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=2,
        help="Maximum tree depth (default: 2)"
    )
    parser.add_argument(
        "--virtual-loss",
        type=float,
        default=1.0,
        help="Virtual loss penalty (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save results (default: results/benchmarks)"
    )
    parser.add_argument(
        "--complexity",
        type=str,
        choices=["simple", "medium", "complex", "all"],
        default="all",
        help="Filter molecules by complexity (default: all)"
    )

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        total_iterations=10 if args.quick else args.iterations,
        max_depth=args.depth,
        max_children_per_expand=5,
        worker_counts=args.workers,
        virtual_loss=args.virtual_loss,
    )

    # Filter molecules
    molecules = TEST_MOLECULES.copy()

    if args.complexity != "all":
        molecules = [m for m in molecules if m.get("complexity") == args.complexity]

    if args.molecules:
        molecules = molecules[:args.molecules]

    if not molecules:
        print("No molecules to benchmark. Check your filters.")
        return

    # Run benchmarks
    runner = BenchmarkRunner(config)
    runner.run_benchmarks(molecules)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / "benchmarks"
    runner.save_results(output_dir)


if __name__ == "__main__":
    main()
