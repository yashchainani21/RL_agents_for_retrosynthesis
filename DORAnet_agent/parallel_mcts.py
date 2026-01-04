"""
Parallel Monte Carlo Tree Search with Virtual Loss for DORAnet.

This module implements a parallelized version of the DORAnet MCTS agent
using virtual loss to prevent redundant exploration across threads.

The virtual loss technique was introduced in:
    Chaslot, G., Winands, M.H.M., & van den Herik, H.J. (2008).
    "Parallel Monte-Carlo Tree Search."

Key features:
- Thread-safe tree operations with proper locking
- Virtual loss to encourage exploration diversity
- Configurable number of worker threads
- Backward compatible with sequential DORAnetMCTS
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from rdkit import Chem

from .mcts import DORAnetMCTS, RetroTideResult, _canonicalize_smiles
from .node import Node


class ParallelDORAnetMCTS(DORAnetMCTS):
    """
    Parallel MCTS with virtual loss for multi-threaded exploration.

    This class extends DORAnetMCTS to support parallel execution using
    multiple worker threads. Virtual loss is applied during selection
    to discourage multiple threads from selecting the same node.

    The parallelization strategy:
    1. Selection phase: Synchronized with lock, applies virtual loss
    2. Expansion phase: Unsynchronized (thread-local computation)
    3. Backpropagation phase: Synchronized, removes virtual loss and applies real rewards

    Example:
        >>> root = Node(fragment=target_mol, parent=None, depth=0, provenance="target")
        >>> agent = ParallelDORAnetMCTS(
        ...     root=root,
        ...     target_molecule=target_mol,
        ...     total_iterations=100,
        ...     num_workers=4,
        ...     virtual_loss=1.0,
        ... )
        >>> agent.run()
        >>> pks_matches = agent.get_pks_matches()
    """

    def __init__(
        self,
        root: Node,
        target_molecule: Chem.Mol,
        num_workers: Optional[int] = None,
        virtual_loss: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Initialize parallel MCTS agent.

        Args:
            root: Starting node containing the target molecule.
            target_molecule: The molecule to fragment.
            num_workers: Number of parallel worker threads. Set to 1 for
                sequential execution. Set to None or 0 to use maximum available
                threads (CPU count - 1). Default is None (max threads).
            virtual_loss: Penalty applied during selection to discourage
                other threads from selecting the same node. Higher values
                encourage more exploration diversity. Default is 1.0.
            **kwargs: Additional arguments passed to DORAnetMCTS.
        """
        super().__init__(root=root, target_molecule=target_molecule, **kwargs)

        # Parallel configuration
        self.num_workers = self._get_optimal_workers(num_workers)
        self.virtual_loss = virtual_loss

        # Thread synchronization primitives
        self._tree_lock = threading.Lock()
        self._node_id_lock = threading.Lock()
        self._results_lock = threading.Lock()

        # Thread-safe node ID counter (replaces class-level counter for parallel safety)
        self._next_node_id = root.node_id + 1

        # Statistics tracking
        self._completed_iterations = 0
        self._failed_iterations = 0
        self._lock_wait_time = 0.0

    def _get_optimal_workers(self, requested_workers: Optional[int]) -> int:
        """
        Determine optimal number of workers based on CPU count.

        Args:
            requested_workers: User-requested number of workers.
                If None or 0, uses maximum available (CPU count - 1).

        Returns:
            Optimal worker count, capped by available CPUs.
        """
        cpu_count = os.cpu_count() or 4
        # Use at most N-1 cores to leave headroom for system
        max_workers = max(1, cpu_count - 1)

        # None or 0 means "use max available"
        if requested_workers is None or requested_workers == 0:
            print(f"[ParallelMCTS] Using max available workers: {max_workers} (CPU count: {cpu_count})")
            return max_workers

        optimal = max(1, min(requested_workers, max_workers))

        if requested_workers > max_workers:
            print(f"[ParallelMCTS] Requested {requested_workers} workers, "
                  f"using {optimal} (CPU count: {cpu_count})")

        return optimal

    def _apply_virtual_loss_to_path(self, leaf: Node) -> None:
        """
        Apply virtual loss from leaf up to root.

        This penalizes the entire selection path to discourage other
        threads from selecting overlapping paths.

        Args:
            leaf: The leaf node selected for expansion.
        """
        current = leaf
        while current is not None:
            current.apply_virtual_loss(self.virtual_loss)
            current = current.parent

    def _remove_virtual_loss_from_path(self, leaf: Node) -> None:
        """
        Remove virtual loss from leaf up to root.

        Called after expansion completes, before applying real rewards.

        Args:
            leaf: The leaf node that was expanded.
        """
        current = leaf
        while current is not None:
            current.remove_virtual_loss(self.virtual_loss)
            current = current.parent

    def _get_next_node_id(self) -> int:
        """
        Get the next unique node ID in a thread-safe manner.

        Returns:
            Unique integer node ID.
        """
        with self._node_id_lock:
            node_id = self._next_node_id
            self._next_node_id += 1
            return node_id

    def _thread_safe_add_node(self, node: Node, parent: Node) -> None:
        """
        Add a node to the tree in a thread-safe manner.

        Args:
            node: The node to add.
            parent: The parent node.
        """
        # Assign thread-safe node ID
        node.node_id = self._get_next_node_id()
        node.parent_id = parent.node_id

        # Add to parent's children (already holding tree lock during backprop)
        parent.add_child(node)

        # Add to global tracking lists
        self.nodes.append(node)
        self.edges.append((parent.node_id, node.node_id))

    def _thread_safe_add_retrotide_result(self, result: RetroTideResult) -> None:
        """
        Add a RetroTide result in a thread-safe manner.

        Args:
            result: The RetroTide result to add.
        """
        with self._results_lock:
            self.retrotide_results.append(result)

    def _parallel_iteration(self, iteration: int) -> bool:
        """
        Execute one MCTS iteration with virtual loss.

        This method is designed to be called by multiple threads concurrently.
        It uses locking to ensure thread safety during tree modifications.

        Args:
            iteration: The iteration number (for logging/tracking).

        Returns:
            True if iteration completed successfully, False if no valid node found.
        """
        # =====================================================================
        # PHASE 1: Selection (synchronized)
        # =====================================================================
        with self._tree_lock:
            leaf = self.select(self.root)

            if leaf is None:
                return False

            # Check if already expanded or at max depth
            if leaf.expanded or leaf.depth >= self.max_depth:
                # Just backpropagate current value
                reward = self.calculate_reward(leaf)
                self.backpropagate(leaf, reward)
                return True

            # Apply virtual loss along path from leaf to root
            # This discourages other threads from selecting this path
            self._apply_virtual_loss_to_path(leaf)

            # Track selection for visualization
            leaf.selected_at_iterations.append(iteration)

        # =====================================================================
        # PHASE 2: Expansion (unsynchronized - thread-local computation)
        # =====================================================================
        # This is the expensive part - DORAnet fragment generation
        # It runs without holding the lock, allowing other threads to proceed

        new_children = []
        pks_matches_found = []

        if not leaf.expanded:
            # Generate fragments using DORAnet (expensive, no lock needed)
            fragment_infos = []

            if self.use_enzymatic:
                for frag_info in self._generate_doranet_fragments(leaf.fragment, "enzymatic"):
                    fragment_infos.append((frag_info, "enzymatic"))

            if self.use_synthetic:
                for frag_info in self._generate_doranet_fragments(leaf.fragment, "synthetic"):
                    fragment_infos.append((frag_info, "synthetic"))

            # Process fragments (filtering, creating nodes)
            for frag_info, provenance in fragment_infos:
                # Check if prohibited
                if self._is_prohibited_chemical(frag_info.smiles):
                    continue

                # Check MW threshold
                if self._exceeds_MW_threshold(frag_info.molecule):
                    continue

                # Create child node (node ID will be assigned thread-safely later)
                child = Node(
                    fragment=frag_info.molecule,
                    parent=None,  # Will be set during thread-safe add
                    provenance=provenance,
                    reaction_smarts=frag_info.reaction_smarts,
                    reaction_name=frag_info.reaction_name,
                    reactants_smiles=frag_info.reactants_smiles,
                    products_smiles=frag_info.products_smiles,
                )
                child.created_at_iteration = iteration

                # Check sink compound status
                sink_type = self._get_sink_compound_type(frag_info.smiles)
                if sink_type:
                    child.is_sink_compound = True
                    child.sink_compound_type = sink_type
                    child.expanded = True

                # Check PKS library match
                elif self._is_in_pks_library(frag_info.smiles):
                    child.is_pks_terminal = True
                    child.expanded = True
                    if self.spawn_retrotide:
                        pks_matches_found.append((frag_info.molecule, child))

                new_children.append((child, leaf))

        # =====================================================================
        # PHASE 3: Backpropagation (synchronized)
        # =====================================================================
        with self._tree_lock:
            # Remove virtual loss from the path
            self._remove_virtual_loss_from_path(leaf)

            # Mark leaf as expanded
            leaf.expanded = True
            leaf.expanded_at_iteration = iteration

            # Add new children to tree (thread-safe)
            for child, parent in new_children:
                self._thread_safe_add_node(child, parent)

                # Calculate and backpropagate reward
                reward = self.calculate_reward(child)
                self.backpropagate(child, reward)

        # =====================================================================
        # PHASE 4: RetroTide spawning (outside lock - independent searches)
        # =====================================================================
        # RetroTide searches are independent and can run in parallel
        for target_mol, source_node in pks_matches_found:
            self._launch_retrotide_agent_parallel(target_mol, source_node)

        return True

    def _launch_retrotide_agent_parallel(self, target: Chem.Mol, source_node: Node) -> None:
        """
        Spawn a RetroTide MCTS search (thread-safe version).

        This is similar to the parent class method but uses thread-safe
        result collection.

        Args:
            target: The fragment molecule to synthesize.
            source_node: The DORAnet node that produced this fragment.
        """
        from RetroTide_agent.mcts import MCTS as RetroTideMCTS
        from RetroTide_agent.node import Node as RetroTideNode

        target_smiles = Chem.MolToSmiles(target)
        print(f"[ParallelMCTS] Spawning RetroTide search for: {target_smiles}")

        root = RetroTideNode(PKS_product=None, PKS_design=None, parent=None, depth=0)
        agent = RetroTideMCTS(
            root=root,
            target_molecule=target,
            **self.retrotide_kwargs,
        )
        agent.run()

        # Extract results
        successful_nodes = getattr(agent, 'successful_nodes', set())
        num_successful = len(successful_nodes)

        best_score = 0.0
        if successful_nodes:
            best_score = 1.0
        else:
            for node in getattr(agent, 'nodes', []):
                if hasattr(node, 'value') and node.visits > 0:
                    avg_value = node.value / node.visits
                    best_score = max(best_score, avg_value)

        # Create result record
        result = RetroTideResult(
            doranet_node_id=source_node.node_id,
            doranet_node_smiles=source_node.smiles or "",
            doranet_node_depth=source_node.depth,
            doranet_node_provenance=source_node.provenance or "unknown",
            doranet_reaction_name=source_node.reaction_name,
            doranet_reaction_smarts=source_node.reaction_smarts,
            doranet_reactants_smiles=source_node.reactants_smiles or [],
            doranet_products_smiles=source_node.products_smiles or [],
            retrotide_target_smiles=target_smiles,
            retrotide_successful=(num_successful > 0),
            retrotide_num_successful_nodes=num_successful,
            retrotide_best_score=best_score,
            retrotide_total_nodes=len(getattr(agent, 'nodes', [])),
            retrotide_agent=agent,
        )

        # Thread-safe result addition
        self._thread_safe_add_retrotide_result(result)

    def run(self) -> None:
        """
        Execute parallel MCTS with virtual loss.

        If num_workers is 1, falls back to sequential execution.
        Otherwise, runs iterations in parallel using a thread pool.
        """
        if self.num_workers <= 1:
            # Fall back to sequential execution
            print("[ParallelMCTS] Running in sequential mode (num_workers=1)")
            super().run()
            return

        pks_mode = bool(self.pks_library)
        mode_str = "PKS library lookup" if pks_mode else "RetroTide spawning"

        print(f"[ParallelMCTS] Starting parallel MCTS with {self.num_workers} workers, "
              f"{self.total_iterations} iterations, virtual_loss={self.virtual_loss}")
        print(f"[ParallelMCTS] Mode: {mode_str}, max_depth={self.max_depth}")

        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all iterations to the thread pool
            futures = {
                executor.submit(self._parallel_iteration, i): i
                for i in range(self.total_iterations)
            }

            # Process results as they complete
            for future in as_completed(futures):
                iteration = futures[future]
                try:
                    if future.result():
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"[ParallelMCTS] Iteration {iteration} failed with error: {e}")
                    failed += 1

                # Progress reporting every 10%
                total_done = completed + failed
                if total_done % max(1, self.total_iterations // 10) == 0:
                    pct = total_done / self.total_iterations * 100
                    print(f"[ParallelMCTS] Progress: {total_done}/{self.total_iterations} "
                          f"({pct:.0f}%) - {completed} completed, {failed} failed")

        self._completed_iterations = completed
        self._failed_iterations = failed

        # Summary statistics
        sink_count = len(self.get_sink_compounds())
        pks_terminal_count = len(self.get_pks_terminal_nodes())

        print(f"[ParallelMCTS] MCTS complete. Total nodes: {len(self.nodes)}, "
              f"PKS terminals: {pks_terminal_count}, Sink compounds: {sink_count}")
        print(f"[ParallelMCTS] Iterations: {completed} completed, {failed} failed")

        # Log SMILES cache statistics
        from .mcts import _canonicalize_smiles
        cache_info = _canonicalize_smiles.cache_info()
        hit_rate = (cache_info.hits / (cache_info.hits + cache_info.misses) * 100
                    if (cache_info.hits + cache_info.misses) > 0 else 0)
        print(f"[ParallelMCTS] SMILES cache: {cache_info.hits} hits, {cache_info.misses} misses "
              f"({hit_rate:.1f}% hit rate), {cache_info.currsize}/{cache_info.maxsize} cached")

    def get_parallel_stats(self) -> dict:
        """
        Get statistics about the parallel execution.

        Returns:
            Dictionary with parallel execution statistics.
        """
        return {
            "num_workers": self.num_workers,
            "virtual_loss": self.virtual_loss,
            "completed_iterations": self._completed_iterations,
            "failed_iterations": self._failed_iterations,
            "total_nodes": len(self.nodes),
            "pks_terminals": len(self.get_pks_terminal_nodes()),
            "sink_compounds": len(self.get_sink_compounds()),
        }
