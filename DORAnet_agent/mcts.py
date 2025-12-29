"""
Simplified Monte Carlo Tree Search agent that explores DORAnet retro-biosynthetic
and retro-chemical transformations, spawning RetroTide forward searches for each
fragment discovered.

This is a minimal implementation with only Selection and Expansion steps.
Rollout and Backpropagation will be added later.
"""

from __future__ import annotations

import csv
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit import RDLogger

import doranet.modules.enzymatic as enzymatic
import doranet.modules.synthetic as synthetic

from .node import Node

# Optional RetroTide imports - may not be available in all environments
try:
    from RetroTide_agent.mcts import MCTS as RetroTideMCTS
    from RetroTide_agent.node import Node as RetroTideNode
    RETROTIDE_AVAILABLE = True
except ImportError:
    RetroTideMCTS = None
    RetroTideNode = None
    RETROTIDE_AVAILABLE = False

# Silence RDKit logs during network builds.
RDLogger.DisableLog("rdApp.*")


@dataclass
class RetroTideResult:
    """Stores results from a RetroTide MCTS run with traceability to DORAnet."""

    # DORAnet context
    doranet_node_id: int
    doranet_node_smiles: str
    doranet_node_depth: int
    doranet_node_provenance: str

    # DORAnet reaction info (how this fragment was created)
    doranet_reaction_name: Optional[str] = None
    doranet_reaction_smarts: Optional[str] = None
    doranet_reactants_smiles: List[str] = field(default_factory=list)
    doranet_products_smiles: List[str] = field(default_factory=list)

    # RetroTide results
    retrotide_target_smiles: str = ""
    retrotide_successful: bool = False
    retrotide_num_successful_nodes: int = 0
    retrotide_best_score: float = 0.0
    retrotide_total_nodes: int = 0

    # Optional: the actual RetroTide agent (for deeper inspection)
    retrotide_agent: Any = field(default=None, repr=False)


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form, returning None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _load_enzymatic_rule_labels() -> List[str]:
    """
    Load enzymatic reaction rule labels from DORAnet.

    Returns:
        List of rule names indexed by operator position.
    """
    try:
        import doranet.modules.enzymatic as enzymatic
        import os
        import pandas as pd

        # Load rule labels from TSV file
        enzymatic_path = os.path.dirname(enzymatic.__file__)
        rules_file = os.path.join(enzymatic_path, 'JN3604IMT_rules.tsv')
        df = pd.read_csv(rules_file, sep='\t')

        # Create list of names indexed by position (same order as in file)
        labels = []
        for _, row in df.iterrows():
            name = row['Name']
            if pd.notna(name):
                labels.append(name)
            else:
                labels.append(None)

        return labels

    except Exception as e:
        print(f"[WARN] Could not load enzymatic rule labels: {e}")
        return []


def _load_synthetic_reaction_labels() -> List[str]:
    """
    Load synthetic reaction labels from DORAnet.

    Returns:
        List of reaction names indexed by operator position.
    """
    try:
        import doranet.modules.synthetic.Reaction_Smarts_Retro as retro_smarts

        # Create list of names indexed by position
        labels = []
        for op_def in retro_smarts.op_retro_smarts:
            if hasattr(op_def, 'name'):
                labels.append(op_def.name)
            else:
                labels.append(None)

        return labels

    except Exception as e:
        print(f"[WARN] Could not load synthetic reaction labels: {e}")
        return []


# Small molecules to exclude (common byproducts, not useful fragments)
DEFAULT_EXCLUDED_FRAGMENTS = (
    "O",           # water
    "O=O",         # O2
    "[H][H]",      # H2
    "O=C=O",       # CO2
    "C=O",         # formaldehyde
    "[C-]#[O+]",   # CO
    "N#N",         # N2
    "N",           # ammonia
    "S",           # sulfur
)


def _load_cofactors_from_csv(csv_path: str) -> Set[str]:
    """
    Load cofactor SMILES from a CSV file and return canonical SMILES set.

    Args:
        csv_path: Path to CSV file with a 'SMILES' column.

    Returns:
        Set of canonical SMILES strings for cofactors.
    """
    cofactors: Set[str] = set()
    path = Path(csv_path)

    if not path.exists():
        print(f"[WARN] Cofactors file not found: {csv_path}")
        return cofactors

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get("SMILES", "").strip()
            if smiles:
                # Skip SMILES with wildcards (e.g., "*c1c...")
                if "*" in smiles:
                    continue
                canonical = _canonicalize_smiles(smiles)
                if canonical:
                    cofactors.add(canonical)

    print(f"[DORAnet] Loaded {len(cofactors)} cofactors from {path.name}")
    return cofactors


def _load_pks_library(pks_file: str) -> Set[str]:
    """
    Load PKS product SMILES from a text file (one SMILES per line).

    Args:
        pks_file: Path to text file with PKS SMILES.

    Returns:
        Set of canonical SMILES strings for PKS products.
    """
    pks_smiles: Set[str] = set()
    path = Path(pks_file)

    if not path.exists():
        print(f"[WARN] PKS library file not found: {pks_file}")
        return pks_smiles

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            smiles = line.strip()
            if smiles and not smiles.startswith("#"):
                canonical = _canonicalize_smiles(smiles)
                if canonical:
                    pks_smiles.add(canonical)

    print(f"[DORAnet] Loaded {len(pks_smiles)} PKS products from {path.name}")
    return pks_smiles


class DORAnetMCTS:
    """
    Simplified MCTS driver for DORAnet retro-fragmentation.

    For each fragment discovered during expansion, a RetroTide forward
    MCTS search is spawned to attempt synthesis from PKS building blocks.

    Current implementation: Selection + Expansion only.
    TODO: Add Rollout and Backpropagation steps.
    """

    def __init__(
        self,
        root: Node,
        target_molecule: Chem.Mol,
        total_iterations: int = 100,
        max_depth: int = 3,
        use_enzymatic: bool = True,
        use_synthetic: bool = True,
        generations_per_expand: int = 1,
        max_children_per_expand: int = 10,
        excluded_fragments: Optional[Iterable[str]] = None,
        cofactors_file: Optional[str] = None,
        cofactors_files: Optional[List[str]] = None,
        pks_library_file: Optional[str] = None,
        spawn_retrotide: bool = True,
        retrotide_kwargs: Optional[Dict] = None,
        enable_visualization: bool = False,
        enable_interactive_viz: bool = False,
        enable_iteration_visualizations: bool = False,
        auto_open_viz: bool = False,
        auto_open_iteration_viz: bool = False,
        visualization_output_dir: Optional[str] = None,
        iteration_viz_interval: int = 1,
    ) -> None:
        """
        Args:
            root: Starting node containing the target molecule.
            target_molecule: The molecule to fragment.
            total_iterations: Number of MCTS iterations to run.
            max_depth: Maximum tree depth for fragmentation.
            use_enzymatic: Whether to use enzymatic retro-transformations.
            use_synthetic: Whether to use synthetic retro-transformations.
            generations_per_expand: DORAnet generations per expansion step.
            max_children_per_expand: Max fragments to retain per expansion.
            excluded_fragments: SMILES of fragments to ignore (small byproducts).
            cofactors_file: Path to CSV file with cofactor SMILES to exclude (deprecated, use cofactors_files).
            cofactors_files: List of paths to CSV files with cofactor SMILES to exclude.
            pks_library_file: Path to text file with PKS product SMILES for reward calculation.
            spawn_retrotide: Whether to spawn RetroTide searches for each fragment.
            retrotide_kwargs: Parameters passed to RetroTide MCTS agents.
            enable_visualization: Whether to automatically generate static visualizations (PNG).
            enable_interactive_viz: Whether to generate interactive HTML visualization.
            enable_iteration_visualizations: If True, generate visualizations after each iteration.
            auto_open_viz: If True, automatically open final interactive visualization in browser.
            auto_open_iteration_viz: If True, automatically open iteration visualizations in browser.
            visualization_output_dir: Directory to save visualizations (default: current directory).
            iteration_viz_interval: Generate iteration visualizations every N iterations (default: 1).
        """
        self.root = root
        self.target_molecule = target_molecule
        self.total_iterations = total_iterations
        self.max_depth = max_depth
        self.use_enzymatic = use_enzymatic
        self.use_synthetic = use_synthetic
        self.generations_per_expand = generations_per_expand
        self.max_children_per_expand = max_children_per_expand
        self.spawn_retrotide = spawn_retrotide and RETROTIDE_AVAILABLE
        self.retrotide_kwargs = retrotide_kwargs or {}
        self.enable_visualization = enable_visualization
        self.enable_interactive_viz = enable_interactive_viz
        self.enable_iteration_visualizations = enable_iteration_visualizations
        self.auto_open_viz = auto_open_viz
        self.auto_open_iteration_viz = auto_open_iteration_viz
        self.visualization_output_dir = visualization_output_dir or "."
        self.iteration_viz_interval = iteration_viz_interval

        if spawn_retrotide and not RETROTIDE_AVAILABLE:
            print("[WARN] RetroTide not available - spawning disabled")

        # Tree tracking
        self.nodes: List[Node] = [root]
        self.edges: List[Tuple[int, int]] = []

        # Track all RetroTide runs with full traceability
        self.retrotide_results: List[RetroTideResult] = []

        # Build excluded fragments set from default small molecules
        excluded_iterable = (
            excluded_fragments if excluded_fragments is not None
            else DEFAULT_EXCLUDED_FRAGMENTS
        )
        self.excluded_fragments: Set[str] = {
            _canonicalize_smiles(smi)
            for smi in excluded_iterable
            if _canonicalize_smiles(smi)
        }

        # Add cofactors from CSV files if provided
        # Support both single file (deprecated) and list of files
        cofactor_files_to_load = []
        if cofactors_file:
            cofactor_files_to_load.append(cofactors_file)
        if cofactors_files:
            cofactor_files_to_load.extend(cofactors_files)

        # Track chemistry helpers separately for DORAnet synthetic network generation
        self.chemistry_helpers: Set[str] = set()

        for cofactor_file_path in cofactor_files_to_load:
            cofactor_smiles = _load_cofactors_from_csv(cofactor_file_path)
            self.excluded_fragments.update(cofactor_smiles)

            # If this is the chemistry_helpers file, also store for DORAnet
            if "chemistry_helpers" in str(cofactor_file_path).lower():
                self.chemistry_helpers.update(cofactor_smiles)

        # Load PKS product library for reward calculation
        self.pks_library: Set[str] = set()
        if pks_library_file:
            self.pks_library = _load_pks_library(pks_library_file)

        # Load reaction label mappings for human-readable names
        self._enzymatic_labels = _load_enzymatic_rule_labels()
        self._synthetic_labels = _load_synthetic_reaction_labels()
        print(f"[DORAnet] Loaded {len(self._enzymatic_labels)} enzymatic rule labels")
        print(f"[DORAnet] Loaded {len(self._synthetic_labels)} synthetic reaction labels")

        # Report chemistry helpers for synthetic networks
        if self.chemistry_helpers:
            print(f"[DORAnet] Using {len(self.chemistry_helpers)} chemistry helpers for synthetic network generation")

    @dataclass
    class FragmentInfo:
        """Information about a generated fragment and the reaction that created it."""
        molecule: Chem.Mol
        smiles: str
        reaction_smarts: Optional[str] = None
        reaction_name: Optional[str] = None
        reactants_smiles: List[str] = field(default_factory=list)
        products_smiles: List[str] = field(default_factory=list)

    def _generate_doranet_fragments(
        self,
        molecule: Chem.Mol,
        mode: str,
    ) -> List["DORAnetMCTS.FragmentInfo"]:
        """
        Run DORAnet in retro mode to generate fragment molecules.

        Args:
            molecule: The molecule to fragment.
            mode: Either "enzymatic" or "synthetic".

        Returns:
            List of FragmentInfo objects with molecule and reaction details.
        """
        if molecule is None:
            return []

        starter_smiles = Chem.MolToSmiles(molecule)
        job_name = f"doranet_{mode}_retro_{uuid.uuid4().hex[:8]}"

        module = enzymatic if mode == "enzymatic" else synthetic
        try:
            # Build network generation kwargs
            network_kwargs = {
                "job_name": job_name,
                "starters": {starter_smiles},
                "gen": self.generations_per_expand,
                "direction": "retro",
            }

            # Add helpers for synthetic network generation
            if mode == "synthetic" and self.chemistry_helpers:
                network_kwargs["helpers"] = self.chemistry_helpers

            network = module.generate_network(**network_kwargs)
        except Exception as exc:
            print(f"[WARN] DORAnet {mode} failed for {starter_smiles}: {exc}")
            return []

        # Build a map from product SMILES to reaction info
        mol_to_reaction: Dict[str, Dict] = {}

        # Convert network mols and ops to lists for indexing
        mols_list = list(network.mols)
        ops_list = list(network.ops)

        for rxn in getattr(network, 'rxns', []):
            # Extract reaction info using indices
            try:
                # Get the operator using its index
                op_idx = rxn.operator
                op = ops_list[op_idx] if op_idx < len(ops_list) else None

                # Extract SMARTS - use uid attribute if available, otherwise convert to string
                if op and hasattr(op, 'uid'):
                    rxn_smarts = op.uid
                else:
                    rxn_smarts = str(op) if op else None

                # Get human-readable reaction label using operator index
                rxn_label = None
                if op_idx is not None:
                    # Look up label by operator index
                    if mode == "enzymatic":
                        if op_idx < len(self._enzymatic_labels):
                            rxn_label = self._enzymatic_labels[op_idx]
                    else:  # synthetic
                        if op_idx < len(self._synthetic_labels):
                            rxn_label = self._synthetic_labels[op_idx]

                # Fallback to truncated SMARTS if no label found
                if not rxn_label and rxn_smarts:
                    rxn_label = rxn_smarts[:60] + "..." if len(rxn_smarts) > 60 else rxn_smarts

                # Get reactants and products using their indices
                reactant_idxs = rxn.reactants
                product_idxs = rxn.products

                reactant_smiles = [
                    str(getattr(mols_list[i], 'uid', mols_list[i]))
                    for i in reactant_idxs if i < len(mols_list)
                ]
                product_smiles = [
                    str(getattr(mols_list[i], 'uid', mols_list[i]))
                    for i in product_idxs if i < len(mols_list)
                ]

                # Map each product to this reaction
                for prod in product_smiles:
                    canonical_prod = _canonicalize_smiles(prod)
                    if canonical_prod and canonical_prod not in mol_to_reaction:
                        mol_to_reaction[canonical_prod] = {
                            'smarts': rxn_smarts,
                            'label': rxn_label,  # Human-readable label
                            'reactants': reactant_smiles,
                            'products': product_smiles,
                        }
            except Exception:
                continue

        fragments: List[DORAnetMCTS.FragmentInfo] = []
        seen: Set[str] = set()

        for mol in network.mols:
            mol_smiles = getattr(mol, "uid", None)
            if mol_smiles is None:
                continue

            # Skip SMILES with wildcards (template patterns, not real molecules)
            mol_smiles_str = str(mol_smiles)
            if "*" in mol_smiles_str:
                continue

            canonical = _canonicalize_smiles(mol_smiles_str)
            if canonical is None:
                continue
            if canonical == _canonicalize_smiles(starter_smiles):
                continue  # Skip the input molecule itself
            if canonical in self.excluded_fragments:
                continue  # Skip cofactors and small byproducts
            if canonical in seen:
                continue  # Skip duplicates

            rd_mol = Chem.MolFromSmiles(canonical)
            if rd_mol is None:
                continue

            seen.add(canonical)

            # Get reaction info for this fragment
            rxn_info = mol_to_reaction.get(canonical, {})

            frag_info = DORAnetMCTS.FragmentInfo(
                molecule=rd_mol,
                smiles=canonical,
                reaction_smarts=rxn_info.get('smarts'),
                reaction_name=rxn_info.get('label'),  # Use the human-readable label
                reactants_smiles=rxn_info.get('reactants', []),
                products_smiles=rxn_info.get('products', []),
            )
            fragments.append(frag_info)

            if len(fragments) >= self.max_children_per_expand:
                break

        return fragments

    def select(self, node: Node) -> Optional[Node]:
        """
        Traverse tree using UCB1 policy to find a leaf node to expand.

        Returns the selected leaf node, or None if no valid node found.
        """
        while node.children:
            best_node: Optional[Node] = None
            best_score = -math.inf
            log_parent_visits = math.log(max(node.visits, 1))

            for child in node.children:
                if child.visits == 0:
                    # Prioritize unvisited nodes
                    score = math.inf
                else:
                    exploit = child.value / child.visits
                    explore = math.sqrt(2 * log_parent_visits / child.visits)
                    score = exploit + explore

                child.selection_score = score

                if score > best_score:
                    best_score = score
                    best_node = child

            if best_node is None:
                return None

            node = best_node

        return node

    def _is_in_pks_library(self, smiles: str) -> bool:
        """Check if a SMILES string is in the PKS library."""
        if not self.pks_library:
            return False
        canonical = _canonicalize_smiles(smiles)
        return canonical is not None and canonical in self.pks_library

    def expand(self, node: Node) -> List[Node]:
        """
        Expand a node by applying DORAnet retro-transformations.

        For each fragment generated, creates a child node. If the fragment
        matches the PKS library and spawn_retrotide is enabled, a RetroTide
        forward MCTS search is spawned to find PKS designs.

        Returns:
            List of newly created child nodes.
        """
        fragment_infos: List[Tuple[DORAnetMCTS.FragmentInfo, str]] = []

        if self.use_enzymatic:
            for frag_info in self._generate_doranet_fragments(node.fragment, "enzymatic"):
                fragment_infos.append((frag_info, "enzymatic"))

        if self.use_synthetic:
            for frag_info in self._generate_doranet_fragments(node.fragment, "synthetic"):
                fragment_infos.append((frag_info, "synthetic"))

        new_children: List[Node] = []

        for frag_info, provenance in fragment_infos:
            # Create child node with reaction information
            child = Node(
                fragment=frag_info.molecule,
                parent=node,
                provenance=provenance,
                reaction_smarts=frag_info.reaction_smarts,
                reaction_name=frag_info.reaction_name,
                reactants_smiles=frag_info.reactants_smiles,
                products_smiles=frag_info.products_smiles,
            )
            node.add_child(child)
            self.nodes.append(child)
            self.edges.append((node.node_id, child.node_id))
            new_children.append(child)

            # Only spawn RetroTide for fragments that match the PKS library
            if self.spawn_retrotide and self._is_in_pks_library(frag_info.smiles):
                print(f"[DORAnet] Fragment {frag_info.smiles} matches PKS library - spawning RetroTide")
                self._launch_retrotide_agent(target=frag_info.molecule, source_node=child)
            elif self.spawn_retrotide:
                print(f"[DORAnet] Fragment {frag_info.smiles} not in PKS library - skipping RetroTide")

        node.expanded = True
        return new_children

    def _launch_retrotide_agent(self, target: Chem.Mol, source_node: Node) -> None:
        """
        Spawn a RetroTide MCTS search to synthesize the given fragment.

        Args:
            target: The fragment molecule to synthesize.
            source_node: The DORAnet node that produced this fragment.
        """
        target_smiles = Chem.MolToSmiles(target)
        print(f"[DORAnet] Spawning RetroTide search for: {target_smiles}")

        root = RetroTideNode(PKS_product=None, PKS_design=None, parent=None, depth=0)
        agent = RetroTideMCTS(
            root=root,
            target_molecule=target,
            **self.retrotide_kwargs,
        )
        agent.run()

        # Extract results from the RetroTide agent
        successful_nodes = getattr(agent, 'successful_nodes', set())
        num_successful = len(successful_nodes)

        # Get best score from successful nodes or from all nodes
        best_score = 0.0
        if successful_nodes:
            best_score = 1.0  # Target was reached
        else:
            # Get best score from any node
            for node in getattr(agent, 'nodes', []):
                if hasattr(node, 'value') and node.visits > 0:
                    avg_value = node.value / node.visits
                    best_score = max(best_score, avg_value)

        # Create result record with full traceability
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
        self.retrotide_results.append(result)

    def calculate_reward(self, node: Node) -> float:
        """
        Calculate reward for a node based on PKS library matching.

        Returns:
            1.0 if the fragment is in the PKS library, 0.0 otherwise.
        """
        if not self.pks_library:
            # No PKS library loaded, return 0 (neutral reward)
            return 0.0

        smiles = node.smiles
        if smiles is None:
            return 0.0

        canonical = _canonicalize_smiles(smiles)
        if canonical and canonical in self.pks_library:
            return 1.0

        return 0.0

    def backpropagate(self, node: Node, reward: float) -> None:
        """
        Backpropagate reward from a node up to the root.

        Args:
            node: The node to start backpropagation from.
            reward: The reward value to propagate.
        """
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def run(self) -> None:
        """
        Execute the MCTS loop: Selection → Expansion → Reward → Backpropagation.

        If a PKS library is provided, rewards are calculated based on whether
        fragments match known PKS products. Otherwise, RetroTide searches are
        spawned for each fragment.
        """
        pks_mode = bool(self.pks_library)
        mode_str = "PKS library lookup" if pks_mode else "RetroTide spawning"
        print(f"[DORAnet] Starting MCTS with {self.total_iterations} iterations, "
              f"max_depth={self.max_depth}, mode={mode_str}")

        # Track current iteration for visualization
        self.current_iteration = 0

        for iteration in range(self.total_iterations):
            self.current_iteration = iteration

            # Selection: find a leaf node using UCB1
            leaf = self.select(self.root)
            if leaf is None:
                print(f"[DORAnet] No valid leaf found at iteration {iteration}, stopping.")
                break

            # Track when this node was selected
            leaf.selected_at_iterations.append(iteration)

            # Check depth limit
            if leaf.depth >= self.max_depth:
                # Backpropagate with current reward and continue
                reward = self.calculate_reward(leaf)
                self.backpropagate(leaf, reward)
                continue

            # Expansion: generate fragments
            if not leaf.expanded:
                leaf.expanded_at_iteration = iteration
                new_children = self.expand(leaf)
                pks_matches = 0

                # Calculate rewards for each child and backpropagate
                for child in new_children:
                    child.created_at_iteration = iteration
                    reward = self.calculate_reward(child)
                    if reward > 0:
                        pks_matches += 1
                    self.backpropagate(child, reward)

                if pks_mode:
                    print(f"[DORAnet] Iteration {iteration}: expanded node {leaf.node_id}, "
                          f"created {len(new_children)} children, {pks_matches} PKS matches")
                else:
                    print(f"[DORAnet] Iteration {iteration}: expanded node {leaf.node_id}, "
                          f"created {len(new_children)} children")
            else:
                # Node already expanded, just backpropagate
                reward = self.calculate_reward(leaf)
                self.backpropagate(leaf, reward)

            # Generate iteration visualization if enabled
            if self.enable_iteration_visualizations:
                if (iteration + 1) % self.iteration_viz_interval == 0:
                    self._generate_iteration_visualization(iteration)

        # Summary statistics
        if pks_mode:
            total_matches = sum(1 for n in self.nodes if self.calculate_reward(n) > 0)
            print(f"[DORAnet] MCTS complete. Total nodes: {len(self.nodes)}, "
                  f"PKS matches: {total_matches}")
        else:
            print(f"[DORAnet] MCTS complete. Total nodes: {len(self.nodes)}, "
                  f"RetroTide runs: {len(self.retrotide_runs)}")

    def get_tree_summary(self) -> str:
        """Return a summary of the search tree."""
        lines = ["DORAnet MCTS Tree Summary:", "=" * 40]
        for node in self.nodes:
            indent = "  " * node.depth
            pks_match = "✓PKS" if self.calculate_reward(node) > 0 else ""
            avg_value = f"{node.value / node.visits:.2f}" if node.visits > 0 else "N/A"
            lines.append(f"{indent}Node {node.node_id}: {node.smiles} "
                        f"(depth={node.depth}, visits={node.visits}, value={avg_value}, "
                        f"via={node.provenance}) {pks_match}")
        return "\n".join(lines)

    def get_pks_matches(self) -> List[Node]:
        """Return nodes whose fragments match the PKS library."""
        return [n for n in self.nodes if self.calculate_reward(n) > 0]

    def get_pathway_to_node(self, node: Node) -> List[Node]:
        """
        Trace the pathway from root to a given node.

        Returns:
            List of nodes from root to the given node (inclusive).
        """
        pathway = []
        current = node
        while current is not None:
            pathway.append(current)
            current = current.parent
        return list(reversed(pathway))

    def format_pathway(self, node: Node) -> str:
        """
        Format the pathway from root to node as a readable string.

        Returns:
            Multi-line string showing each step in the pathway.
        """
        pathway = self.get_pathway_to_node(node)
        if len(pathway) <= 1:
            return f"  Target molecule (no transformations needed)"

        lines = []
        for i, step_node in enumerate(pathway):
            if i == 0:
                lines.append(f"  Step 0 (Target): {step_node.smiles}")
            else:
                rxn_info = ""
                if step_node.reaction_name:
                    # Extract a readable portion of the reaction
                    rxn_short = step_node.reaction_name[:60] + "..." if len(step_node.reaction_name) > 60 else step_node.reaction_name
                    rxn_info = f"\n           Reaction: {rxn_short}"
                lines.append(f"  Step {i} ({step_node.provenance}): {step_node.smiles}{rxn_info}")

        return "\n".join(lines)

    @staticmethod
    def format_pks_module(module, module_num: int) -> str:
        """
        Format a PKS module with detailed domain information.

        Args:
            module: A PKS module object from bcs.Cluster
            module_num: The module number (1-indexed)

        Returns:
            Formatted string describing the module domains.
        """
        lines = [f"      Module {module_num}:"]

        # Try to extract domain information from module
        mod_str = str(module)

        # Parse the module string to extract domain info
        # Typical format: ["AT{'substrate': 'xxx'}", "KR{'type': 'xxx'}", "DH{}", "ER{}", "loading: False"]

        # AT domain (Acyltransferase) - determines substrate
        if "AT{" in mod_str:
            import re
            at_match = re.search(r"AT\{'substrate':\s*'([^']+)'", mod_str)
            if at_match:
                substrate = at_match.group(1)
                # Map substrate codes to readable names
                substrate_names = {
                    'Malonyl-CoA': 'Malonyl-CoA (C2, no branch)',
                    'Methylmalonyl-CoA': 'Methylmalonyl-CoA (C2, methyl branch)',
                    'mxmal': 'Methoxymalonyl-ACP (C2, methoxy branch)',
                    'emal': 'Ethylmalonyl-CoA (C2, ethyl branch)',
                    'allylmal': 'Allylmalonyl-CoA (C2, allyl branch)',
                    'isobutmal': 'Isobutyrylmalonyl-CoA (C2, isobutyl branch)',
                    'D-isobutmal': 'D-Isobutyrylmalonyl-CoA (C2, D-isobutyl branch)',
                    'butmal': 'Butyrylmalonyl-CoA (C2, butyl branch)',
                    'hexmal': 'Hexylmalonyl-CoA (C2, hexyl branch)',
                    'hmal': 'Hydroxymalonyl-ACP (C2, hydroxy branch)',
                    'DCP': 'Dichloropropionyl-ACP',
                }
                substrate_desc = substrate_names.get(substrate, substrate)
                lines.append(f"        AT (Acyltransferase): {substrate_desc}")

        # KR domain (Ketoreductase) - determines stereochemistry
        if "KR{" in mod_str:
            import re
            kr_match = re.search(r"KR\{'type':\s*'([^']+)'", mod_str)
            if kr_match:
                kr_type = kr_match.group(1)
                kr_desc = {
                    'A1': 'A1 (L-hydroxyl, syn methyl)',
                    'A2': 'A2 (L-hydroxyl, anti methyl)',
                    'B1': 'B1 (D-hydroxyl, syn methyl)',
                    'B2': 'B2 (D-hydroxyl, anti methyl)',
                    'B': 'B (D-hydroxyl)',
                    'C1': 'C1 (inactive, syn methyl)',
                    'C2': 'C2 (inactive, anti methyl)',
                }.get(kr_type, kr_type)
                lines.append(f"        KR (Ketoreductase): {kr_desc}")
            else:
                lines.append(f"        KR (Ketoreductase): present")

        # DH domain (Dehydratase) - removes water to form double bond
        if "DH{}" in mod_str:
            lines.append(f"        DH (Dehydratase): active (forms C=C double bond)")

        # ER domain (Enoylreductase) - reduces double bond
        if "ER{}" in mod_str:
            lines.append(f"        ER (Enoylreductase): active (reduces C=C to C-C)")

        # Loading module
        if "loading: True" in mod_str:
            lines.append(f"        Loading Module: Yes (starter unit)")
        elif "loading: False" in mod_str:
            lines.append(f"        Loading Module: No (extension module)")

        return "\n".join(lines)

    @staticmethod
    def format_pks_cluster(cluster, score: float) -> str:
        """
        Format a full PKS cluster with all module details.

        Args:
            cluster: A bcs.Cluster object
            score: The design score

        Returns:
            Formatted string describing the full PKS design.
        """
        lines = [
            f"    Design Score: {score:.4f}",
        ]

        if hasattr(cluster, 'modules'):
            num_modules = len(cluster.modules)
            lines.append(f"    Number of PKS Modules: {num_modules}")
            lines.append(f"    PKS Assembly Line:")

            for k, mod in enumerate(cluster.modules):
                lines.append(DORAnetMCTS.format_pks_module(mod, k + 1))

        return "\n".join(lines)

    @property
    def retrotide_runs(self) -> List:
        """Backwards compatibility: return list of RetroTide agents."""
        return [r.retrotide_agent for r in self.retrotide_results if r.retrotide_agent]

    def get_successful_results(self) -> List[RetroTideResult]:
        """Return only the RetroTide results that found successful PKS designs."""
        return [r for r in self.retrotide_results if r.retrotide_successful]

    def get_results_summary(self) -> str:
        """Return a detailed summary of all RetroTide results."""
        lines = [
            "",
            "=" * 70,
            "RetroTide Results Summary",
            "=" * 70,
            "",
            f"Total RetroTide searches: {len(self.retrotide_results)}",
            f"Successful searches: {len(self.get_successful_results())}",
            "",
        ]

        # Group by success
        successful = self.get_successful_results()
        unsuccessful = [r for r in self.retrotide_results if not r.retrotide_successful]

        if successful:
            lines.append("✅ SUCCESSFUL PKS DESIGNS:")
            lines.append("-" * 70)
            for r in successful:
                lines.append(f"  DORAnet Node {r.doranet_node_id}:")
                lines.append(f"    Fragment SMILES: {r.doranet_node_smiles}")
                lines.append(f"    Depth: {r.doranet_node_depth}, Provenance: {r.doranet_node_provenance}")
                lines.append(f"    RetroTide Target: {r.retrotide_target_smiles}")
                lines.append(f"    Successful Nodes: {r.retrotide_num_successful_nodes}")
                lines.append(f"    Best Score: {r.retrotide_best_score:.3f}")
                lines.append(f"    Total RetroTide Nodes: {r.retrotide_total_nodes}")
                lines.append("")

        if unsuccessful:
            lines.append("❌ UNSUCCESSFUL SEARCHES (top 5 by score):")
            lines.append("-" * 70)
            # Sort by best score descending
            sorted_unsuccessful = sorted(unsuccessful, key=lambda r: r.retrotide_best_score, reverse=True)
            for r in sorted_unsuccessful[:5]:
                lines.append(f"  DORAnet Node {r.doranet_node_id}: {r.doranet_node_smiles[:40]}...")
                lines.append(f"    Provenance: {r.doranet_node_provenance}, Best Score: {r.retrotide_best_score:.3f}")
            lines.append("")

        return "\n".join(lines)

    def save_results(self, output_path: str) -> None:
        """
        Save detailed results to a log file.

        Args:
            output_path: Path to the output file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # Header
            f.write("=" * 70 + "\n")
            f.write("DORAnet + RetroTide Multi-Agent MCTS Results\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n\n")

            # Target info
            f.write("TARGET MOLECULE\n")
            f.write("-" * 70 + "\n")
            f.write(f"SMILES: {Chem.MolToSmiles(self.target_molecule)}\n\n")

            # DORAnet configuration
            f.write("DORANET CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total iterations: {self.total_iterations}\n")
            f.write(f"Max depth: {self.max_depth}\n")
            f.write(f"Use enzymatic: {self.use_enzymatic}\n")
            f.write(f"Use synthetic: {self.use_synthetic}\n")
            f.write(f"Max children per expand: {self.max_children_per_expand}\n")
            f.write(f"Spawn RetroTide: {self.spawn_retrotide}\n\n")

            # DORAnet tree
            f.write("DORANET SEARCH TREE\n")
            f.write("-" * 70 + "\n")
            f.write(self.get_tree_summary() + "\n\n")

            # PKS Library match summary for all DORAnet-generated precursors
            f.write("=" * 70 + "\n")
            f.write("DORANET PRECURSORS - PKS LIBRARY MATCHES\n")
            f.write("=" * 70 + "\n\n")

            if self.pks_library:
                f.write(f"PKS Library Size: {len(self.pks_library)} molecules\n\n")

                # Separate matches and non-matches
                pks_matches = [n for n in self.nodes if self.calculate_reward(n) > 0]
                non_matches = [n for n in self.nodes if self.calculate_reward(n) == 0]

                f.write(f"Total DORAnet nodes: {len(self.nodes)}\n")
                f.write(f"Nodes matching PKS library: {len(pks_matches)}\n")
                f.write(f"Nodes NOT in PKS library: {len(non_matches)}\n\n")

                # List PKS matches with pathways
                if pks_matches:
                    f.write("✅ PRECURSORS FOUND IN PKS LIBRARY:\n")
                    f.write("-" * 70 + "\n")
                    for node in pks_matches:
                        f.write(f"  Node {node.node_id} (depth={node.depth}, {node.provenance}): {node.smiles}\n")
                    f.write("\n")

                    # Detailed pathways for each PKS match
                    f.write("=" * 70 + "\n")
                    f.write("DORANET PATHWAYS TO PKS-SYNTHESIZABLE FRAGMENTS\n")
                    f.write("=" * 70 + "\n\n")

                    for i, node in enumerate(pks_matches):
                        f.write(f"PATHWAY #{i + 1}: Node {node.node_id}\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Final Fragment: {node.smiles}\n")
                        f.write(f"Depth: {node.depth}, Provenance: {node.provenance}\n\n")
                        f.write("Retrosynthetic Route:\n")
                        f.write(self.format_pathway(node) + "\n\n")

                # List non-matches (only first 20 to avoid clutter)
                if non_matches:
                    f.write("❌ PRECURSORS NOT IN PKS LIBRARY:\n")
                    f.write("-" * 70 + "\n")
                    display_limit = min(20, len(non_matches))
                    for node in non_matches[:display_limit]:
                        smiles_display = node.smiles[:50] + "..." if node.smiles and len(node.smiles) > 50 else node.smiles
                        f.write(f"  Node {node.node_id} (depth={node.depth}, {node.provenance}): {smiles_display}\n")
                    if len(non_matches) > display_limit:
                        f.write(f"  ... and {len(non_matches) - display_limit} more\n")
                    f.write("\n")
            else:
                f.write("No PKS library loaded - skipping PKS match analysis.\n\n")

            # RetroTide results summary
            f.write(self.get_results_summary() + "\n")

            # Detailed successful results
            successful = self.get_successful_results()
            if successful:
                f.write("\n" + "=" * 70 + "\n")
                f.write("DETAILED SUCCESSFUL RETROTIDE RESULTS\n")
                f.write("=" * 70 + "\n\n")

                for i, r in enumerate(successful):
                    f.write(f"SUCCESS #{i + 1}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"DORAnet Node ID: {r.doranet_node_id}\n")
                    f.write(f"Fragment SMILES: {r.doranet_node_smiles}\n")
                    f.write(f"Fragment Depth: {r.doranet_node_depth}\n")
                    f.write(f"Fragment Provenance: {r.doranet_node_provenance}\n")

                    # Find the DORAnet node and show the pathway
                    doranet_node = next((n for n in self.nodes if n.node_id == r.doranet_node_id), None)
                    if doranet_node:
                        f.write("\n--- DORAnet Pathway (Target → Fragment) ---\n")
                        f.write(self.format_pathway(doranet_node) + "\n")

                    # DORAnet reaction that created this fragment
                    f.write("\n--- DORAnet Reaction (how fragment was created) ---\n")
                    if r.doranet_reaction_name:
                        f.write(f"Reaction Name: {r.doranet_reaction_name}\n")
                    if r.doranet_reaction_smarts:
                        # Truncate very long SMARTS
                        smarts_display = r.doranet_reaction_smarts[:100] + "..." if len(r.doranet_reaction_smarts) > 100 else r.doranet_reaction_smarts
                        f.write(f"Reaction SMARTS: {smarts_display}\n")
                    if r.doranet_reactants_smiles:
                        f.write(f"Reactants: {' + '.join(r.doranet_reactants_smiles[:3])}\n")
                    if r.doranet_products_smiles:
                        f.write(f"Products: {' + '.join(r.doranet_products_smiles[:3])}\n")

                    f.write("\n--- RetroTide Forward Synthesis Results ---\n")
                    f.write(f"RetroTide Target: {r.retrotide_target_smiles}\n")
                    f.write(f"Successful PKS Nodes: {r.retrotide_num_successful_nodes}\n")
                    f.write(f"Best Score: {r.retrotide_best_score:.4f}\n")
                    f.write(f"Total RetroTide Nodes Explored: {r.retrotide_total_nodes}\n")

                    # If we have the agent, try to get PKS design details
                    if r.retrotide_agent:
                        agent = r.retrotide_agent
                        succ_nodes = getattr(agent, 'successful_nodes', set())
                        if succ_nodes:
                            f.write("\n--- Successful PKS Designs (Exact Matches) ---\n")
                            f.write(f"Found {len(succ_nodes)} design(s) that exactly synthesize the target.\n")
                            for j, pks_node in enumerate(list(succ_nodes)[:5]):  # Limit to 5
                                f.write(f"\n  PKS Design #{j + 1}:\n")
                                if hasattr(pks_node, 'PKS_product') and pks_node.PKS_product:
                                    prod_smi = Chem.MolToSmiles(pks_node.PKS_product)
                                    f.write(f"    Product SMILES: {prod_smi}\n")
                                if hasattr(pks_node, 'PKS_design') and pks_node.PKS_design:
                                    f.write(f"    Design Depth (# modules): {pks_node.depth}\n")
                                    # Use the new formatting function
                                    try:
                                        cluster, score, _ = pks_node.PKS_design
                                        f.write(self.format_pks_cluster(cluster, score) + "\n")
                                    except Exception as e:
                                        f.write(f"    (Could not extract module details: {e})\n")

                        # Also show simulated successful designs with detailed formatting
                        sim_designs = getattr(agent, 'successful_simulated_designs', [])
                        if sim_designs:
                            f.write("\n--- Successful Simulated PKS Designs ---\n")
                            f.write(f"Found {len(sim_designs)} design(s) that reached target in simulation.\n")
                            for j, design in enumerate(sim_designs[:5]):  # Limit to 5
                                f.write(f"\n  Simulated Design #{j + 1}:\n")
                                # Format each module in the design
                                if isinstance(design, (list, tuple)):
                                    f.write(f"    Number of Modules: {len(design)}\n")
                                    for k, mod in enumerate(design):
                                        f.write(self.format_pks_module(mod, k + 1) + "\n")
                                else:
                                    f.write(f"    Modules: {design}\n")

                        # Show best designs even if not successful (for debugging)
                        if not succ_nodes and not sim_designs:
                            f.write("\n--- Best PKS Designs (No Exact Match) ---\n")
                            f.write(f"No exact matches found. Showing best scoring designs:\n")
                            all_nodes = getattr(agent, 'nodes', [])
                            # Sort by value/visits ratio
                            scored_nodes = [(n, n.value / max(n.visits, 1)) for n in all_nodes if hasattr(n, 'PKS_design') and n.PKS_design]
                            scored_nodes.sort(key=lambda x: x[1], reverse=True)
                            for j, (pks_node, avg_score) in enumerate(scored_nodes[:3]):
                                f.write(f"\n  Best Design #{j + 1} (avg score: {avg_score:.4f}):\n")
                                if hasattr(pks_node, 'PKS_product') and pks_node.PKS_product:
                                    prod_smi = Chem.MolToSmiles(pks_node.PKS_product)
                                    f.write(f"    Product SMILES: {prod_smi}\n")
                                if hasattr(pks_node, 'PKS_design') and pks_node.PKS_design:
                                    try:
                                        cluster, score, _ = pks_node.PKS_design
                                        f.write(self.format_pks_cluster(cluster, score) + "\n")
                                    except Exception as e:
                                        f.write(f"    (Could not extract details: {e})\n")

                    f.write("\n")

            # All results table
            f.write("\n" + "=" * 70 + "\n")
            f.write("ALL RETROTIDE RESULTS TABLE\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'Node ID':<10} {'Provenance':<12} {'Success':<10} {'Score':<10} {'Fragment SMILES'}\n")
            f.write("-" * 70 + "\n")

            for r in self.retrotide_results:
                success_str = "✅ YES" if r.retrotide_successful else "❌ NO"
                smiles_short = r.doranet_node_smiles[:35] + "..." if len(r.doranet_node_smiles) > 35 else r.doranet_node_smiles
                f.write(f"{r.doranet_node_id:<10} {r.doranet_node_provenance:<12} {success_str:<10} {r.retrotide_best_score:<10.3f} {smiles_short}\n")

            # Detailed PKS designs for ALL RetroTide searches (including unsuccessful)
            if self.retrotide_results:
                f.write("\n" + "=" * 70 + "\n")
                f.write("DETAILED PKS MODULE INFORMATION FOR ALL SEARCHES\n")
                f.write("=" * 70 + "\n\n")

                for r in self.retrotide_results:
                    f.write(f"Fragment: {r.doranet_node_smiles}\n")
                    f.write(f"DORAnet Node: {r.doranet_node_id} ({r.doranet_node_provenance})\n")
                    f.write(f"Best Score: {r.retrotide_best_score:.4f}\n")
                    f.write(f"Exact Match Found: {'Yes' if r.retrotide_successful else 'No'}\n")

                    # Show DORAnet pathway
                    doranet_node = next((n for n in self.nodes if n.node_id == r.doranet_node_id), None)
                    if doranet_node:
                        f.write(f"\nDORAnet Pathway:\n")
                        f.write(self.format_pathway(doranet_node) + "\n")

                    # Show PKS designs from RetroTide agent
                    if r.retrotide_agent:
                        agent = r.retrotide_agent

                        # Show successful exact match designs first
                        succ_nodes = getattr(agent, 'successful_nodes', set())
                        if succ_nodes:
                            f.write(f"\n✅ EXACT MATCH PKS DESIGNS:\n")
                            for j, pks_node in enumerate(list(succ_nodes)[:3]):
                                f.write(f"\n  Design #{j + 1}:\n")
                                if hasattr(pks_node, 'PKS_product') and pks_node.PKS_product:
                                    f.write(f"    Product: {Chem.MolToSmiles(pks_node.PKS_product)}\n")
                                if hasattr(pks_node, 'PKS_design') and pks_node.PKS_design:
                                    try:
                                        cluster, score, _ = pks_node.PKS_design
                                        f.write(self.format_pks_cluster(cluster, score) + "\n")
                                    except Exception:
                                        pass

                        # Show simulated successful designs
                        sim_designs = getattr(agent, 'successful_simulated_designs', [])
                        if sim_designs:
                            f.write(f"\n✅ SIMULATED SUCCESSFUL PKS DESIGNS:\n")
                            for j, design in enumerate(sim_designs[:3]):
                                f.write(f"\n  Simulated Design #{j + 1}:\n")
                                if isinstance(design, (list, tuple)):
                                    f.write(f"    Number of Modules: {len(design)}\n")
                                    for k, mod in enumerate(design):
                                        f.write(self.format_pks_module(mod, k + 1) + "\n")
                                else:
                                    f.write(f"    Modules: {design}\n")

                        # Show best designs even if no exact match
                        if not succ_nodes:
                            f.write(f"\n📊 BEST SCORING PKS DESIGNS (closest to target):\n")
                            all_nodes = getattr(agent, 'nodes', [])
                            scored_nodes = [
                                (n, n.value / max(n.visits, 1))
                                for n in all_nodes
                                if hasattr(n, 'PKS_design') and n.PKS_design
                            ]
                            scored_nodes.sort(key=lambda x: x[1], reverse=True)

                            for j, (pks_node, avg_score) in enumerate(scored_nodes[:3]):
                                f.write(f"\n  Design #{j + 1} (score: {avg_score:.4f}):\n")
                                if hasattr(pks_node, 'PKS_product') and pks_node.PKS_product:
                                    f.write(f"    Product: {Chem.MolToSmiles(pks_node.PKS_product)}\n")
                                if hasattr(pks_node, 'PKS_design') and pks_node.PKS_design:
                                    try:
                                        cluster, score, _ = pks_node.PKS_design
                                        f.write(self.format_pks_cluster(cluster, score) + "\n")
                                    except Exception as e:
                                        f.write(f"    (Could not extract details)\n")

                    f.write("\n" + "-" * 70 + "\n\n")

        print(f"[DORAnet] Results saved to: {path}")

        # Generate visualizations if enabled
        if self.enable_visualization:
            self._generate_visualizations(output_path)

    def _generate_visualizations(self, results_path: str) -> None:
        """
        Generate tree and pathway visualizations.

        Args:
            results_path: Path to the results text file (used to derive viz filenames).
        """
        try:
            from .visualize import (
                visualize_doranet_tree,
                visualize_pks_pathways,
                create_enhanced_interactive_html
            )

            print("\n[DORAnet] Generating visualizations...")

            # Derive visualization paths from results path
            results_path_obj = Path(results_path)
            base_name = results_path_obj.stem  # filename without extension
            viz_dir = Path(self.visualization_output_dir)
            viz_dir.mkdir(parents=True, exist_ok=True)

            # Full tree visualization
            tree_viz_path = viz_dir / f"{base_name}_tree.png"
            visualize_doranet_tree(self, output_path=str(tree_viz_path))
            print(f"[DORAnet] Tree visualization saved to: {tree_viz_path}")

            # PKS pathways visualization
            pks_viz_path = viz_dir / f"{base_name}_pks_pathways.png"
            visualize_pks_pathways(self, output_path=str(pks_viz_path))
            print(f"[DORAnet] PKS pathways visualization saved to: {pks_viz_path}")

            # Interactive HTML visualization (if enabled)
            if self.enable_interactive_viz:
                interactive_path = viz_dir / f"{base_name}_interactive.html"
                create_enhanced_interactive_html(
                    self,
                    output_path=str(interactive_path),
                    auto_open=self.auto_open_viz
                )
                print(f"[DORAnet] Interactive HTML saved to: {interactive_path}")

        except ImportError as e:
            print(f"[DORAnet] Could not generate visualizations: {e}")
            print("[DORAnet] Install required packages: pip install networkx matplotlib bokeh")
        except Exception as e:
            print(f"[DORAnet] Error generating visualizations: {e}")

    def _generate_iteration_visualization(self, iteration: int) -> None:
        """
        Generate tree visualizations for a specific iteration to show MCTS dynamics.

        This creates iteration-specific visualizations showing the current state of
        the tree, which nodes have been selected, and how the tree is growing.

        Args:
            iteration: The current iteration number (0-indexed).
        """
        if not self.visualization_output_dir:
            return

        try:
            from .visualize import (
                visualize_doranet_tree,
                create_enhanced_interactive_html
            )

            # Create iterations subdirectory
            viz_dir = Path(self.visualization_output_dir)
            iterations_dir = viz_dir / "iterations"
            iterations_dir.mkdir(parents=True, exist_ok=True)

            # Zero-padded iteration number for proper sorting
            iter_str = f"{iteration:05d}"

            # Generate static tree visualization
            tree_viz_path = iterations_dir / f"iteration_{iter_str}_tree.png"
            visualize_doranet_tree(
                self,
                output_path=str(tree_viz_path),
                title=f"DORAnet Tree - Iteration {iteration + 1}/{self.total_iterations}"
            )

            # Generate interactive HTML visualization if enabled
            if self.enable_interactive_viz:
                interactive_path = iterations_dir / f"iteration_{iter_str}_interactive.html"
                create_enhanced_interactive_html(
                    self,
                    output_path=str(interactive_path),
                    auto_open=self.auto_open_iteration_viz,  # Auto-open iteration visualizations
                    title=f"DORAnet Tree - Iteration {iteration + 1}/{self.total_iterations}"
                )

            # Print status every 10 iterations or at specified intervals
            if (iteration + 1) % max(10, self.iteration_viz_interval) == 0:
                print(f"[DORAnet] Generated iteration {iteration + 1} visualizations")

        except ImportError as e:
            # Only print warning once
            if not hasattr(self, '_viz_warning_shown'):
                print(f"[DORAnet] Could not generate iteration visualizations: {e}")
                print("[DORAnet] Install required packages: pip install networkx matplotlib bokeh")
                self._viz_warning_shown = True
        except Exception as e:
            print(f"[DORAnet] Error generating iteration {iteration} visualization: {e}")