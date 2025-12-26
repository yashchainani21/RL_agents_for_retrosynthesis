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
        spawn_retrotide: bool = True,
        retrotide_kwargs: Optional[Dict] = None,
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
            cofactors_file: Path to CSV file with cofactor SMILES to exclude.
            spawn_retrotide: Whether to spawn RetroTide searches for each fragment.
            retrotide_kwargs: Parameters passed to RetroTide MCTS agents.
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

        # Add cofactors from CSV file if provided
        if cofactors_file:
            cofactor_smiles = _load_cofactors_from_csv(cofactors_file)
            self.excluded_fragments.update(cofactor_smiles)

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
            network = module.generate_network(
                job_name=job_name,
                starters={starter_smiles},
                gen=self.generations_per_expand,
                direction="retro",
            )
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
                rxn_smarts = str(op) if op else None

                # Extract a readable name from the SMARTS
                rxn_name = None
                if rxn_smarts:
                    # Use first 50 chars of SMARTS as a name identifier
                    rxn_name = rxn_smarts[:50] if len(rxn_smarts) > 50 else rxn_smarts

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
                            'name': rxn_name,
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
                reaction_name=rxn_info.get('name'),
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

    def expand(self, node: Node) -> List[Node]:
        """
        Expand a node by applying DORAnet retro-transformations.

        For each fragment generated, creates a child node and spawns
        a RetroTide forward MCTS search.

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

            # Optionally spawn RetroTide forward search for this fragment
            if self.spawn_retrotide:
                self._launch_retrotide_agent(target=frag_info.molecule, source_node=child)

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

    def run(self) -> None:
        """
        Execute the simplified MCTS loop: Selection → Expansion.

        Each expansion spawns RetroTide searches for all generated fragments.
        """
        print(f"[DORAnet] Starting MCTS with {self.total_iterations} iterations, max_depth={self.max_depth}")

        for iteration in range(self.total_iterations):
            # Selection: find a leaf node
            leaf = self.select(self.root)
            if leaf is None:
                print(f"[DORAnet] No valid leaf found at iteration {iteration}, stopping.")
                break

            # Check depth limit
            if leaf.depth >= self.max_depth:
                # Update visit count even if we can't expand further
                leaf.visits += 1
                continue

            # Expansion: generate fragments and spawn RetroTide for each
            if not leaf.expanded:
                new_children = self.expand(leaf)
                print(f"[DORAnet] Iteration {iteration}: expanded node {leaf.node_id}, "
                      f"created {len(new_children)} children")

            # Update visit count for the expanded node
            leaf.visits += 1

        print(f"[DORAnet] MCTS complete. Total nodes: {len(self.nodes)}, "
              f"RetroTide runs: {len(self.retrotide_runs)}")

    def get_tree_summary(self) -> str:
        """Return a summary of the search tree."""
        lines = ["DORAnet MCTS Tree Summary:", "=" * 40]
        for node in self.nodes:
            indent = "  " * node.depth
            lines.append(f"{indent}Node {node.node_id}: {node.smiles} "
                        f"(depth={node.depth}, visits={node.visits}, via={node.provenance})")
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

            # RetroTide results summary
            f.write(self.get_results_summary() + "\n")

            # Detailed successful results
            successful = self.get_successful_results()
            if successful:
                f.write("\n" + "=" * 70 + "\n")
                f.write("DETAILED SUCCESSFUL RESULTS\n")
                f.write("=" * 70 + "\n\n")

                for i, r in enumerate(successful):
                    f.write(f"SUCCESS #{i + 1}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"DORAnet Node ID: {r.doranet_node_id}\n")
                    f.write(f"Fragment SMILES: {r.doranet_node_smiles}\n")
                    f.write(f"Fragment Depth: {r.doranet_node_depth}\n")
                    f.write(f"Fragment Provenance: {r.doranet_node_provenance}\n")

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

                    f.write("\n--- RetroTide Results ---\n")
                    f.write(f"RetroTide Target: {r.retrotide_target_smiles}\n")
                    f.write(f"Successful PKS Nodes: {r.retrotide_num_successful_nodes}\n")
                    f.write(f"Best Score: {r.retrotide_best_score:.4f}\n")
                    f.write(f"Total RetroTide Nodes Explored: {r.retrotide_total_nodes}\n")

                    # If we have the agent, try to get PKS design details
                    if r.retrotide_agent:
                        agent = r.retrotide_agent
                        succ_nodes = getattr(agent, 'successful_nodes', set())
                        if succ_nodes:
                            f.write("\n--- Successful PKS Designs ---\n")
                            for j, node in enumerate(list(succ_nodes)[:3]):  # Limit to 3
                                f.write(f"\n  PKS Design #{j + 1}:\n")
                                if hasattr(node, 'PKS_product') and node.PKS_product:
                                    prod_smi = Chem.MolToSmiles(node.PKS_product)
                                    f.write(f"    Product SMILES: {prod_smi}\n")
                                if hasattr(node, 'PKS_design') and node.PKS_design:
                                    f.write(f"    Design Depth: {node.depth}\n")
                                    # Extract module information from PKS design
                                    try:
                                        cluster, score, _ = node.PKS_design
                                        f.write(f"    Design Score: {score:.4f}\n")
                                        if hasattr(cluster, 'modules'):
                                            f.write(f"    PKS Modules:\n")
                                            for k, mod in enumerate(cluster.modules):
                                                mod_str = str(mod)[:80]
                                                f.write(f"      Module {k}: {mod_str}\n")
                                    except Exception:
                                        pass
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

        print(f"[DORAnet] Results saved to: {path}")