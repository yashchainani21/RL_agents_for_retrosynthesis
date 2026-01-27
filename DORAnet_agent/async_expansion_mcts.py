"""
Async expansion MCTS for DORAnet using multiprocessing.

This class offloads the DORAnet expansion step to worker processes while
keeping selection, reward, and tree mutation on the main process.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import uuid
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

from .mcts import (
    DORAnetMCTS,
    RetroTideResult,
    _canonicalize_smiles,
)
from .node import Node
from .policies import (
    RolloutPolicy,
    RewardPolicy,
    NoOpRolloutPolicy,
    SpawnRetroTideOnDatabaseCheck,
    SparseTerminalRewardPolicy,
)


RDLogger.DisableLog("rdApp.*")


def _expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Worker process expansion for a single starter SMILES.

    Returns a list of fragment dicts with reaction metadata and provenance.
    """
    import doranet.modules.enzymatic as enzymatic
    import doranet.modules.synthetic as synthetic

    starter_smiles = payload["starter_smiles"]
    modes = payload["modes"]
    generations_per_expand = payload["generations_per_expand"]
    max_children_per_expand = payload["max_children_per_expand"]
    child_downselection_strategy = payload["child_downselection_strategy"]
    excluded_fragments = set(payload["excluded_fragments"])
    chemistry_helpers = set(payload["chemistry_helpers"])
    pks_library = set(payload["pks_library"])
    sink_compounds = set(payload["sink_compounds"])
    target_mw = payload["target_mw"]
    fragment_cache_dir = payload.get("fragment_cache_dir")

    mol = Chem.MolFromSmiles(starter_smiles)
    if mol is None:
        return []

    # Lazy init scorers for most_thermo_feasible strategy
    dora_model = None
    pathermo_Hf = None
    if child_downselection_strategy == "most_thermo_feasible":
        # Initialize DORA-XGB for enzymatic reactions
        try:
            from DORA_XGB import DORA_XGB
            dora_model = DORA_XGB.feasibility_classifier(
                cofactor_positioning='by_descending_MW'
            )
        except Exception:
            dora_model = None

        # Initialize pathermo for thermodynamic scoring
        try:
            from pathermo.properties import Hf as pathermo_Hf_import
            pathermo_Hf = pathermo_Hf_import
        except Exception:
            pathermo_Hf = None

    def _compute_frag_feasibility(
        frag: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        """
        Compute unified feasibility score for fragment.

        For enzymatic: Uses DORA-XGB probability if available, else sigmoid(ΔH).
        For synthetic: Uses sigmoid-transformed ΔH.
        Unknown: Default to 1.0.

        Returns the fragment dict with score fields added.
        """
        if child_downselection_strategy != "most_thermo_feasible":
            return frag

        reactants = frag.get("reactants_smiles", [])
        products = frag.get("products_smiles", [])

        dora_score = None
        dora_label = None
        delta_h = None
        thermo_label = None

        # Score DORA-XGB for enzymatic reactions
        if mode == "enzymatic" and dora_model is not None:
            try:
                # DORAnet stores reactions in retro direction
                # For forward direction: products_smiles -> reactants_smiles
                rxn_str = ".".join(products) + ">>" + ".".join(reactants)
                prob = float(dora_model.predict_proba(rxn_str))
                dora_score = prob
                dora_label = 1 if prob >= 0.5 else 0
            except Exception:
                pass

        # Score thermodynamic feasibility (both enzymatic and synthetic)
        if pathermo_Hf is not None:
            try:
                # DORAnet stores reactions in retro direction
                # For forward: products_smiles (stored) become reactants
                #              reactants_smiles (stored) become products
                # ΔH = Σ(Hf products) - Σ(Hf reactants)
                products_hf = sum(float(pathermo_Hf(s)) for s in reactants)
                reactants_hf = sum(float(pathermo_Hf(s)) for s in products)
                delta_h = products_hf - reactants_hf
                thermo_label = 1 if delta_h < 15.0 else 0
            except Exception:
                pass

        # Compute unified feasibility score (0-1 scale)
        if mode == "enzymatic" and dora_score is not None:
            feasibility_score = dora_score
        elif delta_h is not None:
            # Sigmoid transform: score = 1 / (1 + exp(0.2 * (ΔH - 15)))
            feasibility_score = 1.0 / (1.0 + math.exp(0.2 * (delta_h - 15.0)))
        else:
            feasibility_score = 1.0  # Unknown, assume feasible

        frag["feasibility_score"] = feasibility_score
        frag["dora_xgb_score"] = dora_score
        frag["dora_xgb_label"] = dora_label
        frag["enthalpy_of_reaction"] = delta_h
        frag["thermodynamic_label"] = thermo_label

        return frag

    def _downselect_fragments(
        fragments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if len(fragments) <= max_children_per_expand:
            return fragments
        if child_downselection_strategy == "first_N":
            return fragments[:max_children_per_expand]

        if child_downselection_strategy == "hybrid":
            scored: List[Tuple[float, int, Dict[str, Any]]] = []
            for idx, frag in enumerate(fragments):
                score = 0.0
                smiles = frag["smiles"]
                if smiles in sink_compounds:
                    score += 1000.0
                elif smiles in pks_library:
                    score += 500.0

                frag_mw = frag.get("mw")
                if frag_mw is None:
                    frag_mol = Chem.MolFromSmiles(smiles)
                    if frag_mol is not None:
                        frag_mw = Descriptors.MolWt(frag_mol)
                if frag_mw is not None:
                    mw_score = max(0, 100 * (1 - frag_mw / max(target_mw, 1)))
                    score += mw_score

                scored.append((score, -idx, frag))

            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return [frag for _, _, frag in scored[:max_children_per_expand]]

        elif child_downselection_strategy == "most_thermo_feasible":
            # Score fragments by thermodynamic feasibility with priority bonuses
            scored: List[Tuple[float, int, Dict[str, Any]]] = []
            for idx, frag in enumerate(fragments):
                # Use pre-computed feasibility score, default to 1.0 if not available
                feas_score = frag.get("feasibility_score", 1.0)
                if feas_score is None:
                    feas_score = 1.0

                # Priority bonuses (same as hybrid) to preserve terminal prioritization
                smiles = frag["smiles"]
                if smiles in sink_compounds:
                    score = feas_score + 1000.0
                elif smiles in pks_library:
                    score = feas_score + 500.0
                else:
                    score = feas_score

                scored.append((score, -idx, frag))

            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return [frag for _, _, frag in scored[:max_children_per_expand]]

        else:
            # Unknown strategy, fall back to first_N
            return fragments[:max_children_per_expand]

    def _generate_fragments_for_mode(mode: str) -> List[Dict[str, Any]]:
        if mode == "enzymatic":
            module = enzymatic
        else:
            module = synthetic

        job_name = f"doranet_{mode}_retro_{uuid.uuid4().hex[:8]}"

        cache_file = None
        if fragment_cache_dir:
            excluded_hash = hashlib.md5(
                "\n".join(sorted(excluded_fragments)).encode()
            ).hexdigest()[:12]
            helpers_hash = hashlib.md5(
                "\n".join(sorted(chemistry_helpers)).encode()
            ).hexdigest()[:12]
            cache_key = hashlib.md5(
                f"{starter_smiles}|{mode}|gen={generations_per_expand}"
                f"|max_children={max_children_per_expand}|strategy={child_downselection_strategy}"
                f"|excluded={excluded_hash}|helpers={helpers_hash}|target_mw={target_mw:.3f}".encode()
            ).hexdigest()[:16]
            cache_dir = Path(fragment_cache_dir)
            cache_file = cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    cached = pickle.loads(cache_file.read_bytes())
                    return cached
                except Exception:
                    pass

        try:
            network_kwargs = {
                "job_name": job_name,
                "starters": {starter_smiles},
                "gen": generations_per_expand,
                "direction": "retro",
            }
            if mode == "synthetic" and chemistry_helpers:
                network_kwargs["helpers"] = chemistry_helpers
            network = module.generate_network(**network_kwargs)
        except Exception:
            return []

        mol_to_reaction: Dict[str, Dict[str, Any]] = {}
        mols_list = list(network.mols)
        ops_list = list(network.ops)

        for rxn in getattr(network, "rxns", []):
            try:
                op_idx = rxn.operator
                op = ops_list[op_idx] if op_idx < len(ops_list) else None
                if op and hasattr(op, "uid"):
                    rxn_smarts = op.uid
                else:
                    rxn_smarts = str(op) if op else None

                # Get human-readable reaction label from network metadata
                rxn_label = None
                if op_idx is not None:
                    try:
                        meta = network.ops.meta(op_idx)
                        rxn_label = meta.get('name')
                    except Exception:
                        pass
                # Fallback to truncated SMARTS if no label found
                if not rxn_label and rxn_smarts:
                    rxn_label = rxn_smarts[:60] + "..." if len(rxn_smarts) > 60 else rxn_smarts

                reactant_idxs = rxn.reactants
                product_idxs = rxn.products
                reactant_smiles = [
                    str(getattr(mols_list[i], "uid", mols_list[i]))
                    for i in reactant_idxs if i < len(mols_list)
                ]
                product_smiles = [
                    str(getattr(mols_list[i], "uid", mols_list[i]))
                    for i in product_idxs if i < len(mols_list)
                ]

                for prod in product_smiles:
                    canonical_prod = _canonicalize_smiles(prod)
                    if canonical_prod and canonical_prod not in mol_to_reaction:
                        mol_to_reaction[canonical_prod] = {
                            "smarts": rxn_smarts,
                            "label": rxn_label,
                            "reactants": reactant_smiles,
                            "products": product_smiles,
                        }
            except Exception:
                continue

        fragments: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        starter_canonical = _canonicalize_smiles(starter_smiles)

        for mol_obj in network.mols:
            mol_smiles = getattr(mol_obj, "uid", None)
            if mol_smiles is None:
                continue
            mol_smiles_str = str(mol_smiles)
            if "*" in mol_smiles_str:
                continue
            canonical = _canonicalize_smiles(mol_smiles_str)
            if canonical is None:
                continue
            if canonical == starter_canonical:
                continue
            if canonical in excluded_fragments:
                continue
            if canonical in seen:
                continue

            rd_mol = Chem.MolFromSmiles(canonical)
            if rd_mol is None:
                continue

            seen.add(canonical)
            rxn_info = mol_to_reaction.get(canonical, {})
            frag = {
                "smiles": canonical,
                "reaction_smarts": rxn_info.get("smarts"),
                "reaction_name": rxn_info.get("label"),
                "reactants_smiles": rxn_info.get("reactants", []),
                "products_smiles": rxn_info.get("products", []),
                "provenance": mode,
                "mw": Descriptors.MolWt(rd_mol),
            }

            # Compute feasibility scores for most_thermo_feasible strategy
            frag = _compute_frag_feasibility(frag, mode)

            fragments.append(frag)

            if child_downselection_strategy == "first_N":
                if len(fragments) >= max_children_per_expand:
                    break

        fragments = _downselect_fragments(fragments)
        for frag in fragments:
            frag.pop("mw", None)

        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(pickle.dumps(fragments))
            except Exception:
                pass

        return fragments

    all_fragments: List[Dict[str, Any]] = []
    for mode in modes:
        all_fragments.extend(_generate_fragments_for_mode(mode))

    return all_fragments


class AsyncExpansionDORAnetMCTS(DORAnetMCTS):
    """
    DORAnet MCTS with asynchronous expansion using multiprocessing.

    Expansion is offloaded to worker processes while selection, rollout,
    reward calculation, and backpropagation happen on the main process.

    Supports the same rollout_policy and reward_policy parameters as
    DORAnetMCTS. Rollouts are performed on the main process after
    expansion results are integrated (Option B architecture).
    """

    def __init__(
        self,
        root: Node,
        target_molecule: Chem.Mol,
        num_workers: Optional[int] = None,
        max_inflight_expansions: Optional[int] = None,
        reward_fn: Optional[Callable[[Node], float]] = None,  # Deprecated
        **kwargs,
    ) -> None:
        """
        Args:
            root: Starting node containing the target molecule.
            target_molecule: The molecule to fragment.
            num_workers: Number of worker processes. Defaults to CPU count - 1.
            max_inflight_expansions: Max concurrent expansions. Defaults to num_workers.
            reward_fn: DEPRECATED. Use reward_policy instead via kwargs.
            **kwargs: Additional arguments passed to DORAnetMCTS.__init__().
        """
        super().__init__(root=root, target_molecule=target_molecule, **kwargs)
        self.num_workers = self._get_optimal_workers(num_workers)
        self.max_inflight_expansions = max_inflight_expansions or self.num_workers
        self._pending: Dict[Any, Dict[str, Any]] = {}

        # Handle deprecated reward_fn parameter
        if reward_fn is not None:
            print("[DEPRECATED] reward_fn parameter is deprecated. Use reward_policy instead.")
            self._legacy_reward_fn = reward_fn
        else:
            self._legacy_reward_fn = None

    def _get_optimal_workers(self, requested_workers: Optional[int]) -> int:
        cpu_count = os.cpu_count() or 4
        max_workers = max(1, cpu_count - 1)
        if requested_workers is None or requested_workers == 0:
            print(f"[AsyncExpansion] Using max available workers: {max_workers} (CPU count: {cpu_count})")
            return max_workers
        optimal = max(1, min(requested_workers, max_workers))
        if requested_workers > max_workers:
            print(f"[AsyncExpansion] Requested {requested_workers} workers, using {optimal} (CPU count: {cpu_count})")
        return optimal

    def select(self, node: Node) -> Optional[Node]:
        # Same as parent but skip nodes with expansion pending.
        UNVISITED_BASE_SCORE = 1000.0

        while node.children:
            best_node: Optional[Node] = None
            best_score = -math.inf
            log_parent_visits = math.log(max(node.visits, 1))

            for child in node.children:
                if child.is_sink_compound or child.is_pks_terminal or child.is_expansion_pending:
                    continue

                if self.selection_policy == "UCB1":
                    if child.visits == 0:
                        score = math.inf
                    else:
                        exploit = child.value / child.visits
                        explore = math.sqrt(2 * log_parent_visits / child.visits)
                        score = exploit + explore
                else:
                    depth_bonus = self.depth_bonus_coefficient * child.depth
                    if child.visits == 0:
                        score = UNVISITED_BASE_SCORE + depth_bonus
                    else:
                        exploit = child.value / child.visits
                        explore = math.sqrt(2 * log_parent_visits / child.visits)
                        score = exploit + explore + depth_bonus

                child.selection_score = score
                if score > best_score:
                    best_score = score
                    best_node = child

            if best_node is None:
                return None

            node = best_node

        if node.is_sink_compound or node.is_pks_terminal or node.is_expansion_pending:
            return None

        return node

    def _build_expansion_payload(self, leaf: Node) -> Dict[str, Any]:
        modes: List[str] = []
        if self.use_enzymatic:
            modes.append("enzymatic")
        if self.use_synthetic:
            modes.append("synthetic")

        return {
            "starter_smiles": leaf.smiles or "",
            "modes": modes,
            "generations_per_expand": self.generations_per_expand,
            "max_children_per_expand": self.max_children_per_expand,
            "child_downselection_strategy": self.child_downselection_strategy,
            "excluded_fragments": list(self.excluded_fragments),
            "chemistry_helpers": list(self.chemistry_helpers),
            "bio_cofactors": list(self.bio_cofactors),
            "pks_library": list(self.pks_library),
            "sink_compounds": list(self.sink_compounds),
            "target_mw": self.target_MW,
            "fragment_cache_dir": str(self.fragment_cache_dir),
        }

    def _submit_expansion(self, leaf: Node, iteration: int, executor: ProcessPoolExecutor) -> None:
        payload = self._build_expansion_payload(leaf)
        future = executor.submit(_expand_worker, payload)
        leaf.is_expansion_pending = True
        self._pending[future] = {"leaf": leaf, "iteration": iteration}

    def _integrate_expansion_results(
        self, leaf: Node, fragments: List[Dict[str, Any]], iteration: int
    ) -> None:
        """
        Integrate expansion results from a worker process.

        Creates child nodes from fragments, then applies rollout and reward
        policies on the main process.

        Args:
            leaf: The parent node that was expanded.
            fragments: List of fragment dictionaries from the worker.
            iteration: The iteration number when expansion was submitted.
        """
        new_children: List[Node] = []
        context = self._build_rollout_context()

        for frag in fragments:
            smiles = frag["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            if self._is_prohibited_chemical(smiles):
                print(f"[DORAnet] Fragment {smiles} is a PROHIBITED CHEMICAL - skipping")
                continue

            if self._exceeds_MW_threshold(mol):
                frag_mw = Descriptors.MolWt(mol)
                print(f"[DORAnet] Fragment {smiles} exceeds MW threshold "
                      f"({frag_mw:.1f} > {self.max_fragment_MW:.1f}) - skipping")
                continue

            child = Node(
                fragment=mol,
                parent=leaf,
                provenance=frag.get("provenance"),
                reaction_smarts=frag.get("reaction_smarts"),
                reaction_name=frag.get("reaction_name"),
                reactants_smiles=frag.get("reactants_smiles", []),
                products_smiles=frag.get("products_smiles", []),
            )

            provenance = frag.get("provenance")

            # Check if scores were pre-computed by worker (for most_thermo_feasible)
            if frag.get("feasibility_score") is not None:
                # Use pre-computed scores
                child.feasibility_score = frag["feasibility_score"]
                child.feasibility_label = frag.get("dora_xgb_label")
                child.enthalpy_of_reaction = frag.get("enthalpy_of_reaction")
                child.thermodynamic_label = frag.get("thermodynamic_label")
            else:
                # Compute scores (for other strategies)
                # Score feasibility for enzymatic reactions using DORA-XGB
                if provenance == "enzymatic":
                    score, label = self.feasibility_scorer.score_reaction(
                        reactants_smiles=frag.get("reactants_smiles", []),
                        products_smiles=frag.get("products_smiles", []),
                        provenance=provenance
                    )
                    child.feasibility_score = score
                    child.feasibility_label = label

                # Score thermodynamic feasibility using pathermo (both enzymatic and synthetic)
                delta_h, thermo_label = self.thermodynamic_scorer.score_reaction(
                    reactants_smiles=frag.get("reactants_smiles", []),
                    products_smiles=frag.get("products_smiles", []),
                    provenance=provenance
                )
                child.enthalpy_of_reaction = delta_h
                child.thermodynamic_label = thermo_label

            child.created_at_iteration = iteration
            leaf.add_child(child)
            self.nodes.append(child)
            self.edges.append((leaf.node_id, child.node_id))
            new_children.append(child)

            # Check if this fragment is a sink compound (known terminal)
            sink_type = self._get_sink_compound_type(smiles)
            if sink_type:
                child.is_sink_compound = True
                child.sink_compound_type = sink_type
                child.expanded = True
                type_label = "BIOLOGICAL" if sink_type == "biological" else "CHEMICAL"
                print(f"[DORAnet] Fragment {smiles} is a {type_label} BUILDING BLOCK")
                # Note: reward calculated below via policy

        leaf.expanded = True
        leaf.expanded_at_iteration = iteration
        leaf.is_expansion_pending = False

        # Apply rollout and reward policies to each child (on main process)
        for child in new_children:
            # Check if this node matches PKS library (potential for RetroTide verification)
            is_pks_library_match = self._is_in_pks_library(child.smiles or "")

            if is_pks_library_match:
                # PKS library matches: always run rollout (even if sink compound)
                print(f"[DORAnet] Fragment {child.smiles} is PKS library match - "
                      f"attempting RetroTide (sink={child.is_sink_compound})")

                result = self.rollout_policy.rollout(child, context)

                if result.terminal:
                    child.is_pks_terminal = True
                    child.expanded = True

                    if "retrotide_agent" in result.metadata:
                        self._store_retrotide_result_from_rollout(child, result)

                    reward = result.reward
                else:
                    # RetroTide failed - fall back to sink compound reward if applicable
                    if child.is_sink_compound:
                        reward = self.reward_policy.calculate_reward(child, context)
                    else:
                        reward = result.reward

            elif child.is_sink_compound:
                # Pure sink compound (not in PKS library) - use reward policy directly
                reward = self.reward_policy.calculate_reward(child, context)
            else:
                # Non-sink, non-PKS: standard rollout
                result = self.rollout_policy.rollout(child, context)

                if result.terminal:
                    child.is_pks_terminal = True
                    child.expanded = True

                    if "retrotide_agent" in result.metadata:
                        self._store_retrotide_result_from_rollout(child, result)

                reward = result.reward

            self.backpropagate(child, reward)

    def _drain_completed(self, block: bool) -> None:
        if not self._pending:
            return
        futures = list(self._pending.keys())
        done, _ = wait(
            futures,
            timeout=None if block else 0,
            return_when=FIRST_COMPLETED,
        )
        if not done:
            return
        for future in done:
            info = self._pending.pop(future, None)
            if not info:
                continue
            leaf = info["leaf"]
            iteration = info["iteration"]
            try:
                fragments = future.result()
            except Exception as exc:
                print(f"[AsyncExpansion] Expansion failed for node {leaf.node_id}: {exc}")
                leaf.is_expansion_pending = False
                continue
            self._integrate_expansion_results(leaf, fragments, iteration)

    def run(self) -> None:
        """
        Execute async-expansion MCTS using a multiprocessing pool.

        Expansion is offloaded to worker processes. Rollouts and reward
        calculations happen on the main process after integration.
        """
        print(f"[AsyncExpansion] Starting async MCTS with {self.num_workers} workers, "
              f"{self.total_iterations} iterations, max_inflight={self.max_inflight_expansions}")
        print(f"[AsyncExpansion] max_depth={self.max_depth}")

        self.current_iteration = 0

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for iteration in range(self.total_iterations):
                self.current_iteration = iteration

                # Drain completed expansions without blocking.
                self._drain_completed(block=False)

                # If inflight is full, wait for one expansion to finish.
                if len(self._pending) >= self.max_inflight_expansions:
                    self._drain_completed(block=True)

                leaf = self.select(self.root)
                if leaf is None:
                    if self._pending:
                        self._drain_completed(block=True)
                        continue
                    print(f"[AsyncExpansion] No valid leaf found at iteration {iteration}, stopping.")
                    break

                leaf.selected_at_iterations.append(iteration)

                # Handle legacy reward_fn if provided (deprecated)
                if self._legacy_reward_fn is not None:
                    reward = self._legacy_reward_fn(leaf)
                    self.backpropagate(leaf, reward)
                    # When using legacy reward_fn, skip normal depth check
                    continue

                if leaf.depth >= self.max_depth:
                    reward = self.calculate_reward(leaf)
                    self.backpropagate(leaf, reward)
                    continue

                if not leaf.expanded and not leaf.is_expansion_pending:
                    self._submit_expansion(leaf, iteration, executor)

            # Drain all remaining expansions before finishing.
            while self._pending:
                self._drain_completed(block=True)

        sink_count = len(self.get_sink_compounds())
        pks_terminal_count = len(self.get_pks_terminal_nodes())
        print(f"[AsyncExpansion] MCTS complete. Total nodes: {len(self.nodes)}, "
              f"PKS terminals: {pks_terminal_count}, Sink compounds: {sink_count}, "
              f"RetroTide results: {len(self.retrotide_results)}")
