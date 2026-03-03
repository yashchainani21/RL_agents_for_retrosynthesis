"""
Unit tests for AsyncExpansionDORAnetMCTS.
"""

from pathlib import Path
import sys

from concurrent.futures import Future
from typing import Any, Dict, List

import pytest
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent.async_expansion_mcts import AsyncExpansionDORAnetMCTS
from DORAnet_agent.node import Node


class _FakeExecutor:
    """Synchronous executor that returns already-completed futures."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs) -> Future:
        fut: Future = Future()
        try:
            result = fn(*args, **kwargs)
            fut.set_result(result)
        except Exception as exc:
            fut.set_exception(exc)
        return fut


@pytest.fixture
def sample_molecule():
    return Chem.MolFromSmiles("CCCCC(=O)O")


@pytest.fixture
def root_node(sample_molecule):
    Node.node_counter = 0
    return Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")


def test_select_skips_pending_node(root_node, sample_molecule):
    child = Node(fragment=sample_molecule, parent=root_node, depth=1, provenance="enzymatic")
    child.is_expansion_pending = True
    root_node.add_child(child)

    agent = AsyncExpansionDORAnetMCTS(
        root=root_node,
        target_molecule=sample_molecule,
        total_iterations=1,
        max_depth=1,
        use_enzymatic=False,
        use_synthetic=False,
        num_workers=1,
    )

    assert agent.select(agent.root) is None


def test_async_integration_adds_child(monkeypatch, root_node, sample_molecule):
    def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{
            "smiles": "CCO",
            "reaction_smarts": "dummy",
            "reaction_name": "dummy",
            "reactants_smiles": ["CCO"],
            "products_smiles": ["CCO"],
            "provenance": "enzymatic",
        }]

    monkeypatch.setattr(
        "DORAnet_agent.async_expansion_mcts._expand_worker",
        fake_expand_worker,
    )
    monkeypatch.setattr(
        "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
        _FakeExecutor,
    )

    agent = AsyncExpansionDORAnetMCTS(
        root=root_node,
        target_molecule=sample_molecule,
        total_iterations=1,
        max_depth=1,
        use_enzymatic=True,
        use_synthetic=False,
        num_workers=1,
    )

    agent.run()

    assert len(agent.nodes) == 2
    assert agent.nodes[1].smiles == "CCO"
    assert agent.root.expanded
    assert not agent.root.is_expansion_pending



# =============================================================================
# Tests for most_thermo_feasible downselection strategy
# =============================================================================


class TestMostThermoFeasibleStrategy:
    """Tests for most_thermo_feasible downselection strategy.

    Note: Since we monkeypatch _expand_worker, the downselection happens
    in the fake worker. These tests verify that:
    1. The strategy is accepted and properly configured
    2. Pre-computed scores from worker are preserved in nodes
    3. Multiple fragments returned by worker are all integrated
    """

    def test_strategy_accepted_and_configured(self, monkeypatch, sample_molecule):
        """The most_thermo_feasible strategy should be accepted without error."""
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            # Verify the strategy was passed to the worker
            assert payload["child_downselection_strategy"] == "most_thermo_feasible"
            # Return a single fragment (simulating downselection already happened)
            return [{
                "smiles": "CCC",
                "reaction_smarts": "smarts1",
                "reaction_name": "reaction1",
                "reactants_smiles": ["CCC"],
                "products_smiles": ["CCCCC(=O)O"],
                "provenance": "enzymatic",
                "feasibility_score": 0.9,
                "dora_xgb_score": 0.9,
                "dora_xgb_label": 1,
                "enthalpy_of_reaction": 5.0,
                "thermodynamic_label": 1,
            }]

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        # Should not raise an error
        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            max_children_per_expand=5,
            child_downselection_strategy="most_thermo_feasible",
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
        )

        assert agent.child_downselection_strategy == "most_thermo_feasible"
        agent.run()

        # Should have 2 nodes (root + 1 child)
        assert len(agent.nodes) == 2

    def test_multiple_fragments_all_integrated(self, monkeypatch, sample_molecule):
        """When worker returns multiple fragments, all should be integrated as nodes."""
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [
                {
                    "smiles": "CCC",
                    "reaction_smarts": "smarts1",
                    "reaction_name": "reaction1",
                    "reactants_smiles": ["CCC"],
                    "products_smiles": ["CCCCC(=O)O"],
                    "provenance": "enzymatic",
                    "feasibility_score": 0.9,
                    "dora_xgb_score": 0.9,
                    "dora_xgb_label": 1,
                    "enthalpy_of_reaction": 5.0,
                    "thermodynamic_label": 1,
                },
                {
                    "smiles": "CCCC",
                    "reaction_smarts": "smarts2",
                    "reaction_name": "reaction2",
                    "reactants_smiles": ["CCCC"],
                    "products_smiles": ["CCCCC(=O)O"],
                    "provenance": "enzymatic",
                    "feasibility_score": 0.3,
                    "dora_xgb_score": 0.3,
                    "dora_xgb_label": 0,
                    "enthalpy_of_reaction": 20.0,
                    "thermodynamic_label": 0,
                },
            ]

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            max_children_per_expand=5,  # Allow multiple
            child_downselection_strategy="most_thermo_feasible",
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
        )

        agent.run()

        # Should have 3 nodes (root + 2 children)
        assert len(agent.nodes) == 3

        # Verify both fragments were created as nodes with their scores
        child_smiles = {n.smiles for n in agent.nodes[1:]}
        assert "CCC" in child_smiles
        assert "CCCC" in child_smiles

    def test_precomputed_scores_override_recomputation(self, monkeypatch, sample_molecule):
        """Pre-computed scores from worker should be used instead of recomputing."""
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [{
                "smiles": "CCC",
                "reaction_smarts": "smarts1",
                "reaction_name": "reaction1",
                "reactants_smiles": ["CCC"],
                "products_smiles": ["CCCCC(=O)O"],
                "provenance": "enzymatic",
                # Pre-computed scores that would be different from recomputation
                "feasibility_score": 0.12345,
                "dora_xgb_score": 0.12345,
                "dora_xgb_label": 0,
                "enthalpy_of_reaction": 99.99,
                "thermodynamic_label": 0,
            }]

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            child_downselection_strategy="most_thermo_feasible",
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
        )

        agent.run()

        child = agent.nodes[1]
        # These specific values from worker should be preserved, not recomputed
        assert child.feasibility_score == 0.12345
        assert child.feasibility_label == 0
        assert child.enthalpy_of_reaction == 99.99
        assert child.thermodynamic_label == 0


# =============================================================================
# Tests for strict_thermo_filtering strategy
# =============================================================================


class TestStrictThermoFilteringStrategy:
    """Tests for strict_thermo_filtering downselection strategy.

    Note: These tests simulate what the real worker would return after applying
    the strict_thermo_filtering downselection. The fake worker must return
    already-filtered results because the actual filtering happens inside the
    worker process which we monkeypatch.
    """

    def test_filters_infeasible_enzymatic(self, monkeypatch, sample_molecule):
        """Enzymatic fragments with DORA-XGB label=0 should be filtered.

        The fake worker simulates returning only feasible fragments (label=1).
        """
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            strategy = payload.get("child_downselection_strategy")
            # Simulate the worker's strict_thermo_filtering:
            # Only return feasible fragments (label=1)
            all_frags = [
                {
                    "smiles": "CCO",
                    "reaction_smarts": "smarts1",
                    "reaction_name": "rxn1",
                    "reactants_smiles": ["CCO"],
                    "products_smiles": ["CCO"],
                    "provenance": "enzymatic",
                    "feasibility_score": 0.9,
                    "dora_xgb_score": 0.9,
                    "dora_xgb_label": 1,
                    "enthalpy_of_reaction": 5.0,
                    "thermodynamic_label": 1,
                },
                {
                    "smiles": "CCCO",
                    "reaction_smarts": "smarts2",
                    "reaction_name": "rxn2",
                    "reactants_smiles": ["CCCO"],
                    "products_smiles": ["CCCO"],
                    "provenance": "enzymatic",
                    "feasibility_score": 0.1,
                    "dora_xgb_score": 0.1,
                    "dora_xgb_label": 0,  # Infeasible
                    "enthalpy_of_reaction": 5.0,
                    "thermodynamic_label": 1,
                },
            ]
            if strategy == "strict_thermo_filtering":
                # Filter to only feasible (label=1)
                return [f for f in all_frags if f.get("dora_xgb_label") == 1]
            return all_frags

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            max_children_per_expand=None,
            child_downselection_strategy="strict_thermo_filtering",
        )

        agent.run()

        # Should only have root + 1 child (the feasible one)
        assert len(agent.nodes) == 2
        assert agent.nodes[1].smiles == "CCO"

    def test_filters_infeasible_synthetic(self, monkeypatch, sample_molecule):
        """Synthetic fragments with ΔH > 15 should be filtered.

        The fake worker simulates returning only feasible fragments (ΔH <= 15).
        """
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            strategy = payload.get("child_downselection_strategy")
            all_frags = [
                {
                    "smiles": "CCO",
                    "reaction_smarts": "smarts1",
                    "reaction_name": "rxn1",
                    "reactants_smiles": ["CCO"],
                    "products_smiles": ["CCO"],
                    "provenance": "synthetic",
                    "feasibility_score": 0.9,
                    "dora_xgb_score": None,
                    "dora_xgb_label": None,
                    "enthalpy_of_reaction": 5.0,  # Feasible
                    "thermodynamic_label": 1,
                },
                {
                    "smiles": "CCCO",
                    "reaction_smarts": "smarts2",
                    "reaction_name": "rxn2",
                    "reactants_smiles": ["CCCO"],
                    "products_smiles": ["CCCO"],
                    "provenance": "synthetic",
                    "feasibility_score": 0.1,
                    "dora_xgb_score": None,
                    "dora_xgb_label": None,
                    "enthalpy_of_reaction": 20.0,  # Infeasible (> 15)
                    "thermodynamic_label": 0,
                },
            ]
            if strategy == "strict_thermo_filtering":
                # Filter to only feasible (ΔH <= 15)
                return [f for f in all_frags
                        if f.get("provenance") == "synthetic"
                        and f.get("enthalpy_of_reaction") is not None
                        and f.get("enthalpy_of_reaction") <= 15.0]
            return all_frags

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=True,
            num_workers=1,
            max_children_per_expand=None,
            child_downselection_strategy="strict_thermo_filtering",
        )

        agent.run()

        # Should only have root + 1 child (the feasible one)
        assert len(agent.nodes) == 2
        assert agent.nodes[1].smiles == "CCO"

    def test_filters_none_scores(self, monkeypatch, sample_molecule):
        """Fragments with None scores should be filtered.

        The fake worker simulates filtering out fragments with None scores.
        """
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            strategy = payload.get("child_downselection_strategy")
            all_frags = [
                {
                    "smiles": "CCO",
                    "reaction_smarts": "smarts1",
                    "reaction_name": "rxn1",
                    "reactants_smiles": ["CCO"],
                    "products_smiles": ["CCO"],
                    "provenance": "enzymatic",
                    "feasibility_score": 0.9,
                    "dora_xgb_score": 0.9,
                    "dora_xgb_label": 1,
                    "enthalpy_of_reaction": 5.0,
                    "thermodynamic_label": 1,
                },
                {
                    "smiles": "CCCO",
                    "reaction_smarts": "smarts2",
                    "reaction_name": "rxn2",
                    "reactants_smiles": ["CCCO"],
                    "products_smiles": ["CCCO"],
                    "provenance": "enzymatic",
                    "feasibility_score": None,
                    "dora_xgb_score": None,
                    "dora_xgb_label": None,  # None score - infeasible
                    "enthalpy_of_reaction": None,
                    "thermodynamic_label": None,
                },
            ]
            if strategy == "strict_thermo_filtering":
                # Filter out fragments with None scores
                return [f for f in all_frags if f.get("dora_xgb_label") == 1]
            return all_frags

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            max_children_per_expand=None,
            child_downselection_strategy="strict_thermo_filtering",
        )

        agent.run()

        # Should only have root + 1 child (the one with valid scores)
        assert len(agent.nodes) == 2
        assert agent.nodes[1].smiles == "CCO"

    def test_with_count_limit(self, monkeypatch, sample_molecule):
        """When count limit set, filter first then truncate.

        The fake worker simulates filtering then truncating to limit.
        """
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            max_children = payload.get("max_children_per_expand")
            strategy = payload.get("child_downselection_strategy")

            # Create 10 feasible fragments (small molecules to avoid MW filter)
            small_smiles = ["CO", "CCO", "C(C)O", "CCCO", "CCCCO", "C(C)(C)O", "OCC", "OCCO", "CC", "CCC"]
            all_frags = []
            for i, smi in enumerate(small_smiles):
                all_frags.append({
                    "smiles": smi,
                    "reaction_smarts": f"smarts{i}",
                    "reaction_name": f"rxn{i}",
                    "reactants_smiles": [smi],
                    "products_smiles": [smi],
                    "provenance": "enzymatic",
                    "feasibility_score": 0.9,
                    "dora_xgb_score": 0.9,
                    "dora_xgb_label": 1,
                    "enthalpy_of_reaction": 5.0,
                    "thermodynamic_label": 1,
                })

            if strategy == "strict_thermo_filtering":
                # All are feasible, so just truncate
                if max_children is not None:
                    return all_frags[:max_children]
            return all_frags

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            max_children_per_expand=5,  # Limit to 5
            child_downselection_strategy="strict_thermo_filtering",
        )

        agent.run()

        # Should have root + 5 children (limited from 10 feasible)
        assert len(agent.nodes) == 6

    def test_without_count_limit(self, monkeypatch, sample_molecule):
        """When max_children=None, keep all feasible fragments.

        The fake worker simulates filtering out infeasible and keeping all feasible.
        """
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            strategy = payload.get("child_downselection_strategy")

            # Small molecules to avoid MW filter
            small_smiles = ["CO", "CCO", "C(C)O", "CCCO", "CCCCO"]

            # Create 5 feasible + 3 infeasible fragments
            all_frags = []
            for i, smi in enumerate(small_smiles):
                all_frags.append({
                    "smiles": smi,
                    "reaction_smarts": f"smarts{i}",
                    "reaction_name": f"rxn{i}",
                    "reactants_smiles": [smi],
                    "products_smiles": [smi],
                    "provenance": "enzymatic",
                    "feasibility_score": 0.9,
                    "dora_xgb_score": 0.9,
                    "dora_xgb_label": 1,  # Feasible
                    "enthalpy_of_reaction": 5.0,
                    "thermodynamic_label": 1,
                })
            # Add infeasible ones
            infeasible_smiles = ["CC", "CCC", "CCCC"]
            for i, smi in enumerate(infeasible_smiles):
                all_frags.append({
                    "smiles": smi,
                    "reaction_smarts": f"infeasible{i}",
                    "reaction_name": f"infeasible{i}",
                    "reactants_smiles": [smi],
                    "products_smiles": [smi],
                    "provenance": "enzymatic",
                    "feasibility_score": 0.1,
                    "dora_xgb_score": 0.1,
                    "dora_xgb_label": 0,  # Infeasible
                    "enthalpy_of_reaction": 5.0,
                    "thermodynamic_label": 1,
                })

            if strategy == "strict_thermo_filtering":
                # Filter to only feasible
                return [f for f in all_frags if f.get("dora_xgb_label") == 1]
            return all_frags

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            max_children_per_expand=None,  # No limit
            child_downselection_strategy="strict_thermo_filtering",
        )

        agent.run()

        # Should have root + 5 children (all feasible ones)
        # Note: some SMILES might canonicalize to duplicates so we check >= 4
        assert len(agent.nodes) >= 4
        # Verify configuration
        assert agent.max_children_per_expand is None
        assert agent.child_downselection_strategy == "strict_thermo_filtering"


# =============================================================================
# Tests for max_children_per_expand=None validation
# =============================================================================


class TestMaxChildrenNoneValidation:
    """Tests for max_children_per_expand=None validation."""

    def test_none_with_strict_thermo_allowed(self, sample_molecule):
        """max_children=None + strict_thermo_filtering should work."""
        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
            num_workers=1,
            max_children_per_expand=None,
            child_downselection_strategy="strict_thermo_filtering",
        )
        assert agent.max_children_per_expand is None
        assert agent.child_downselection_strategy == "strict_thermo_filtering"

    def test_none_with_none_strategy_allowed(self, sample_molecule):
        """max_children=None + strategy=None should work."""
        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=0,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
            num_workers=1,
            max_children_per_expand=None,
            child_downselection_strategy=None,
        )
        assert agent.max_children_per_expand is None
        assert agent.child_downselection_strategy is None

    def test_none_with_first_n_raises(self, sample_molecule):
        """max_children=None + first_N should raise ValueError."""
        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        with pytest.raises(ValueError, match="only valid with"):
            AsyncExpansionDORAnetMCTS(
                root=root,
                target_molecule=sample_molecule,
                total_iterations=0,
                max_depth=1,
                use_enzymatic=False,
                use_synthetic=False,
                num_workers=1,
                max_children_per_expand=None,
                child_downselection_strategy="first_N",
            )

    def test_none_with_hybrid_raises(self, sample_molecule):
        """max_children=None + hybrid should raise ValueError."""
        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        with pytest.raises(ValueError, match="only valid with"):
            AsyncExpansionDORAnetMCTS(
                root=root,
                target_molecule=sample_molecule,
                total_iterations=0,
                max_depth=1,
                use_enzymatic=False,
                use_synthetic=False,
                num_workers=1,
                max_children_per_expand=None,
                child_downselection_strategy="hybrid",
            )

    def test_none_with_most_thermo_raises(self, sample_molecule):
        """max_children=None + most_thermo_feasible should raise ValueError."""
        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        with pytest.raises(ValueError, match="only valid with"):
            AsyncExpansionDORAnetMCTS(
                root=root,
                target_molecule=sample_molecule,
                total_iterations=0,
                max_depth=1,
                use_enzymatic=False,
                use_synthetic=False,
                num_workers=1,
                max_children_per_expand=None,
                child_downselection_strategy="most_thermo_feasible",
            )


# =============================================================================
# Tests for None downselection strategy
# =============================================================================


class TestNoneDownselectionStrategy:
    """Tests for child_downselection_strategy=None."""

    def test_none_strategy_with_limit(self, monkeypatch, sample_molecule):
        """strategy=None with integer limit should keep all up to limit.

        Note: Since we monkeypatch _expand_worker, we simulate what the real
        worker would return after applying downselection (5 fragments).
        """
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            # Simulate what the real worker would return after downselection
            # With strategy=None and max_children=5, it would return first 5
            max_children = payload.get("max_children_per_expand", 10)
            fragments = []
            for i in range(10):
                fragments.append({
                    "smiles": f"C{'C' * i}O",
                    "reaction_smarts": f"smarts{i}",
                    "reaction_name": f"rxn{i}",
                    "reactants_smiles": [f"C{'C' * i}O"],
                    "products_smiles": [f"C{'C' * i}O"],
                    "provenance": "enzymatic",
                })
            # Apply the downselection that the real worker would do
            if max_children is not None:
                return fragments[:max_children]
            return fragments

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            max_children_per_expand=5,  # Limit to 5
            child_downselection_strategy=None,
        )

        agent.run()

        # Should have root + 5 children
        assert len(agent.nodes) == 6

    def test_none_strategy_without_limit(self, monkeypatch, sample_molecule):
        """strategy=None with max_children=None should keep all fragments.

        Note: Since we monkeypatch _expand_worker, we simulate what the real
        worker would return after applying downselection (all 10 fragments).
        We use small SMILES to avoid MW filtering.
        """
        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            # Simulate what the real worker would return - with no limit, all fragments
            # Use small molecules (methanol, ethanol, propanol variants) to avoid MW filtering
            max_children = payload.get("max_children_per_expand")
            small_smiles = ["CO", "CCO", "C(C)O", "C(O)C", "OCC", "COC", "OCCO", "C(C)(C)O", "CC(O)C", "CCC"]
            fragments = []
            for i, smi in enumerate(small_smiles):
                fragments.append({
                    "smiles": smi,
                    "reaction_smarts": f"smarts{i}",
                    "reaction_name": f"rxn{i}",
                    "reactants_smiles": [smi],
                    "products_smiles": [smi],
                    "provenance": "enzymatic",
                })
            # Apply the downselection that the real worker would do
            if max_children is not None:
                return fragments[:max_children]
            return fragments

        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts._expand_worker",
            fake_expand_worker,
        )
        monkeypatch.setattr(
            "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
            _FakeExecutor,
        )

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            max_children_per_expand=None,  # No limit
            child_downselection_strategy=None,  # No filtering
        )

        agent.run()

        # Should have root + 10 children (all fragments, minus any duplicates after canonicalization)
        # Some SMILES like "C(C)O", "C(O)C", "OCC" canonicalize to "CCO", so fewer unique nodes
        assert len(agent.nodes) >= 5  # At least some nodes created
        # Verify the agent was configured correctly
        assert agent.max_children_per_expand is None
        assert agent.child_downselection_strategy is None
