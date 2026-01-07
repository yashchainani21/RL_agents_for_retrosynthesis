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


def test_reward_fn_applied_to_selected_leaf(monkeypatch, root_node, sample_molecule):
    monkeypatch.setattr(
        "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
        _FakeExecutor,
    )

    rewards: List[float] = []

    def reward_fn(node: Node) -> float:
        rewards.append(0.7)
        return 0.7

    agent = AsyncExpansionDORAnetMCTS(
        root=root_node,
        target_molecule=sample_molecule,
        total_iterations=1,
        max_depth=0,
        use_enzymatic=False,
        use_synthetic=False,
        num_workers=1,
        reward_fn=reward_fn,
    )

    agent.run()

    assert rewards == [0.7]
    assert agent.root.visits == 1
    assert agent.root.value == pytest.approx(0.7)
