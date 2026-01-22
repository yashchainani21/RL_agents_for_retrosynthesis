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
from DORAnet_agent.policies import PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck


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


# =============================================================================
# Tests for PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck with async MCTS
# =============================================================================


@pytest.fixture
def pks_building_blocks_path():
    """Path to the PKS building blocks test fixture (100 entries for fast testing)."""
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "test_pks_building_blocks.txt"
    if not fixture_path.exists():
        # Create fixture from full file if it doesn't exist
        full_path = REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES.txt"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "r") as src, open(fixture_path, "w") as dst:
            for i, line in enumerate(src):
                if i >= 100:
                    break
                dst.write(line)
    return fixture_path


@pytest.fixture
def sample_pks_smiles(pks_building_blocks_path):
    """Get a real PKS building block SMILES for testing."""
    with open(pks_building_blocks_path, "r") as f:
        # Return the first valid SMILES
        for line in f:
            smiles = line.strip()
            if smiles and Chem.MolFromSmiles(smiles):
                return smiles
    pytest.skip("No valid PKS SMILES found in building blocks file")


def test_pks_sim_score_policy_initializes_with_async_agent(
    monkeypatch, root_node, sample_molecule, pks_building_blocks_path
):
    """
    Test that PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck correctly initializes
    when passed to AsyncExpansionDORAnetMCTS.
    """
    monkeypatch.setattr(
        "DORAnet_agent.async_expansion_mcts.ProcessPoolExecutor",
        _FakeExecutor,
    )

    # Create the PKS similarity policy with test fixture (100 entries)
    pks_policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
        pks_building_blocks_path=str(pks_building_blocks_path),
        similarity_threshold=0.9,
        mcs_timeout=1.0,
        atom_count_tolerance=0.5,
    )

    agent = AsyncExpansionDORAnetMCTS(
        root=root_node,
        target_molecule=sample_molecule,
        total_iterations=0,  # Don't run any iterations, just test initialization
        max_depth=3,
        use_enzymatic=False,
        use_synthetic=False,
        num_workers=1,
        rollout_policy=pks_policy,
    )

    # Verify the policy is set correctly
    assert agent.rollout_policy is pks_policy
    # Name includes parameters, so check it starts with the class name
    assert "PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck" in agent.rollout_policy.name
    
    # Verify PKS building blocks are loaded (using test fixture with 100 entries)
    assert len(pks_policy._pks_building_blocks) == 100


def test_pks_sim_score_policy_rollout_after_async_expansion(
    monkeypatch, root_node, sample_molecule, pks_building_blocks_path
):
    """
    Test that PKS policy rollout is called after async expansion completes
    and returns appropriate rewards for non-PKS molecules.
    """
    def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Return ethanol - a small molecule that won't match PKS library well
        return [{
            "smiles": "CCO",
            "reaction_smarts": "dummy_smarts",
            "reaction_name": "dummy_reaction",
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

    # Create PKS policy with a small subset for faster testing
    pks_policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
        pks_building_blocks_path=str(pks_building_blocks_path),
        similarity_threshold=0.9,
        mcs_timeout=0.1,  # Short timeout for testing
        atom_count_tolerance=0.5,
    )

    agent = AsyncExpansionDORAnetMCTS(
        root=root_node,
        target_molecule=sample_molecule,
        total_iterations=1,
        max_depth=3,
        use_enzymatic=True,
        use_synthetic=False,
        num_workers=1,
        rollout_policy=pks_policy,
    )

    agent.run()

    # Verify child was created
    assert len(agent.nodes) == 2
    child = agent.nodes[1]
    assert child.smiles == "CCO"
    
    # Verify rollout was applied (backpropagation should have updated values)
    # The child should have been visited and have some value
    assert child.visits >= 1
    
    # Ethanol (CCO) is very small and unlikely to match PKS library well
    # so it should have a low similarity reward (< 0.9 threshold)
    assert not child.is_pks_terminal  # Not terminal since similarity < 0.9


def test_pks_policy_terminal_detection_in_async_context(
    monkeypatch, root_node, sample_molecule, pks_building_blocks_path, sample_pks_smiles
):
    """
    Test that when a fragment has high PKS similarity, it receives high reward.
    
    Note: Terminal marking only occurs when the SMILES exactly matches the
    agent's pks_library AND RetroTide succeeds. This test verifies that high
    similarity (1.0 for exact match) results in high reward via backpropagation.
    """
    def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Return an exact PKS building block match
        return [{
            "smiles": sample_pks_smiles,
            "reaction_smarts": "pks_match_smarts",
            "reaction_name": "PKS_match_reaction",
            "reactants_smiles": [sample_pks_smiles],
            "products_smiles": [sample_pks_smiles],
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

    pks_policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
        pks_building_blocks_path=str(pks_building_blocks_path),
        similarity_threshold=0.9,
        mcs_timeout=1.0,
        atom_count_tolerance=0.5,
    )

    agent = AsyncExpansionDORAnetMCTS(
        root=root_node,
        target_molecule=sample_molecule,
        total_iterations=1,
        max_depth=3,
        use_enzymatic=True,
        use_synthetic=False,
        num_workers=1,
        rollout_policy=pks_policy,
        MW_multiple_to_exclude=4.0,  # PKS building blocks are typically 200-400 MW
    )

    agent.run()

    # Verify child was created
    assert len(agent.nodes) == 2
    child = agent.nodes[1]
    assert child.smiles == sample_pks_smiles
    
    # An exact PKS match should have similarity = 1.0
    # The reward should be high (similarity = 1.0) even without terminal marking
    # (Terminal marking requires pks_library match + RetroTide success)
    assert child.visits >= 1
    assert child.value >= 0.9  # Should have received high similarity reward


def test_pks_policy_with_multiple_inflight_expansions(
    monkeypatch, pks_building_blocks_path, sample_pks_smiles
):
    """
    Test that PKS policy works correctly when multiple expansions complete
    and need rollout evaluation (simulating concurrent expansion behavior).
    
    This test verifies:
    1. Multiple expansions are processed correctly
    2. PKS similarity rewards are computed for each child
    3. High-similarity nodes get higher rewards than low-similarity nodes
    """
    # Track which payloads were expanded
    expansion_calls: List[str] = []
    
    def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        starter = payload["starter_smiles"]
        expansion_calls.append(starter)
        
        if "CCCCC" in starter:  # Root molecule pattern
            # First expansion: return two fragments with different PKS similarity
            return [
                {
                    "smiles": "CCCC",  # Butane - low PKS similarity (small, no PKS features)
                    "reaction_smarts": "smarts1",
                    "reaction_name": "reaction1",
                    "reactants_smiles": ["CCCC"],
                    "products_smiles": ["CCCC"],
                    "provenance": "enzymatic",
                },
                {
                    "smiles": sample_pks_smiles,  # Exact PKS match - high similarity
                    "reaction_smarts": "smarts2",
                    "reaction_name": "reaction2",
                    "reactants_smiles": [sample_pks_smiles],
                    "products_smiles": [sample_pks_smiles],
                    "provenance": "enzymatic",
                },
            ]
        else:
            # Second expansion from butane
            return [{
                "smiles": "CCC",  # Propane - low PKS similarity
                "reaction_smarts": "smarts3",
                "reaction_name": "reaction3",
                "reactants_smiles": ["CCC"],
                "products_smiles": ["CCC"],
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

    # Create a fresh root node for this test
    Node.node_counter = 0
    root_mol = Chem.MolFromSmiles("CCCCC(=O)O")
    root = Node(fragment=root_mol, parent=None, depth=0, provenance="target")

    pks_policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
        pks_building_blocks_path=str(pks_building_blocks_path),
        similarity_threshold=0.9,
        mcs_timeout=0.5,
        atom_count_tolerance=0.5,
    )

    agent = AsyncExpansionDORAnetMCTS(
        root=root,
        target_molecule=root_mol,
        total_iterations=3,  # Enough iterations to expand multiple nodes
        max_depth=3,
        max_inflight_expansions=2,  # Allow concurrent expansions
        use_enzymatic=True,
        use_synthetic=False,
        num_workers=2,
        rollout_policy=pks_policy,
        MW_multiple_to_exclude=4.0,  # PKS building blocks are typically 200-400 MW
    )

    agent.run()

    # Verify multiple nodes were created
    assert len(agent.nodes) >= 3  # Root + at least 2 children
    
    # Find nodes by SMILES
    pks_node = None
    small_nodes = []
    for node in agent.nodes:
        if node.smiles == sample_pks_smiles:
            pks_node = node
        elif node.smiles in ["CCCC", "CCC"]:
            small_nodes.append(node)
    
    # Verify PKS-matching node exists and has high reward
    assert pks_node is not None, "PKS-matching node should be created"
    assert pks_node.visits >= 1
    assert pks_node.value >= 0.9, "Exact PKS match should have similarity ~1.0"
    
    # Verify small molecules (butane, propane) have lower rewards
    # They have few atoms and won't match PKS building blocks well
    for node in small_nodes:
        assert node.visits >= 1
        # Small alkanes should have low PKS similarity (typically < 0.5)
        if node.visits > 0:
            avg_reward = node.value / node.visits
            assert avg_reward < pks_node.value / pks_node.visits, \
                f"Small molecule {node.smiles} should have lower reward than PKS match"
