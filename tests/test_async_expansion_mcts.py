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
        full_path = REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V3.txt"
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
# Tests for PKS library match priority over sink compound short-circuit
# =============================================================================

from DORAnet_agent.mcts import _canonicalize_smiles
from DORAnet_agent.policies.base import RolloutResult, RolloutPolicy, RewardPolicy


class MockRolloutPolicy(RolloutPolicy):
    """Mock rollout policy for testing that tracks calls and returns configurable results."""

    def __init__(self, result: RolloutResult = None):
        self._result = result or RolloutResult(reward=0.5, terminal=False)
        self.calls: List[str] = []

    def rollout(self, node: "Node", context: Dict[str, Any]) -> RolloutResult:
        self.calls.append(node.smiles or "")
        return self._result

    @property
    def name(self) -> str:
        return "MockRolloutPolicy"


class MockRewardPolicy(RewardPolicy):
    """Mock reward policy for testing that tracks calls."""

    def __init__(self, reward: float = 1.0):
        self._reward = reward
        self.calls: List[str] = []

    def calculate_reward(self, node: "Node", context: Dict[str, Any]) -> float:
        self.calls.append(node.smiles or "")
        return self._reward

    @property
    def name(self) -> str:
        return "MockRewardPolicy"


class TestPKSSinkCompoundPriority:
    """Tests for PKS library matches that are also sink compounds."""

    def test_pks_match_triggers_rollout_even_if_sink_compound(
        self, monkeypatch, sample_molecule
    ):
        """Nodes that are both sink compounds AND PKS matches should trigger rollout.

        When the rollout result is not terminal, the reward policy is called as fallback
        for sink compounds.
        """
        # Test SMILES that will be both a sink compound and PKS library match
        test_smiles = "CCCCO"  # 1-butanol
        canonical_smiles = _canonicalize_smiles(test_smiles)

        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [{
                "smiles": test_smiles,
                "reaction_smarts": "smarts",
                "reaction_name": "reaction",
                "reactants_smiles": [test_smiles],
                "products_smiles": [test_smiles],
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

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        # Create mock policies to track calls
        # Mock rollout returns non-terminal, so reward policy should be called as fallback
        mock_rollout = MockRolloutPolicy(
            RolloutResult(reward=0.8, terminal=False)
        )
        mock_reward = MockRewardPolicy(reward=1.0)

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            rollout_policy=mock_rollout,
            reward_policy=mock_reward,
        )

        # Set sink_compounds and pks_library using canonical SMILES
        # _get_sink_compound_type checks biological_sink_compounds and chemical_sink_compounds
        agent.sink_compounds = {canonical_smiles}
        agent.chemical_sink_compounds = {canonical_smiles}
        agent.pks_library = {canonical_smiles}

        agent.run()

        # Verify rollout was called even though it's a sink compound
        # (because it's also in PKS library - this is the key behavior we're testing)
        assert canonical_smiles in mock_rollout.calls, \
            "Rollout should be called for sink compound that matches PKS library"
        # Since rollout returned terminal=False, reward policy is called as fallback
        # for sink compounds (this is correct behavior per the implementation)
        assert canonical_smiles in mock_reward.calls, \
            "Reward policy should be called as fallback when rollout is not terminal"

    def test_pks_terminal_marked_when_retrotide_succeeds_on_sink(
        self, monkeypatch, sample_molecule
    ):
        """Sink compound + PKS match + RetroTide success = is_pks_terminal=True."""
        test_smiles = "CCCCO"
        canonical_smiles = _canonicalize_smiles(test_smiles)

        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [{
                "smiles": test_smiles,
                "reaction_smarts": "smarts",
                "reaction_name": "reaction",
                "reactants_smiles": [test_smiles],
                "products_smiles": [test_smiles],
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

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        # Mock rollout that simulates successful RetroTide
        mock_rollout = MockRolloutPolicy(
            RolloutResult(
                reward=1.0,
                terminal=True,
                terminal_type="retrotide_success",
                metadata={"retrotide_agent": {"best_similarity": 1.0}}
            )
        )

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            rollout_policy=mock_rollout,
        )

        # Set sink_compounds and pks_library using canonical SMILES
        # _get_sink_compound_type checks biological_sink_compounds and chemical_sink_compounds
        agent.sink_compounds = {canonical_smiles}
        agent.chemical_sink_compounds = {canonical_smiles}
        agent.pks_library = {canonical_smiles}

        agent.run()

        # Find the child node
        child = agent.nodes[1]
        assert child.smiles == canonical_smiles
        assert child.is_sink_compound, "Node should be marked as sink compound"
        assert child.is_pks_terminal, "Node should be marked as PKS terminal"
        assert child.expanded, "Node should be marked as expanded"

    def test_sink_reward_fallback_when_retrotide_fails(
        self, monkeypatch, sample_molecule
    ):
        """Sink compound + PKS match + RetroTide failure = use sink reward."""
        test_smiles = "CCCCO"
        canonical_smiles = _canonicalize_smiles(test_smiles)

        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [{
                "smiles": test_smiles,
                "reaction_smarts": "smarts",
                "reaction_name": "reaction",
                "reactants_smiles": [test_smiles],
                "products_smiles": [test_smiles],
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

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        # Mock rollout that simulates failed RetroTide (low similarity)
        mock_rollout = MockRolloutPolicy(
            RolloutResult(reward=0.3, terminal=False)  # Not terminal
        )
        mock_reward = MockRewardPolicy(reward=1.0)

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            rollout_policy=mock_rollout,
            reward_policy=mock_reward,
        )

        # Set sink_compounds and pks_library using canonical SMILES
        # _get_sink_compound_type checks biological_sink_compounds and chemical_sink_compounds
        agent.sink_compounds = {canonical_smiles}
        agent.chemical_sink_compounds = {canonical_smiles}
        agent.pks_library = {canonical_smiles}

        agent.run()

        # Verify rollout was called (because PKS match)
        assert canonical_smiles in mock_rollout.calls

        # Verify reward policy was called as fallback (because rollout wasn't terminal
        # but node is a sink compound)
        assert canonical_smiles in mock_reward.calls, \
            "Reward policy should be called as fallback when RetroTide fails on sink"

        # Verify the child used the reward policy reward
        child = agent.nodes[1]
        assert child.is_sink_compound
        assert not child.is_pks_terminal, "Should not be PKS terminal when RetroTide fails"

    def test_pure_sink_compound_skips_rollout(
        self, monkeypatch, sample_molecule
    ):
        """Pure sink compounds (not in PKS library) should still skip rollout."""
        test_smiles = "CCO"  # Ethanol - only a sink compound, not in PKS library
        canonical_smiles = _canonicalize_smiles(test_smiles)

        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [{
                "smiles": test_smiles,
                "reaction_smarts": "smarts",
                "reaction_name": "reaction",
                "reactants_smiles": [test_smiles],
                "products_smiles": [test_smiles],
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

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        mock_rollout = MockRolloutPolicy()
        mock_reward = MockRewardPolicy(reward=1.0)

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            rollout_policy=mock_rollout,
            reward_policy=mock_reward,
        )

        # Set sink_compounds using canonical SMILES, empty PKS library
        # _get_sink_compound_type checks biological_sink_compounds and chemical_sink_compounds
        agent.sink_compounds = {canonical_smiles}  # In sink compounds
        agent.chemical_sink_compounds = {canonical_smiles}
        agent.pks_library = set()  # Empty PKS library

        agent.run()

        # Verify rollout was NOT called (pure sink compound)
        assert canonical_smiles not in mock_rollout.calls, \
            "Rollout should NOT be called for pure sink compound"
        # Verify reward policy WAS called
        assert canonical_smiles in mock_reward.calls, \
            "Reward policy should be called for pure sink compound"

    def test_non_sink_non_pks_uses_standard_rollout(
        self, monkeypatch, sample_molecule
    ):
        """Non-sink, non-PKS compounds should use standard rollout."""
        test_smiles = "CCCCCC"  # Hexane - neither sink nor PKS
        canonical_smiles = _canonicalize_smiles(test_smiles)

        def fake_expand_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [{
                "smiles": test_smiles,
                "reaction_smarts": "smarts",
                "reaction_name": "reaction",
                "reactants_smiles": [test_smiles],
                "products_smiles": [test_smiles],
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

        Node.node_counter = 0
        root = Node(fragment=sample_molecule, parent=None, depth=0, provenance="target")

        mock_rollout = MockRolloutPolicy(
            RolloutResult(reward=0.5, terminal=False)
        )
        mock_reward = MockRewardPolicy()

        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=sample_molecule,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=True,
            use_synthetic=False,
            num_workers=1,
            rollout_policy=mock_rollout,
            reward_policy=mock_reward,
        )

        # Set sink_compounds and pks_library to empty sets
        agent.sink_compounds = set()  # Empty
        agent.pks_library = set()  # Empty

        agent.run()

        # Verify rollout WAS called (standard case)
        assert canonical_smiles in mock_rollout.calls, \
            "Rollout should be called for non-sink, non-PKS compounds"
        # Verify reward policy was NOT called
        assert canonical_smiles not in mock_reward.calls, \
            "Reward policy should not be called when rollout handles reward"
