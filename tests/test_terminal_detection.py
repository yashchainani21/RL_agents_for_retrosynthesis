"""Tests for DORAnet_agent.policies.terminal_detection — terminal detectors."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from DORAnet_agent.node import Node
from DORAnet_agent.policies.base import TerminalDetectionResult
from DORAnet_agent.policies.terminal_detection import (
    NoOpTerminalDetector,
    SimilarityGuidedRetroTideDetector,
    VerifyWithRetroTide,
    _run_retrotide,
)
from DORAnet_agent.policies.utils import generate_morgan_fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(smiles="CCO", parent=None, provenance="enzymatic"):
    """Create a DORAnet Node from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    node = Node(
        fragment=mol,
        parent=parent,
        depth=0 if parent is None else parent.depth + 1,
        provenance=provenance,
    )
    return node


def _make_none_fragment_node():
    """Create a DORAnet Node with fragment=None."""
    node = Node(fragment=None, parent=None, depth=0, provenance="target")
    return node


# ---------------------------------------------------------------------------
# TestNoOpTerminalDetector
# ---------------------------------------------------------------------------

class TestNoOpTerminalDetector:

    def test_detect_returns_not_terminal(self):
        detector = NoOpTerminalDetector()
        node = _make_node()
        result = detector.detect(node, {})
        assert result.terminal is False

    def test_name(self):
        assert NoOpTerminalDetector().name == "NoOp"


# ---------------------------------------------------------------------------
# TestRunRetrotide
# ---------------------------------------------------------------------------

class TestRunRetrotide:
    """Tests for the module-level _run_retrotide() helper."""

    def test_none_fragment_returns_unsuccessful(self):
        node = _make_none_fragment_node()
        target = Chem.MolFromSmiles("CCCCC(=O)O")
        result = _run_retrotide(node, target, {})
        assert result["successful"] is False
        assert result["num_successful_nodes"] == 0

    def test_successful_retrotide_run(self):
        node = _make_node("CCO")
        target = Chem.MolFromSmiles("CCCCC(=O)O")

        mock_successful_node = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run.return_value = None
        mock_agent.successful_nodes = {mock_successful_node}
        mock_agent.successful_simulated_designs = []
        mock_agent.nodes = [MagicMock()]

        mock_mcts_cls = MagicMock(return_value=mock_agent)
        mock_node_cls = MagicMock()

        mock_mcts_mod = MagicMock()
        mock_mcts_mod.MCTS = mock_mcts_cls
        mock_node_mod = MagicMock()
        mock_node_mod.Node = mock_node_cls

        with patch.dict(sys.modules, {
            "RetroTide_agent": MagicMock(),
            "RetroTide_agent.mcts": mock_mcts_mod,
            "RetroTide_agent.node": mock_node_mod,
        }):
            result = _run_retrotide(node, target, {"max_depth": 6})

        assert result["successful"] is True
        assert result["num_successful_nodes"] == 1
        assert result["best_score"] == 1.0

    def test_unsuccessful_retrotide_run(self):
        node = _make_node("CCO")
        target = Chem.MolFromSmiles("CCCCC(=O)O")

        mock_agent = MagicMock()
        mock_agent.run.return_value = None
        mock_agent.successful_nodes = set()
        mock_agent.successful_simulated_designs = []
        # One explored node with low value
        explored = MagicMock()
        explored.value = 0.2
        explored.visits = 1
        mock_agent.nodes = [explored]

        mock_mcts_cls = MagicMock(return_value=mock_agent)
        mock_node_cls = MagicMock()

        mock_mcts_mod = MagicMock()
        mock_mcts_mod.MCTS = mock_mcts_cls
        mock_node_mod = MagicMock()
        mock_node_mod.Node = mock_node_cls

        with patch.dict(sys.modules, {
            "RetroTide_agent": MagicMock(),
            "RetroTide_agent.mcts": mock_mcts_mod,
            "RetroTide_agent.node": mock_node_mod,
        }):
            result = _run_retrotide(node, target, {})

        assert result["successful"] is False
        assert result["best_score"] == pytest.approx(0.2)

    def test_kwargs_passed_to_mcts(self):
        node = _make_node("CCO")
        target = Chem.MolFromSmiles("CCCCC(=O)O")

        mock_agent = MagicMock()
        mock_agent.run.return_value = None
        mock_agent.successful_nodes = set()
        mock_agent.successful_simulated_designs = []
        mock_agent.nodes = []

        mock_mcts_cls = MagicMock(return_value=mock_agent)
        mock_node_cls = MagicMock()

        mock_mcts_mod = MagicMock()
        mock_mcts_mod.MCTS = mock_mcts_cls
        mock_node_mod = MagicMock()
        mock_node_mod.Node = mock_node_cls

        with patch.dict(sys.modules, {
            "RetroTide_agent": MagicMock(),
            "RetroTide_agent.mcts": mock_mcts_mod,
            "RetroTide_agent.node": mock_node_mod,
        }):
            _run_retrotide(node, target, {"max_depth": 10, "total_iterations": 200})

        # Verify MCTS was constructed with our kwargs
        call_kwargs = mock_mcts_cls.call_args
        assert call_kwargs.kwargs["max_depth"] == 10
        assert call_kwargs.kwargs["total_iterations"] == 200


# ---------------------------------------------------------------------------
# TestVerifyWithRetroTide
# ---------------------------------------------------------------------------

class TestVerifyWithRetroTide:

    def test_init_defaults(self):
        detector = VerifyWithRetroTide()
        assert detector._pks_library is None
        assert detector._retrotide_kwargs == {}

    def test_get_pks_library_from_instance(self):
        detector = VerifyWithRetroTide(pks_library={"CCO"})
        assert detector._get_pks_library({}) == {"CCO"}

    def test_get_pks_library_from_context(self):
        detector = VerifyWithRetroTide()
        ctx = {"pks_library": {"CCC"}}
        assert detector._get_pks_library(ctx) == {"CCC"}

    def test_get_retrotide_kwargs_merges(self):
        detector = VerifyWithRetroTide(retrotide_kwargs={"max_depth": 10, "a": 1})
        ctx = {"retrotide_kwargs": {"max_depth": 15, "b": 2}}
        merged = detector._get_retrotide_kwargs(ctx)
        # Context overrides instance for max_depth
        assert merged["max_depth"] == 15
        assert merged["a"] == 1
        assert merged["b"] == 2

    def test_is_pks_match_true(self):
        detector = VerifyWithRetroTide()
        node = _make_node("CCO")
        assert detector._is_pks_match(node, {"CCO"}) is True

    def test_is_pks_match_false(self):
        detector = VerifyWithRetroTide()
        node = _make_node("CCC")
        assert detector._is_pks_match(node, {"CCO"}) is False

    def test_is_pks_match_empty_library(self):
        detector = VerifyWithRetroTide()
        node = _make_node("CCO")
        assert detector._is_pks_match(node, set()) is False

    def test_is_pks_match_none_smiles(self):
        detector = VerifyWithRetroTide()
        node = _make_none_fragment_node()
        assert detector._is_pks_match(node, {"CCO"}) is False

    def test_detect_retrotide_unavailable(self):
        detector = VerifyWithRetroTide()
        detector._retrotide_available = False
        node = _make_node("CCO")
        result = detector.detect(node, {"pks_library": {"CCO"}})
        assert result.terminal is False

    def test_detect_already_attempted(self):
        detector = VerifyWithRetroTide()
        node = _make_node("CCO")
        node.retrotide_attempted = True
        result = detector.detect(node, {"pks_library": {"CCO"}})
        assert result.terminal is False

    def test_detect_no_pks_match(self):
        detector = VerifyWithRetroTide()
        node = _make_node("CCC")
        result = detector.detect(node, {"pks_library": {"CCO"}})
        assert result.terminal is False

    @patch("DORAnet_agent.policies.terminal_detection._run_retrotide")
    def test_detect_pks_match_retrotide_success(self, mock_run):
        mock_run.return_value = {
            "successful": True,
            "num_successful_nodes": 3,
            "best_score": 1.0,
            "total_nodes": 50,
            "agent": MagicMock(),
            "target_smiles": "CCO",
        }
        detector = VerifyWithRetroTide()
        node = _make_node("CCO")
        target = Chem.MolFromSmiles("CCCCC(=O)O")
        ctx = {"pks_library": {"CCO"}, "target_molecule": target}
        result = detector.detect(node, ctx)
        assert result.terminal is True
        assert result.terminal_type == "pks_terminal"
        assert result.metadata["retrotide_successful"] is True
        assert result.metadata["retrotide_num_successful_nodes"] == 3

    @patch("DORAnet_agent.policies.terminal_detection._run_retrotide")
    def test_detect_pks_match_retrotide_failure(self, mock_run):
        mock_run.return_value = {
            "successful": False,
            "num_successful_nodes": 0,
            "best_score": 0.3,
            "total_nodes": 100,
            "agent": MagicMock(),
            "target_smiles": "CCO",
        }
        detector = VerifyWithRetroTide()
        node = _make_node("CCO")
        target = Chem.MolFromSmiles("CCCCC(=O)O")
        ctx = {"pks_library": {"CCO"}, "target_molecule": target}
        result = detector.detect(node, ctx)
        assert result.terminal is False
        assert result.metadata["retrotide_successful"] is False

    def test_detect_no_target_molecule_in_context(self):
        detector = VerifyWithRetroTide()
        node = _make_node("CCO")
        ctx = {"pks_library": {"CCO"}}  # no target_molecule
        result = detector.detect(node, ctx)
        assert result.terminal is False
        assert node.retrotide_attempted is True


# ---------------------------------------------------------------------------
# TestSimilarityGuidedRetroTideDetector
# ---------------------------------------------------------------------------

class TestSimilarityGuidedRetroTideDetector:

    @pytest.fixture
    def tanimoto_detector(self):
        """Create a detector with mocked file loading (tanimoto)."""
        with patch.object(SimilarityGuidedRetroTideDetector, "_load_pks_building_blocks"):
            det = SimilarityGuidedRetroTideDetector(
                pks_library={"CCO"},
                similarity_method="tanimoto",
                retrotide_spawn_threshold=0.9,
            )
        det._pks_building_blocks = []
        return det

    @pytest.fixture
    def mcs_detector(self):
        """Create a detector with mocked file loading (mcs)."""
        with patch.object(SimilarityGuidedRetroTideDetector, "_load_pks_building_blocks"):
            det = SimilarityGuidedRetroTideDetector(
                pks_library={"CCO"},
                similarity_method="mcs",
                retrotide_spawn_threshold=0.9,
            )
        det._pks_building_blocks = []
        return det

    def test_invalid_similarity_method_raises(self):
        with pytest.raises(ValueError, match="Invalid similarity_method"):
            with patch.object(SimilarityGuidedRetroTideDetector, "_load_pks_building_blocks"):
                SimilarityGuidedRetroTideDetector(similarity_method="invalid")

    # --- Tanimoto similarity ---

    def test_tanimoto_empty_building_blocks(self, tanimoto_detector):
        node = _make_node("CCO")
        sim, meta = tanimoto_detector._compute_tanimoto_similarity(node)
        assert sim == 0.0
        assert meta["pks_building_blocks_checked"] == 0

    def test_tanimoto_finds_match(self, tanimoto_detector):
        # Pre-populate with ethanol fingerprint
        mol = Chem.MolFromSmiles("CCO")
        fp = generate_morgan_fingerprint(mol)
        tanimoto_detector._pks_building_blocks = [("CCO", fp)]

        node = _make_node("CCO")
        sim, meta = tanimoto_detector._compute_tanimoto_similarity(node)
        assert sim == pytest.approx(1.0)
        assert meta["pks_building_blocks_checked"] == 1

    def test_tanimoto_none_fragment(self, tanimoto_detector):
        node = _make_none_fragment_node()
        sim, meta = tanimoto_detector._compute_tanimoto_similarity(node)
        assert sim == 0.0

    # --- MCS similarity ---

    def test_mcs_empty_building_blocks(self, mcs_detector):
        node = _make_node("CCO")
        sim, meta = mcs_detector._compute_mcs_similarity(node)
        assert sim == 0.0
        assert meta["pks_building_blocks_checked"] == 0

    def test_mcs_atom_count_filtering(self, mcs_detector):
        # Huge molecule (naphthalene, 10 atoms) vs tiny query (methane, 1 atom)
        big_mol = Chem.MolFromSmiles("c1cccc2ccccc12")
        mcs_detector._pks_building_blocks = [(big_mol, big_mol.GetNumAtoms())]

        node = _make_node("C")  # methane, 1 atom
        sim, meta = mcs_detector._compute_mcs_similarity(node)
        assert meta["pks_building_blocks_skipped_size"] > 0

    # --- detect() ---

    def test_detect_already_attempted(self, tanimoto_detector):
        node = _make_node("CCO")
        node.retrotide_attempted = True
        result = tanimoto_detector.detect(node, {})
        assert result.terminal is False

    def test_detect_no_match_low_similarity(self, tanimoto_detector):
        node = _make_node("c1ccccc1")  # benzene — not in PKS library {"CCO"}
        result = tanimoto_detector.detect(node, {})
        assert result.terminal is False
        assert result.metadata["pks_match"] is False

    def test_detect_exact_match_retrotide_unavailable(self, tanimoto_detector):
        tanimoto_detector._retrotide_available = False
        node = _make_node("CCO")
        # Make similarity high enough by adding ethanol fingerprint
        mol = Chem.MolFromSmiles("CCO")
        fp = generate_morgan_fingerprint(mol)
        tanimoto_detector._pks_building_blocks = [("CCO", fp)]

        result = tanimoto_detector.detect(node, {"pks_library": {"CCO"}})
        assert result.terminal is False
        assert result.metadata.get("retrotide_available") is False

    @patch("DORAnet_agent.policies.terminal_detection._run_retrotide")
    def test_detect_high_similarity_retrotide_success(self, mock_run, tanimoto_detector):
        mock_run.return_value = {
            "successful": True,
            "num_successful_nodes": 2,
            "best_score": 1.0,
            "total_nodes": 30,
            "agent": MagicMock(),
            "target_smiles": "CCO",
        }
        # Pre-populate building blocks for high tanimoto similarity
        mol = Chem.MolFromSmiles("CCO")
        fp = generate_morgan_fingerprint(mol)
        tanimoto_detector._pks_building_blocks = [("CCO", fp)]

        node = _make_node("CCO")
        target = Chem.MolFromSmiles("CCCCC(=O)O")
        ctx = {"pks_library": {"CCO"}, "target_molecule": target}
        result = tanimoto_detector.detect(node, ctx)
        assert result.terminal is True
        assert result.terminal_type == "pks_terminal"
        assert result.metadata["retrotide_successful"] is True
