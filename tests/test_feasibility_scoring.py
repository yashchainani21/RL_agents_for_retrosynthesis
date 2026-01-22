"""Tests for DORA-XGB feasibility scoring integration."""
import pytest
from DORAnet_agent.mcts import (
    _reverse_reaction_string,
    FeasibilityScorer,
    DORA_XGB_AVAILABLE,
)


class TestReverseReactionString:
    """Tests for the _reverse_reaction_string helper function."""

    def test_simple_reversal(self):
        """Test reversing a simple reaction string."""
        assert _reverse_reaction_string("A.B>>C.D") == "C.D>>A.B"

    def test_single_reactant_product(self):
        """Test reversing with single reactant and product."""
        assert _reverse_reaction_string("A>>B") == "B>>A"

    def test_no_arrow(self):
        """Test that strings without >> are returned unchanged."""
        assert _reverse_reaction_string("ABC") == "ABC"

    def test_empty_string(self):
        """Test that empty strings are returned unchanged."""
        assert _reverse_reaction_string("") == ""

    def test_complex_smiles(self):
        """Test with actual SMILES strings."""
        retro = "CC(=O)O.O>>CC(=O)OC"
        forward = "CC(=O)OC>>CC(=O)O.O"
        assert _reverse_reaction_string(retro) == forward

    def test_multiple_arrows(self):
        """Test that only the first >> is used for splitting."""
        # Edge case: if there are multiple >>, only split on first
        result = _reverse_reaction_string("A>>B>>C")
        assert result == "B>>C>>A"


class TestFeasibilityScorer:
    """Tests for the FeasibilityScorer class."""

    def test_synthetic_returns_none(self):
        """Synthetic reactions should return None for score and label."""
        scorer = FeasibilityScorer()
        score, label = scorer.score_reaction(
            reactants_smiles=["CCO", "O"],
            products_smiles=["CCCO"],
            provenance="synthetic"
        )
        assert score is None
        assert label is None

    def test_missing_reactants_returns_none(self):
        """Missing reactants should return None for score and label."""
        scorer = FeasibilityScorer()
        score, label = scorer.score_reaction(
            reactants_smiles=[],
            products_smiles=["CCO"],
            provenance="enzymatic"
        )
        assert score is None
        assert label is None

    def test_missing_products_returns_none(self):
        """Missing products should return None for score and label."""
        scorer = FeasibilityScorer()
        score, label = scorer.score_reaction(
            reactants_smiles=["CCO"],
            products_smiles=[],
            provenance="enzymatic"
        )
        assert score is None
        assert label is None

    def test_unknown_provenance_returns_none(self):
        """Unknown provenance should return None (only enzymatic is scored)."""
        scorer = FeasibilityScorer()
        score, label = scorer.score_reaction(
            reactants_smiles=["CCO"],
            products_smiles=["CCCO"],
            provenance="unknown"
        )
        assert score is None
        assert label is None

    def test_none_provenance_returns_none(self):
        """None provenance should return None."""
        scorer = FeasibilityScorer()
        score, label = scorer.score_reaction(
            reactants_smiles=["CCO"],
            products_smiles=["CCCO"],
            provenance=None
        )
        assert score is None
        assert label is None

    @pytest.mark.skipif(not DORA_XGB_AVAILABLE, reason="DORA-XGB not installed")
    def test_enzymatic_scoring_returns_valid_values(self):
        """With DORA-XGB available, enzymatic reactions should return valid scores."""
        scorer = FeasibilityScorer()
        score, label = scorer.score_reaction(
            reactants_smiles=["CC(=O)O", "[NH3]"],  # acetate + ammonia
            products_smiles=["CC(=O)N"],  # acetamide
            provenance="enzymatic"
        )
        # Score should be a probability between 0 and 1
        assert score is not None
        assert 0.0 <= score <= 1.0
        # Label should be binary
        assert label in [0, 1]

    @pytest.mark.skipif(not DORA_XGB_AVAILABLE, reason="DORA-XGB not installed")
    def test_scorer_lazy_initialization(self):
        """Scorer should lazily initialize the DORA-XGB model."""
        scorer = FeasibilityScorer()
        # Model should not be initialized yet
        assert scorer._model is None
        assert scorer._initialized is False

        # After scoring, model should be initialized
        scorer.score_reaction(
            reactants_smiles=["CCO"],
            products_smiles=["CC=O"],
            provenance="enzymatic"
        )
        assert scorer._initialized is True

    @pytest.mark.skipif(DORA_XGB_AVAILABLE, reason="Test only when DORA-XGB is NOT installed")
    def test_enzymatic_without_dora_xgb_returns_none(self):
        """Without DORA-XGB, enzymatic reactions should return None."""
        scorer = FeasibilityScorer()
        score, label = scorer.score_reaction(
            reactants_smiles=["CCO"],
            products_smiles=["CC=O"],
            provenance="enzymatic"
        )
        # Without DORA-XGB, should return None
        assert score is None
        assert label is None


class TestNodeFeasibilityAttributes:
    """Tests for feasibility attributes on Node class."""

    def test_node_has_feasibility_attributes(self):
        """Node should have feasibility_score and feasibility_label attributes."""
        from DORAnet_agent.node import Node
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None, depth=0, provenance="enzymatic")

        # Attributes should exist and be None by default
        assert hasattr(node, 'feasibility_score')
        assert hasattr(node, 'feasibility_label')
        assert node.feasibility_score is None
        assert node.feasibility_label is None

    def test_node_feasibility_can_be_set(self):
        """Node feasibility attributes should be settable."""
        from DORAnet_agent.node import Node
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None, depth=0, provenance="enzymatic")

        # Set values
        node.feasibility_score = 0.85
        node.feasibility_label = 1

        assert node.feasibility_score == 0.85
        assert node.feasibility_label == 1
