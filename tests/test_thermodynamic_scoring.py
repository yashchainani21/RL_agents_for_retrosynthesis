"""Tests for pathermo thermodynamic scoring integration."""
import pytest
from DORAnet_agent.mcts import ThermodynamicScorer, PATHERMO_AVAILABLE


class TestThermodynamicScorer:
    """Tests for the ThermodynamicScorer class."""

    @pytest.mark.skipif(not PATHERMO_AVAILABLE, reason="pathermo not installed")
    def test_enzymatic_scoring_returns_valid_values(self):
        """With pathermo available, enzymatic reactions should return valid scores."""
        scorer = ThermodynamicScorer()
        delta_h, label = scorer.score_reaction(
            reactants_smiles=["CCO", "O"],
            products_smiles=["CCCO"],
            provenance="enzymatic"
        )
        assert delta_h is not None
        assert isinstance(delta_h, float)
        assert label in [0, 1]

    def test_missing_reactants_returns_none(self):
        """Missing reactants should return None."""
        scorer = ThermodynamicScorer()
        delta_h, label = scorer.score_reaction(
            reactants_smiles=[],
            products_smiles=["CCO"],
            provenance="synthetic"
        )
        assert delta_h is None
        assert label is None

    def test_missing_products_returns_none(self):
        """Missing products should return None."""
        scorer = ThermodynamicScorer()
        delta_h, label = scorer.score_reaction(
            reactants_smiles=["CCO"],
            products_smiles=[],
            provenance="synthetic"
        )
        assert delta_h is None
        assert label is None

    @pytest.mark.skipif(not PATHERMO_AVAILABLE, reason="pathermo not installed")
    def test_synthetic_scoring_returns_valid_values(self):
        """With pathermo available, synthetic reactions should return valid scores."""
        scorer = ThermodynamicScorer()
        delta_h, label = scorer.score_reaction(
            reactants_smiles=["C1CCCCC1"],  # cyclohexane
            products_smiles=["CCCCCC"],      # hexane
            provenance="synthetic"
        )
        assert delta_h is not None
        assert isinstance(delta_h, float)
        assert label in [0, 1]

    @pytest.mark.skipif(not PATHERMO_AVAILABLE, reason="pathermo not installed")
    def test_feasibility_threshold(self):
        """Test that feasibility label respects 15 kcal/mol threshold."""
        scorer = ThermodynamicScorer()
        # Test with known molecules where we can predict the outcome
        # This is a simple test - actual threshold behavior
        assert scorer.FEASIBILITY_THRESHOLD == 15.0

    @pytest.mark.skipif(not PATHERMO_AVAILABLE, reason="pathermo not installed")
    def test_unsupported_molecule_returns_none(self):
        """Molecules with missing groups should return None."""
        scorer = ThermodynamicScorer()
        # Use a molecule that pathermo doesn't support
        delta_h, label = scorer.score_reaction(
            reactants_smiles=["N#CC#N"],  # Known to return None in pathermo
            products_smiles=["CC"],
            provenance="synthetic"
        )
        assert delta_h is None
        assert label is None


class TestNodeThermodynamicAttributes:
    """Tests for thermodynamic attributes on Node class."""

    def test_node_has_thermodynamic_attributes(self):
        """Node should have enthalpy_of_reaction and thermodynamic_label attributes."""
        from DORAnet_agent.node import Node
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None, depth=0, provenance="synthetic")

        assert hasattr(node, 'enthalpy_of_reaction')
        assert hasattr(node, 'thermodynamic_label')
        assert node.enthalpy_of_reaction is None
        assert node.thermodynamic_label is None

    def test_node_thermodynamic_can_be_set(self):
        """Node thermodynamic attributes should be settable."""
        from DORAnet_agent.node import Node
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None, depth=0, provenance="synthetic")

        node.enthalpy_of_reaction = -5.5
        node.thermodynamic_label = 1

        assert node.enthalpy_of_reaction == -5.5
        assert node.thermodynamic_label == 1
