"""
Tests for DORAnet MCTS policies module.

Tests cover:
- Reward policies (Sparse, SAScore+Terminal)
- SA Score helpers and utilities
- Terminal detection and PKS database contents
- MCTS integration (preprocessing, sink compounds, pathways)
"""

import pytest
from unittest.mock import MagicMock, patch
from rdkit import Chem

from DORAnet_agent.policies import (
    # Base classes
    RewardPolicy,
    # Terminal detection
    NoOpTerminalDetector,
    # Reward policies
    SparseTerminalRewardPolicy,
    SAScore_and_TerminalRewardPolicy,
    # Thermodynamic scaling
    ThermodynamicScaledRewardPolicy,
)
from DORAnet_agent.node import Node


class TestSparseTerminalRewardPolicy:
    """Tests for SparseTerminalRewardPolicy."""

    def test_sink_compound_returns_sink_reward(self):
        policy = SparseTerminalRewardPolicy(sink_terminal_reward=0.5)
        node = MagicMock()
        node.is_sink_compound = True
        node.is_pks_terminal = False
        node.smiles = "CCO"

        reward = policy.calculate_reward(node, {})
        assert reward == 0.5

    def test_pks_terminal_returns_one(self):
        policy = SparseTerminalRewardPolicy(sink_terminal_reward=0.5)
        node = MagicMock()
        node.is_sink_compound = False
        node.is_pks_terminal = True
        node.smiles = "CCO"

        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_pks_library_match_returns_one(self):
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False

        policy = SparseTerminalRewardPolicy(pks_library={"CCO"})
        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_non_terminal_returns_zero(self):
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False

        policy = SparseTerminalRewardPolicy(pks_library={"CCC"})  # no match
        reward = policy.calculate_reward(node, {})
        assert reward == 0.0



class TestSAScoreHelpers:
    """Tests for SA Score helper functions."""
    
    def test_sa_score_to_reward_formula(self):
        """Test the SA Score to reward conversion formula."""
        from DORAnet_agent.policies.utils import sa_score_to_reward
        
        # Easy synthesis (SA Score = 1) -> high reward
        assert sa_score_to_reward(1.0) == pytest.approx(0.9)
        
        # Moderate synthesis (SA Score = 5) -> medium reward
        assert sa_score_to_reward(5.0) == pytest.approx(0.5)
        
        # Hard synthesis (SA Score = 10) -> low reward
        assert sa_score_to_reward(10.0) == pytest.approx(0.0)
        
        # Typical range
        assert sa_score_to_reward(3.0) == pytest.approx(0.7)
        assert sa_score_to_reward(8.0) == pytest.approx(0.2)
    
    def test_sa_score_to_reward_with_max(self):
        """Test SA Score reward capping with sa_max_reward."""
        from DORAnet_agent.policies.utils import sa_score_to_reward
        
        # With default max_reward=1.0, no cap effect
        assert sa_score_to_reward(1.0, max_reward=1.0) == pytest.approx(0.9)
        
        # With max_reward=0.5, cap takes effect
        assert sa_score_to_reward(1.0, max_reward=0.5) == pytest.approx(0.5)
        assert sa_score_to_reward(5.0, max_reward=0.5) == pytest.approx(0.5)
        
        # With max_reward=0.3
        assert sa_score_to_reward(1.0, max_reward=0.3) == pytest.approx(0.3)
        assert sa_score_to_reward(8.0, max_reward=0.3) == pytest.approx(0.2)  # below cap
    
    def test_calculate_sa_score_for_simple_molecules(self):
        """Test SA Score calculation for simple molecules."""
        from DORAnet_agent.policies.utils import calculate_sa_score, _SA_SCORE_AVAILABLE
        
        if not _SA_SCORE_AVAILABLE:
            pytest.skip("SA Score not available in this environment")
        
        # Simple molecules should have low SA scores (easy to synthesize)
        mol = Chem.MolFromSmiles("CCCCCCCCC(=O)O")  # nonanoic acid
        sa_score = calculate_sa_score(mol)
        assert sa_score is not None
        assert 1.0 <= sa_score <= 10.0
        assert sa_score < 3.0  # Should be easy to synthesize
        
        # Ethanol - very simple
        mol = Chem.MolFromSmiles("CCO")
        sa_score = calculate_sa_score(mol)
        assert sa_score is not None
        assert sa_score < 2.5  # Very easy


class TestSAScore_and_TerminalRewardPolicy:
    """Tests for SAScore_and_TerminalRewardPolicy."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        policy = SAScore_and_TerminalRewardPolicy()
        assert policy.sink_terminal_reward == 1.0
        assert policy.pks_terminal_reward == 1.0
        assert policy.sa_max_reward == 1.0
        assert policy.sa_fallback_reward == 0.0
        assert "SAScore" in policy.name
        assert "Terminal" in policy.name

    def test_sink_compound_gets_terminal_reward(self):
        """Sink compounds should get full terminal reward."""
        policy = SAScore_and_TerminalRewardPolicy(sink_terminal_reward=1.0)
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = True
        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_pks_terminal_gets_terminal_reward(self):
        """PKS terminals should get full terminal reward."""
        policy = SAScore_and_TerminalRewardPolicy(pks_terminal_reward=1.0)
        mol = Chem.MolFromSmiles("CCCCCC(=O)O")
        node = Node(fragment=mol, parent=None)
        node.is_pks_terminal = True
        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_non_terminal_gets_sa_score(self):
        """Non-terminal compounds should get SA score reward."""
        policy = SAScore_and_TerminalRewardPolicy()
        mol = Chem.MolFromSmiles("CCO")  # Ethanol - easy to synthesize
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False
        reward = policy.calculate_reward(node, {})
        # SA score for ethanol is ~1.5, so reward should be ~0.85
        assert 0.5 < reward < 1.0

    def test_complex_non_terminal_lower_reward(self):
        """Complex non-terminals should get lower SA score reward."""
        policy = SAScore_and_TerminalRewardPolicy()
        # Complex natural product-like structure
        mol = Chem.MolFromSmiles("CC1=C2C(=O)C3=C(C=CC=C3O)C(=O)C2=CC=C1")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False
        reward = policy.calculate_reward(node, {})
        assert 0.0 <= reward <= 1.0

    def test_priority_sink_over_pks(self):
        """Sink compound takes priority over PKS terminal."""
        policy = SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=0.8,
            pks_terminal_reward=0.9,
        )
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = True
        node.is_pks_terminal = True  # Both flags set
        reward = policy.calculate_reward(node, {})
        assert reward == 0.8  # Sink reward takes priority

    def test_pks_library_match_fallback(self):
        """PKS library match should give terminal reward."""
        pks_library = {"CCO"}  # Ethanol in PKS library
        policy = SAScore_and_TerminalRewardPolicy(
            pks_terminal_reward=1.0,
            pks_library=pks_library,
        )
        mol = Chem.MolFromSmiles("CCO")
        node = Node(fragment=mol, parent=None)
        # node.smiles is computed from fragment, so it should be "CCO"
        node.is_sink_compound = False
        node.is_pks_terminal = False  # Not marked, but in library
        reward = policy.calculate_reward(node, {})
        assert reward == 1.0

    def test_sa_max_reward_cap(self):
        """SA score should be capped at sa_max_reward."""
        policy = SAScore_and_TerminalRewardPolicy(sa_max_reward=0.5)
        mol = Chem.MolFromSmiles("C")  # Methane - trivially easy
        node = Node(fragment=mol, parent=None)
        node.is_sink_compound = False
        node.is_pks_terminal = False
        reward = policy.calculate_reward(node, {})
        assert reward <= 0.5

    def test_inherits_from_reward_policy(self):
        """Test proper inheritance."""
        policy = SAScore_and_TerminalRewardPolicy()
        assert isinstance(policy, RewardPolicy)

    def test_thermodynamic_scaling_wrapper(self):
        """Test wrapping with ThermodynamicScaledRewardPolicy."""
        base = SAScore_and_TerminalRewardPolicy()
        scaled = ThermodynamicScaledRewardPolicy(
            base_policy=base,
            feasibility_weight=0.5,
        )
        assert "ThermodynamicScaled" in scaled.name
        assert "SAScore" in scaled.name



class TestPKSDatabaseContents:
    """
    Tests to verify specific molecules are present in the PKS database.

    These tests use the actual PKS database (expanded_PKS_SMILES_V3.txt)
    to verify that key target molecules are included.
    """

    @pytest.fixture
    def pks_library(self):
        """Load the PKS library as a set of canonical SMILES."""
        from DORAnet_agent.policies.utils import canonicalize_smiles
        from pathlib import Path
        pks_file = Path(__file__).resolve().parents[1] / "data" / "processed" / "expanded_PKS_SMILES_V3.txt"
        if not pks_file.exists():
            pytest.skip("PKS library file not found")
        pks_set = set()
        with open(pks_file) as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    canon = canonicalize_smiles(smiles)
                    if canon:
                        pks_set.add(canon)
        return pks_set

    def _is_in_pks(self, smiles: str, pks_library) -> bool:
        """Check if a SMILES is in the PKS library."""
        from DORAnet_agent.policies.utils import canonicalize_smiles
        canon = canonicalize_smiles(smiles)
        return canon is not None and canon in pks_library

    def test_4_hydroxybutyric_acid_in_database(self, pks_library):
        """Test that 4-hydroxybutyric acid is in the PKS database."""
        assert self._is_in_pks("OCCCC(=O)O", pks_library), \
            "4-hydroxybutyric acid should be in PKS database"

    def test_5_ketohexanoic_acid_in_database(self, pks_library):
        """Test that 5-ketohexanoic acid is in the PKS database."""
        assert self._is_in_pks("CC(=O)CCCC(=O)O", pks_library), \
            "5-ketohexanoic acid should be in PKS database"

    def test_tiglic_acid_in_database(self, pks_library):
        """Test that tiglic acid is in the PKS database."""
        assert self._is_in_pks("CC=CC(=O)O", pks_library), \
            "Tiglic acid should be in PKS database"

    def test_gamma_valerolactone_not_in_database(self, pks_library):
        """Test that gamma-valerolactone is NOT in the PKS database."""
        assert not self._is_in_pks("CC1CCC(=O)O1", pks_library), \
            "Gamma-valerolactone should NOT be in PKS database"

    def test_hydroxyethyl_furanone_in_database(self, pks_library):
        """Test that 5-(2-hydroxyethylidene)furan-2(5H)-one is in the PKS database."""
        assert self._is_in_pks("O=C1C=CC(=CCO)O1", pks_library), \
            "Hydroxyethyl furanone should be in PKS database"

    def test_styryl_lactone_in_database(self, pks_library):
        """Test that 6-styryl-5,6-dihydro-2H-pyran-2-one is in the PKS database."""
        assert self._is_in_pks("O=C1C=CCC(C=Cc2ccccc2)O1", pks_library), \
            "Styryl lactone should be in PKS database"

    def test_database_loads_correct_molecule_count(self, pks_library):
        """Test that the PKS database loads the expected number of molecules."""
        num_molecules = len(pks_library)
        assert num_molecules > 900000, \
            f"PKS database should have >900k molecules, got {num_molecules}"
        assert num_molecules < 1100000, \
            f"PKS database should have <1.1M molecules, got {num_molecules}"


class TestPreprocessTargetMolecule:
    """Tests for preprocess_target_molecule helper function."""

    def test_removes_stereochemistry(self):
        """Test that stereochemistry is removed from molecules."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Molecule with chiral center (2-butanol)
        mol_with_stereo = Chem.MolFromSmiles("C[C@H](O)CC")
        preprocessed, smiles = preprocess_target_molecule(mol_with_stereo)

        # Resulting SMILES should not contain @ symbols
        assert "@" not in smiles
        # The canonical SMILES for 2-butanol without stereo is "CCC(C)O"
        assert smiles == "CCC(C)O"

    def test_removes_ez_stereochemistry(self):
        """Test that E/Z double bond stereochemistry is removed."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Molecule with E/Z stereochemistry
        mol_with_ez = Chem.MolFromSmiles("C/C=C/C")
        preprocessed, smiles = preprocess_target_molecule(mol_with_ez)

        # Resulting SMILES should not contain / or \ symbols
        assert "/" not in smiles
        assert "\\" not in smiles

    def test_produces_canonical_smiles(self):
        """Test that output SMILES is canonical."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Two equivalent but different SMILES representations
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("OCC")

        _, smiles1 = preprocess_target_molecule(mol1)
        _, smiles2 = preprocess_target_molecule(mol2)

        # Both should produce the same canonical SMILES
        assert smiles1 == smiles2

    def test_sanitizes_molecule(self):
        """Test that molecule is sanitized."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Valid molecule
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        preprocessed, smiles = preprocess_target_molecule(mol)

        # Should produce valid aromatic SMILES
        assert preprocessed is not None
        assert "c" in smiles.lower()  # Should contain aromatic carbons

    def test_raises_on_none_molecule(self):
        """Test that None molecule raises ValueError."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        with pytest.raises(ValueError, match="Cannot preprocess None"):
            preprocess_target_molecule(None)

    def test_returns_valid_mol_object(self):
        """Test that returned molecule is a valid RDKit Mol."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        mol = Chem.MolFromSmiles("CCO")
        preprocessed, smiles = preprocess_target_molecule(mol)

        # Should be a valid Mol object
        assert preprocessed is not None
        assert isinstance(preprocessed, Chem.Mol)
        assert preprocessed.GetNumAtoms() == mol.GetNumAtoms()

    def test_complex_molecule_with_multiple_stereocenters(self):
        """Test preprocessing of complex molecule with multiple stereocenters."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Molecule with multiple chiral centers (e.g., a sugar-like structure)
        mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](O)[C@H](O)C")
        preprocessed, smiles = preprocess_target_molecule(mol)

        # No stereochemistry in output
        assert "@" not in smiles

        # Atom count should be preserved
        assert preprocessed.GetNumAtoms() == mol.GetNumAtoms()

    def test_kavalactone_with_stereo(self):
        """Test preprocessing of kavalactone-like structure with stereochemistry."""
        from DORAnet_agent.mcts import preprocess_target_molecule

        # Kavain with added stereochemistry
        mol = Chem.MolFromSmiles("COC1=CC(OC(/C=C/C2=CC=CC=C2)C1)=O")
        preprocessed, smiles = preprocess_target_molecule(mol)

        # Should succeed and remove E/Z stereochemistry
        assert preprocessed is not None
        assert "/" not in smiles
        assert "\\" not in smiles


class TestGetSinkCompoundType:
    """Tests for _get_sink_compound_type method which labels byproducts in pathway output."""

    @pytest.fixture
    def mcts_agent(self):
        """Create a minimal DORAnetMCTS agent for testing sink compound type labeling."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        mol = Chem.MolFromSmiles("CCCCC(=O)O")  # pentanoic acid
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        # Manually set up test data for the various categories
        agent.biological_sink_compounds = {"CCO", "CCC"}  # ethanol, propane
        agent.chemical_sink_compounds = {"C", "CC"}  # methane, ethane
        agent.bio_cofactors = {
            "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CSCC[C@H](N)C(=O)O)[C@@H](O)[C@H]1O"  # SAH (canonical)
        }
        agent.chemistry_helpers = {"O", "[H][H]"}  # water, hydrogen

        return agent

    def test_biological_sink_compound(self, mcts_agent):
        """Test that biological building blocks return 'biological'."""
        result = mcts_agent._get_sink_compound_type("CCO")  # ethanol
        assert result == "biological"

    def test_chemical_sink_compound(self, mcts_agent):
        """Test that chemical building blocks return 'chemical'."""
        result = mcts_agent._get_sink_compound_type("C")  # methane
        assert result == "chemical"

    def test_bio_cofactor(self, mcts_agent):
        """Test that biology cofactors return 'bio_cofactor'."""
        # SAH (S-adenosyl-L-homocysteine) - a common enzymatic cofactor byproduct
        sah_smiles = "NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](CSCC[C@H](N)C(=O)O)[C@@H](O)[C@H]1O"
        result = mcts_agent._get_sink_compound_type(sah_smiles)
        assert result == "bio_cofactor"

    def test_chemistry_helper(self, mcts_agent):
        """Test that chemistry helpers return 'chem_helper'."""
        result = mcts_agent._get_sink_compound_type("O")  # water
        assert result == "chem_helper"

        result = mcts_agent._get_sink_compound_type("[H][H]")  # hydrogen
        assert result == "chem_helper"

    def test_unknown_compound_returns_none(self, mcts_agent):
        """Test that unknown compounds return None."""
        result = mcts_agent._get_sink_compound_type("CCCCCCCCCC")  # decane
        assert result is None

    def test_invalid_smiles_returns_none(self, mcts_agent):
        """Test that invalid SMILES return None."""
        result = mcts_agent._get_sink_compound_type("invalid_smiles_xyz")
        assert result is None

    def test_priority_biological_over_bio_cofactor(self, mcts_agent):
        """Test that biological sink compounds take priority over bio_cofactors."""
        # Add the same compound to both sets
        test_smiles = "CCCC"
        mcts_agent.biological_sink_compounds.add(test_smiles)
        mcts_agent.bio_cofactors.add(test_smiles)

        result = mcts_agent._get_sink_compound_type(test_smiles)
        assert result == "biological"  # biological takes priority

    def test_priority_chemical_over_chem_helper(self, mcts_agent):
        """Test that chemical sink compounds take priority over chem_helpers."""
        # Add the same compound to both sets
        test_smiles = "CCCCC"
        mcts_agent.chemical_sink_compounds.add(test_smiles)
        mcts_agent.chemistry_helpers.add(test_smiles)

        result = mcts_agent._get_sink_compound_type(test_smiles)
        assert result == "chemical"  # chemical takes priority

    def test_pks_library_match(self, mcts_agent):
        """Test that PKS library matches return 'pks'."""
        # Add a test SMILES to pks_library
        mcts_agent.pks_library = {"CCCCC(=O)O"}  # pentanoic acid
        result = mcts_agent._get_sink_compound_type("CCCCC(=O)O")
        assert result == "pks"

    def test_pks_library_lower_priority_than_building_blocks(self, mcts_agent):
        """Test that PKS library has lower priority than building blocks."""
        # Add the same compound to biological and pks_library
        test_smiles = "CCCCCC"  # hexane
        mcts_agent.biological_sink_compounds.add(test_smiles)
        mcts_agent.pks_library = {test_smiles}

        result = mcts_agent._get_sink_compound_type(test_smiles)
        assert result == "biological"  # building block takes priority over pks

    def test_pks_library_lower_priority_than_cofactors(self, mcts_agent):
        """Test that PKS library has lower priority than cofactors."""
        # Add the same compound to bio_cofactors and pks_library
        test_smiles = "CCCCCCC"  # heptane
        mcts_agent.bio_cofactors.add(test_smiles)
        mcts_agent.pks_library = {test_smiles}

        result = mcts_agent._get_sink_compound_type(test_smiles)
        assert result == "bio_cofactor"  # cofactor takes priority over pks


class TestBioCofactorsTracking:
    """Tests for bio_cofactors set initialization and tracking."""

    def test_bio_cofactors_set_exists(self):
        """Test that bio_cofactors set is created during initialization."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        assert hasattr(agent, 'bio_cofactors')
        assert isinstance(agent.bio_cofactors, set)

    def test_bio_cofactors_populated_from_cofactors_file(self, tmp_path):
        """Test that cofactor files populate bio_cofactors set."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        # Create a temporary cofactors CSV file (name must contain 'cofactors')
        # Note: CSV reader expects column named "SMILES" (uppercase)
        cofactors_file = tmp_path / "biology_cofactors.csv"
        cofactors_file.write_text("SMILES,name\nCCO,ethanol\nCCC,propane\n")

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
            cofactors_files=[str(cofactors_file)],
        )

        # bio_cofactors should contain the loaded cofactors
        assert "CCO" in agent.bio_cofactors
        assert "CCC" in agent.bio_cofactors
        # They should also be in excluded_fragments
        assert "CCO" in agent.excluded_fragments
        assert "CCC" in agent.excluded_fragments

    def test_chemistry_helpers_not_in_bio_cofactors(self, tmp_path):
        """Test that chemistry_helpers file populates chemistry_helpers, not bio_cofactors."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        # Create a temporary chemistry_helpers CSV file
        # Note: CSV reader expects column named "SMILES" (uppercase)
        helpers_file = tmp_path / "test_chemistry_helpers.csv"
        helpers_file.write_text("SMILES,name\nO,water\n[H][H],hydrogen\n")

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
            cofactors_files=[str(helpers_file)],
        )

        # chemistry_helpers should contain the loaded helpers
        assert "O" in agent.chemistry_helpers
        # They should NOT be in bio_cofactors (file name has 'chemistry_helpers')
        assert "O" not in agent.bio_cofactors
        # They should also be in excluded_fragments
        assert "O" in agent.excluded_fragments


class TestSaveSuccessfulPathways:
    """Tests for save_successful_pathways method."""

    def test_leaf_node_without_sink_or_pks_excluded(self, tmp_path):
        """Test that leaf nodes without sink/PKS status are excluded from successful pathways."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        # Create a leaf node that is NOT a sink compound
        leaf_mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC")  # Long chain, not in any library
        leaf = Node(fragment=leaf_mol, parent=root, depth=1, provenance="enzymatic")
        leaf.is_sink_compound = False
        leaf.is_pks_terminal = False
        root.children.append(leaf)
        agent.nodes.append(leaf)

        # Save successful pathways
        output_file = tmp_path / "test_successful.txt"
        agent.save_successful_pathways(str(output_file))

        # Read the file and verify the leaf node is NOT included
        content = output_file.read_text()
        assert "CCCCCCCCCCCCCCCC" not in content

    def test_leaf_node_with_sink_compound_included(self, tmp_path):
        """Test that leaf nodes with sink compound status ARE included in successful pathways."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        # Add a known sink compound to the biological_sink_compounds set
        sink_smiles = "CCO"  # ethanol
        agent.biological_sink_compounds.add(sink_smiles)

        # Create a leaf node that IS a sink compound
        leaf_mol = Chem.MolFromSmiles(sink_smiles)
        leaf = Node(fragment=leaf_mol, parent=root, depth=1, provenance="enzymatic")
        leaf.is_sink_compound = True
        root.children.append(leaf)
        agent.nodes.append(leaf)

        # Save successful pathways
        output_file = tmp_path / "test_successful.txt"
        agent.save_successful_pathways(str(output_file))

        # Read the file and verify the leaf node IS included
        content = output_file.read_text()
        assert "CCO" in content or "OCC" in content  # canonical SMILES may vary

    def test_leaf_node_with_pks_terminal_included(self, tmp_path):
        """Test that leaf nodes with PKS terminal status ARE included in successful pathways."""
        from DORAnet_agent.mcts import DORAnetMCTS
        from DORAnet_agent.node import Node
        from DORAnet_agent.mcts import RetroTideResult

        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        agent = DORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=1,
            max_depth=1,
            use_enzymatic=False,
            use_synthetic=False,
        )

        # Create a leaf node that is a PKS terminal
        pks_smiles = "CCCC(=O)O"  # butyric acid
        leaf_mol = Chem.MolFromSmiles(pks_smiles)
        leaf = Node(fragment=leaf_mol, parent=root, depth=1, provenance="enzymatic")
        leaf.is_sink_compound = False
        leaf.is_pks_terminal = True
        root.children.append(leaf)
        agent.nodes.append(leaf)

        # Add a successful RetroTide result for this node
        agent.retrotide_results.append(
            RetroTideResult(
                doranet_node_id=1,
                doranet_node_smiles=pks_smiles,
                doranet_node_depth=1,
                doranet_node_provenance="enzymatic",
                retrotide_successful=True,
            )
        )

        # Save successful pathways
        output_file = tmp_path / "test_successful.txt"
        agent.save_successful_pathways(str(output_file))

        # Read the file and verify the leaf node IS included
        content = output_file.read_text()
        assert "CCCC(=O)O" in content or pks_smiles in content

    @pytest.mark.slow
    def test_pentanoic_acid_integration_all_modalities(self, tmp_path):
        """
        Integration test: Run minimal MCTS on pentanoic acid and verify
        all successful pathways have covered terminal fragments and byproducts.

        This test runs a real (but minimal) search to verify end-to-end
        pathway validation logic.
        """
        from pathlib import Path
        import re
        from DORAnet_agent.async_expansion_mcts import AsyncExpansionDORAnetMCTS
        from DORAnet_agent.node import Node

        REPO_ROOT = Path(__file__).resolve().parents[1]

        # Target: pentanoic acid
        target_smiles = "CCCCC(=O)O"
        mol = Chem.MolFromSmiles(target_smiles)
        root = Node(fragment=mol, parent=None, depth=0, provenance="target")

        # Minimal MCTS configuration
        agent = AsyncExpansionDORAnetMCTS(
            root=root,
            target_molecule=mol,
            total_iterations=20,
            max_depth=2,
            max_children_per_expand=10,
            use_enzymatic=True,
            use_synthetic=True,
            use_chem_building_blocksDB=True,
            use_bio_building_blocksDB=True,
            use_PKS_building_blocksDB=True,
            cofactors_files=[
                str(REPO_ROOT / "data" / "raw" / "all_cofactors.csv"),
                str(REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv"),
            ],
            pks_library_file=str(REPO_ROOT / "data" / "processed" / "expanded_PKS_SMILES_V3.txt"),
            sink_compounds_files=[
                str(REPO_ROOT / "data" / "processed" / "biological_building_blocks.txt"),
                str(REPO_ROOT / "data" / "processed" / "chemical_building_blocks.txt"),
            ],
            terminal_detector=NoOpTerminalDetector(),
            reward_policy=SparseTerminalRewardPolicy(sink_terminal_reward=1.0),
            num_workers=2,
        )

        # Run the search
        agent.run()

        # Save successful pathways
        output_file = tmp_path / "successful_pathways.txt"
        agent.save_successful_pathways(str(output_file))

        # Read and parse the output
        content = output_file.read_text()

        # === VERIFICATION 1: File format ===
        assert "SUCCESSFUL PATHWAYS (PKS OR SINK PRODUCTS ONLY)" in content
        assert "RUN CONFIGURATION" in content
        assert "PATHWAY TYPE BREAKDOWN" in content

        # Extract total pathways count from header
        total_match = re.search(r"Total pathways:\s*(\d+)", content)
        assert total_match is not None, "Could not find 'Total pathways' in output"
        total_pathways = int(total_match.group(1))

        # Count actual pathway blocks
        pathway_blocks = re.findall(r"PATHWAY #(\d+):", content)
        assert len(pathway_blocks) == total_pathways, \
            f"Header says {total_pathways} pathways but found {len(pathway_blocks)} blocks"

        # If no pathways found, test passes (nothing to validate)
        if total_pathways == 0:
            return

        # === VERIFICATION 2: All terminal fragments are covered ===
        # Build the coverage sets
        coverage_sets = (
            agent.biological_sink_compounds |
            agent.chemical_sink_compounds |
            agent.pks_library |
            agent.excluded_fragments
        )

        # Extract terminal fragments from each pathway
        terminal_pattern = r"Terminal Fragment:\s*(\S+)"
        terminals = re.findall(terminal_pattern, content)

        for terminal_smiles in terminals:
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(terminal_smiles))
            assert canonical in coverage_sets, \
                f"Terminal fragment '{terminal_smiles}' (canonical: {canonical}) is not covered"

        # === VERIFICATION 3: All byproducts are covered ===
        # Extract products from each step, check non-primary ones
        # Pattern matches lines like: "- SMILES [branch, sink=chemical]"
        product_pattern = r"-\s*(\S+)\s*\[(\w+),\s*sink=(\w+)\]"
        products = re.findall(product_pattern, content)

        for smiles, role, sink_type in products:
            if role == "primary":
                continue  # Primary products continue along pathway
            # Byproducts must be covered
            if sink_type == "No":
                # If marked as sink=No, it should NOT be in successful pathways
                # This would indicate a bug
                pytest.fail(f"Byproduct '{smiles}' marked as sink=No in successful pathway")
