"""Tests for DORAnet_agent.policies.utils — pure utility functions."""

from unittest.mock import patch

import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS

from DORAnet_agent.policies.utils import (
    calculate_mcs_similarity_without_stereo,
    calculate_tanimoto_similarity,
    canonicalize_smiles,
    generate_morgan_fingerprint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ethanol_mol():
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def propanol_mol():
    return Chem.MolFromSmiles("CCCO")


@pytest.fixture
def benzene_mol():
    return Chem.MolFromSmiles("c1ccccc1")


# ---------------------------------------------------------------------------
# TestCanonicalizeSmiles
# ---------------------------------------------------------------------------

class TestCanonicalizeSmiles:
    """Tests for canonicalize_smiles()."""

    def test_non_canonical_input(self):
        # OCC is non-canonical for ethanol; canonical is CCO
        assert canonicalize_smiles("OCC") == "CCO"

    def test_already_canonical(self):
        assert canonicalize_smiles("CCO") == "CCO"

    def test_invalid_smiles_returns_none(self):
        assert canonicalize_smiles("not_a_smiles") is None

    def test_empty_string_returns_empty(self):
        # RDKit treats "" as a valid (empty) molecule
        assert canonicalize_smiles("") == ""

    def test_complex_molecule_canonicalizes(self):
        # Non-canonical propionic acid → canonical form
        result = canonicalize_smiles("C(C(=O)O)C")
        assert result == "CCC(=O)O"


# ---------------------------------------------------------------------------
# TestGenerateMorganFingerprint
# ---------------------------------------------------------------------------

class TestGenerateMorganFingerprint:
    """Tests for generate_morgan_fingerprint()."""

    def test_valid_mol_returns_bitvect(self, ethanol_mol):
        fp = generate_morgan_fingerprint(ethanol_mol)
        assert isinstance(fp, DataStructs.ExplicitBitVect)

    def test_default_length_is_2048(self, ethanol_mol):
        fp = generate_morgan_fingerprint(ethanol_mol)
        assert fp.GetNumBits() == 2048

    def test_custom_radius_and_bits(self, ethanol_mol):
        fp = generate_morgan_fingerprint(ethanol_mol, radius=3, n_bits=1024)
        assert fp.GetNumBits() == 1024

    def test_different_molecules_different_fingerprints(self, ethanol_mol, benzene_mol):
        fp_eth = generate_morgan_fingerprint(ethanol_mol)
        fp_ben = generate_morgan_fingerprint(benzene_mol)
        # At least one bit position must differ
        assert fp_eth != fp_ben


# ---------------------------------------------------------------------------
# TestCalculateTanimotoSimilarity
# ---------------------------------------------------------------------------

class TestCalculateTanimotoSimilarity:
    """Tests for calculate_tanimoto_similarity()."""

    def test_identical_molecule_returns_one(self, ethanol_mol):
        fp = generate_morgan_fingerprint(ethanol_mol)
        assert calculate_tanimoto_similarity(fp, fp) == 1.0

    def test_different_molecules_less_than_one(self, ethanol_mol, benzene_mol):
        fp_eth = generate_morgan_fingerprint(ethanol_mol)
        fp_ben = generate_morgan_fingerprint(benzene_mol)
        sim = calculate_tanimoto_similarity(fp_eth, fp_ben)
        assert 0.0 <= sim < 1.0

    def test_symmetry(self, ethanol_mol, benzene_mol):
        fp_a = generate_morgan_fingerprint(ethanol_mol)
        fp_b = generate_morgan_fingerprint(benzene_mol)
        assert calculate_tanimoto_similarity(fp_a, fp_b) == calculate_tanimoto_similarity(fp_b, fp_a)

    def test_returns_float(self, ethanol_mol, benzene_mol):
        fp_a = generate_morgan_fingerprint(ethanol_mol)
        fp_b = generate_morgan_fingerprint(benzene_mol)
        assert isinstance(calculate_tanimoto_similarity(fp_a, fp_b), float)


# ---------------------------------------------------------------------------
# TestCalculateMcsSimilarityWithoutStereo
# ---------------------------------------------------------------------------

class TestCalculateMcsSimilarityWithoutStereo:
    """Tests for calculate_mcs_similarity_without_stereo()."""

    def test_identical_molecules_returns_one(self, ethanol_mol):
        sim = calculate_mcs_similarity_without_stereo(ethanol_mol, ethanol_mol)
        assert sim == pytest.approx(1.0)

    def test_substructure_similarity(self, ethanol_mol, propanol_mol):
        sim = calculate_mcs_similarity_without_stereo(ethanol_mol, propanol_mol)
        # Ethanol (3 heavy atoms) vs propanol (4 heavy atoms)
        # MCS should share ~3 atoms, union = 3+4-3 = 4, so sim ~ 3/4 = 0.75
        assert sim is not None
        assert 0.5 < sim < 1.0

    def test_very_different_molecules(self, benzene_mol):
        methane = Chem.MolFromSmiles("C")
        sim = calculate_mcs_similarity_without_stereo(methane, benzene_mol)
        assert sim is not None
        assert sim < 0.5

    def test_no_common_substructure_returns_zero(self):
        # Two very different atoms with matchValences=True can yield numAtoms=0
        # Fluorine atom vs silicon atom
        mol_f = Chem.MolFromSmiles("[F]")
        mol_si = Chem.MolFromSmiles("[Si]")
        sim = calculate_mcs_similarity_without_stereo(mol_f, mol_si)
        assert sim == 0.0

    @patch("DORAnet_agent.policies.utils.rdFMCS.FindMCS", side_effect=RuntimeError("boom"))
    def test_exception_returns_none(self, mock_find_mcs, ethanol_mol, propanol_mol):
        sim = calculate_mcs_similarity_without_stereo(ethanol_mol, propanol_mol)
        assert sim is None
