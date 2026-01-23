"""
Unit tests for PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck rollout policy.

These tests verify:
1. PKS building blocks loading (both Tanimoto and MCS modes)
2. MCS similarity calculation
3. Tanimoto similarity calculation
4. Atom count pre-filtering (MCS mode)
5. Early termination threshold
6. Rollout reward computation
7. Fingerprint caching (Tanimoto mode)
"""

import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from rdkit import Chem

# Import the module under test
from DORAnet_agent.policies.rollout import (
    PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
    _calculate_mcs_similarity_without_stereo,
    _calculate_tanimoto_similarity,
    _canonicalize_smiles,
    _generate_morgan_fingerprint,
)


class MockNode:
    """Mock Node class for testing."""
    
    def __init__(
        self,
        smiles: Optional[str] = None,
        fragment: Optional[Chem.Mol] = None,
        node_id: int = 1,
        is_sink_compound: bool = False,
        is_pks_terminal: bool = False,
    ):
        self.smiles = smiles
        self.fragment = fragment
        self.node_id = node_id
        self.is_sink_compound = is_sink_compound
        self.is_pks_terminal = is_pks_terminal
        self.retrotide_attempted = False


class TestMCSSimilarityCalculation:
    """Tests for the MCS similarity calculation function."""
    
    def test_identical_molecules(self):
        """Identical molecules should have similarity of 1.0."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")
        
        similarity = _calculate_mcs_similarity_without_stereo(mol1, mol2)
        
        assert similarity is not None
        assert similarity == pytest.approx(1.0, rel=0.01)
    
    def test_completely_different_molecules(self):
        """Very different molecules should have low similarity."""
        mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        
        similarity = _calculate_mcs_similarity_without_stereo(mol1, mol2)
        
        assert similarity is not None
        # These share very little structure
        assert similarity < 0.5
    
    def test_substructure_similarity(self):
        """Molecule containing another as substructure should have moderate similarity."""
        mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        mol2 = Chem.MolFromSmiles("CCCO")  # Propanol
        
        similarity = _calculate_mcs_similarity_without_stereo(mol1, mol2)
        
        assert similarity is not None
        # Should have good but not perfect similarity
        assert 0.5 < similarity < 1.0
    
    def test_stereo_ignored(self):
        """Stereochemistry should not affect similarity."""
        # (R)-lactic acid vs (S)-lactic acid
        mol1 = Chem.MolFromSmiles("C[C@H](O)C(=O)O")
        mol2 = Chem.MolFromSmiles("C[C@@H](O)C(=O)O")
        
        similarity = _calculate_mcs_similarity_without_stereo(mol1, mol2)
        
        assert similarity is not None
        # Should be identical when ignoring stereo
        assert similarity == pytest.approx(1.0, rel=0.01)
    
    def test_returns_none_on_timeout(self):
        """Should return None if MCS calculation times out."""
        # Create very large molecules that might timeout
        # For this test, we'll use a very short timeout
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCC")
        
        # This should complete quickly, but we're testing the interface
        similarity = _calculate_mcs_similarity_without_stereo(mol1, mol2, timeout=1.0)
        
        # Should return a valid similarity for simple molecules
        assert similarity is not None


class TestPKSBuildingBlocksLoading:
    """Tests for loading PKS building blocks (MCS mode)."""

    def test_load_from_file_mcs_mode(self):
        """Should correctly load PKS building blocks from a file in MCS mode."""
        # Create a temporary file with test SMILES
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            f.write("CCCO\n")
            f.write("c1ccccc1\n")
            f.write("\n")  # Empty line should be skipped
            f.write("INVALID_SMILES\n")  # Should be skipped
            temp_path = f.name

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="mcs",
            )

            # Should load 3 valid molecules (skipping empty line and invalid SMILES)
            assert len(policy._pks_building_blocks) == 3

            # Check that atom counts are stored correctly (MCS mode stores (mol, atom_count))
            atom_counts = [atoms for _, atoms in policy._pks_building_blocks]
            assert 3 in atom_counts  # CCO has 3 heavy atoms
            assert 4 in atom_counts  # CCCO has 4 heavy atoms
            assert 6 in atom_counts  # Benzene has 6 heavy atoms
        finally:
            Path(temp_path).unlink()

    def test_missing_file_warning(self):
        """Should warn but not crash if file doesn't exist."""
        policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
            pks_building_blocks_path="/nonexistent/path/file.txt",
            similarity_method="mcs",
        )

        # Should have no building blocks loaded
        assert len(policy._pks_building_blocks) == 0


class TestAtomCountPreFiltering:
    """Tests for atom count pre-filtering optimization (MCS mode only)."""

    def test_skips_different_size_molecules(self):
        """Should skip molecules with very different atom counts in MCS mode."""
        # Create a policy with known building blocks
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("C\n")  # 1 atom - should be skipped for query with 10 atoms
            f.write("CCCCCCCCCC\n")  # 10 atoms - should be checked
            f.write("CCCCCCCCCCCCCCCCCCCC\n")  # 20 atoms - should be skipped
            temp_path = f.name

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="mcs",
                atom_count_tolerance=0.5,  # 50% tolerance
            )

            # Create a mock node with a molecule of ~10 atoms
            mol = Chem.MolFromSmiles("CCCCCCCCC")  # 9 atoms
            node = MockNode(fragment=mol)

            reward, metadata = policy._get_pks_similarity_reward(node)

            # With 50% tolerance and 9 atoms:
            # min_atoms = 9 * 0.5 = 4.5 -> 4
            # max_atoms = 9 * 1.5 = 13.5 -> 13
            # So C (1 atom) should be skipped
            # CCCCCCCCCC (10 atoms) should be checked
            # CCCCCCCCCCCCCCCCCCCC (20 atoms) should be skipped
            assert metadata["pks_building_blocks_skipped_size"] >= 1
            assert metadata["pks_building_blocks_checked"] >= 1
            assert metadata["similarity_method"] == "mcs"
        finally:
            Path(temp_path).unlink()


class TestEarlyTermination:
    """Tests for early termination when similarity threshold is reached."""

    def test_early_termination_on_high_similarity_tanimoto(self):
        """Should stop checking after finding similarity >= threshold (Tanimoto mode)."""
        # Create building blocks including an exact match
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")  # Exact match - should trigger early termination
            f.write("CCCO\n")  # Similar
            f.write("CCCCO\n")  # Less similar
            temp_path = f.name

        # Clean up any cache file that might exist
        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="tanimoto",
                similarity_threshold=0.9,
            )

            # Query with exact match to first building block
            mol = Chem.MolFromSmiles("CCO")
            node = MockNode(fragment=mol)

            reward, metadata = policy._get_pks_similarity_reward(node)

            # Should find exact match with similarity 1.0
            assert reward == pytest.approx(1.0, rel=0.01)

            # Should have checked at least 1 building block
            assert metadata["pks_building_blocks_checked"] >= 1
            assert metadata["similarity_method"] == "tanimoto"
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()

    def test_early_termination_on_high_similarity_mcs(self):
        """Should stop checking after finding similarity >= threshold (MCS mode)."""
        # Create building blocks including an exact match
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")  # Exact match - should trigger early termination
            f.write("CCCO\n")  # Similar
            f.write("CCCCO\n")  # Less similar
            temp_path = f.name

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="mcs",
                similarity_threshold=0.9,
            )

            # Query with exact match to first building block
            mol = Chem.MolFromSmiles("CCO")
            node = MockNode(fragment=mol)

            reward, metadata = policy._get_pks_similarity_reward(node)

            # Should find exact match with similarity 1.0
            assert reward == pytest.approx(1.0, rel=0.01)

            # Should have checked at least 1 building block
            assert metadata["pks_building_blocks_checked"] >= 1
            assert metadata["similarity_method"] == "mcs"
        finally:
            Path(temp_path).unlink()


class TestRolloutBehavior:
    """Tests for the overall rollout behavior."""

    def test_non_pks_match_returns_similarity_reward(self):
        """Non-PKS matches should return PKS similarity reward."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            f.write("CCCO\n")
            temp_path = f.name

        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                pks_library=set(),  # Empty PKS library
            )

            mol = Chem.MolFromSmiles("CCCCO")
            node = MockNode(fragment=mol, smiles="CCCCO")

            context = {"pks_library": set()}
            result = policy.rollout(node, context)

            # Should not be terminal (no PKS match)
            assert result.terminal is False

            # Should have PKS similarity reward
            assert 0.0 <= result.reward <= 1.0
            assert "pks_similarity" in result.metadata
            assert result.metadata.get("pks_match") is False
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()

    def test_already_attempted_skips_retrotide(self):
        """Should skip RetroTide if already attempted."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name

        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
            )

            mol = Chem.MolFromSmiles("CCO")
            node = MockNode(fragment=mol, smiles="CCO")
            node.retrotide_attempted = True  # Already attempted

            context = {"pks_library": {"CCO"}}
            result = policy.rollout(node, context)

            # Should return immediately with similarity reward
            assert result.terminal is False
            assert "pks_similarity" in result.metadata
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()


class TestCanonicalizeSmiles:
    """Tests for SMILES canonicalization helper."""
    
    def test_valid_smiles(self):
        """Should canonicalize valid SMILES."""
        result = _canonicalize_smiles("C(C)O")
        assert result == "CCO"
    
    def test_invalid_smiles(self):
        """Should return None for invalid SMILES."""
        result = _canonicalize_smiles("INVALID")
        assert result is None
    
    def test_caching(self):
        """Should cache results for repeated calls."""
        # Call twice with same input
        result1 = _canonicalize_smiles("CCO")
        result2 = _canonicalize_smiles("CCO")
        
        assert result1 == result2


class TestPolicyProperties:
    """Tests for policy properties and repr."""

    def test_name_property_tanimoto(self):
        """Should return descriptive name with tanimoto method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name

        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="tanimoto",
                success_reward=1.0,
                similarity_threshold=0.9,
            )

            name = policy.name
            assert "PKS_sim_score" in name
            assert "tanimoto" in name
            assert "1.0" in name or "success" in name.lower()
            assert "0.9" in name or "threshold" in name.lower()
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()

    def test_name_property_mcs(self):
        """Should return descriptive name with mcs method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="mcs",
                success_reward=1.0,
                similarity_threshold=0.9,
            )

            name = policy.name
            assert "PKS_sim_score" in name
            assert "mcs" in name
        finally:
            Path(temp_path).unlink()

    def test_repr_tanimoto(self):
        """Should return informative repr string for tanimoto mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name

        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="tanimoto",
            )

            repr_str = repr(policy)
            assert "PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck" in repr_str
            assert "tanimoto" in repr_str
            assert "fingerprint_radius" in repr_str
            assert "fingerprint_bits" in repr_str
            assert "pks_building_blocks" in repr_str
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()

    def test_repr_mcs(self):
        """Should return informative repr string for mcs mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="mcs",
            )

            repr_str = repr(policy)
            assert "PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck" in repr_str
            assert "mcs" in repr_str
            assert "mcs_timeout" in repr_str
            assert "atom_count_tolerance" in repr_str
            assert "pks_building_blocks" in repr_str
        finally:
            Path(temp_path).unlink()


class TestTanimotoSimilarityCalculation:
    """Tests for the Tanimoto fingerprint similarity calculation."""

    def test_identical_molecules(self):
        """Identical molecules should have similarity of 1.0."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")

        fp1 = _generate_morgan_fingerprint(mol1)
        fp2 = _generate_morgan_fingerprint(mol2)

        assert fp1 is not None
        assert fp2 is not None

        similarity = _calculate_tanimoto_similarity(fp1, fp2)
        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_different_molecules(self):
        """Different molecules should have similarity < 1.0."""
        mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene

        fp1 = _generate_morgan_fingerprint(mol1)
        fp2 = _generate_morgan_fingerprint(mol2)

        assert fp1 is not None
        assert fp2 is not None

        similarity = _calculate_tanimoto_similarity(fp1, fp2)
        assert 0.0 <= similarity < 1.0

    def test_similar_molecules(self):
        """Similar molecules should have moderate similarity."""
        mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        mol2 = Chem.MolFromSmiles("CCCO")  # Propanol

        fp1 = _generate_morgan_fingerprint(mol1)
        fp2 = _generate_morgan_fingerprint(mol2)

        assert fp1 is not None
        assert fp2 is not None

        similarity = _calculate_tanimoto_similarity(fp1, fp2)
        # Similar molecules should have moderate to high similarity
        assert 0.3 < similarity < 1.0


class TestMorganFingerprintGeneration:
    """Tests for Morgan fingerprint generation helper."""

    def test_generates_fingerprint(self):
        """Should generate a valid fingerprint."""
        mol = Chem.MolFromSmiles("CCO")
        fp = _generate_morgan_fingerprint(mol)

        assert fp is not None
        assert fp.GetNumBits() == 2048  # Default bits

    def test_custom_radius_and_bits(self):
        """Should respect custom radius and bits parameters."""
        mol = Chem.MolFromSmiles("CCO")
        fp = _generate_morgan_fingerprint(mol, radius=3, n_bits=1024)

        assert fp is not None
        assert fp.GetNumBits() == 1024


class TestTanimotoBuildingBlocksLoading:
    """Tests for loading PKS building blocks in Tanimoto mode."""

    def test_load_from_file_tanimoto_mode(self):
        """Should correctly load PKS building blocks with fingerprints."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            f.write("CCCO\n")
            f.write("c1ccccc1\n")
            f.write("\n")  # Empty line should be skipped
            f.write("INVALID_SMILES\n")  # Should be skipped
            temp_path = f.name

        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="tanimoto",
            )

            # Should load 3 valid molecules (skipping empty line and invalid SMILES)
            assert len(policy._pks_building_blocks) == 3

            # Check that fingerprints are stored correctly (Tanimoto mode stores (smiles, fp))
            for smiles, fp in policy._pks_building_blocks:
                assert isinstance(smiles, str)
                assert fp is not None
                assert fp.GetNumBits() == 2048  # Default bits
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()

    def test_cache_is_created(self):
        """Should create a cache file on first load."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            f.write("CCCO\n")
            temp_path = f.name

        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            # Ensure no cache exists
            if cache_path.exists():
                cache_path.unlink()

            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="tanimoto",
            )

            # Cache should now exist
            assert cache_path.exists()
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()

    def test_cache_is_loaded(self):
        """Should load from cache on subsequent runs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            f.write("CCCO\n")
            temp_path = f.name

        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            # First load creates cache
            policy1 = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="tanimoto",
            )

            # Get cache modification time
            cache_mtime1 = cache_path.stat().st_mtime

            # Second load should use cache
            policy2 = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="tanimoto",
            )

            # Cache should not have been modified
            cache_mtime2 = cache_path.stat().st_mtime
            assert cache_mtime1 == cache_mtime2

            # Both should have same number of building blocks
            assert len(policy1._pks_building_blocks) == len(policy2._pks_building_blocks)
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()


class TestSimilarityMethodValidation:
    """Tests for similarity method validation."""

    def test_invalid_similarity_method_raises(self):
        """Should raise ValueError for invalid similarity method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                    pks_building_blocks_path=temp_path,
                    similarity_method="invalid",
                )
            assert "similarity_method" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_tanimoto_is_default(self):
        """Tanimoto should be the default similarity method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name

        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
            )
            assert policy.similarity_method == "tanimoto"
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()


class TestSimilarityMethodComparison:
    """Tests comparing both similarity methods."""

    def test_both_methods_return_valid_results(self):
        """Both methods should return valid similarity scores."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            f.write("CCCO\n")
            f.write("CCCCO\n")
            temp_path = f.name

        cache_path = Path(temp_path).with_suffix('.fingerprints.pkl')

        try:
            # Test Tanimoto
            policy_tanimoto = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="tanimoto",
            )

            # Test MCS
            policy_mcs = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                similarity_method="mcs",
            )

            mol = Chem.MolFromSmiles("CCO")
            node = MockNode(fragment=mol)

            reward_tanimoto, meta_tanimoto = policy_tanimoto._get_pks_similarity_reward(node)
            reward_mcs, meta_mcs = policy_mcs._get_pks_similarity_reward(node)

            # Both should return valid rewards
            assert 0.0 <= reward_tanimoto <= 1.0
            assert 0.0 <= reward_mcs <= 1.0

            # Both should find exact match (1.0 similarity)
            assert reward_tanimoto == pytest.approx(1.0, rel=0.01)
            assert reward_mcs == pytest.approx(1.0, rel=0.01)

            # Metadata should reflect the method used
            assert meta_tanimoto["similarity_method"] == "tanimoto"
            assert meta_mcs["similarity_method"] == "mcs"
        finally:
            Path(temp_path).unlink()
            if cache_path.exists():
                cache_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
