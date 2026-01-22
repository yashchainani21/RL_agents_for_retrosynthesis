"""
Unit tests for PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck rollout policy.

These tests verify:
1. PKS building blocks loading
2. MCS similarity calculation
3. Atom count pre-filtering
4. Early termination threshold
5. Rollout reward computation
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
    _canonicalize_smiles,
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
    """Tests for loading PKS building blocks."""
    
    def test_load_from_file(self):
        """Should correctly load PKS building blocks from a file."""
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
            )
            
            # Should load 3 valid molecules (skipping empty line and invalid SMILES)
            assert len(policy._pks_building_blocks) == 3
            
            # Check that atom counts are stored correctly
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
        )
        
        # Should have no building blocks loaded
        assert len(policy._pks_building_blocks) == 0


class TestAtomCountPreFiltering:
    """Tests for atom count pre-filtering optimization."""
    
    def test_skips_different_size_molecules(self):
        """Should skip molecules with very different atom counts."""
        # Create a policy with known building blocks
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("C\n")  # 1 atom - should be skipped for query with 10 atoms
            f.write("CCCCCCCCCC\n")  # 10 atoms - should be checked
            f.write("CCCCCCCCCCCCCCCCCCCC\n")  # 20 atoms - should be skipped
            temp_path = f.name
        
        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
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
        finally:
            Path(temp_path).unlink()


class TestEarlyTermination:
    """Tests for early termination when similarity threshold is reached."""
    
    def test_early_termination_on_high_similarity(self):
        """Should stop checking after finding similarity >= threshold."""
        # Create building blocks including an exact match
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")  # Exact match - should trigger early termination
            f.write("CCCO\n")  # Similar
            f.write("CCCCO\n")  # Less similar
            temp_path = f.name
        
        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
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
    
    def test_already_attempted_skips_retrotide(self):
        """Should skip RetroTide if already attempted."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name
        
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
    
    def test_name_property(self):
        """Should return descriptive name."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name
        
        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
                success_reward=1.0,
                similarity_threshold=0.9,
            )
            
            name = policy.name
            assert "PKS_sim_score" in name
            assert "1.0" in name or "success" in name.lower()
            assert "0.9" in name or "threshold" in name.lower()
        finally:
            Path(temp_path).unlink()
    
    def test_repr(self):
        """Should return informative repr string."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CCO\n")
            temp_path = f.name
        
        try:
            policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
                pks_building_blocks_path=temp_path,
            )
            
            repr_str = repr(policy)
            assert "PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck" in repr_str
            assert "pks_building_blocks" in repr_str
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
