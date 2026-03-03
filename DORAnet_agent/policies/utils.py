"""
Shared utility functions for MCTS policies.

This module consolidates helper functions used across reward policies,
terminal detectors, and other policy components to eliminate code duplication.

Includes:
    - SMILES canonicalization
    - SA Score computation and reward conversion
    - Morgan fingerprint generation
    - Tanimoto and MCS similarity calculations
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Optional

from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, rdFMCS

# SA Score import setup
_SA_SCORE_PATH = os.path.join(RDConfig.RDContribDir, 'SA_Score')
if _SA_SCORE_PATH not in sys.path:
    sys.path.insert(0, _SA_SCORE_PATH)

try:
    import sascorer  # type: ignore[import-not-found]
    _SA_SCORE_AVAILABLE = True
except ImportError:
    _SA_SCORE_AVAILABLE = False
    sascorer = None


@lru_cache(maxsize=50000)
def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form, returning None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def calculate_sa_score(mol: Chem.Mol) -> Optional[float]:
    """
    Calculate SA Score for a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        SA Score (1-10, 1=easy, 10=hard) or None if calculation fails
    """
    if not _SA_SCORE_AVAILABLE or sascorer is None:
        return None
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return None


def sa_score_to_reward(sa_score: float, max_reward: float = 1.0) -> float:
    """
    Convert SA Score to reward.

    Formula: (10 - sa_score) / 10
    This gives range 0.0-0.9 for typical SA scores (1-10).

    Args:
        sa_score: SA Score from RDKit (1-10 scale)
        max_reward: Optional cap on the reward (default 1.0, no effective cap)

    Returns:
        Reward in range [0.0, min(0.9, max_reward)]
    """
    reward = (10.0 - sa_score) / 10.0
    return min(reward, max_reward)


def generate_morgan_fingerprint(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[DataStructs.ExplicitBitVect]:
    """
    Generate Morgan (ECFP-like) fingerprint for a molecule.

    Args:
        mol: RDKit Mol object
        radius: Fingerprint radius (default 2, equivalent to ECFP4)
        n_bits: Number of bits in fingerprint (default 2048)

    Returns:
        Morgan fingerprint as ExplicitBitVect, or None if generation fails
    """
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def calculate_tanimoto_similarity(
    query_fp: DataStructs.ExplicitBitVect,
    reference_fp: DataStructs.ExplicitBitVect,
) -> float:
    """Calculate Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(query_fp, reference_fp)


def calculate_mcs_similarity_without_stereo(
    query_mol: Chem.Mol,
    reference_mol: Chem.Mol,
    timeout: float = 1.0,
) -> Optional[float]:
    """
    Calculate MCS-based similarity without stereochemistry matching.

    Uses the Maximum Common Substructure (MCS) algorithm to find structural
    similarity between two molecules. This is based on RetroTide's
    'mcs_without_stereo' similarity metric.

    Formula: numAtoms / (query_atoms + reference_atoms - numAtoms)

    Args:
        query_mol: The query molecule to compare
        reference_mol: The reference molecule (e.g., PKS building block)
        timeout: Maximum time in seconds for MCS calculation (default 1.0)

    Returns:
        Similarity score in range [0.0, 1.0], or None if calculation times out
    """
    try:
        result = rdFMCS.FindMCS(
            [query_mol, reference_mol],
            timeout=int(timeout),  # MCS timeout in seconds
            matchValences=True,
            matchChiralTag=False,  # No stereochemistry matching
            bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact,
        )

        # Check if MCS timed out
        if result.canceled:
            return None

        if result.numAtoms == 0:
            return 0.0

        # Calculate Tanimoto-like similarity from MCS
        query_atoms = query_mol.GetNumAtoms()
        reference_atoms = reference_mol.GetNumAtoms()
        union_atoms = query_atoms + reference_atoms - result.numAtoms

        if union_atoms == 0:
            return 0.0

        score = result.numAtoms / union_atoms
        return score

    except Exception:
        return None
