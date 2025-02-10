from rdkit import Chem
from rdkit.Chem import rdFMCS

def subgraph_mcs_reward(rolledout_mol: Chem.rdchem.Mol,
                        target_mol: Chem.rdchem.Mol,
                        match_valences: bool = True,
                        match_chiral: bool = False):
    """
    Returns a reward in the range [0,1]:
        - 1.0 if "rolledout_mol" is fully a subgraph of "target_mol"
        - Otherwise, default to calculating the normalized MCS score
    """

    # perform a substructure check first
    if target_mol.HasSubstructMatch(rolledout_mol):
        return 1.0

    # if "rolledout_mol" is not a subgraph of "target_mol", compute MCS score
    result = rdFMCS.FindMCS([target_mol, rolledout_mol],
                            timeout=1,
                            matchValences = match_valences,
                            matchChiralTag = match_chiral,
                            bondCompare = Chem.rdFMCS.BondCompare.CompareOrderExact)  # search for 1 second max

    if result.canceled:
        print('MCS timeout')
        return 0.0

    return result.numAtoms / (len(rolledout_mol.GetAtoms()) + len(target_mol.GetAtoms()) - result.numAtoms)

