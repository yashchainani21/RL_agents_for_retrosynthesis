from rdkit import Chem

def are_isomorphic(mol1: Chem.Mol, mol2: Chem.Mol, consider_stereo: bool = False):
    if consider_stereo:
        is_isomorphic = mol1.HasSubstructMatch(mol2, useChirality = True) and mol2.HasSubstructMatch(mol1, useChirality = True)
    else:
        is_isomorphic = mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)
        
    return is_isomorphic