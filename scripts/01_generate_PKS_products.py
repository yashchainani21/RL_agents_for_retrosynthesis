import bcs
from retrotide import retrotide, structureDB
from rdkit import Chem
from rdkit.Chem import AllChem

def _pks_release_reaction(pks_release_mechanism: str,
                          bound_product_mol: Chem.Mol) -> Chem.Mol:
    """
    Run an offloading reaction to release a bound PKS product.
    Two types of offloading reactions are currently supported: thiolysis and cyclization.
    A thiolysis offloading reaction will result in the formation of a carboxylic acid.
    Meanwhile, a cyclization offloading reaction results in the formation of a lactone ring.
    """

    if pks_release_mechanism == 'thiolysis':
        Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to produce terminal acid group
        rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
        unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
        Chem.SanitizeMol(unbound_product_mol)
        return unbound_product_mol

    if pks_release_mechanism == 'cyclization':
        Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to cyclize bound substrate
        rxn = AllChem.ReactionFromSmarts('([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])>>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]')
        try:
            unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
            Chem.SanitizeMol(unbound_product_mol)
            return unbound_product_mol

        # if the bound substrate cannot be cyclized, then return None
        except:
            raise ValueError("\nUnable to perform cyclization reaction")

starters_codes_list = list(bcs.starters.keys())
extender_codes_list = list(bcs.extenders.keys())

print(extender_codes_list)