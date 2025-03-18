import bcs
from retrotide import retrotide, structureDB
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import OrderedDict

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

all_starters_list = list(bcs.starters.keys())
all_extenders_list = list(bcs.extenders.keys())

# iterate through all starter units
for starter in all_starters_list:

    # initialize a loading module with each PKS starter
    loading_AT_domain = bcs.AT(active = True,
                               substrate = starter)

    loading_domains_dict = OrderedDict({bcs.AT: loading_AT_domain})
    loading_mod = bcs.Module(domains = loading_domains_dict,
                             loading = True)

    print(loading_mod)