import numpy as np
import pandas as pd
import bcs
import doranet.modules.enzymatic as enzymatic
from rdkit import Chem
from rdkit.Chem import AllChem
from retrotide import structureDB
from collections import OrderedDict

cofactors_df = pd.read_csv("../data/raw/all_cofactors.csv")
cofactors_list = list(cofactors_df["SMILES"])
def _pks_release_reaction(pks_release_mechanism: str, bound_product_mol: Chem.Mol) -> Chem.Mol:
    """
    Run a PKS offloading reaction to release a PKS product bound to its synthase via either a thiolysis or cyclization reaction
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

starter_codes_list = list(bcs.starters.keys())

PKS_products_list = [] # initialize an empty list to store PKS products

for starter_code in starter_codes_list:

    # initialize a loading module with each PKS starter
    loading_AT_domain = bcs.AT(active = True, substrate = starter_code)
    loading_domains_dict = OrderedDict({bcs.AT: loading_AT_domain})
    loading_mod = bcs.Module(domains = loading_domains_dict, loading=True)

    # create a bcs Cluster object and compute bound PKS product
    cluster = bcs.Cluster(modules = [loading_mod])
    loading_mod_product = cluster.computeProduct(structureDB)

    loading_mod_product = _pks_release_reaction(pks_release_mechanism = "thiolysis",
                                                bound_product_mol = loading_mod_product)

    PKS_products_list.append(Chem.MolToSmiles(loading_mod_product))

DORAnet_products = []

for PKS_product_SMILES in PKS_products_list:
    forward_network = enzymatic.generate_network(job_name = "test",
                                                 starters = PKS_product_SMILES,
                                                 gen = 1,
                                                 direction = "forward")

    for mol in forward_network.mols:
        if mol.uid not in cofactors_list and mol.uid!= PKS_product_SMILES:
            DORAnet_products.append(mol.uid) # store SMILES str of DORAnet product

print(f"Final number of all DORAnet products generated: {len(DORAnet_products)}")

final_products_list = PKS_products_list + DORAnet_products
labels_array = np.concatenate(( np.ones(len(PKS_products_list)), np.zeros(len(DORAnet_products)) ))
assert len(final_products_list) == len(labels_array)

final_products_df = pd.DataFrame({"SMILES": final_products_list,
                                  "Label": labels_array})

final_products_df.to_parquet('../data/raw/PKS_loading_1bio_cpds.parquet')
