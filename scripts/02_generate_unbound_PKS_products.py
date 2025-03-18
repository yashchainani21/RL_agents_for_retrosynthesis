import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

def run_pks_release_reaction(pks_release_mechanism: str,
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

module = "Mod1" # pick from "LM", "Mod1", "Mod2", or "Mod3"

input_filepath = f'../data/raw/PKS_designs_and_products_{module}.pkl'
output_dict_filepath = f'../data/interim/PKS_designs_and_unbound_products_{module}.pkl'
output_unique_PKS_products_no_stereo_filepath = f'../data/interim/unique_PKS_products_no_stereo_{module}.txt'

with open(input_filepath, "rb") as f:
    PKS_designs_and_products_dict = pickle.load(f)

# initialize a new dictionary to store PKS products after they have been released
PKS_designs_and_unbound_products_dict = {}

# initialize a list to store only unique PKS products and without their stereochemistry
unique_PKS_products_no_stereo_list = []

for key in PKS_designs_and_products_dict.keys():
    PKS_design = key
    PKS_product_smiles = PKS_designs_and_products_dict[key]
    PKS_product_mol = Chem.MolFromSmiles(PKS_product_smiles)

    # first, we try to offload the PKS product via a thiolysis reaction
    acid_product_mol = run_pks_release_reaction(pks_release_mechanism = "thiolysis",
                                            bound_product_mol = PKS_product_mol)

    # store the PKS design and corresponding product released from thiolysis
    PKS_designs_and_unbound_products_dict[tuple(PKS_design)] = Chem.MolToSmiles(acid_product_mol)

    # also store the product after removing its stereochemistry
    rdmolops.RemoveStereochemistry(acid_product_mol)

    acid_product_smiles = Chem.MolToSmiles(acid_product_mol)

    if acid_product_smiles not in unique_PKS_products_no_stereo_list:
        unique_PKS_products_no_stereo_list.append(acid_product_smiles)

    # now, we try to offload the PKS product via a cyclization reaction
    try:
        cyclized_product_mol = run_pks_release_reaction(pks_release_mechanism = 'cyclization',
                                                         bound_product_mol = PKS_product_mol)

        # if cyclization is possible, store the resulting lactone as well as its
        PKS_designs_and_unbound_products_dict[tuple(PKS_design)] = Chem.MolToSmiles(cyclized_product_mol)
        rdmolops.RemoveStereochemistry(cyclized_product_mol)

        cyclized_product_smiles = Chem.MolToSmiles(cyclized_product_mol)

        if cyclized_product_smiles not in unique_PKS_products_no_stereo_list:
            unique_PKS_products_no_stereo_list.append(cyclized_product_smiles)

    # do nothing if cyclization is not possible
    except:
        pass

# store PKS designs and their corresponding unbound products
with open(output_dict_filepath, "wb") as f:
    pickle.dump(PKS_designs_and_unbound_products_dict, f)

# store the unique PKS products without any stereochemistry
with open(output_unique_PKS_products_no_stereo_filepath, "w") as f:
    for item in unique_PKS_products_no_stereo_list:
        f.write(item + "\n")

print(f'Number of unique PKS products stored without their stereochemistry: {len(unique_PKS_products_no_stereo_list)}')
