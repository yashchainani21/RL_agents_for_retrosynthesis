import pickle
import multiprocessing
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

def run_pks_release_reaction(pks_release_mechanism: str,
                             bound_product_mol: Chem.Mol) -> Chem.Mol:
    """
    Run an offloading reaction to release a bound PKS product.
    Two types of offloading reactions are currently supported: thiolysis and cyclization.
    """

    if pks_release_mechanism == 'thiolysis':
        Chem.SanitizeMol(bound_product_mol)
        rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
        unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
        Chem.SanitizeMol(unbound_product_mol)
        return unbound_product_mol

    if pks_release_mechanism == 'cyclization':
        Chem.SanitizeMol(bound_product_mol)
        rxn = AllChem.ReactionFromSmarts('([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])>>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]')
        try:
            unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
            Chem.SanitizeMol(unbound_product_mol)
            return unbound_product_mol
        except:
            raise ValueError("\nUnable to perform cyclization reaction")

def process_pks_design(args):
    """Process a single PKS design and return results."""
    key, smiles = args
    PKS_product_mol = Chem.MolFromSmiles(smiles)

    # Dictionary to store unbound products
    local_results = {}
    unique_products = set()

    # Try thiolysis
    try:
        acid_product_mol = run_pks_release_reaction("thiolysis", PKS_product_mol)
        local_results[tuple(key)] = Chem.MolToSmiles(acid_product_mol)

        # Remove stereochemistry
        rdmolops.RemoveStereochemistry(acid_product_mol)
        unique_products.add(Chem.MolToSmiles(acid_product_mol))

    except Exception as e:
        print(f"Error in thiolysis for {key}: {e}")

    # Try cyclization
    try:
        cyclized_product_mol = run_pks_release_reaction("cyclization", PKS_product_mol)
        local_results[tuple(key)] = Chem.MolToSmiles(cyclized_product_mol)

        # Remove stereochemistry
        rdmolops.RemoveStereochemistry(cyclized_product_mol)
        unique_products.add(Chem.MolToSmiles(cyclized_product_mol))

    except:
        pass  # Ignore errors in cyclization

    return local_results, unique_products

def main():
    max_module = "M3"  # Pick from "LM", "M1", "M2", or "M3"

    input_filepath = f'../data/raw/PKS_designs_and_products_{max_module}.pkl'
    output_dict_filepath = f'../data/interim/PKS_designs_and_unbound_products_{max_module}.pkl'
    output_unique_PKS_products_no_stereo_filepath = f'../data/interim/unique_PKS_products_no_stereo_{max_module}.txt'

    # Load PKS designs
    with open(input_filepath, "rb") as f:
        PKS_designs_and_products_dict = pickle.load(f)

    # Use multiprocessing to process the data
    num_workers = multiprocessing.cpu_count() - 1  # Use all but one core
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(process_pks_design, PKS_designs_and_products_dict.items())

    # Aggregate results
    PKS_designs_and_unbound_products_dict = {}
    unique_PKS_products_no_stereo_list = set()

    for res_dict, unique_set in results:
        PKS_designs_and_unbound_products_dict.update(res_dict)
        unique_PKS_products_no_stereo_list.update(unique_set)

    # Store results
    with open(output_dict_filepath, "wb") as f:
        pickle.dump(PKS_designs_and_unbound_products_dict, f)

    with open(output_unique_PKS_products_no_stereo_filepath, "w") as f:
        for item in unique_PKS_products_no_stereo_list:
            f.write(item + "\n")

    print(f'Number of unique PKS products stored without their stereochemistry: {len(unique_PKS_products_no_stereo_list)}')

if __name__ == "__main__":
    main()
