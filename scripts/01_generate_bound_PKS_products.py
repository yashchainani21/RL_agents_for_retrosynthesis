import pickle
import bcs
import multiprocessing
from retrotide import structureDB
from rdkit import Chem
from collections import OrderedDict

# Get list of all starter units
all_starters_list = list(bcs.starters.keys())
num_processes = multiprocessing.cpu_count()

max_num_of_modules = "M2"

def generate_pks_designs(starter: str):
    """
    Generates all PKS designs for a given starter unit.
    Returns a dictionary with PKS designs as keys and bound products as values.
    """
    max_module = "M2"
    all_PKS_designs_and_products_dict = {}

    # Initialize the loading module
    loading_AT_domain = bcs.AT(active = True, substrate = starter)
    loading_module = bcs.Module(domains=OrderedDict({bcs.AT: loading_AT_domain}), loading=True)

    # Convert this loading module into a bcs cluster object and obtain the corresponding product
    loading_modules_list = [loading_module]
    LM_cluster = bcs.Cluster(modules=loading_modules_list)
    bound_LM_product = Chem.MolToSmiles(LM_cluster.computeProduct(structureDB))

    # Store the loading module design
    all_PKS_designs_and_products_dict[tuple(loading_modules_list)] = bound_LM_product

    # stop here if generating PKS designs only up until the loading module
    if max_module == "LM":
        return all_PKS_designs_and_products_dict

    # Iterate through extension module combinations (up to 3)
    for key1 in structureDB.keys():
        extension_module_1 = key1
        extension_modules_list_1 = [loading_module, extension_module_1]
        cluster_1 = bcs.Cluster(modules=extension_modules_list_1)
        bound_PKS_product_1 = Chem.MolToSmiles(cluster_1.computeProduct(structureDB))
        all_PKS_designs_and_products_dict[tuple(extension_modules_list_1)] = bound_PKS_product_1

        # stop here if generating PKS designs only up until the first extension module
        if max_module == "M1":
            pass
        else:

            for key2 in structureDB.keys():
                extension_module_2 = key2
                extension_modules_list_2 = [loading_module, extension_module_1, extension_module_2]
                cluster_2 = bcs.Cluster(modules=extension_modules_list_2)
                bound_PKS_product_2 = Chem.MolToSmiles(cluster_2.computeProduct(structureDB))
                all_PKS_designs_and_products_dict[tuple(extension_modules_list_2)] = bound_PKS_product_2

                # stop here if generating PKS designs only up until the second extension module
                if max_module == "M2":
                    pass
                else:

                    for key3 in structureDB.keys():
                        extension_module_3 = key3
                        extension_modules_list_3 = [loading_module, extension_module_1, extension_module_2, extension_module_3]
                        cluster_3 = bcs.Cluster(modules=extension_modules_list_3)
                        bound_PKS_product_3 = Chem.MolToSmiles(cluster_3.computeProduct(structureDB))
                        all_PKS_designs_and_products_dict[tuple(extension_modules_list_3)] = bound_PKS_product_3

    return all_PKS_designs_and_products_dict

if __name__ == "__main__":
    # use multiprocessing to distribute computation across multiple cores
    with multiprocessing.Pool(processes = num_processes) as pool:
        results = pool.map(generate_pks_designs, all_starters_list)

    # merge all results into a single dictionary
    all_PKS_designs_and_products_dict = {}
    for result in results:
        all_PKS_designs_and_products_dict.update(result)

    output_filepath = f'../data/raw/PKS_designs_and_products_{max_num_of_modules}.pkl'

    with open(output_filepath, "wb") as f:
        pickle.dump(all_PKS_designs_and_products_dict, f)

    print(f"\nGenerated {len(all_PKS_designs_and_products_dict)} PKS designs and saved to file.")
