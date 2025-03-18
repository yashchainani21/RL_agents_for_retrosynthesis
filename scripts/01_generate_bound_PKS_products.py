import pickle
import bcs
import multiprocessing
from retrotide import structureDB
from rdkit import Chem
from collections import OrderedDict

# Get list of all starter units
all_starters_list = list(bcs.starters.keys())
num_extension_modules = 3  # Number of extension modules
num_processes = min(multiprocessing.cpu_count(), len(all_starters_list))  # Use available CPU cores


def generate_pks_designs(starter):
    """
    Function to generate all PKS designs for a given starter unit.
    Returns a dictionary with PKS designs as keys and bound products as values.
    """
    all_PKS_designs_and_products_dict = {}

    # Initialize the loading module
    loading_AT_domain = bcs.AT(active=True, substrate=starter)
    loading_module = bcs.Module(domains=OrderedDict({bcs.AT: loading_AT_domain}), loading=True)

    # Convert this loading module into a bcs cluster object and obtain the corresponding product
    loading_modules_list = [loading_module]
    LM_cluster = bcs.Cluster(modules=loading_modules_list)
    bound_LM_product = Chem.MolToSmiles(LM_cluster.computeProduct(structureDB))

    # Store the loading module design
    all_PKS_designs_and_products_dict[tuple(loading_modules_list)] = bound_LM_product

    # Iterate through extension module combinations (up to 3)
    for key1 in structureDB.keys():
        extension_module_1 = key1
        extension_modules_list_1 = [loading_module, extension_module_1]
        cluster_1 = bcs.Cluster(modules=extension_modules_list_1)
        bound_PKS_product_1 = Chem.MolToSmiles(cluster_1.computeProduct(structureDB))
        all_PKS_designs_and_products_dict[tuple(extension_modules_list_1)] = bound_PKS_product_1

        for key2 in structureDB.keys():
            extension_module_2 = key2
            extension_modules_list_2 = [loading_module, extension_module_1, extension_module_2]
            cluster_2 = bcs.Cluster(modules=extension_modules_list_2)
            bound_PKS_product_2 = Chem.MolToSmiles(cluster_2.computeProduct(structureDB))
            all_PKS_designs_and_products_dict[tuple(extension_modules_list_2)] = bound_PKS_product_2

            for key3 in structureDB.keys():
                extension_module_3 = key3
                extension_modules_list_3 = [loading_module, extension_module_1, extension_module_2, extension_module_3]
                cluster_3 = bcs.Cluster(modules=extension_modules_list_3)
                bound_PKS_product_3 = Chem.MolToSmiles(cluster_3.computeProduct(structureDB))
                all_PKS_designs_and_products_dict[tuple(extension_modules_list_3)] = bound_PKS_product_3

    return all_PKS_designs_and_products_dict


if __name__ == "__main__":
    # Use multiprocessing to distribute computation across multiple cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(generate_pks_designs, all_starters_list)

    # Merge all results into a single dictionary
    all_PKS_designs_and_products_dict = {}
    for result in results:
        all_PKS_designs_and_products_dict.update(result)

    # Pickle the generated data
    with open('../data/raw/PKS_designs_and_products.pkl', "wb") as f:
        pickle.dump(all_PKS_designs_and_products_dict, f)

    print(f"Generated {len(all_PKS_designs_and_products_dict)} PKS designs and saved to file.")
