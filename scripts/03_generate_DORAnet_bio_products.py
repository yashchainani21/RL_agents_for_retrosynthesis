import multiprocessing as mp
from rdkit import Chem
import doranet.modules.enzymatic as enzymatic
import pandas as pd

# Read in cofactors and prepare a list of their canonical SMILES
cofactors_df = pd.read_csv('../data/raw/all_cofactors.csv')
cofactors_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in cofactors_df["SMILES"]]

# Define file paths based on modification type
max_module = "LM"
modify_PKS_products = True
DORAnet_product_type_to_modify = None

if modify_PKS_products:
    precursors_filepath = f'../data/interim/unique_PKS_products_no_stereo_{max_module}.txt'
    output_filepath = f'../data/interim/DORAnet_BIO1_products_from_{max_module}_polyketides.txt'

if DORAnet_product_type_to_modify == 'BIO':
    precursors_filepath = f'../data/interim/DORAnet_BIO1_products_from_{max_module}_polyketides.txt'
    output_filepath = f'../data/interim/DORAnet_BIO2_products_from_{max_module}_polyketides.txt'

if DORAnet_product_type_to_modify == 'CHEM':
    precursors_filepath = f'../data/interim/DORAnet_CHEM1_products_from_{max_module}_polyketides.txt'
    output_filepath = f'../data/interim/DORAnet_CHEM2_products_from_BIO1.txt'

# Read precursor SMILES
with open(precursors_filepath, 'r') as precursors_file:
    precursors_list = [s.strip('\n') for s in precursors_file.readlines()]


def perform_DORAnet_bio_1step(precursor_smiles: str):
    """Generates one-step DORAnet products for a given precursor SMILES string."""
    forward_network = enzymatic.generate_network(
        job_name=precursor_smiles,
        starters={precursor_smiles},
        gen=1,
        direction="forward"
    )

    generated_bioproducts_list = []

    for mol in forward_network.mols:
        generated_bioproduct_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol.uid))

        # Store only non-cofactor unique products
        if generated_bioproduct_smiles and generated_bioproduct_smiles not in cofactors_list:
            generated_bioproducts_list.append(generated_bioproduct_smiles)

    return generated_bioproducts_list


def process_precursors_parallel(precursors, num_workers=mp.cpu_count()):
    """Parallelize the processing of precursor SMILES using multiprocessing."""
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(perform_DORAnet_bio_1step, precursors)

    # Flatten results (convert list of lists into a single list)
    all_bioproducts_list = [prod for sublist in results for prod in sublist]

    return all_bioproducts_list


if __name__ == "__main__":
    # Run in parallel
    print(f"Running with {mp.cpu_count()} cores...")
    all_bioproducts_list = process_precursors_parallel(precursors_list, num_workers=mp.cpu_count())

    print(f"\nNumber of total bioproducts generated: {len(all_bioproducts_list)}\n")

    # Save results
    with open(output_filepath, 'w') as output_file:
        output_file.write('\n'.join(all_bioproducts_list))
