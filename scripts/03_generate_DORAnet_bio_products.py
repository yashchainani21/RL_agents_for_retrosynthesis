from mpi4py import MPI
from rdkit import Chem
import doranet.modules.enzymatic as enzymatic
import pandas as pd

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define file paths based on modification type
input_products = "LM" # choose from "LM", "M1", "M2", "M3", "BIO1", "CHEM1"
output_products = "BIO1" # choose from either "BIO1", or "BIO2"
module = "LM" # choose from "LM", "M1", "M2", "M3"
modify_PKS_products = True
DORAnet_product_type_to_modify = None

# set input and output filepaths as None first and modify later
output_filepath = None
precursors_filepath = None

# if performing DORAnet modifications on polyketides
if input_products in ("LM", "M1", "M2", "M3"):
    precursors_filepath = f'../data/interim/unique_PKS_products_no_stereo_{input_products}.txt'
    output_filepath = f'../data/interim/DORAnet_{output_products}_from_{input_products}.txt'

# if performing DORAnet modifications for a second-step on DORAnet products from the first step
elif input_products in ("BIO1", "CHEM1"):
    precursors_filepath = f'../data/interim/DORAnet_{input_products}_from_{module}.txt'
    output_filepath = f'../data/interim/DORAnet_{output_products}_from_{input_products}_from{module}.txt'

else:
    raise ValueError("invalid input products name provided")

# only rank 0 reads files and sets up initial data
if rank == 0:

    # read in cofactors and prepare a list of their canonical SMILES
    cofactors_df = pd.read_csv('../data/raw/all_cofactors.csv')
    cofactors_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in cofactors_df["SMILES"]]


    # read precursor SMILES
    with open(precursors_filepath, 'r') as precursors_file:
        precursors_list = [s.strip('\n') for s in precursors_file.readlines()]

else:
    cofactors_list = None
    precursors_list = None

# broadcast the cofactors list and precursors list to all processes
cofactors_list = comm.bcast(cofactors_list, root = 0)
precursors_list = comm.bcast(precursors_list, root = 0)

# define a helper function to evenly split the precursors list into n chunks
def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[i::n] for i in range(n)]

# scatter

def perform_DORAnet_bio_1step(precursor_smiles: str):
    """Generates one-step DORAnet products for a given precursor SMILES string."""
    forward_network = enzymatic.generate_network(
        job_name = precursor_smiles,
        starters = {precursor_smiles},
        gen = 1,
        direction = "forward")

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

    print(f"Running with {mp.cpu_count()} cores...")
    all_bioproducts_list = process_precursors_parallel(precursors_list, num_workers = mp.cpu_count())
    all_bioproducts_list = list(set(all_bioproducts_list))
    print(f"\nNumber of total, unique bioproducts generated: {len(all_bioproducts_list)}\n")

    # save all biologically modified products
    with open(output_filepath, 'w') as output_file:
        output_file.write('\n'.join(all_bioproducts_list))
