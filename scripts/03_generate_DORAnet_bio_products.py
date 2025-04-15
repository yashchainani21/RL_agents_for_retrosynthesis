import uuid
from mpi4py import MPI
from rdkit import Chem
import doranet.modules.enzymatic as enzymatic
import pandas as pd

if __name__ == '__main__':

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
        """Split lst into n (roughly) equal-sized chunks. Avoid creating empty chunks."""
        n = min(n, len(lst))  # avoid creating more chunks than data
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    # scatter the data (only rank 0 prepares the chunks)
    if rank == 0:
        chunks = chunkify(precursors_list, size)
    else:
        chunks = None

    # each process then receives its chunk of precursor SMILES
    my_precursors = comm.scatter(chunks, root = 0)

    print(f"[Rank {rank}] received {len(my_precursors)} precursors.")

    def perform_DORAnet_bio_1step(precursor_smiles: str):
        """Generates one-step DORAnet products for a given precursor SMILES string."""

        unique_id = str(uuid.uuid4())
        unique_jobname = f'{precursor_smiles}_{unique_id}'

        forward_network = enzymatic.generate_network(
            job_name = unique_jobname,
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

    # process each precursor in the assigned chunk
    my_results = []
    for precursor in my_precursors:
        my_results.append(perform_DORAnet_bio_1step(precursor))

    # flatten the results for this rank
    my_results_flat = [item for sublist in my_results for item in sublist]

    # gather the lists of results from all processes to the master process
    all_results = comm.gather(my_results_flat, root=0)

    if rank == 0:
        # Combine and deduplicate results
        combined_results = set()
        for sublist in all_results:
            combined_results.update(sublist)

        print(f"\nTotal unique bioproducts generated: {len(combined_results)}\n")

        # Save the results to the output file
        with open(output_filepath, 'w') as output_file:
            output_file.write('\n'.join(combined_results))