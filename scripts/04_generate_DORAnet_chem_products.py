import uuid
from mpi4py import MPI
from rdkit import Chem
import doranet.modules.synthetic as synthetic
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

if __name__ == '__main__':

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define file paths based on modification type
    input_products = "LM"  # choose from "LM", "M1", "M2", "M3", "BIO1", "CHEM1"
    output_products = "CHEM1"  # choose from either "CHEM1", or "CHEM2"
    module = "LM"  # choose from "LM", "M1", "M2", "M3"
    modify_PKS_products = True
    DORAnet_product_type_to_modify = None

    output_filepath = None
    precursors_filepath = None

    if input_products in ("LM", "M1", "M2", "M3"):
        precursors_filepath = f'../data/interim/unique_PKS_products_no_stereo_{input_products}.txt'
        output_filepath = f'../data/interim/DORAnet_{output_products}_from_{input_products}.txt'

    elif input_products in ("BIO1", "CHEM1"):
        precursors_filepath = f'../data/interim/DORAnet_{input_products}_from_{module}.txt'
        output_filepath = f'../data/interim/DORAnet_{output_products}_from_{input_products}_from{module}.txt'
    else:
        raise ValueError("invalid input products name provided")

    # Root process reads helper molecules and precursors
    if rank == 0:
        cofactors_df = pd.read_csv('../data/raw/all_cofactors.csv')
        cofactors_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in cofactors_df["SMILES"]]

        with open(precursors_filepath, 'r') as f:
            precursors_list = [line.strip() for line in f.readlines()]
    else:
        cofactors_list = None
        precursors_list = None

    # Broadcast to all processes
    cofactors_list = comm.bcast(cofactors_list, root=0)
    precursors_list = comm.bcast(precursors_list, root=0)

    # Helper SMILES to remove
    helper_smiles_set = set((
        "O", "O=O", "[H][H]", "O=C=O", "C=O", "[C-]#[O+]", "Br", "[Br][Br]", "CO", "C=C",
        "O=S(O)O", "N", "O=S(=O)(O)O", "O=NO", "N#N", "O=[N+]([O-])O", "NO", "C#N", "S", "O=S=O"
    ))

    # Split precursors into chunks
    def chunkify(lst, n):
        n = min(n, len(lst))
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    if rank == 0:
        chunks = chunkify(precursors_list, size)
        num_active_ranks = len(chunks)
    else:
        chunks = None
        num_active_ranks = None

    num_active_ranks = comm.bcast(num_active_ranks, root=0)

    if rank < num_active_ranks:
        my_precursors = comm.scatter(chunks, root=0)
    else:
        my_precursors = []

    print(f"[Rank {rank}] received {len(my_precursors)} precursors.", flush=True)

    def perform_DORAnet_chem_1step(precursor_smiles: str):
        """Generates one-step DORAnet products for a given precursor SMILES string."""
        unique_id = str(uuid.uuid4())
        unique_jobname = f'{precursor_smiles}_{unique_id}'

        forward_network = synthetic.generate_network(
            job_name=unique_jobname,
            starters={precursor_smiles},
            helpers=tuple(helper_smiles_set),
            gen=1,
            direction="forward"
        )

        generated_chem_products_list = []
        for mol in forward_network.mols:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol.uid))
            if smiles and smiles not in helper_smiles_set:
                generated_chem_products_list.append(smiles)
        return generated_chem_products_list

    # Run modifications
    my_results = []
    for precursor in my_precursors:
        my_results.append(perform_DORAnet_chem_1step(precursor))

    # Flatten results
    my_results_flat = [item for sublist in my_results for item in sublist]

    # Gather results to root
    all_results = comm.gather(my_results_flat, root=0)

    if rank == 0:
        combined_results = set()
        for sublist in all_results:
            combined_results.update(sublist)

        print(f"\nTotal unique chemical products generated: {len(combined_results)}\n", flush=True)

        with open(output_filepath, 'w') as f:
            f.write('\n'.join(combined_results))
