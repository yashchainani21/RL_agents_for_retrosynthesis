from mpi4py import MPI
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
radius = 2  # ECFP4 radius
n_bits = 2048  # Fingerprint length

def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits = n_bits)
    arr = np.zeros((n_bits,), dtype = np.uint8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


if rank == 0:
    # Master rank reads the full dataset
    dataset_type = 'training'  # choose from 'training' or 'testing' / 'validation'
    input_filepath = f'../data/{dataset_type}/{dataset_type}_reactant_template_pairs.csv'

    data = pd.read_csv(input_filepath)

    # Extract reactants and label indices
    smiles_list = data['Reactant'].tolist()
    label_indices = data['Label Index'].tolist()

    # Split into approximately equal-sized chunks
    smiles_chunks = np.array_split(smiles_list, size)
    labels_chunks = np.array_split(label_indices, size)
else:
    smiles_chunks = None
    labels_chunks = None

# Scatter SMILES and labels to each process
my_smiles_chunk = comm.scatter(smiles_chunks, root=0)
my_labels_chunk = comm.scatter(labels_chunks, root=0)

# Each process featurizes its chunk
my_features = []
my_labels = []

for smi, label in zip(my_smiles_chunk, my_labels_chunk):
    fp = featurize_smiles(smi)
    if fp is not None:
        my_features.append(fp)
        my_labels.append(label)

# Gather results from all ranks
all_features = comm.gather(my_features, root=0)
all_labels = comm.gather(my_labels, root=0)

if rank == 0:
    # Flatten the gathered lists
    all_features = [item for sublist in all_features for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]

    X = np.vstack(all_features)
    y = np.array(all_labels)

    X_df = pd.DataFrame(X.astype(np.uint8))  # cast to uint8 to save space
    y_df = pd.DataFrame(y, columns=["Label Index"])

    X_df.to_parquet(f'../data/{dataset_type}/reactant_ecfp4_fingerprints_stratified_by_{stratification_type}.parquet', index=False)
    y_df.to_parquet(f'../data/{dataset_type}/template_labels_stratified_by_{stratification_type}.parquet', index=False)
