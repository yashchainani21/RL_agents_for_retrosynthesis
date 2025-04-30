"""
In this script, we use MPI to featurize PKS and non-PKS product molecules using ECFP4 fingerprints.
ECFP4 fingerprints are used to later train baseline binary classification XGBoost models.
Given an input molecule, at inference time, these models will predict the probability of the molecule being a polyketide.
"""

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
module = "LM"

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
    input_filepath =  f'../data/{dataset_type}/{dataset_type}_{module}_PKS_and_non_PKS_products.parquet'

    data = pd.read_parquet(input_filepath)

    # Extract reactants and label indices
    smiles_list = data['SMILES'].tolist()
    label_indices = data['label'].tolist()

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

    X_df.to_parquet(f'../data/{dataset_type}/{dataset_type}_PKS_and_non_PKS_products_fingerprints.parquet', index=False)
    y_df.to_parquet(f'../data/{dataset_type}/{dataset_type}_PKS_and_non_PKS_products_labels.parquet', index=False)
