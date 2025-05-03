from mpi4py import MPI
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

radius = 2
n_bits = 2048
module = "M2"
ratio_of_non_PKS_to_PKS_products = 15 # set the ratio of PKS to non-PKS products for down sampling
def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

if rank == 0:
    dataset_type = 'training'
    input_filepath = f'../data/{dataset_type}/{dataset_type}_{module}_PKS_and_non_PKS_products.parquet'
    all_products_df = pd.read_parquet(input_filepath)

    PKS_products_df = all_products_df[all_products_df["labels"] == 1]
    non_PKS_products_df = all_products_df[all_products_df["labels"] == 0]

    # downsample the set of non-PKS products to the pre-determine ratio of non-PKS to PKS products
    num_non_PKS_products_to_sample = int(non_PKS_products_df.shape[0]*ratio_of_non_PKS_to_PKS_products)
    non_PKS_products_df_index = non_PKS_products_df.index.values
    subsampled_indices = np.random.choice(non_PKS_products_df_index, size = num_non_PKS_products_to_sample, replace = False)
    non_PKS_products_df_subsampled = non_PKS_products_df.iloc[subsampled_indices,:]

    # merge both dataframes back together
    data = pd.concat([PKS_products_df, non_PKS_products_df_subsampled], axis = 0).sample(frac=1).reset_index(drop=True)

    smiles_list = data['SMILES'].tolist()
    label_list = data['labels'].tolist()

    assert len(smiles_list) == len(label_list), "SMILES and labels length mismatch."

    # Explicitly ensure chunk size matches MPI ranks
    #smiles_chunks = np.array_split(smiles_list, size)
    # labels_chunks = np.array_split(label_indices, size)

    # split list of SMILES strings into chunks
    smiles_chunks = [[] for _ in range(size)]
    for i, chunk in enumerate(smiles_list):
        smiles_chunks[i % size].append(chunk)

    # split list of binary labels into chunks
    labels_chunks = [[] for _ in range(size)]
    for i, chunk in enumerate(label_list):
        labels_chunks[i % size].append(chunk)

    for idx, chunk in enumerate(smiles_chunks):
        print(f"Chunk {idx} size: {len(chunk)}")

    # Check serialization explicitly
    import pickle
    try:
        pickle.dumps(smiles_chunks)
        pickle.dumps(labels_chunks)
    except Exception as e:
        print("Serialization Error:", e)
        comm.Abort()
else:
    smiles_chunks = None
    labels_chunks = None

# Scatter safely
my_smiles_chunk = comm.scatter(smiles_chunks, root=0)
my_labels_chunk = comm.scatter(labels_chunks, root=0)

print(f"Rank {rank}: received {len(my_smiles_chunk)} SMILES.")

# Featurization
my_features = []
my_labels = []

for smi, label in zip(my_smiles_chunk, my_labels_chunk):
    fp = featurize_smiles(smi)
    if fp is not None:
        my_features.append(fp)
        my_labels.append(label)

# Gather results
all_features = comm.gather(my_features, root=0)
all_labels = comm.gather(my_labels, root=0)

if rank == 0:
    all_features_flat = [fp for sublist in all_features for fp in sublist]
    all_labels_flat = [label for sublist in all_labels for label in sublist]

    print(f"Total featurized molecules: {len(all_features_flat)}")

    X = np.vstack(all_features_flat).astype(np.uint8)
    y = np.array(all_labels_flat).astype(np.uint8)

    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)

    X_df.to_parquet(f'../data/{dataset_type}/{dataset_type}_{module}_PKS_and_non_PKS_products_fingerprints_with_downsampling_ratio_{ratio_of_non_PKS_to_PKS_products}.parquet', index=False)
    y_df.to_parquet(f'../data/{dataset_type}/{dataset_type}_{module}_PKS_and_non_PKS_products_labels.parquet_with_downsampling_ratio_{ratio_of_non_PKS_to_PKS_products}', index=False)
