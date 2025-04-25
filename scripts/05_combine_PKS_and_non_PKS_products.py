from typing import List, Set
import pandas as pd
from multiprocessing import Pool, cpu_count
from rdkit import Chem
import os

module = 'LM'

PKS_products_filepath = f'../data/interim/unique_PKS_products_no_stereo_{module}.txt'
BIO_products_filepath = f'../data/interim/DORAnet_BIO1_from_{module}.txt'
CHEM_products_filepath = f'../data/interim/DORAnet_CHEM1_from_{module}.txt'

outfile_path = f'../data/processed/{module}_labeled_products.parquet'


# ----------------------------
# Canonicalization Utilities
# ----------------------------
def safe_canonicalize(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None


def canonicalize_in_chunks(smiles_list: List[str], chunk_size: int = 100_000) -> Set[str]:
    total_unique = set()
    n_chunks = (len(smiles_list) + chunk_size - 1) // chunk_size
    print(f"📦 Canonicalizing {len(smiles_list)} SMILES in {n_chunks} chunks...")

    for i in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[i:i + chunk_size]
        with Pool(cpu_count()) as pool:
            canon_chunk = pool.map(safe_canonicalize, chunk)

        chunk_unique = set(filter(None, canon_chunk))
        total_unique.update(chunk_unique)
        print(f"  ✅ Processed chunk {i // chunk_size + 1}/{n_chunks} — {len(chunk_unique)} unique")

    return total_unique


# ----------------------------
# File I/O
# ----------------------------
def load_smiles(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


# ----------------------------
# Main script
# ----------------------------
if __name__ == '__main__':
    # Load all SMILES
    PKS_smiles = load_smiles(PKS_products_filepath)
    BIO_smiles = load_smiles(BIO_products_filepath)
    CHEM_smiles = load_smiles(CHEM_products_filepath)

    # Canonicalize
    canon_PKS = canonicalize_in_chunks(PKS_smiles)
    canon_BIO = canonicalize_in_chunks(BIO_smiles)
    canon_CHEM = canonicalize_in_chunks(CHEM_smiles)

    # Combine DORAnet (BIO + CHEM)
    all_DORAnet = canon_BIO.union(canon_CHEM)
    overlap = len(canon_BIO) + len(canon_CHEM) - len(all_DORAnet)
    print(f"\n🔁 Combined DORAnet size: {len(all_DORAnet)} (overlap: {overlap})")

    # Label and combine
    label_dict = {smiles: 1 for smiles in canon_PKS}
    for smiles in all_DORAnet:
        if smiles not in label_dict:
            label_dict[smiles] = 0

    combined_df = pd.DataFrame({
        'SMILES': list(label_dict.keys()),
        'labels': list(label_dict.values())
    })

    # Save
    combined_df.to_parquet(outfile_path, index=False)
    print(f"\n✅ Saved {len(combined_df)} molecules to {outfile_path}")
