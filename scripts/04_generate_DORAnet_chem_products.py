from rdkit import Chem
import doranet.modules.enzymatic as enzymatic
import pandas as pd

# read in cofactors and prepare a list of their canonical SMILES
cofactors_df = pd.read_csv('../data/raw/all_cofactors.csv')
cofactors_list = list(cofactors_df["SMILES"])
cofactors_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in cofactors_list]

# read in synthetically generated PKS products
max_module = "LM"
modify_PKS_products = True
DORAnet_product_type_to_modify = 'BIO'

# if modifying DORAnet PKS products for one chem step to get DORAnet CHEM1 products
if modify_PKS_products:
    precursors_filepath = f'../data/interim/unique_PKS_products_no_stereo_{max_module}.txt'
    output_filepath = f'../data/interim/DORAnet_CHEM1_products_from_{max_module}_polyketides.txt'

if DORAnet_product_type_to_modify == 'BIO':
    precursors_filepath = f'../data/interim/DORAnet_BIO1_products_from_{max_module}_polyketides.txt'
    output_filepath = f'../data/interim/DORAnet_BIO2_products_from_{max_module}_polyketides.txt'

if DORAnet_product_type_to_modify == 'CHEM':
    precursors_filepath = f'../data/interim/DORAnet_CHEM1_products_from_{max_module}_polyketides.txt'
    output_filepath = f'../data/interim/DORAnet_CHEM2_products_from_BIO1.txt'