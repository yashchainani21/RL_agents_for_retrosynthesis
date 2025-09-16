"""
In this script, we split previously combined PKS and non-PKS products into train/test/validation sets.
Splits will be stratified by the label for each molecule which determines if a product is a polyketide (1) or not (0).
This ensures that the distribution of polyketides to non-polyketides is retained across all sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

module = "LM"
input_PKS_and_non_PKS_products_path = f'../data/processed/{module}_labeled_products.parquet'

train_outfile_path = f'../data/training/training_{module}_PKS_and_non_PKS_products.parquet'
test_outfile_path = f'../data/testing/testing_{module}_PKS_and_non_PKS_products.parquet'
val_outfile_path = f'../data/validation/validation_{module}_PKS_and_non_PKS_products.parquet'

PKS_and_non_PKS_products_df = pd.read_parquet(input_PKS_and_non_PKS_products_path)

# first, we split into 80% training and 20% testing + validation
train, test_and_val_combined = train_test_split(
    PKS_and_non_PKS_products_df,
    test_size = 0.2,
    stratify = PKS_and_non_PKS_products_df['labels'], # stratify by whether molecules are PKSs or PKS-modified products
    random_state = 42)

# then, the 20% testing and validation are divided further into 10% testing and 10% validation
val, test = train_test_split(
    test_and_val_combined,
    test_size = 0.5,
    stratify = test_and_val_combined['labels'], # stratify by whether molecules are PKSs or PKS-modified products
    random_state = 42)

print(f"Train size: {len(train)}")
print(f"Validation size: {len(val)}")
print(f"Test size: {len(test)}")

train.to_parquet(train_outfile_path, index=False)
test.to_parquet(test_outfile_path, index=False)
val.to_parquet(val_outfile_path, index=False)

