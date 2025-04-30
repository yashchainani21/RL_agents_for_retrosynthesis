"""
In this script, we split previously combined PKS and non-PKS products into train/test/validation sets.
Splits will be stratified by the label for each molecule which determines if a product is a polyketide (1) or not (0).
This ensures that the distribution of polyketides to non-polyketides is retained across all sets.
"""

import dask.dataframe as dd

module = "LM"
input_PKS_and_non_PKS_products_path = f'../data/processed/{module}_labeled_products.parquet'

PKS_and_non_PKS_products_df = dd.read_parquet(input_PKS_and_non_PKS_products_path)
