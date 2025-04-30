"""
In this script, we featurize previously combined PKS and non-PKS products.
"""

import pandas as pd

module = "LM"
outfile_path = f'../data/processed/{module}_labeled_products.parquet'
