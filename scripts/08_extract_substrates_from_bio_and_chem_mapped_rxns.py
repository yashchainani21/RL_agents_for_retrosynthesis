"""
In the previous script, all mapped enzymatic reactions and synthetic chemistry reactions were combined.
Here, we parse those mapped reactions to extract out the reactants involved in each mapped reaction.
This will eventually give us a table of reactant structures and corresponding templates.
Such structure-template mapping data will later enable us to train a multi-class template prioritizer model.
At inference time, such a supervised multi-class classifier can help predict which templates apply to a given reactant.
"""
import pandas as pd
from rdkit import Chem

input_filepath = '../data/processed/all_unique_bio_and_chem_mapped_rxns.csv'
all_bio_and_chem_mapped_rxns = pd.read_csv(input_filepath)

print(all_bio_and_chem_mapped_rxns.columns)