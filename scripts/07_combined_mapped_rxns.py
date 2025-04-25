"""
In this script, mapped enzymatic and chemical reactions are vertically stacked together to combine them.
Enzymatic reactions are obtained from the EnzymeMap versions of BRENDA, KEGG, and MetaCyc and mapped with JN3604IMT.
Meanwhile, chemical reactions are obtained from USPTO50K and mapped with around 323 synthetic chemistry templates.
"""
import numpy as np
import pandas as pd

# read in all enzymatically mapped reactions
brenda_mappings = pd.read_csv("../data/processed/EnzymeMap_all_BRENDA_imt_rule_mappings.csv")
brenda_mappings = brenda_mappings.drop(labels = ['Unnamed: 0'], axis = 1)

kegg_mappings = pd.read_csv("../data/processed/EnzymeMap_KEGG_imt_rule_mappings.csv")

metacyc_mappings = pd.read_csv("../data/processed/EnzymeMap_MetaCyc_imt_rule_mappings.csv")

# vertically stack all enzymatically mapped reactions first
all_bio_mappings_df = pd.concat([brenda_mappings, kegg_mappings, metacyc_mappings],
                                axis = 0,
                                ignore_index = True)

# remove any duplicates from the union of mapped brenda, kegg, and metacyc reactions
duplicates = all_bio_mappings_df.duplicated()
print(f'\nRemoving {sum(duplicates)} duplicate mapped reactions from the union of BRENDA, KEGG, and MetaCyc mappings')

all_bio_mappings_df = all_bio_mappings_df[~duplicates].reset_index(drop = True)
print(f'\nNumber of unique mapped enzymatic reactions across BRENDA, KEGG, and MetaCyc: {all_bio_mappings_df.shape[0]}')

# insert a column within the mapped enzymatic reactions to indicate reaction type (helps with stratification later)
all_bio_mappings_df["type"] = np.repeat("bio", all_bio_mappings_df.shape[0])

# next, read in all synthetic chemistry mapped reactions
USPTO50K_CHO_mappings = pd.read_csv("../data/processed/mapped_USPTO50K_CHO.csv")
USPTO50K_CHO_mappings = USPTO50K_CHO_mappings.drop(labels = ['Unnamed: 0'], axis = 1)

USPTO50K_N_mappings = pd.read_csv("../data/processed/mapped_USPTO50K_N.csv")

USPTO50K_S_mappings = pd.read_csv("../data/processed/mapped_USPTO50K_onlyS.csv")
USPTO50K_S_mappings = USPTO50K_S_mappings.drop(labels = ['Unnamed: 0'], axis = 1)

# vertically stack all chemically mapped reactions
all_chem_mappings_df = pd.concat([USPTO50K_CHO_mappings, USPTO50K_N_mappings, USPTO50K_S_mappings],
                                 axis = 0,
                                 ignore_index = True)

# remove any duplicates from the union of mapped CHO, N, and S synthetic chemistry reactions
duplicates = all_chem_mappings_df.duplicated()
print(f'\nRemoving {sum(duplicates)} duplicate mapped reactions from the union of all mapped chemistry reactions')

all_chem_mappings_df = all_chem_mappings_df[~duplicates].reset_index(drop = True)
print(f'\nNumber of unique mapped synthetic chemistry reactions across USPTO50K: {all_chem_mappings_df.shape[0]}')

# insert a column within the mapped synthetic chemistry reactions to indicate reaction type
all_chem_mappings_df["type"] = np.repeat("chem", all_chem_mappings_df.shape[0])

# combine both bio and chem mappings, then remove any duplicates one last time
bio_and_chem_mapped_rxns_df = pd.concat([all_bio_mappings_df, all_chem_mappings_df], axis = 0, ignore_index = True)
duplicates = bio_and_chem_mapped_rxns_df.duplicated()
print(f'\nRemoving {sum(duplicates)} duplicate mapped reactions from all bio and chem mapped reactions')

#

