import pandas as pd

def test_if_EnzymeMap_BRENDA_mappings_are_unique():
    mapped_brenda_df = pd.read_csv("../data/processed/EnzymeMap_all_BRENDA_imt_rule_mappings.csv")
    assert sum(mapped_brenda_df.duplicated()) == 0

def test_if_EnzymeMap_KEGG_mappings_are_unique():
    mapped_brenda_df = pd.read_csv("../data/processed/EnzymeMap_KEGG_imt_rule_mappings.csv")
    assert sum(mapped_brenda_df.duplicated()) == 0

def test_if_EnzymeMap_MetaCyc_mappings_are_unique():
    mapped_brenda_df = pd.read_csv("../data/processed/EnzymeMap_MetaCyc_imt_rule_mappings.csv")
    assert sum(mapped_brenda_df.duplicated()) == 0