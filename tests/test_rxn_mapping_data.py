import pandas as pd

def test_if_EnzymeMap_BRENDA_mappings_are_unique():
    """Ensure all mapped BRENDA reactions are unique."""
    mapped_brenda_df = pd.read_csv("../data/processed/EnzymeMap_all_BRENDA_imt_rule_mappings.csv")
    assert sum(mapped_brenda_df.duplicated()) == 0

def test_if_EnzymeMap_KEGG_mappings_are_unique():
    """Ensure all mapped KEGG reactions are unique."""
    mapped_brenda_df = pd.read_csv("../data/processed/EnzymeMap_KEGG_imt_rule_mappings.csv")
    assert sum(mapped_brenda_df.duplicated()) == 0

def test_if_EnzymeMap_MetaCyc_mappings_are_unique():
    """Ensure all mapped MetaCyc reactions are unique."""
    mapped_brenda_df = pd.read_csv("../data/processed/EnzymeMap_MetaCyc_imt_rule_mappings.csv")
    assert sum(mapped_brenda_df.duplicated()) == 0

def test_if_USPTO50K_CHO_mappings_are_unique():
    """Ensure all mapped USPTO50K reactions with C, H, and O atoms only are unique."""
    mapped_USPTO50K_CHO_df = pd.read_csv("../data/processed/mapped_USPTO50K_CHO.csv")
    assert sum(mapped_USPTO50K_CHO_df.duplicated()) == 0

def test_if_USPTO50K_N_mappings_are_unique():
    """Ensure all mapped USPTO50K reactions with C, H, O, and N atoms are unique."""
    mapped_USPTO50K_N_df = pd.read_csv("../data/processed/mapped_USPTO50K_N.csv")
    assert sum(mapped_USPTO50K_N_df.duplicated()) == 0

def test_if_USPTO50K_S_mappings_are_unique():
    """Ensure all mapped USPTO50K reactions with C, H, O, and S atoms are unique."""
    mapped_USPTO50K_S_df = pd.read_csv("../data/processed/mapped_USPTO50K_onlyS.csv")
    assert sum(mapped_USPTO50K_S_df.duplicated()) == 0