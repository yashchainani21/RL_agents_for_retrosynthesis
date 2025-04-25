import pandas as pd

def test_if_EnzymeMap_BRENDA_mappings_are_unique():
    """Ensure all mapped BRENDA reaction-template pairs are unique."""
    mapped_brenda_df = pd.read_csv("../data/processed/EnzymeMap_all_BRENDA_imt_rule_mappings.csv")
    assert sum(mapped_brenda_df.duplicated()) == 0

def test_if_EnzymeMap_KEGG_mappings_are_unique():
    """Ensure all mapped KEGG reaction-template pairs are unique."""
    mapped_brenda_df = pd.read_csv("../data/processed/EnzymeMap_KEGG_imt_rule_mappings.csv")
    assert sum(mapped_brenda_df.duplicated()) == 0

def test_if_EnzymeMap_MetaCyc_mappings_are_unique():
    """Ensure all mapped MetaCyc reaction-template pairs are unique."""
    mapped_brenda_df = pd.read_csv("../data/processed/EnzymeMap_MetaCyc_imt_rule_mappings.csv")
    assert sum(mapped_brenda_df.duplicated()) == 0

def test_if_USPTO50K_CHO_mappings_are_unique():
    """Ensure all mapped USPTO50K reaction-template pairs with C, H, and O atoms only are unique."""
    mapped_USPTO50K_CHO_df = pd.read_csv("../data/processed/mapped_USPTO50K_CHO.csv")
    assert sum(mapped_USPTO50K_CHO_df.duplicated()) == 0

def test_if_USPTO50K_N_mappings_are_unique():
    """Ensure all mapped USPTO50K reaction-template pairs with C, H, O, and N atoms are unique."""
    mapped_USPTO50K_N_df = pd.read_csv("../data/processed/mapped_USPTO50K_N.csv")
    assert sum(mapped_USPTO50K_N_df.duplicated()) == 0

def test_if_USPTO50K_S_mappings_are_unique():
    """Ensure all mapped USPTO50K reaction-template pairs with C, H, O, and S atoms are unique."""
    mapped_USPTO50K_S_df = pd.read_csv("../data/processed/mapped_USPTO50K_onlyS.csv")
    assert sum(mapped_USPTO50K_S_df.duplicated()) == 0

def test_if_biochem_mappings_are_unique():
    """Ensure all mapped biochemistry reaction-template pairs arising from the above datasets are unique."""
    all_mapped_bio_and_chem_rxns = pd.read_csv("../data/processed/all_unique_bio_and_chem_mapped_rxns.csv")
    assert sum(all_mapped_bio_and_chem_rxns.duplicated()) == 0