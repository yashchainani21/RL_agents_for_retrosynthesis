import pytest
import pandas as pd

def test_no_overlap_between_splits_of_reactant_template_pairs_01():
    """
    Ensures that there is no overlap between the train, validation, and test sets
    based on the full row contents (Reactant + Template Label + Type).
    """
    # Define a unique identifier for each row

    train_data = pd.read_csv("../data/training/reactant_template_pairs_stratified_by_bio_or_chem.csv")
    test_data = pd.read_csv("../data/testing/reactant_template_pairs_stratified_by_bio_or_chem.csv")
    val_data = pd.read_csv("../data/validation/reactant_template_pairs_stratified_by_bio_or_chem.csv")

    train_keys = set(train_data.apply(lambda row: (row['Reactant'], row['Template Label'], row['Type']), axis=1))
    val_keys = set(val_data.apply(lambda row: (row['Reactant'], row['Template Label'], row['Type']), axis=1))
    test_keys = set(test_data.apply(lambda row: (row['Reactant'], row['Template Label'], row['Type']), axis=1))

    # Ensure disjoint sets
    assert train_keys.isdisjoint(val_keys), "Train and Validation sets overlap!"
    assert train_keys.isdisjoint(test_keys), "Train and Test sets overlap!"
    assert val_keys.isdisjoint(test_keys), "Validation and Test sets overlap!"

def test_no_overlap_between_splits_of_reactant_template_pairs_02():
    """
    Ensures that there is no overlap between the train, validation, and test sets
    based on the full row contents (Reactant + Template Label + Type).
    """
    # Define a unique identifier for each row

    train_data = pd.read_csv("../data/training/reactant_template_pairs_stratified_by_specific_rule.csv")
    test_data = pd.read_csv("../data/testing/reactant_template_pairs_stratified_by_specific_rule.csv")
    val_data = pd.read_csv("../data/validation/reactant_template_pairs_stratified_by_specific_rule.csv")

    train_keys = set(train_data.apply(lambda row: (row['Reactant'], row['Template Label'], row['Type']), axis=1))
    val_keys = set(val_data.apply(lambda row: (row['Reactant'], row['Template Label'], row['Type']), axis=1))
    test_keys = set(test_data.apply(lambda row: (row['Reactant'], row['Template Label'], row['Type']), axis=1))

    # Ensure disjoint sets
    assert train_keys.isdisjoint(val_keys), "Train and Validation sets overlap!"
    assert train_keys.isdisjoint(test_keys), "Train and Test sets overlap!"
    assert val_keys.isdisjoint(test_keys), "Validation and Test sets overlap!"
