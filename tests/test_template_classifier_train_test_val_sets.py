import pytest
import pandas as pd

def test_no_missing_values_in_reactant_template_dataset_before_splits():
    reactant_template_pairs_before_splits_df = pd.read_csv('../data/processed/all_bio_and_chem_unique_reactant_template_pairs_no_stereo_w_integer_labels.csv')
    assert reactant_template_pairs_before_splits_df.isnull().values.any() == False

def test_each_template_in_reactant_template_dataset_has_minimally_10_examples_before_splits():
    reactant_template_pairs_before_splits_df = pd.read_csv('../data/processed/all_bio_and_chem_unique_reactant_template_pairs_no_stereo_w_integer_labels.csv')
    template_frequency_counts = reactant_template_pairs_before_splits_df['Template Label'].value_counts()

    for idx, count in enumerate(template_frequency_counts):
        assert count >= 10

def test_no_overlap_between_train_test_val_split_of_reactant_template_pairs():
    """
    Ensures that there is no overlap between the train, validation, and test sets
    based on the full row contents (Reactant + Template Label + Type).
    """
    # Define a unique identifier for each row

    train_data = pd.read_csv("../data/training/training_reactant_template_pairs.csv")
    test_data = pd.read_csv("../data/testing/testing_reactant_template_pairs.csv")
    val_data = pd.read_csv("../data/validation/validation_reactant_template_pairs.csv")

    train_keys = set(train_data.apply(lambda row: (row['Reactant'], row['Template Label'], row['Type'], row['Label Index']), axis=1))
    val_keys = set(val_data.apply(lambda row: (row['Reactant'], row['Template Label'], row['Type'], row['Label Index']), axis=1))
    test_keys = set(test_data.apply(lambda row: (row['Reactant'], row['Template Label'], row['Type'], row['Label Index']),axis=1))

    # Ensure disjoint sets
    assert train_keys.isdisjoint(val_keys), "Train and Validation sets overlap!"
    assert train_keys.isdisjoint(test_keys), "Train and Test sets overlap!"
    assert val_keys.isdisjoint(test_keys), "Validation and Test sets overlap!"
