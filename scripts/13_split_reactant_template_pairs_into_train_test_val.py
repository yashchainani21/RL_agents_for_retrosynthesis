"""
In this script, we split our dataset of reactant-template pairs into train/test/val.
An 80/10/10 split is used, stratified using the 'Template Label' column.
Previously, we ensured only template-reactant pairs for which templates have >=10 reactants mapped to them are retained.
Since there are at least 10 examples for each template, such a stratified split is possible.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

reactant_template_pairs_filepath = '../data/processed/all_bio_and_chem_unique_reactant_template_pairs_no_stereo_w_integer_labels.csv'
reactant_template_pairs_df = pd.read_csv(reactant_template_pairs_filepath)

train_outfile_path = f'../data/training/training_reactant_template_pairs.csv'
test_outfile_path = f'../data/testing/testing_reactant_template_pairs.csv'
val_outfile_path = f'../data/validation/validation_reactant_template_pairs.csv'

# split reactant-template pairs stratified by 'Template Label'
# at this point, only templates for which there are at least 10 examples have been retained
train, test_and_val_combined = train_test_split(
    reactant_template_pairs_df ,
    test_size = 0.2,
    stratify = reactant_template_pairs_df ['Template Label'],
    random_state = 42)

val, test = train_test_split(
    test_and_val_combined,
    test_size = 0.5,
    stratify = test_and_val_combined['Template Label'],
    random_state = 42)

print(f"Train size: {len(train)}")
print(f"Validation size: {len(val)}")
print(f"Test size: {len(test)}")

train.to_csv(train_outfile_path, index=False)
test.to_csv(test_outfile_path, index=False)
val.to_csv(val_outfile_path, index=False)