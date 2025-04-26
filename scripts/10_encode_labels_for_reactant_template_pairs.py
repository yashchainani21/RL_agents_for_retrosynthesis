"""
In this script, we one-hot-encode reaction templates assigned to each reactant structure.
The total set of reaction templates available to us has 3927 elements (323 chemistry + 3604 biology).
We will first read in both biology and chemical templates in order to combine them into a single list.
Each template in this list will then be assigned an integer from 0 to 3926.
"""
import pandas as pd

# read in chemical and biological templates
enzymatic_templates_df = pd.read_csv("../data/raw/JN3604IMT_rules.tsv", delimiter = '\t')
enzymatic_templates_df.drop(labels = ["Reactants", "SMARTS", "Products", "Comments"], axis = 1, inplace = True)
enzymatic_template_names_list = list(enzymatic_templates_df["Name"])

chemical_templates_df = pd.read_csv("../data/raw/synthetic_chemistry_rules.csv")
chemical_templates_df.drop(labels = ["SMARTS", "Unnamed: 0"], axis = 1, inplace = True)
chemistry_template_names_list = list(chemical_templates_df["Name"])

all_templates = enzymatic_template_names_list + chemistry_template_names_list
template_to_idx = {template: i for i, template in enumerate(all_templates)}

# Save this mapping for future use
template_to_idx_mapping_df = pd.DataFrame({
    "Rule": list(template_to_idx.keys()),
    "Rule_index": list(template_to_idx.values())})

template_to_idx_mapping_df.to_csv("../data/processed/template_to_idx_mapping.csv", index = False)

dataset_type = 'training' # choose from 'training', 'testing', or 'validation'
stratification_type = 'bio_or_chem' # chose from 'bio_or_chem' or 'specific_rule'
input_filepath = f'../data/{dataset_type}/reactant_template_pairs_stratified_by_{stratification_type}.csv'
output_filepath = f'../data/{dataset_type}/reactant_template_pairs_stratified_by_{stratification_type}_labels_encoded.csv'

input_data = pd.read_csv(input_filepath)

# make sure the input columns are as we would expect
assert {'Reactant', 'Template Label', 'Type'}.issubset(input_data.columns), "Input data does not have expected columns."



