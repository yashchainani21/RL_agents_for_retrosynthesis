"""
In this script, we split the unique reactant-template pairs obtained previously into train, test, and validation sets.
We can stratify the reactant-template pairs dataset in two ways.
The first is by the type of the reaction, i.e., by 'bio' or 'chem'.
This ensures that the ratio of biology to chemistry reactions is retained in each set.
Alternatively, we can also stratify by the specific template label.
To do this, we first isolate template labels for which we have at least 10 reactant examples.
These common reactant-template pairs are then split into train, test, and val sets, stratified by template label.
Reactant-template pairs for which the templates have less than 10 reactants mapped to each of them are then lumped together.
Using this second set of template reactant pairs, a random 80/10/10 split is performed.
Both splits are then recombined together.
"""

import pandas as pd

# create a mapping between templates and integers
reactant_template_pairs_filepath = '../data/processed/all_bio_and_chem_unique_reactant_template_pairs_no_stereo.csv'
reactant_template_pairs_df = pd.read_csv(reactant_template_pairs_filepath)

unique_templates_set = set(reactant_template_pairs_df['Template Label'])
template_to_idx = {template: i for i, template in enumerate(unique_templates_set)}

# Save this mapping for future use
template_to_idx_mapping_df = pd.DataFrame({
                                "Rule": list(template_to_idx.keys()),
                                "Rule_index": list(template_to_idx.values())})

template_to_idx_mapping_df.to_csv("../data/processed/template_to_idx_mapping.csv", index = False)

# insert integer-encoded labels into dataframe of reactant-template pairs
reactant_template_pairs_df['Label Index'] = reactant_template_pairs_df['Template Label'].map(template_to_idx)

# save the dataframe with integer-encoded labels to later split into train/ test/ val sets
reactant_template_pairs_df.to_csv("../data/processed/all_bio_and_chem_unique_reactant_template_pairs_no_stereo_w_integer_labels.csv", index = False)
