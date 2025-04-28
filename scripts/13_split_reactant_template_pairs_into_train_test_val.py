import pandas as pd
from sklearn.model_selection import train_test_split

reactant_template_pairs_filepath = '../data/processed/all_bio_and_chem_unique_reactant_template_pairs_no_stereo_w_integer_labels.csv'
reactant_template_pairs_df = pd.read_csv(reactant_template_pairs_filepath)

stratify_with = 'by_specific_rule' # choose between 'by_bio_or_chem' or 'by_specific_rule'

train_outfile_path = f'../data/training/reactant_template_pairs_stratified_{stratify_with}.csv'
test_outfile_path = f'../data/testing/reactant_template_pairs_stratified_{stratify_with}.csv'
val_outfile_path = f'../data/validation/reactant_template_pairs_stratified_{stratify_with}.csv'

if stratify_with == 'by_bio_or_chem':

    # first, we split into train (80%) and temp (20%)
    train_data, temp_data = train_test_split(
        reactant_template_pairs_df,
        test_size = 0.2,
        stratify = reactant_template_pairs_df['Type'],
        random_state = 42)

    # then, we split temp_data into validation (10%) and test (10%)
    val_data, test_data = train_test_split(
        temp_data,
        test_size = 0.5,
        stratify = temp_data['Type'],
        random_state = 42)

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    train_data.to_csv(train_outfile_path, index=False)
    test_data.to_csv(test_outfile_path, index=False)
    val_data.to_csv(val_outfile_path, index=False)

if stratify_with == 'by_specific_rule':

    # identify common and rare templates first
    template_counts = reactant_template_pairs_df["Template Label"].value_counts()
    common_templates = template_counts[template_counts >= 10].index.tolist()
    rare_templates = template_counts[template_counts < 10].index.tolist()

    common_data = reactant_template_pairs_df[reactant_template_pairs_df['Template Label'].isin(common_templates)]
    rare_data = reactant_template_pairs_df[reactant_template_pairs_df['Template Label'].isin(rare_templates)]

    # then, we split common data stratified by 'Template Label'
    # here, common data refers to templates for which there are >=10 reactant structures
    train_common, temp_common = train_test_split(
        common_data,
        test_size = 0.2,
        stratify = common_data['Template Label'],
        random_state = 42)

    val_common, test_common = train_test_split(
        temp_common,
        test_size = 0.5,
        stratify = temp_common['Template Label'],
        random_state = 42)

    # now, we split the rare data randomly
    # here, rare data refers to templates for which there are <10 reactant structures
    train_rare, temp_rare = train_test_split(
        rare_data,
        test_size = 0.2,
        random_state = 42)

    val_rare, test_rare = train_test_split(
        temp_rare,
        test_size = 0.5,
        random_state = 42)

    # Step 4: Combine splits
    train_data = pd.concat([train_common, train_rare]).reset_index(drop=True)
    val_data = pd.concat([val_common, val_rare]).reset_index(drop=True)
    test_data = pd.concat([test_common, test_rare]).reset_index(drop=True)

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    train_data.to_csv(train_outfile_path, index=False)
    test_data.to_csv(test_outfile_path, index=False)
    val_data.to_csv(val_outfile_path, index=False)