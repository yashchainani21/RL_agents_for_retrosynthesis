"""
In the previous script, all mapped enzymatic reactions and synthetic chemistry reactions were combined.
Here, we parse those mapped reactions to extract out the reactants involved in each mapped reaction.
This will eventually give us a table of reactant structures and corresponding templates.
Note that reactants which belong to cofactors in biology of helper molecules in chemistry are not stored.
Such structure-template mapping data will later enable us to train a multi-class template prioritizer model.
At inference time, such a supervised multi-class classifier can help predict which templates apply to a given reactant.
"""
import pandas as pd
from rdkit import Chem

input_filepath = '../data/processed/all_unique_bio_and_chem_mapped_rxns.csv'
all_bio_and_chem_mapped_rxns_df = pd.read_csv(input_filepath).drop(labels = ["Template", "Unnamed: 0"], axis = 1)
ignore_stereo = True

# read in cofactors list for biology and set up a list of helper molecules for chemistry
# any reactants which fall within these two lists will not be stored
cofactors_filepath = '../data/raw/all_cofactors.csv'
all_cofactors_list = list(pd.read_csv(cofactors_filepath)['SMILES'])
chemistry_helpers = ["O", "O=O", "[H][H]", "O=C=O", "C=O", "[C-]#[O+]", "Br", "[Br][Br]", "CO", "[C-]#[O+]", "C=C",
                     "O=S(O)O", "N", "O=S(=O)(O)O", "O=NO", "N#N", "O=[N+]([O-])O", "NO", "C#N", "S", "O=S=O"]

all_rxns = list(all_bio_and_chem_mapped_rxns_df['Reaction'])
all_template_labels = list(all_bio_and_chem_mapped_rxns_df['Mapped Rule'])
all_types = list(all_bio_and_chem_mapped_rxns_df['type'])

final_reactants_list = [] # initialize empty list to store all reactant structures
final_template_labels = [] # initialize empty list to store all template labels
final_rxn_types = [] # initialize empty list to store if reaction is 'bio' or 'chem'

for i, rxn in enumerate(all_rxns):

    rxn_type = all_types[i] # extract the rxn type first ('bio', or 'chem')
    rxn_template_label = all_template_labels[i]
    rxn_lhs, _ = rxn.split('>>')

    # considering only the LHS of the reaction (since we care about reactant structures only)
    if '.' in rxn_lhs:
        reactants_list = rxn_lhs.split('.')

        # iterate over each reactant in the reactants list
        for reactant_smiles in reactants_list:
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)

            try:
                Chem.SanitizeMol(reactant_mol) # sanitize reactant mol
                if ignore_stereo:
                    Chem.RemoveStereochemistry(reactant_mol) # remove stereochemistry
                reactant_smiles_canonicalized = Chem.MolToSmiles(reactant_mol)

                # do not save reactant structure if it is a cofactor in biology or a helper molecule in chemistry
                if reactant_smiles_canonicalized not in all_cofactors_list and reactant_smiles_canonicalized not in chemistry_helpers:

                    final_reactants_list.append(reactant_smiles_canonicalized)
                    final_template_labels.append(rxn_template_label)
                    final_rxn_types.append(rxn_type)

            except Chem.rdchem.MolSanitizeException as e:
                pass # do nothing if there are any sanitization errors
    else:
        reactant_mol = Chem.MolFromSmiles(rxn_lhs)

        try:
            Chem.SanitizeMol(reactant_mol) # sanitize reactant mol
            if ignore_stereo:
                Chem.RemoveStereochemistry(reactant_mol)  # remove stereochemistry
            reactant_smiles_canonicalized = Chem.MolToSmiles(reactant_mol)

            # do not save reactant structure if it is a cofactor in biology or a helper molecule in chemistry
            if reactant_smiles_canonicalized not in all_cofactors_list and reactant_smiles_canonicalized not in chemistry_helpers:

                final_reactants_list.append(reactant_smiles_canonicalized)
                final_template_labels.append(rxn_template_label)
                final_rxn_types.append(rxn_type)

        except:
            pass # do nothing if there are sanitization errors

reactant_template_mappings_df = pd.DataFrame(data = {"Reactant": final_reactants_list,
                                                     "Template Label": final_template_labels,
                                                     "Type": final_rxn_types})

duplicates = reactant_template_mappings_df.duplicated()
print(f"\n{sum(duplicates)} duplicate reactant-templates found across both biological and chemical reactions")

all_bio_and_chem_mappings_df = reactant_template_mappings_df[~duplicates].reset_index(drop = True)

print(f'\nNumber of unique reactant-template pairs found (without stereochemistry): {all_bio_and_chem_mappings_df.shape[0]}')

oufile_path = '../data/processed/all_bio_and_chem_unique_reactant_template_pairs_no_stereo.csv'
print(f'Saving to: {oufile_path}')

all_bio_and_chem_mappings_df.to_csv(oufile_path, index = False)