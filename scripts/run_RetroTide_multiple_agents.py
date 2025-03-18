import multiprocessing
from rdkit import Chem
from RetroTide_agent.node import Node
from RetroTide_agent.mcts import MCTS

def run_mcts(target_smiles):
    """Function to run a single MCTS search for a given molecule."""
    root = Node(PKS_product = None,
                PKS_design = None,
                parent = None,
                depth = 0)

    mcts = MCTS(root = root,
                target_molecule = Chem.MolFromSmiles(target_smiles),
                max_depth = 5,
                total_iterations = 15000,
                maxPKSDesignsRetroTide = 3000,
                selection_policy =" UCB1")

    mcts.run()
    #mcts.save_results()

    print(f'\nMCTS search completed for molecule: {target_smiles}')
    print('\nSuccessful nodes:')
    for node in mcts.successful_nodes:
        print(node, '\n')

    print('\nSuccessful PKS designs:')
    for design in mcts.successful_simulated_designs:
        print(design, '\n')

# List of molecules to run MCTS searches on
target_molecules = ["CCCCCC(=O)O",
                    "O=C1C=CCC(CO)O1"]

multiprocessing.set_start_method('fork')

if __name__ == "__main__":

    processes = []

    for smiles in target_molecules:
        p = multiprocessing.Process(target = run_mcts, args = (smiles,))
        processes.append(p)
        p.start()

    # # wait for all processes to finish
    # for p in processes:
    #     p.join()
