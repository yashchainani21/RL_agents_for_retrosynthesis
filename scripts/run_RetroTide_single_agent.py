from bokeh.io import output_notebook
output_notebook()

from rdkit import Chem
from RetroTide_agent.node import Node
from RetroTide_agent.mcts import MCTS

root = Node(PKS_product = None,
            PKS_design = None,
            parent = None,
            depth = 0)

mcts = MCTS(root = root,
            target_molecule = Chem.MolFromSmiles("CCCCCC(=O)O"), # OC(CC(O)CC(O)=O)/C=C/C1=CC=CC=C1 # CCCCCC(=O)O # O=C1C=CCC(CO)O1 # OC(CC(O)CC(O)=O)C=CC1=CC=CC=C1
            max_depth = 5,
            total_iterations = 15000,
            maxPKSDesignsRetroTide = 3000,
            selection_policy = "UCB1",
            save_logs = False)

mcts.run()
mcts.save_results()

print('\nFollowing are the successful nodes that were actually reached by the RetroTide MCTS agent:\n')
for node in mcts.successful_nodes:
    print(node)
    print('')

print('\nFollowing are the successful PKS designs that were reached in simulation:\n')
for design in mcts.successful_simulated_designs:
    print(design)
    print('')