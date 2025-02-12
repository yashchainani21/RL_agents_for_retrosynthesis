from rdkit import Chem
from RetroTide_agent.node import Node
from RetroTide_agent.mcts import MCTS

root = Node(PKS_product = None,
            PKS_design = None,
            parent = None,
            depth = 0)

mcts = MCTS(root = root,
            target_molecule = Chem.MolFromSmiles("CCCC"),
            max_depth = 50,
            maxPKSDesignsRetroTide = 3000,
            selection_policy = "UCB1")

mcts.run()

# selected_node = mcts.select(node = root)
# mcts.expand(node = selected_node)
# mcts.expand(node = root.children[0])
#
# print(root.children[0])
# print('')
# print(root.children[1])
# print('')
# print('-----\n')
# print(root.children[1000])
# print(mcts.simulate_and_get_reward(root.children[1000]))
#
# mcts.backpropagate(root.children[1000], reward = 1)
# print(root.value)
# print(root.visits)



