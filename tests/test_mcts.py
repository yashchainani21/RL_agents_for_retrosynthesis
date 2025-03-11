from rdkit import Chem
from RetroTide_agent.node import Node
from RetroTide_agent.mcts import MCTS

def test_UCB1_selection_policy():
    """
    Test whether the UCB1 selection policy correctly prioritizes child node with the highest value from UCB1 formula.
    The UCB1 policy balances exploration and exploitation.
    In this scenario, two child nodes have the same number of visits assigned to them but different values.
    This test checks if UCB1 policy correctly identifies child node that best balances exploration & exploitation.
    For exploration, we want to visit less-visited nodes.
    Meanwhile, for exploitation, we want to visit more-visited nodes.
    """

    root = Node(PKS_product = None,
                PKS_design = None,
                depth = 0)

    child1 = Node(PKS_product = Chem.MolFromSmiles("CC"),
                  PKS_design = None,
                  depth = 1)

    child2 = Node(PKS_product = Chem.MolFromSmiles("CCCCCCCC"),
                  PKS_design = None,
                  depth = 1)

    # manually set visits to simulate prior exploration
    child1.visits = 1
    child2.visits = 1

    # manually set value to simulate prior exploitation
    child1.value = 100
    child2.value = 1

    root.add_child(child1)
    root.add_child(child2)

    mcts = MCTS(root = root,
                target_molecule = Chem.MolFromSmiles("CCCCCCCC"),
                max_depth = 10,
                selection_policy = "UCB1")

    selected_node = mcts.select(root)

    # we expect child1 to be selected since it has a significantly higher value.
    assert selected_node == child1


