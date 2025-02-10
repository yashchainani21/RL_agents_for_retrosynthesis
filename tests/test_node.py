from RL_agents_for_retrosynthesis.RetroTide_agent.node import Node
from rdkit import Chem

def test_node_initialization():
    """
    Test the initialization of a Node instance.
    Verifies that the Node instance is created with the correct attributes.
    """
    node = Node(PKS_product = Chem.MolFromSmiles("CCCCCCC"),
                PKS_design = None,
                parent = None,
                depth = 0)

    assert node.PKS_product is not None
    assert node.PKS_design is None