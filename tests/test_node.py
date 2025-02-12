from RetroTide_agent.node import Node
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

def test_add_child_and_update_parent():
    """
    Test the add_child method of a Node instance.
    Ensures that when a child is added to a parent node, the child node is correctly list in the parent's children list.
    Also ensures that the child's parent attribute is set to the correct parent node.
    """
    parent = Node(PKS_product = Chem.MolFromSmiles("C"), # PKS product here is methane as example only
                  PKS_design = None,
                  depth = 0) # assume parent is root node so depth = 0

    child = Node(PKS_product = Chem.MolFromSmiles("CC"), # add another carbon as example only
                 PKS_design = None,
                 depth = 1) # assume child is one node below so depth > 0

    parent.add_child(child)

    assert child in parent.children
    assert child.parent is parent

def test_add_child_idempotent():
    """
    Ensures the idempotence of the add_child method  of a Node instance.
    Ensures that attempting to add the exact same child node more than once doesn't result in duplicates.
    """
    parent = Node(PKS_product = Chem.MolFromSmiles("C"),
                  PKS_design = None,
                  depth = 0)

    child = Node(PKS_product = Chem.MolFromSmiles("CC"),
                 PKS_design = None,
                 depth = 1)

    parent.add_child(child)
    parent.add_child(child) # let's repeat this deliberately

    assert len(parent.children) == 1

def test_update_node():
    """
    Test the update method of a Node instance.
    Verifies that updating a node with a reward correctly updates its value and visits count.
    """
    node = Node(PKS_product = Chem.MolFromSmiles("C"),
                PKS_design = None,
                depth = 0)

    node.update(reward = 3) # update with an initial reward

    assert node.value == 3
    assert node.visits == 1

    node.update(reward = 2) # update with another reward

    assert node.value == 5 # the cumulative value of the node should be updated
    assert node.visits == 2 # the total visit count of the node should be updated