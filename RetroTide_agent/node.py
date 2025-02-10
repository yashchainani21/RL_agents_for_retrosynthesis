from typing import List, Dict, Optional, Tuple
import bcs
from rdkit import Chem
from rdkit import RDLogger
import uuid

# disable RDKit warnings
logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)

class Node:
    """
    A node in the Monte Carlo Tree Search for chemical synthesis
    """
    def __init__(self,
                 PKS_product: Optional[Chem.rdchem.Mol] = None,
                 PKS_design: Optional[bcs.Cluster] = None,
                 parent: any = None,
                 depth: int = 0):

        self.PKS_product = PKS_product
        self.PKS_design = PKS_design
        self.parent = parent
        self.children = []
        self.depth = depth
        self.uuid = str(uuid.uuid4()) # give this node a unique identifier for debugging purposes

        # track the number of visits to and the cumulative value of this node
        self.visits = 0
        self.value = 0

    def add_child(self, child: any) -> None:
        """
        Adds a child to this node.
        Ensures that the same child node instance is not added more than once.
        Also sets the child's parent attribute to this node if not already set.
        """
        if child not in self.children:
            if child.parent is None:
                child.parent = self
        self.children.append(child)
        child.depth = self.depth + 1

    def update(self, reward: float) -> None:
        """
        Updates this node's value and visit count based on a new reward.
        """
        self.value += reward
        self.visits += 1

    def __repr__(self):
        """
        Provides a string representation of this node for debugging and logging purposes.
        """
        return f"Node {self.uuid} with PKS design {self.PKS_design} and PKS product {Chem.MolToSmiles(self.PKS_product)}"