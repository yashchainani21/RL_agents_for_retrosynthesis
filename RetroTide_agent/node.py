from typing import List, Optional, Tuple
import bcs
from rdkit import Chem
from rdkit import RDLogger

# disable RDKit warnings
logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)

class Node:
    """
    A node in the Monte Carlo Tree Search for chemical synthesis
    """

    node_counter = 0 # unique node ID tracker
    def __init__(self,
                 PKS_product: Optional[Chem.rdchem.Mol] = None,
                 PKS_design: Optional[Tuple[bcs.Cluster, float, Chem.rdchem.Mol]] = None,
                 parent: Optional["Node"] = None, # quotes here help prevent circular import issues
                 depth: int = 0):

        self.PKS_product: Chem.Mol = PKS_product
        self.PKS_design: Optional[Tuple[bcs.Cluster, float, Chem.rdchem.Mol]] = PKS_design
        self.parent: Optional["Node"] = parent
        self.children: List["Node"] = []
        self.depth: int = depth
        self.visits: int = 0 # track the number of visits to this node
        self.value: float = 0.0 # track the cumulative value of this node
        self.selection_score: Optional[float] = None
        self.expand: bool = False

        # assign unique ID
        self.node_id = Node.node_counter
        self.parent_id = parent.node_id if parent else None
        Node.node_counter += 1

    def add_child(self, child: "Node") -> None:
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

    def __repr__(self) -> str:
        """
        Provides a string representation of this node for debugging and logging purposes.
        """
        design_info = self.PKS_design[0].modules if self.PKS_design else "No design"
        product_info = Chem.MolToSmiles(self.PKS_product) if self.PKS_product else "None"

        return f"Node ID: {self.node_id}, Depth: {self.depth}, PKS Design: {design_info}, PKS Product: {product_info}"