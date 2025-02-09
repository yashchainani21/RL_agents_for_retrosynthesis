import bcs
from rdkit import Chem
import uuid

class MCTSNode:
    """
    A singe node in the MCTS search tree.
    """
    def __init__(self, state, parent = None):
        self.id = uuid.uuid4() # give this node unique ID for easier debugging/ visualization
        self.state = state # PKSState(cluster, mol) representing current PKS product and its associated design
        self.parent = parent

        # store the child nodes as a list of (childNode, action)
        self.children = []

        # track properties for MCTS
        self.visits = 0
        self.total_value = 0.0 # sum of rewards from all rollouts

        # populated when this node is expanded for the first time
        # typically, a set/ list of possible "next actions" (modules with acyl-CoA units) that haven't been tried
        self.untried_actions = None

    def is_fully_expanded(self):
        """
        Returns True if all possible actions have already been applied to expand this node.
        False if there are any untried actions remaining at all.
        """
        if self.untried_actions is not None and len(self.untried_actions) == 0:
            return True
        else:
            return False

    def add_child(self, child_state, action):
        """
        Create a new MCTSNode for 'child_state', which results from taking an 'action'.
        Attach this child node to this node and return the newly created child node.
        """
        child_node = MCTSNode(state = child_state, parent = self)
        self.children.append((child_node, action))
        return child_node

    def update_child(self, reward):
        """
        Update this node's statistics after a rollout/ simulation step:
            - visits += 1
            - total_value += reward
        """
        self.visits += 1
        self.total_value += reward

    @property
    def q_value(self):
        """
        Returns the average value (Q) of this node.
        """
        if self.visits == 0:
            return 0.0
        return self.total_value/ self.visits

    def __repr__(self):
        return f"MCTSNode(id={self.id}, visits={self.visits}, value={self.q_value:.3f})"