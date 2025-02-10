import math
import random
from typing import Tuple
from rdkit import Chem
from RL_agents_for_retrosynthesis.PKS_agent.node import MCTSNode, PKSState
from RL_agents_for_retrosynthesis.PKS_agent.rewards import subgraph_mcs_reward
from retrotide import retrotide, structureDB

def ucb_score(parent: MCTSNode,
              child: Tuple[MCTSNode, any],
              exploration_const: float = 1.44) -> float:

    """
    Standard UCB1 formula:
        Q(child) + exploration_const * sqrt( ln(N(parent)) / N(child) )

    Here, child is actually a tuple of the form (child_node, action).
    """

    if child[0].visits == 0:
        return float('inf')  # encourages exploring unvisited children
    return (child[0].q_value
            + exploration_const * math.sqrt(math.log(parent.visits) / child[0].visits))

class MCTS:
    def __init__(self,
                 target_mol: Chem.rdchem.Mol,
                 exploration_const: float = 1.44,
                 max_depth: int = 10,
                 rollout_depth: int = 3,
                 debug: bool = False):

        self.root = None
        self.target_mol = target_mol
        self.C = exploration_const
        self.max_depth = max_depth
        self.rollout_depth = rollout_depth
        self.debug = debug

    def search(self,
               root_state,
               n_iterations = 1000):
        """
        Run MCTS starting from the given root_state (a PKSState type object).
        """
        self.root = MCTSNode(root_state)

        for i in range(n_iterations):

            # 1. Selection: pick a leaf node
            leaf = self._select(self.root)
            if self.debug:
                print(f"[search - iteration {i + 1}] Selected leaf node: {leaf}\n")

            # 2. Expansion: expand that leaf node by adding children if possible
            if not leaf.is_fully_expanded():
                leaf = self._expand(leaf)
                if self.debug:
                    print(f"[search - iteration {i + 1}] Expanded leaf node -> {leaf}")

            # 3. Rollout: simulate from that leaf node to get a reward
            reward = self._rollout(leaf)
            if self.debug:
                print(f"[search - iteration {i + 1}] Rollout reward = {reward:.3f}")

            # 4. Backpropagation: propagate the earned reward back up
            self._backpropagate(leaf, reward)

        # after MCTS finishes, you can pick the best child node of the root note
        # e.g., by the highest Q-value or the most visits
        best_child, _ = max(self.root.children, key = lambda c: c[0].q_value)
        if self.debug:
            print(f"[search] Best child of root has Q-value = {best_child.q_value:.3f}")
        return best_child.state

    def _select(self,
                node: MCTSNode):
        """
        Follow UCB from the root down to a leaf node.
        """

        # keep going down the tree while the current node is fully expanded
        # AND has children (i.e., not a terminal).
        while node.is_fully_expanded() and node.children:

            # if the max depth has been hit, break early
            if len(node.state.cluster.modules) >= self.max_depth:
                break

            # Among the children, pick the one with the best UCB score
            node, action = max(node.children, key=lambda c: ucb_score(node, c, self.C))

            if self.debug:
                print(f"[select] Moved down to child {node}, via action={action}")

        return node

    def _expand(self,
                node: MCTSNode):
        """
        Expand the node by taking one untried action to create a new child node.
        """
        if node.untried_actions is None:
            node.untried_actions = self._get_untried_actions(node.state)
            if self.debug:
                print(f"[expand] Node {node} untried actions: {node.untried_actions}")

        if not node.untried_actions:
            # if there are no untried actions then we cannot expand
            return node

        # pick just one action to apply
        action = node.untried_actions.pop()
        if self.debug:
            print(f"[expand] Expanding node {node}, applying action: {action}")

        # apply the action to get a child state
        child_state = self._apply_action(node.state, action)

        # create a new child node
        child_node = node.add_child(child_state = child_state,
                                    action = action)

        return child_node

    def _rollout(self,
                 node: MCTSNode) -> float:
        """
        Simulate (rollout) from this node.
        For example, we can use a short multistep greedy approach
        """
        depth = 0
        current_state = node.state

        if self.debug:
            print(f"[rollout] Starting rollout from node {node}")

        # while we haven't reached rollout_depth,
        # or some stopping condition, we keep going:
        while depth < self.rollout_depth:
            best_extension = self._pick_greedy_action(current_state)
            if best_extension is None:
                if self.debug:
                    print(f"[rollout] No further expansions possible at depth {depth}.")
                break

            if self.debug:
                print(f"[rollout] Depth {depth}, best extension = {best_extension}")

            current_state = self._apply_action(current_state, best_extension)
            depth += 1

        # evaluate final product with subgraph-based or partial MCS-based reward
        reward = subgraph_mcs_reward(current_state.PKS_product, self.target_mol)
        if self.debug:
            print(f"[rollout] Final reward after rollout = {reward:.3f}")
        return reward

    def _backpropagate(self, node, reward):
        """
        Propagate the reward up to the root.
        """
        current = node
        while current is not None:
            current.backprop_update(reward)
            current = current.parent

    def _get_untried_actions(self,
                             state: PKSState):
        """
        Returns all possible one-step expansions from state
        """
        # if there are no modules in state.cluster, we are at the root node
        if len(state.cluster.modules) == 0:
            previousDesigns = None
        else:
            previousDesigns = [(state.cluster, 0.0, state.PKS_product)]

        expansions = retrotide.designPKS_onestep(targetMol = self.target_mol,
                                                 previousDesigns = previousDesigns,
                                                 maxDesignsPerRound = 25,
                                                 similarity = 'mcs_without_stereo')

        actions = []
        for (cluster, score, product) in expansions:
            new_module = cluster.modules #[-1] # grab only the last module since it was newly added
            actions.append(new_module)

        return actions

    def _apply_action(self,
                      state: PKSState,
                      action) -> PKSState:
        """
        Given the current state and a new module 'action',
        create the next state, a new PKS product
        """
        new_cluster = type(state.cluster)(modules=state.cluster.modules + [action])
        new_product = new_cluster.computeProduct(structureDB, chain=state.PKS_product)
        return PKSState(new_cluster, new_product)

    def _pick_greedy_action(self,
                            state: PKSState):
        """
        For rollout, pick the single best next module by some similarity metric.
        Then, we return that module, or None if no expansions are possible.
        """
        if len(state.cluster.modules) == 0:
            previousDesigns = None
        else:
            previousDesigns = [(state.cluster, 0.0, state.PKS_product)]

        expansions = retrotide.designPKS_onestep(targetMol = self.target_mol,
                                                 previousDesigns = previousDesigns,
                                                 maxDesignsPerRound = 25,
                                                 similarity = 'mcs_without_stereo')

        if not expansions:
            return None

        # expansions are scored by descending similarity scores so expansions[0] are best
        (best_cluster, best_score, best_product) = expansions[0]
        return best_cluster.modules #[-1]