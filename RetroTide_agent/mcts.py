import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from typing import Optional, Tuple, List
from retrotide import retrotide, structureDB
from RetroTide_agent.node import Node

class MCTS:
    def __init__(self,
                 root: Node,
                 target_molecule: Chem.Mol,
                 max_depth: int = 10,
                 maxPKSDesignsRetroTide: int = 25,
                 selection_policy: Optional[str] = 'UCB1'):

        self.root = root
        self.target_molecule = target_molecule
        self.max_depth = max_depth
        self.maxPKSDesigns = maxPKSDesignsRetroTide
        self.selection_policy = selection_policy

        bag_of_graphs = self.create_bag_of_graphs_from_target()
        self.bag_of_graphs = bag_of_graphs

    @staticmethod
    def run_pks_release_reaction(pks_release_mechanism: str,
                                 bound_product_mol: Chem.Mol) -> Chem.Mol:
        """
        Run a PKS offloading reaction to release a PKS product bound to its synthase via either a thiolysis or cyclization reaction
        """

        if pks_release_mechanism == 'thiolysis':
            Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to produce terminal acid group
            rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
            unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
            Chem.SanitizeMol(unbound_product_mol)
            return unbound_product_mol

        if pks_release_mechanism == 'cyclization':
            Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to cyclize bound substrate
            rxn = AllChem.ReactionFromSmarts('([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])>>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]')
            try:
                unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
                Chem.SanitizeMol(unbound_product_mol)
                return unbound_product_mol

            # if the bound substrate cannot be cyclized, then return None
            except:
                raise ValueError("\nUnable to perform cyclization reaction")

        if pks_release_mechanism == 'reduction':
            Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to cyclize bound substrate
            rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1]')
            unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
            Chem.SanitizeMol(unbound_product_mol)
            return unbound_product_mol

    @staticmethod
    def getSubmolRadN(mol: Chem.Mol,
                      radius: int):

        atoms = mol.GetAtoms()
        submols = []
        for atom in atoms:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx())
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            subsmi = Chem.MolToSmiles(submol, rootedAtAtom=amap[atom.GetIdx()], canonical=False)
            submols.append(Chem.MolFromSmiles(subsmi, sanitize=False))
        return submols

    @staticmethod
    def are_isomorphic(mol1: Chem.Mol,
                       mol2: Chem.Mol,
                       consider_stereo: bool = False) -> bool:

        if consider_stereo:
            is_isomorphic = mol1.HasSubstructMatch(mol2, useChirality=True) and mol2.HasSubstructMatch(mol1,
                                                                                                       useChirality=True)
        else:
            is_isomorphic = mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)

        return is_isomorphic

    def create_bag_of_graphs_from_target(self) -> List[Chem.Mol]:

        # first, we get the longest distance between any two atoms within the target molecule
        dist_matrix = rdmolops.GetDistanceMatrix(self.target_molecule)
        dist_array = np.array(dist_matrix)
        longest_distance = dist_array.max()

        # using this longest distance, create a bag of graphs by decomposing the target molecule across various lengths
        all_submols = []
        for i in range(1, int(longest_distance + 1)):
            try:
                submols = self.getSubmolRadN(mol = self.target_molecule,
                                             radius = i)
                all_submols.extend(submols)
            except:
                pass

        return all_submols

    def is_PKS_product_in_bag_of_graphs(self,
                                        PKS_product: Chem.Mol,
                                        consider_stereo: bool) -> bool:

        for submol in self.bag_of_graphs:
            if self.are_isomorphic(mol1 = submol,
                                mol2 = PKS_product,
                                consider_stereo = consider_stereo):

                return True # returns True only if a match is found

        return False # returns False is no match

    def calculate_reward(self,
                         node: Node) -> int:

        PKS_product = node.PKS_product

        fully_reduced_reward = 0
        fully_carboxylated_reward = 0
        fully_cyclized_reward = 0

        try:
            # first, try fully reducing the PKS product at this node
            fully_reduced_product = self.run_pks_release_reaction(pks_release_mechanism="reduction",
                                                                  bound_product_mol = PKS_product)

            # then, check to see if this fully reduced product is in our bag of graphs
            if self.is_PKS_product_in_bag_of_graphs(fully_reduced_product,
                                                    consider_stereo=False):
                fully_reduced_reward += 1

        except Exception as e:
            print(e)
            pass

        try:
            # now, try releasing the PKS product at this node via a condensation reaction
            fully_carboxylated_product = self.run_pks_release_reaction(pks_release_mechanism="thiolysis",
                                                                       bound_product_mol = PKS_product)

            # then, check to see if this fully carboxylated product is in our bag of graphs
            if self.is_PKS_product_in_bag_of_graphs(fully_carboxylated_product,
                                                    consider_stereo=False):
                fully_carboxylated_reward += 1

        except:
            pass

        try:
            # finally, try releasing the PKS product at this node via a cycliation reaction
            fully_cyclized_product = self.run_pks_release_reaction(pks_release_mechanism="cyclization",
                                                                   bound_product_mol = PKS_product)

            # then, check to see if this fully cyclized product is in our bag of graphs
            if self.is_PKS_product_in_bag_of_graphs(fully_cyclized_product,
                                                    consider_stereo=False):
                fully_cyclized_reward += 1

        except:
            pass

        # lastly, if any of these rewards is non-zero, add a 1 to the final reward
        if fully_carboxylated_reward > 0 or fully_cyclized_reward > 0 or fully_reduced_reward > 0:
            final_reward = 1

        else:
            final_reward = 0

        return final_reward

    def select(self,
               node: Node) -> Node:
        """
        Selection step starts with the root node & the synthesis tree is then traversed until a leaf node is reached.
        This leaf node would have untried actions and therefore, would not have been expanded upon.
        This traversal can be done by following a path determined by the Upper Confidence Bound (1) applied to trees.
        Or by taking a path guided by a custom selection policy that further modifies the basic UCB1 policy.
        """
        while node.children: # continue until a leaf node is reached

            if self.selection_policy == 'UCB1':
                log_parent_visits = math.log(max(node.visits, 1))  # Avoid log(0) by ensuring a minimum of 1 visit
                node = max(node.children,
                           key=lambda child: self.calculate_reward(child) if child.visits == 0 else
                           (child.value / child.visits + math.sqrt(2 * log_parent_visits / child.visits)))

            if self.selection_policy == 'explore_least_visited':
                # prioritize nodes with the fewest visits
                node = min(node.children, key=lambda x: (x.visits, -x.value))

        return node

    def expand(self,
               node: Node) -> None:

        # for the expansion step, run RetroTide for just one step/ module
        # note that if there are no previous designs, this function generates loading + first extension module
        if node.PKS_design is None:
            new_designs = retrotide.designPKS_onestep(targetMol = self.target_molecule,
                                                      previousDesigns = None,
                                                      maxDesignsPerRound = self.maxPKSDesigns,
                                                      similarity = 'mcs_without_stereo')
        else:
            new_designs = retrotide.designPKS_onestep(targetMol = self.target_molecule,
                                                      previousDesigns = [node.PKS_design], # pass this in as list for RetroTide
                                                      maxDesignsPerRound=self.maxPKSDesigns,
                                                      similarity = 'mcs_without_stereo')

        # create a new child node from each PKS design
        for design in new_designs:

            new_node = Node(PKS_product = design[-1],
                            PKS_design = design,
                            parent = node, # set the node currently being expanded as the parent
                            depth = node.depth + 1) # all new child nodes are one level deeper

            # formally add this new node as a child to the current node
            node.add_child(new_node)

    def simulate_and_get_reward(self,
                                node: Node) -> int:

        # calculate reward based on bag of graphs analysis, thereby skipping expensive simulations
         reward = self.calculate_reward(node)

         return reward

    def backpropagate(self,
                      node: Node,
                      reward: int) -> None:
        # Propagate the simulation result (reward) back up the tree.
        while node is not None:
            node.visits += 1  # increment visit count
            node.value += reward  # update value with reward from simulation
            node = node.parent  # move to the parent node

    def run(self):
        """
        Executes the MCTS algorithm for the given number of iterations.

        Stops when:
        - The tree reaches `max_depth`
        - The number of iterations is exhausted
        """

        for i in range(self.max_depth):

            # Step 1: Selection - Start at root and follow UCB until a leaf is found
            leaf = self.select(self.root)
            print(f"[Iteration {i + 1}] Selected leaf node at depth {leaf.depth}")

            # Check stopping condition
            if leaf.depth >= self.max_depth:
                print("[Stopping] Maximum tree depth reached.")
                break  # Do NOT return; just stop further iterations

            # Step 2: Expansion - Expand only if unexpanded
            if not leaf.children:  # Expand only if this node has not been expanded before
                self.expand(leaf)
                print(f"[Iteration {i + 1}] Expanded leaf node: {len(leaf.children)} new children")

            # Step 3: Simulation - Evaluate leaf node and get reward
            reward = self.simulate_and_get_reward(leaf)
            print(f"[Iteration {i + 1}] Simulation reward = {reward}")

            # Step 4: Backpropagation - Propagate reward up the tree
            self.backpropagate(node=leaf, reward=reward)
            print(f"[Iteration {i + 1}] Backpropagation complete.")

        print("[MCTS Completed] All iterations exhausted.")

