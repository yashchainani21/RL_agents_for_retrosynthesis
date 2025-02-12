from rdkit import Chem

# Import your MCTS logic and data structures
from RL_agents_for_retrosynthesis.PKS_agent.mcts import MCTS
from RL_agents_for_retrosynthesis.PKS_agent.node import PKSState
import bcs
def main():
    # 1. Define or load your target molecule
    #    For example, let's say we have a small target SMILES
    target_smiles = "CCCCCC"  # ethanol just as a trivial example
    target_mol = Chem.MolFromSmiles(target_smiles)
    Chem.SanitizeMol(target_mol)

    # 2. Create an *empty* PKS cluster as the root state
    #    i.e., no modules yet
    empty_cluster = bcs.Cluster(modules=[])
    empty_mol = None  # No polyketide product yet

    # Alternatively, if you want to start from a minimal cluster/product,
    # you could do that here.

    # 3. Build a PKSState for the root
    #    If empty_mol is None, you might want to treat that carefully.
    #    But let's just pass None for now (some code checks for that).
    root_state = PKSState(cluster = empty_cluster, PKS_product = empty_mol)

    # 4. Instantiate MCTS with desired parameters
    mcts = MCTS(
        target_mol = target_mol,
        exploration_const = 1.44,
        max_depth = 10,  # limit to how many modules we want
        rollout_depth = 3,  # number of steps in the rollout
        debug = True)  # print debug info to console

    # 5. Run the MCTS search
    final_state = mcts.search(root_state, n_iterations=50)

    # 6. Print the result
    print("===== MCTS FINISHED =====")
    print("Final PKS design has modules:")
    for i, module in enumerate(final_state.cluster.modules):
        print(f"  Module {i + 1} -> {module}")

    if final_state.PKS_product:
        print("Final PKS product SMILES:", Chem.MolToSmiles(final_state.PKS_product))
    else:
        print("No final PKS product (None)")

    print("=========================")


if __name__ == "__main__":
    main()
