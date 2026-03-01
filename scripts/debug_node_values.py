"""Debug script to check node values after MCTS run."""
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from DORAnet_agent.mcts import DORAnetMCTS
from DORAnet_agent.node import Node
from DORAnet_agent.policies.reward import PKSSimilarityRewardPolicy
from DORAnet_agent.policies.terminal_detection import SimilarityGuidedRetroTideDetector

def main():
    # Create target molecule and root node
    target = Chem.MolFromSmiles('CCCCC(=O)O')  # pentanoic acid
    root = Node(fragment=target)

    # Test 1: Default SparseTerminalRewardPolicy (sparse rewards)
    print('\n' + '='*60)
    print('TEST 1: Default SparseTerminalRewardPolicy')
    print('='*60)
    
    agent1 = DORAnetMCTS(
        root=root,
        target_molecule=target,
        max_depth=3,
        total_iterations=5,
    )
    agent1.run()

    print('\nNODE VALUES:')
    for node in agent1.nodes[:10]:
        avg = node.value / node.visits if node.visits > 0 else 0
        print(f'  Node {node.node_id:3d}: visits={node.visits:3d}, value={node.value:.4f}, avg={avg:.4f}')

    nodes_with_value = sum(1 for n in agent1.nodes if n.value > 0)
    print(f'\nTotal nodes: {len(agent1.nodes)}, Nodes with value > 0: {nodes_with_value}')
    print('Note: Sparse rewards only give value when sink/PKS terminals found')
    
    # Test 2: SimilarityGuidedRetroTideDetector + PKSSimilarityRewardPolicy (dense rewards)
    print('\n' + '='*60)
    print('TEST 2: SimilarityGuidedRetroTideDetector + PKSSimilarityRewardPolicy (dense rewards)')
    print('='*60)
    
    root2 = Node(fragment=target)
    root2.__class__.node_counter = 0  # Reset counter
    
    pks_detector = SimilarityGuidedRetroTideDetector()
    
    agent2 = DORAnetMCTS(
        root=root2,
        target_molecule=target,
        max_depth=3,
        total_iterations=5,
        terminal_detector=pks_detector,
    )
    agent2.run()

    print('\nNODE VALUES:')
    for node in agent2.nodes[:10]:
        avg = node.value / node.visits if node.visits > 0 else 0
        print(f'  Node {node.node_id:3d}: visits={node.visits:3d}, value={node.value:.4f}, avg={avg:.4f}')

    nodes_with_value2 = sum(1 for n in agent2.nodes if n.value > 0)
    print(f'\nTotal nodes: {len(agent2.nodes)}, Nodes with value > 0: {nodes_with_value2}')
    print('Note: Dense rewards use PKS similarity and SA score to compute rewards')

if __name__ == '__main__':
    main()
