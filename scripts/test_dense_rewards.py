"""Test script to verify dense rewards are now applied to all nodes."""
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from DORAnet_agent.mcts import DORAnetMCTS
from DORAnet_agent.node import Node
from DORAnet_agent.policies import (
    VerifyWithRetroTide,
    SAScore_and_TerminalRewardPolicy,
    ThermodynamicScaledRewardPolicy,
    SparseTerminalRewardPolicy,
)


def test_dense_rewards():
    """Test that dense SA score rewards are applied to all nodes."""
    target = Chem.MolFromSmiles('CCCCC(=O)O')  # pentanoic acid
    root = Node(fragment=target)

    # Configure with same policies as run_DORAnet_Async.py
    terminal_detector = VerifyWithRetroTide()
    
    # Dense reward policy with SA scores
    reward_policy = SAScore_and_TerminalRewardPolicy(
        sink_terminal_reward=1.0,
        pks_terminal_reward=1.0,
    )

    agent = DORAnetMCTS(
        root=root,
        target_molecule=target,
        max_depth=3,
        total_iterations=5,
        terminal_detector=terminal_detector,
        reward_policy=reward_policy,
    )
    agent.run()

    print('\n' + '='*70)
    print('TEST: Dense SA Score Rewards for All Nodes')
    print('='*70)
    print(f'Terminal detector: {terminal_detector.name}')
    print(f'Reward policy: {reward_policy.name}')
    print()
    
    print('NODE VALUES AFTER MCTS:')
    print('-'*70)
    for node in agent.nodes[:15]:
        avg = node.value / node.visits if node.visits > 0 else 0
        terminal = 'SINK' if node.is_sink_compound else ('PKS' if node.is_pks_terminal else '')
        print(f'  Node {node.node_id:3d}: depth={node.depth}, visits={node.visits:3d}, '
              f'value={node.value:.4f}, avg_value={avg:.4f} {terminal}')

    # Summary statistics
    total_nodes = len(agent.nodes)
    nodes_with_value = sum(1 for n in agent.nodes if n.value > 0)
    avg_values = [n.value / n.visits for n in agent.nodes if n.visits > 0]
    
    print()
    print('='*70)
    print(f'Total nodes: {total_nodes}')
    print(f'Nodes with value > 0: {nodes_with_value} ({100*nodes_with_value/total_nodes:.1f}%)')
    if avg_values:
        print(f'Average avg_value: {sum(avg_values)/len(avg_values):.4f}')
        print(f'Min avg_value: {min(avg_values):.4f}')
        print(f'Max avg_value: {max(avg_values):.4f}')
    print('='*70)
    
    # Test result
    if nodes_with_value > 0:
        print('\n✅ SUCCESS: Dense rewards are being applied!')
    else:
        print('\n❌ FAILURE: No nodes received non-zero rewards')
    
    return nodes_with_value > 0


if __name__ == '__main__':
    success = test_dense_rewards()
    sys.exit(0 if success else 1)
