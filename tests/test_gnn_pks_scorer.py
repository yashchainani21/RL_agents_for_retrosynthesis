"""Tests for GNN Polyketide Scorer."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

# Featurization functions are always available (only need rdkit)
from DORAnet_agent.policies.gnn_pks_scorer import (
    ATOM_MAP,
    DEGREE_MAP,
    CHARGE_MAP,
    NUM_H_MAP,
    HYB_MAP,
    EDGE_FEAT_DIM,
    atom_to_feature,
    bond_to_feature,
    smiles_to_graph,
)

torch = pytest.importorskip("torch")

from DORAnet_agent.policies.gnn_pks_scorer import (
    GNNPolyketideScorer,
    SupervisedGNNClassifier,
    TORCH_AVAILABLE,
)


# Path to the trained checkpoint
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "gnn_pks_classifier", "best_model.pt",
)
CHECKPOINT_EXISTS = os.path.exists(CHECKPOINT_PATH)


# =============================================================================
# Featurization tests (no checkpoint needed)
# =============================================================================


class TestFeaturization:
    """Tests for graph featurization functions."""

    def test_atom_feature_dimensions(self):
        """Atom features should be 40-dimensional."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        feat = atom_to_feature(mol.GetAtomWithIdx(0))
        # 13 (atom) + 7 (degree) + 6 (charge) + 6 (numH) + 6 (hyb) + 1 (aromatic) + 1 (ring)
        expected_dim = (
            len(ATOM_MAP) + 1
            + len(DEGREE_MAP) + 1
            + len(CHARGE_MAP) + 1
            + len(NUM_H_MAP) + 1
            + len(HYB_MAP) + 1
            + 1  # aromatic
            + 1  # ring
        )
        assert feat.shape == (expected_dim,)
        assert feat.dtype == np.float32

    def test_bond_feature_dimensions(self):
        """Edge features should be 5-dimensional (4 bond types + 1 self-loop)."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        bond = mol.GetBondWithIdx(0)
        feat = bond_to_feature(bond)
        assert feat.shape == (EDGE_FEAT_DIM,)
        assert feat.dtype == np.float32
        assert feat.sum() == 1.0  # one-hot

    def test_self_loop_feature(self):
        """Self-loop bond feature should have last element set."""
        feat = bond_to_feature(None)
        assert feat[-1] == 1.0
        assert feat[:-1].sum() == 0.0

    def test_smiles_to_graph_shapes(self):
        """Graph tensors should have correct shapes for ethanol (3 atoms, 2 bonds)."""
        node_feat, edge_index, edge_attr = smiles_to_graph("CCO")
        n_atoms = 3
        n_bonds = 2
        n_edges = 2 * n_bonds + n_atoms  # directed edges + self-loops

        assert node_feat.shape[0] == n_atoms
        assert edge_index.shape == (2, n_edges)
        assert edge_attr.shape == (n_edges, EDGE_FEAT_DIM)

    def test_smiles_to_graph_invalid(self):
        """Invalid SMILES should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_graph("not_a_smiles")

    def test_smiles_to_graph_benzene(self):
        """Benzene should have aromatic features set."""
        node_feat, edge_index, edge_attr = smiles_to_graph("c1ccccc1")
        assert node_feat.shape[0] == 6  # 6 carbons


# =============================================================================
# Model architecture tests (torch required, no checkpoint)
# =============================================================================


class TestModelArchitecture:
    """Tests for the GNN model (no checkpoint needed)."""

    def test_model_forward_pass(self):
        """Model should produce logit and embedding outputs."""
        model = SupervisedGNNClassifier(
            node_dim=40, edge_dim=5, hidden_dim=32, heads=2, num_layers=2
        )
        model.eval()

        node_feat, edge_index, edge_attr = smiles_to_graph("CCO")
        node_feat_t = torch.from_numpy(node_feat)
        edge_index_t = torch.from_numpy(edge_index)
        edge_attr_t = torch.from_numpy(edge_attr)
        batch_t = torch.zeros(node_feat.shape[0], dtype=torch.long)

        with torch.no_grad():
            logit, embed = model(node_feat_t, edge_index_t, batch_t, edge_attr_t)

        assert logit.shape == (1, 1)
        assert embed.shape == (1, 32)

    def test_model_sigmoid_output_range(self):
        """Sigmoid of model logit should be in [0, 1]."""
        model = SupervisedGNNClassifier(
            node_dim=40, edge_dim=5, hidden_dim=16, heads=2, num_layers=1
        )
        model.eval()

        for smi in ["CCO", "c1ccccc1", "CC(=O)O", "CC(O)CC(=O)O"]:
            node_feat, edge_index, edge_attr = smiles_to_graph(smi)
            with torch.no_grad():
                logit, _ = model(
                    torch.from_numpy(node_feat),
                    torch.from_numpy(edge_index),
                    torch.zeros(node_feat.shape[0], dtype=torch.long),
                    torch.from_numpy(edge_attr),
                )
            prob = torch.sigmoid(logit).item()
            assert 0.0 <= prob <= 1.0


# =============================================================================
# Scorer wrapper tests (checkpoint required)
# =============================================================================


@pytest.mark.skipif(not CHECKPOINT_EXISTS, reason="GNN checkpoint not found")
class TestGNNPolyketideScorer:
    """Tests for the GNNPolyketideScorer wrapper (requires trained checkpoint)."""

    @pytest.fixture(scope="class")
    def scorer(self):
        """Create a scorer instance (shared across tests in this class)."""
        return GNNPolyketideScorer(
            checkpoint_path=CHECKPOINT_PATH,
            device="cpu",
            cache_size=100,
            fallback_score=0.0,
        )

    def test_import_available(self):
        """GNNPolyketideScorer should be importable when torch is present."""
        assert TORCH_AVAILABLE
        from DORAnet_agent.policies import GNNPolyketideScorer as Imported
        assert Imported is not None

    def test_scorer_output_range(self, scorer):
        """Predictions should be floats in [0, 1]."""
        for smi in ["CCO", "c1ccccc1", "CC(=O)O", "CC(O)CC(=O)CC(O)CC(=O)O"]:
            node = SimpleNamespace(smiles=smi)
            score = scorer(node)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {smi}"

    def test_scorer_caching(self, scorer):
        """Repeated calls with the same SMILES should use the cache."""
        smi = "CCCCCC"
        node = SimpleNamespace(smiles=smi)
        score1 = scorer(node)
        # Verify it's in the cache
        assert smi in scorer._cache
        score2 = scorer(node)
        assert score1 == score2

    def test_scorer_invalid_smiles(self, scorer):
        """Invalid SMILES should return fallback_score."""
        node = SimpleNamespace(smiles="not_a_smiles_string")
        score = scorer(node)
        assert score == scorer._fallback_score

    def test_scorer_none_smiles_node(self, scorer):
        """Node with no smiles attribute should return fallback_score."""
        node = SimpleNamespace()  # no smiles attr
        score = scorer(node)
        assert score == scorer._fallback_score

        node2 = SimpleNamespace(smiles=None)
        score2 = scorer(node2)
        assert score2 == scorer._fallback_score

    def test_scorer_name(self, scorer):
        """Scorer name should be GNN_PKS."""
        assert scorer.name == "GNN_PKS"

    def test_integration_with_reward_policy(self, scorer):
        """Scorer should work as non_terminal_scorer in SAScore_and_TerminalRewardPolicy."""
        from DORAnet_agent.policies import SAScore_and_TerminalRewardPolicy

        policy = SAScore_and_TerminalRewardPolicy(
            sink_terminal_reward=1.0,
            pks_terminal_reward=1.0,
            non_terminal_scorer=scorer,
        )

        # Check name includes GNN_PKS
        assert "GNN_PKS" in policy.name

        # Non-terminal node should get a GNN score
        node = SimpleNamespace(
            smiles="CCO",
            is_sink_compound=False,
            is_pks_terminal=False,
            fragment=None,
        )
        reward = policy.calculate_reward(node, context={})
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
