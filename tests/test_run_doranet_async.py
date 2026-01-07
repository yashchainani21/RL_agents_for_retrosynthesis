"""
Integration-style tests for the async runner script.
"""

from pathlib import Path
import sys

import pytest
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_async_runner_invokes_agent(monkeypatch, tmp_path):
    import scripts.run_DORAnet_Async as runner

    calls = {"run": 0, "save_results": 0, "save_final": 0, "save_success": 0}

    class DummyAgent:
        def __init__(self, *args, **kwargs):
            self.nodes = []
            self._summary = "summary"

        def run(self):
            calls["run"] += 1

        def get_tree_summary(self):
            return self._summary

        def save_results(self, path):
            calls["save_results"] += 1
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("results")

        def save_finalized_pathways(self, path, total_runtime_seconds=None):
            calls["save_final"] += 1
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("finalized")

        def save_successful_pathways(self, path):
            calls["save_success"] += 1
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("successful")

    def dummy_agent_factory(*args, **kwargs):
        return DummyAgent()

    monkeypatch.setattr(runner, "AsyncExpansionDORAnetMCTS", dummy_agent_factory)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "create_enhanced_interactive_html", lambda **kwargs: None)
    monkeypatch.setattr(runner, "create_pathways_interactive_html", lambda **kwargs: None)
    _orig_mol_from_smiles = Chem.MolFromSmiles
    monkeypatch.setattr(runner.Chem, "MolFromSmiles", lambda s: _orig_mol_from_smiles("CCO"))

    runner.main()

    assert calls["run"] == 1
    assert calls["save_results"] == 1
    assert calls["save_final"] == 1
    assert calls["save_success"] == 1


def test_async_runner_respects_invalid_smiles(monkeypatch, tmp_path):
    import scripts.run_DORAnet_Async as runner

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner.Chem, "MolFromSmiles", lambda s: None)

    with pytest.raises(ValueError):
        runner.main()
