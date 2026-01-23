# RL Agents for Retrosynthesis

A hierarchical multi-agent reinforcement learning system for retrosynthesis planning that combines **three complementary synthesis modalities**:

1. **Synthetic Organic Chemistry**: Complex carbon-carbon bond formations (Suzuki, Negishi couplings, etc.)
2. **Monofunctional Enzymes**: Regioselective and stereoselective substrate modifications
3. **Polyketide Synthases (PKS)**: Programmatic C-C bond formation from acyl-CoA building blocks

This multi-modal approach accesses a wider chemical space than any single modality alone.

## Scientific Background

The system uses Monte Carlo Tree Search (MCTS) to explore retrosynthetic pathways:

- **DORAnet Agent**: Performs retrosynthetic fragmentation using both enzymatic (from the JN3604IMT enzyme database) and synthetic transformations (retro-chemical SMARTS)
- **RetroTide Agent**: Forward PKS synthesis verification—when DORAnet fragments match the PKS library, RetroTide is spawned to design and verify PKS module sequences that can synthesize the fragment

This hierarchical architecture leverages the complementary strengths:
- Monofunctional enzymes excel at regio/stereoselective modifications but cannot form complex C-C bonds
- Synthetic chemistry catalyzes complex coupling reactions
- PKS modules can assemble carbon backbones from simple acyl-CoA precursors

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd RL_agents_for_retrosynthesis

# Create conda environment
conda create -n retrosynthesis python=3.9
conda activate retrosynthesis

# Install in development mode
pip install -e .
```

### Dependencies

Core dependencies (see `pyproject.toml` for full list):
- `doranet` - Reaction network expansion (enzymatic + synthetic transformations)
- `rdkit>=2022.03.1` - Cheminformatics toolkit
- `networkx>=2.6.0` - Graph operations
- `plotly>=5.0.0` - Interactive visualizations

Optional:
- `retrotide` - PKS synthesis verification (for RetroTide spawning)
- `bcs` - Biosynthetic cluster scoring
- `DORA_XGB` - Enzymatic reaction feasibility prediction (DORA-XGB model)
- `pathermo` - Thermodynamic property estimation (group contribution method for ΔH)

## Directory Structure

```
RL_agents_for_retrosynthesis/
├── DORAnet_agent/
│   ├── mcts.py                    # DORAnetMCTS class (sequential MCTS)
│   ├── async_expansion_mcts.py    # AsyncExpansionDORAnetMCTS (parallel)
│   ├── node.py                    # Tree node with MCTS statistics
│   ├── visualize.py               # Interactive HTML visualization
│   └── policies/
│       ├── base.py                # RolloutPolicy, RewardPolicy base classes
│       ├── rollout.py             # Rollout policy implementations
│       ├── reward.py              # Reward policy implementations
│       ├── thermodynamic.py       # Thermodynamic-scaled wrapper policies
│       └── tests/                 # Policy unit tests
├── RetroTide_agent/
│   ├── mcts.py                    # Forward PKS synthesis MCTS
│   └── node.py                    # PKS design state node
├── scripts/
│   ├── run_DORAnet_single_agent.py   # Sequential MCTS runner
│   ├── run_DORAnet_Async.py          # Async MCTS runner (recommended)
│   ├── run_DORAnet_Async_batch.py    # Batch processing runner
│   └── run_RetroTide_single_agent.py # Standalone RetroTide runner
├── tests/
│   ├── test_async_expansion_mcts.py
│   ├── test_policies.py
│   └── fixtures/
├── data/
│   ├── building_blocks/
│   │   ├── biological_building_blocks.txt    # 334 metabolites
│   │   ├── chemical_building_blocks.txt      # 278,779 commercial compounds
│   │   ├── expanded_pks_building_blocks.txt  # 106,496 PKS products
│   │   ├── pks_building_blocks.txt           # 13,312 original PKS
│   │   ├── prohibited_building_blocks.txt    # 652 hazardous (excluded)
│   │   └── cofactors/
│   └── processed/                 # Alternative location for building blocks
├── results/                       # Output directory for runs
├── ARCHITECTURE_AND_ROADMAP.md
├── CLAUDE.md                      # AI assistant guidance
└── pyproject.toml
```

## MCTS Implementations

### 1. AsyncExpansionDORAnetMCTS (Recommended)

Multiprocessing MCTS that parallelizes DORAnet expansion while keeping tree operations thread-safe via virtual loss.

```python
from rdkit import Chem
from DORAnet_agent import AsyncExpansionDORAnetMCTS, Node
from DORAnet_agent.policies import (
    PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck,
    SparseTerminalRewardPolicy,
)

# Create target molecule
target_smiles = "COC1=CC(OC(C=CC2=CC=CC=C2)C1)=O"  # kavain
target_mol = Chem.MolFromSmiles(target_smiles)

# Create root node
root = Node(fragment=target_mol, parent=None, depth=0, provenance="target")

# Initialize MCTS with policies
agent = AsyncExpansionDORAnetMCTS(
    root=root,
    target_molecule=target_mol,
    total_iterations=1000,
    max_depth=3,
    max_children_per_expand=50,
    use_enzymatic=True,
    use_synthetic=True,

    # Building block files
    sink_compounds_files=[
        "data/building_blocks/biological_building_blocks.txt",
        "data/building_blocks/chemical_building_blocks.txt",
    ],
    pks_library_file="data/building_blocks/expanded_pks_building_blocks.txt",
    prohibited_chemicals_file="data/building_blocks/prohibited_building_blocks.txt",

    # Policy configuration
    rollout_policy=PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
        pks_building_blocks_path="data/building_blocks/expanded_pks_building_blocks.txt"
    ),
    reward_policy=SparseTerminalRewardPolicy(sink_terminal_reward=1.0),

    # Selection and downselection configuration
    selection_policy="depth_biased",  # or "UCB1"
    depth_bonus_coefficient=4.0,
    child_downselection_strategy="most_thermo_feasible",  # Prioritize thermodynamically feasible reactions

    # Async configuration
    num_workers=None,  # Auto-detect CPU count
)

# Run search
agent.run()

# Get results
print(agent.get_tree_summary())
agent.save_results("results/output.txt")
agent.save_finalized_pathways("results/pathways.txt")
```

### 2. DORAnetMCTS (Sequential)

Single-threaded MCTS for debugging or when parallelization isn't needed.

```python
from DORAnet_agent import DORAnetMCTS, Node

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_mol,
    total_iterations=100,
    max_depth=3,
    # ... same parameters as async
)
agent.run()
```

## Rollout Policies

Rollout policies determine how leaf nodes are evaluated during MCTS simulation.

### NoOpRolloutPolicy

Returns neutral score (0.0). Useful for testing pure MCTS exploration.

```python
from DORAnet_agent.policies import NoOpRolloutPolicy
policy = NoOpRolloutPolicy()
```

### SpawnRetroTideOnDatabaseCheck

Sparse rewards with RetroTide spawning. Checks if fragments match PKS library; on match, spawns RetroTide for forward synthesis verification.

```python
from DORAnet_agent.policies import SpawnRetroTideOnDatabaseCheck
policy = SpawnRetroTideOnDatabaseCheck(
    success_reward=1.0,   # Reward for successful RetroTide PKS designs
    failure_reward=0.0,
)
```

### SAScore_and_SpawnRetroTideOnDatabaseCheck

Dense rewards using Synthetic Accessibility (SA) Score plus PKS database checking. Better training signal but may bias toward chemical synthesis routes.

```python
from DORAnet_agent.policies import SAScore_and_SpawnRetroTideOnDatabaseCheck
policy = SAScore_and_SpawnRetroTideOnDatabaseCheck(
    success_reward=1.0,
    sa_max_reward=1.0,
)
```

### PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck

**Recommended for biosynthetic targets.** Uses Maximum Common Substructure (MCS) similarity to PKS building blocks instead of SA Score. Addresses SA Score's bias toward chemical synthesis.

```python
from DORAnet_agent.policies import PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck
policy = PKS_sim_score_and_SpawnRetroTideOnDatabaseCheck(
    pks_building_blocks_path="data/building_blocks/expanded_pks_building_blocks.txt",
    similarity_threshold=0.7,
)
```

## Reward Policies

Reward policies define how to compute rewards for terminal states.

```python
from DORAnet_agent.policies import (
    SparseTerminalRewardPolicy,
    SinkCompoundRewardPolicy,
    PKSLibraryRewardPolicy,
    ComposedRewardPolicy,
)

# Sparse: 1.0 for sink compounds/PKS terminals, 0.0 otherwise
reward_policy = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)

# Composed: combine multiple policies with weights
reward_policy = ComposedRewardPolicy([
    (SinkCompoundRewardPolicy(reward_value=1.0), 0.5),
    (PKSLibraryRewardPolicy(), 0.5),
])
```

## Thermodynamic Scoring

The system incorporates thermodynamic feasibility scoring to prioritize chemically realistic reactions. Two scoring mechanisms are available:

### DORA-XGB Feasibility Scoring (Enzymatic Reactions)

For enzymatic reactions, the DORA-XGB machine learning model predicts reaction feasibility. This model was trained on known enzymatic transformations and outputs a probability score (0.0-1.0) indicating likelihood of feasibility.

**Requirements**: Install the optional `DORA_XGB` package.

```python
# Scores are automatically computed when DORA_XGB is available
# Node attributes after expansion:
node.feasibility_score  # float 0.0-1.0 (probability)
node.feasibility_label  # int 0 or 1 (binary classification)
```

### Pathermo Thermodynamic Scoring (All Reactions)

For synthetic reactions (and as a fallback for enzymatic), the `pathermo` library computes enthalpy of reaction (ΔH) using group contribution methods. A sigmoid transformation converts ΔH to a 0-1 score:

```
score = 1.0 / (1.0 + exp(k * (ΔH - threshold)))
```

Where `k=0.2` and `threshold=15.0 kcal/mol` by default. Exothermic reactions (negative ΔH) score near 1.0; highly endothermic reactions score near 0.0.

**Requirements**: Install the optional `pathermo` package.

```python
# Node attributes after expansion:
node.enthalpy_of_reaction  # float (ΔH in kcal/mol)
node.thermodynamic_label   # int 0 or 1 (0 = unfavorable, 1 = favorable)
```

### Scoring Priority

| Provenance | Primary Scorer | Fallback |
|------------|----------------|----------|
| Enzymatic  | DORA-XGB | Pathermo sigmoid(ΔH) |
| Synthetic  | Pathermo sigmoid(ΔH) | None |
| Unknown    | Default 1.0 | - |

## Thermodynamic Policies

Wrapper policies that scale rewards by pathway thermodynamic feasibility. These can wrap any base rollout or reward policy.

### ThermodynamicScaledRolloutPolicy

Scales rollout rewards by the thermodynamic feasibility of the pathway from root to the current node.

```python
from DORAnet_agent.policies import (
    ThermodynamicScaledRolloutPolicy,
    SAScore_and_SpawnRetroTideOnDatabaseCheck,
)

policy = ThermodynamicScaledRolloutPolicy(
    base_policy=SAScore_and_SpawnRetroTideOnDatabaseCheck(
        success_reward=1.0,
        sa_max_reward=1.0,
    ),
    feasibility_weight=0.8,      # 0.0=ignore feasibility, 1.0=full scaling
    sigmoid_k=0.2,               # Steepness of sigmoid for ΔH
    sigmoid_threshold=15.0,      # Center point in kcal/mol
    use_dora_xgb_for_enzymatic=True,  # Use DORA-XGB for enzymatic reactions
    aggregation="geometric_mean",     # How to aggregate pathway scores
)
```

### ThermodynamicScaledRewardPolicy

Scales terminal rewards by pathway thermodynamic feasibility.

```python
from DORAnet_agent.policies import (
    ThermodynamicScaledRewardPolicy,
    SparseTerminalRewardPolicy,
)

policy = ThermodynamicScaledRewardPolicy(
    base_policy=SparseTerminalRewardPolicy(sink_terminal_reward=1.0),
    feasibility_weight=0.8,
    sigmoid_k=0.2,
    sigmoid_threshold=15.0,
    use_dora_xgb_for_enzymatic=True,
    aggregation="geometric_mean",
)
```

### Aggregation Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| `geometric_mean` | (∏ scores)^(1/n) | Balanced—penalizes weak links (default) |
| `product` | ∏ scores | Strict—pathway is only as good as worst step |
| `minimum` | min(scores) | Very strict—single bad step dominates |
| `arithmetic_mean` | Σ scores / n | Lenient—averages out bad steps |

### Combined Example

Use both thermodynamic-scaled rollout and reward policies together:

```python
from DORAnet_agent.policies import (
    ThermodynamicScaledRolloutPolicy,
    ThermodynamicScaledRewardPolicy,
    SAScore_and_SpawnRetroTideOnDatabaseCheck,
    SparseTerminalRewardPolicy,
)

# Rollout policy with thermodynamic scaling
rollout_policy = ThermodynamicScaledRolloutPolicy(
    base_policy=SAScore_and_SpawnRetroTideOnDatabaseCheck(
        success_reward=1.0,
        sa_max_reward=1.0,
    ),
    feasibility_weight=0.8,
    aggregation="geometric_mean",
)

# Reward policy with thermodynamic scaling
reward_policy = ThermodynamicScaledRewardPolicy(
    base_policy=SparseTerminalRewardPolicy(sink_terminal_reward=1.0),
    feasibility_weight=0.8,
    aggregation="geometric_mean",
)

agent = AsyncExpansionDORAnetMCTS(
    root=root,
    target_molecule=target_mol,
    rollout_policy=rollout_policy,
    reward_policy=reward_policy,
    # ... other parameters
)
```

## Usage

### Running the Scripts

The runner scripts contain hardcoded configurations. Edit the `main()` call at the bottom of each script to change parameters:

```python
# In scripts/run_DORAnet_Async.py
if __name__ == "__main__":
    main(
        target_smiles="COC1=CC(OC(C=CC2=CC=CC=C2)C1)=O",  # kavain
        molecule_name="kavain",
        total_iterations=1000,
        max_depth=3,
        max_children_per_expand=50,
        results_subfolder="kavain_experiment",
    )
```

Then run:

```bash
# Async MCTS (recommended)
python scripts/run_DORAnet_Async.py

# Sequential MCTS
python scripts/run_DORAnet_single_agent.py

# Batch processing
python scripts/run_DORAnet_Async_batch.py
```

## MCTS Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | Node | required | Root node containing target molecule |
| `target_molecule` | Chem.Mol | required | RDKit molecule to fragment |
| `total_iterations` | int | 100 | Number of MCTS iterations |
| `max_depth` | int | 3 | Maximum tree depth |
| `max_children_per_expand` | int | 10 | Max children per expansion |
| `use_enzymatic` | bool | True | Enable enzymatic transformations |
| `use_synthetic` | bool | True | Enable synthetic transformations |
| `selection_policy` | str | "depth_biased" | "UCB1" or "depth_biased" |
| `depth_bonus_coefficient` | float | 2.0 | Depth bias strength (depth_biased only) |
| `child_downselection_strategy` | str | "first_N" | "first_N", "hybrid", or "most_thermo_feasible" |
| `MW_multiple_to_exclude` | float | 1.5 | Filter fragments > target_MW × this |
| `num_workers` | int | None | Worker processes (None = auto) |

### Selection Policies

- **UCB1**: Standard Upper Confidence Bound—balances exploration and exploitation
- **depth_biased**: UCB1 + depth bonus—encourages finding complete pathways to building blocks

### Child Downselection Strategies

When DORAnet generates fragments during expansion, the `child_downselection_strategy` controls which fragments are kept (up to `max_children_per_expand`):

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `first_N` | Keep first N fragments in DORAnet's generation order | Fastest, minimal overhead |
| `hybrid` | Prioritize: sink compounds (+1000) > PKS matches (+500) > smaller MW | Balances terminals with simpler fragments |
| `most_thermo_feasible` | **Recommended.** Prioritize by thermodynamic feasibility score, with bonuses for sink compounds (+1000) and PKS matches (+500) | Best for finding chemically realistic pathways |

**`most_thermo_feasible` details:**

This strategy computes feasibility scores during fragment generation (before downselection), then sorts by:
1. Sink compounds: feasibility_score + 1000 (highest priority)
2. PKS library matches: feasibility_score + 500
3. Other fragments: raw feasibility_score (0.0-1.0)

Feasibility scores use DORA-XGB for enzymatic reactions and sigmoid-transformed ΔH for synthetic reactions. This ensures thermodynamically favorable fragments are kept even when many fragments are generated.

```python
agent = AsyncExpansionDORAnetMCTS(
    root=root,
    target_molecule=target_mol,
    child_downselection_strategy="most_thermo_feasible",  # Recommended
    max_children_per_expand=50,
    # ... other parameters
)
```

## Building Block Libraries

| Library | Count | Description |
|---------|-------|-------------|
| Chemical | 278,779 | Commercial building blocks |
| Biological | 334 | Biologically-derived compounds |
| PKS Original | 13,312 | Core PKS building blocks |
| PKS Expanded | 106,496 | Expanded PKS library with intermediates |
| Prohibited | 652 | Hazardous compounds (excluded) |
| Cofactors | 47 | Enzymatic cofactors (excluded from search) |

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_async_expansion_mcts.py -v

# Run with coverage
pytest tests/ --cov=DORAnet_agent --cov-report=term-missing
```

## Output Files

Each run generates:

```
results/<subfolder>/
├── doranet_results_<name>_<timestamp>.txt       # Full tree and node details
├── finalized_pathways_<name>_<timestamp>.txt    # Extracted synthesis pathways
├── successful_pathways_<name>_<timestamp>.txt   # Verified PKS pathways
├── doranet_interactive_<name>_<timestamp>.html  # Interactive tree visualization
└── doranet_pathways_<name>_<timestamp>.html     # Pathways-only visualization
```

## Architecture

### Hierarchical Agent Flow

```
Target Molecule (SMILES)
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│              DORAnet MCTS (Retrosynthetic)                 │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────────┐  │
│  │  Select  │→ │  Expand  │→ │  Rollout Policy Check   │  │
│  │  (UCB1)  │  │ (DORAnet)│  │  (PKS Library Match?)   │  │
│  └──────────┘  └──────────┘  └─────────────────────────┘  │
│       ▲              │                    │               │
│       │              │         ┌──────────┴──────────┐    │
│       │              │         ▼                     ▼    │
│       │              │    ┌─────────┐         ┌──────────┐│
│       │              │    │ NO: Use │         │YES: Spawn││
│       │              │    │ Reward  │         │ RetroTide││
│       │              │    │ Policy  │         │  MCTS    ││
│       │              │    └────┬────┘         └────┬─────┘│
│       │              │         │                   │      │
│       │              ▼         ▼                   ▼      │
│       │        ┌───────────────────────────────────────┐  │
│       └────────│           Backpropagate              │  │
│                └───────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
              Terminal Conditions:
              • Sink compound (building block)
              • PKS terminal (RetroTide verified)
              • Max depth reached
```

### MDP Formulation

**DORAnet Agent (Retrosynthetic)**:
- **State**: Molecular fragment (RDKit Mol + canonical SMILES)
- **Action**: DORAnet expansion (enzymatic or synthetic mode)
- **Reward**: Sparse—1.0 for terminals (sink compounds, PKS matches), 0.0 otherwise
- **Selection**: Depth-biased UCB1 (default) or standard UCB1

**RetroTide Agent (Forward PKS Synthesis)**:
- **State**: PKS intermediate (product, module design, depth)
- **Action**: Add PKS extension module (condensation, reduction, dehydration domains)
- **Reward**: 1.0 for exact target match (graph isomorphism), else MCS similarity
- **Selection**: UCB1 with subgraph-guided pruning

See [ARCHITECTURE_AND_ROADMAP.md](ARCHITECTURE_AND_ROADMAP.md) for detailed architecture documentation.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `pytest tests/` to ensure all tests pass
5. Submit a pull request

## Citation

If you use this work, please cite:

```bibtex
@software{rl_agents_retrosynthesis,
  title={RL Agents for Retrosynthesis},
  author={Chainani, Yash},
  year={2024},
  url={https://github.com/...}
}
```
