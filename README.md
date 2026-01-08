# RL Agents for Retrosynthesis

Multi-agent reinforcement learning system for designing synthesis pathways of complex molecules using synthetic organic chemistry, monofunctional enzymes, and multifunctional polyketide synthases (PKS).

## Overview

This project implements a **Monte Carlo Tree Search (MCTS)** framework that combines backward retrosynthetic analysis with forward biosynthetic planning. The system fragments complex target molecules into simpler precursors and evaluates whether those precursors can be synthesized via:

1. **Commercial building blocks** (278K+ chemical compounds, 334 biological metabolites)
2. **Polyketide synthase (PKS) assembly lines** (13K+ known PKS products)
3. **Enzymatic transformations** (biosynthetic reaction rules)
4. **Synthetic organic reactions** (traditional chemistry)

## Architecture

```
Target Molecule (SMILES)
        │
        ▼
┌───────────────────────────────────┐
│     DORAnet MCTS (Backward)       │
│  ─────────────────────────────    │
│  • Fragments target molecule      │
│  • Uses enzymatic + synthetic     │
│    transformation rules           │
│  • Filters prohibited chemicals   │
│  • Checks against PKS library     │
│  • Checks against sink compounds  │
└───────────────────────────────────┘
        │
        │ (if fragment matches PKS library)
        ▼
┌───────────────────────────────────┐
│    RetroTide MCTS (Forward)       │
│  ─────────────────────────────    │
│  • Designs PKS assembly lines     │
│  • Configures module domains      │
│  • Evaluates product similarity   │
│  • Returns viable PKS designs     │
└───────────────────────────────────┘
        │
        ▼
   Synthesis Pathways + Visualizations
```

## Directory Structure

```
RL_agents_for_retrosynthesis/
├── DORAnet_agent/              # DORAnet MCTS implementation
│   ├── mcts.py                 # Main MCTS driver
│   ├── async_expansion_mcts.py # Async multiprocessing MCTS
│   ├── node.py                 # Tree node representation
│   ├── visualize.py            # Interactive HTML visualization
│   └── policies/               # Modular rollout/reward policies
│       ├── __init__.py
│       ├── base.py             # RolloutPolicy, RewardPolicy ABCs
│       ├── rollout.py          # Rollout policy implementations
│       └── reward.py           # Reward policy implementations
├── RetroTide_agent/            # RetroTide MCTS for PKS design
│   ├── mcts.py                 # RetroTide MCTS implementation
│   └── node.py                 # PKS design node representation
├── scripts/                    # Executable scripts
│   ├── run_DORAnet_single_agent.py
│   ├── run_DORAnet_Async.py
│   └── run_RetroTide_single_agent.py
├── data/
│   ├── raw/                    # Source data files
│   │   ├── all_cofactors.csv
│   │   ├── chemistry_helpers.csv
│   │   ├── common_metabolites.csv
│   │   ├── JN3604IMT_rules.tsv
│   │   ├── Building_Blocks_US.sdf
│   │   └── prohibited_chemicals.json
│   └── processed/              # Processed libraries
│       ├── PKS_smiles.txt
│       ├── biological_building_blocks.txt
│       ├── chemical_building_blocks.txt
│       └── prohibited_chemical_SMILES.txt
├── utils/                      # Utility functions
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks
├── results/                    # Output files
└── figures/                    # Generated visualizations
```

## Workflows

### 1. DORAnet MCTS (Backward Retrosynthesis)

The DORAnet agent performs **backward retrosynthetic search** using Monte Carlo Tree Search:

**Selection Phase**
- Navigates the search tree using UCB1 or depth-biased selection policy
- Prioritizes unexplored nodes while balancing exploitation of promising paths

**Expansion Phase**
- Calls DORAnet neural network to generate molecular fragments
- Supports two transformation modes:
  - **Enzymatic**: Biosynthetic reaction rules (oxidations, reductions, condensations, etc.)
  - **Synthetic**: Organic chemistry transformations (Aldol, Diels-Alder, etc.)

**Filtering Pipeline**
- Removes small byproducts (water, CO2, ammonia)
- Excludes cofactors (ATP, FAD, NAD+, etc.)
- Filters molecules exceeding MW threshold (default: 1.5× target MW)
- Blocks prohibited/hazardous chemicals (652 substances)
- Removes duplicates and invalid SMILES

**Terminal Detection**
- **PKS Terminal**: Fragment matches the 13K PKS product library
- **Sink Compound**: Fragment is commercially available (building blocks)

**Reward & Backpropagation**
- Sink compounds receive configurable reward (default: 2.0)
- PKS library matches receive reward of 1.0
- Rewards propagate up the tree to update node values

### 2. RetroTide MCTS (Forward PKS Synthesis)

When DORAnet discovers a fragment matching the PKS library, it can spawn a **RetroTide agent** to design the PKS assembly line:

**PKS Module Architecture**
- **Loading Module**: Selects initial substrate
- **Extension Modules**: Add 2-carbon units with modifications
  - AT (Acyltransferase): Substrate selection
  - KR (Ketoreductase): Stereochemistry control
  - DH (Dehydratase): Double bond formation
  - ER (Enoylreductase): Double bond reduction
- **Thioesterase (TE)**: Product release via thiolysis, cyclization, or reduction

**Algorithm**
- MCTS navigates PKS design space
- Evaluates designs by product similarity to target
- Returns viable PKS configurations with module details

## Chemical Libraries

| Library | Entries | Description |
|---------|---------|-------------|
| PKS Products | 13,312 | Known PKS-synthesizable molecules |
| Chemical Building Blocks | 278,779 | Commercially available synthons |
| Biological Building Blocks | 334 | Natural metabolic precursors |
| Cofactors | 47 | Excluded from fragmentation (ATP, FAD, etc.) |
| Prohibited Chemicals | 652 | Hazardous/controlled substances |

## Usage

### Running DORAnet MCTS

```bash
python scripts/run_DORAnet_single_agent.py --name "molecule_name"
```

**Command Line Options**
- `--name`: Name for the target molecule (used in output filenames)
- `--visualize`: Generate interactive HTML visualizations
- `--iteration-viz`: Create visualizations at intervals during search
- `--iteration-interval`: Interval between iteration visualizations

### Running RetroTide Standalone

```bash
python scripts/run_RetroTide_single_agent.py
```

## Configuration

Key parameters in `DORAnetMCTS`:

```python
DORAnetMCTS(
    # Search parameters
    total_iterations=40,              # MCTS iterations
    max_depth=3,                      # Maximum fragmentation depth

    # Transformation modes
    use_enzymatic=True,               # Enable biosynthetic rules
    use_synthetic=True,               # Enable organic chemistry rules
    generations_per_expand=1,         # DORAnet generations per expansion
    max_children_per_expand=10,       # Max fragments retained per expansion

    # Filtering
    MW_multiple_to_exclude=1.5,       # Max MW = target_MW × this value

    # Selection policy
    selection_policy="depth_biased",  # "UCB1" or "depth_biased"
    depth_bonus_coefficient=2.0,      # Depth exploration bonus

    # Rewards
    sink_terminal_reward=2.0,         # Reward for commercial precursors

    # RetroTide integration
    spawn_retrotide=True,             # Auto-launch RetroTide for PKS matches
    retrotide_kwargs={
        "max_depth": 10,
        "total_iterations": 200,
    },

    # Visualization
    enable_interactive_viz=True,
)
```

## Async Expansion

For faster exploration on multi-core systems, use `AsyncExpansionDORAnetMCTS` which offloads expansion to multiprocessing workers:

### Quick Start

```python
from DORAnet_agent import AsyncExpansionDORAnetMCTS, Node

# Create root node
root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

# Create async expansion agent
agent = AsyncExpansionDORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    total_iterations=100,
    max_depth=3,

    # Async expansion parameters
    num_workers=4,                 # Number of worker processes
    max_inflight_expansions=4,     # Cap in-flight expansions

    # Standard parameters work the same as DORAnetMCTS
    use_enzymatic=True,
    use_synthetic=True,
)

# Run parallel search
agent.run()

# Access results (same API as sequential)
pks_matches = agent.get_pks_matches()
sink_compounds = agent.get_sink_compounds()

# Results are available on the agent as usual
pks_matches = agent.get_pks_matches()
sink_compounds = agent.get_sink_compounds()
```

### Scripts

The runner scripts are designed to be edited and run from your IDE:

- `scripts/run_DORAnet_single_agent.py` for the sequential run
- `scripts/run_DORAnet_Async.py` for async expansion

### Async Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_workers` | int | CPU-1 | Number of worker processes |
| `max_inflight_expansions` | int | `num_workers` | Maximum number of expansions queued at once |

## Modular Policy System

The MCTS implementation uses a **modular policy architecture** that allows swapping rollout and reward strategies without modifying core MCTS logic.

### Policy Types

| Policy Type | Purpose | Default |
|-------------|---------|---------|
| `RolloutPolicy` | Determines what happens after expansion (e.g., spawn RetroTide) | `NoOpRolloutPolicy` |
| `RewardPolicy` | Computes reward for terminal/sink nodes | `SparseTerminalRewardPolicy` |

### Built-in Policies

**Rollout Policies:**
- `NoOpRolloutPolicy`: Returns zero reward, no side effects (default)
- `SpawnRetroTideOnDatabaseCheck`: Spawns RetroTide MCTS when fragment matches PKS library (sparse rewards)
- `SAScore_and_SpawnRetroTideOnDatabaseCheck`: Dense rewards based on synthetic accessibility (SA Score) + RetroTide spawning for PKS matches (RECOMMENDED for better training signals)

**Reward Policies:**
- `SparseTerminalRewardPolicy`: Returns reward for sink compounds and PKS matches
- `SinkCompoundRewardPolicy`: Returns reward only for sink compounds  
- `PKSLibraryRewardPolicy`: Returns reward for PKS library matches
- `ComposedRewardPolicy`: Weighted combination of multiple reward policies

### SA Score Rewards (Dense Signals)

The `SAScore_and_SpawnRetroTideOnDatabaseCheck` policy provides dense intermediate rewards based on synthetic accessibility:

- **SA Score Formula**: `reward = (10 - sa_score) / 10`
- **Reward Range**: 0.0-0.9 (simple molecules get ~0.8-0.9, complex molecules get ~0.2-0.4)
- **Logic**:
  1. Terminal nodes (sink/PKS designs): `success_reward` (default 1.0)
  2. PKS library match + RetroTide succeeds: `success_reward` (1.0)
  3. PKS library match + RetroTide fails: SA Score reward (dense signal!)
  4. Non-PKS nodes: SA Score reward (dense signal!)

This provides valuable training signal for all nodes, not just terminal states.

### Using Custom Policies

```python
from DORAnet_agent import DORAnetMCTS
from DORAnet_agent.policies import (
    SpawnRetroTideOnDatabaseCheck,
    SAScore_and_SpawnRetroTideOnDatabaseCheck,
    SparseTerminalRewardPolicy,
    ComposedRewardPolicy,
    SinkCompoundRewardPolicy,
    PKSLibraryRewardPolicy,
)

# Option 1: Use backward-compatible spawn_retrotide flag
agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    spawn_retrotide=True,  # Creates SpawnRetroTideOnDatabaseCheck automatically
    # ... other parameters
)

# Option 2: Sparse rewards - Explicit SpawnRetroTideOnDatabaseCheck
rollout = SpawnRetroTideOnDatabaseCheck(
    pks_library=pks_smiles,
    retrotide_kwargs={"max_depth": 10, "total_iterations": 200},
    success_reward=1.0,
    failure_reward=0.0,
)
reward = SparseTerminalRewardPolicy(sink_terminal_reward=2.0)

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    rollout_policy=rollout,
    reward_policy=reward,
    # ... other parameters
)

# Option 3: Dense rewards - SA Score + RetroTide (RECOMMENDED)
rollout = SAScore_and_SpawnRetroTideOnDatabaseCheck(
    success_reward=1.0,   # Reward for successful PKS designs
    sa_max_reward=1.0,    # Optional cap on SA rewards (default 1.0)
)
reward = SparseTerminalRewardPolicy(sink_terminal_reward=1.0)

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    rollout_policy=rollout,
    reward_policy=reward,
    # ... other parameters
)

# Option 4: Composed reward policy for custom weighting
composed_reward = ComposedRewardPolicy([
    (SinkCompoundRewardPolicy(reward_value=2.0), 0.7),  # 70% weight
    (PKSLibraryRewardPolicy(), 0.3),                     # 30% weight
])

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    reward_policy=composed_reward,
)
```

### Creating Custom Policies

Implement the abstract base classes to create custom policies:

```python
from DORAnet_agent.policies import RolloutPolicy, RewardPolicy, RolloutResult

class MyRolloutPolicy(RolloutPolicy):
    @property
    def name(self) -> str:
        return "MyCustomRollout"
    
    def rollout(self, node, context):
        # Custom simulation/evaluation logic
        reward = evaluate_node(node)
        return RolloutResult(
            reward=reward,
            terminal=is_terminal(node),
            terminal_type="custom",
            metadata={"custom_data": "value"}
        )

class MyRewardPolicy(RewardPolicy):
    @property  
    def name(self) -> str:
        return "MyCustomReward"
    
    def calculate_reward(self, node, context):
        # Custom reward calculation
        return compute_reward(node, context)
```

### Deprecation Notice

The `reward_fn` parameter in `AsyncExpansionDORAnetMCTS` is deprecated. Use `reward_policy` instead:

```python
# Deprecated (still works but not recommended)
agent = AsyncExpansionDORAnetMCTS(
    reward_fn=lambda node: my_reward(node),  # Deprecated
    ...
)

# Recommended
agent = AsyncExpansionDORAnetMCTS(
    reward_policy=MyRewardPolicy(),
    ...
)
```

### How It Works

Async expansion lets the main thread keep selecting while expansions run in worker processes:

1. **Selection Phase** (main thread): select a leaf as usual, skip nodes already pending expansion
2. **Expansion Phase** (workers): DORAnet fragment generation runs in parallel
3. **Integration Phase** (main thread): returned fragments are attached to the tree and backpropagated
3. **Backpropagation Phase** (synchronized): Virtual loss removed, real rewards applied

Virtual loss temporarily penalizes selected nodes, making them appear less attractive to other threads. This encourages different threads to explore different parts of the tree.

### Expected Speedup

| Workers | Typical Speedup |
|---------|-----------------|
| 1 | 1.0x (sequential) |
| 2 | 1.6-1.8x |
| 4 | 2.5-3.5x |
| 8 | 3.5-5.0x |

Actual speedup depends on the complexity of fragment generation and tree structure.

## Output Files

| File Pattern | Description |
|--------------|-------------|
| `doranet_results_*.txt` | Detailed search tree and all pathways |
| `finalized_pathways_*.txt` | Complete successful synthesis routes |
| `successful_pathways_*.txt` | PKS-synthesizable routes only |
| `doranet_interactive_*.html` | Interactive tree visualization |
| `doranet_pathways_*.html` | Filtered pathways visualization |

### Interactive Visualizations

The HTML visualizations feature:
- Mouse wheel zoom and drag-to-pan navigation
- Hover tooltips showing molecule structures and metadata
- Color-coded nodes:
  - **Orange**: Target molecule
  - **Blue**: Enzymatic transformation products
  - **Purple**: Synthetic transformation products
  - **Green**: PKS library matches
- Edge labels showing reaction SMARTS

## Dependencies

| Package | Purpose |
|---------|---------|
| rdkit | Molecular representation and cheminformatics |
| doranet | Retrosynthesis neural networks |
| retrotide | PKS design engine |
| bcs | Biochemistry synthesis library |
| networkx | Graph algorithms for MCTS tree |
| plotly | Interactive HTML visualizations |
| numpy | Numerical computing |
| pandas | Data manipulation |

## Caching

The system implements efficient caching to avoid redundant computations:

- **Fragment Cache**: DORAnet expansions cached by MD5 hash of input parameters
- **Library Cache**: Large chemical libraries (>10K entries) cached as pickle files
- Cache location: `data/processed/.cache/`

## Safety Features

The system includes multiple safety checks:

1. **Prohibited Chemical Filtering**: Blocks 652 hazardous/controlled substances
2. **Target Validation**: Raises error if target molecule is prohibited
3. **MW Threshold**: Prevents generation of unrealistically large intermediates
4. **Cofactor Exclusion**: Removes metabolic cofactors from pathway intermediates

## Testing

```bash
pytest tests/
```

Tests cover:
- MCTS selection policies
- Node creation and manipulation
- Data loading and validation
- Fragment filtering logic

## License

[Add license information]

## Citation

[Add citation information]
