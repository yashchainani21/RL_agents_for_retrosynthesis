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
│   ├── run_RetroTide_single_agent.py # Standalone RetroTide runner
│   └── benchmark_runtimes.py         # Runtime benchmarking script
├── tests/
│   ├── test_async_expansion_mcts.py
│   ├── test_categorize_pathway.py      # Pathway categorization tests
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

### Rollout Phase: Database Checking Order

After a node is expanded and children are created, each child undergoes database checking in a specific order to determine whether to run the rollout policy or use the reward policy directly. **PKS library membership is checked first** to ensure PKS-eligible fragments always get RetroTide verification, even if they are also sink compounds.

```
For each child node after expansion:

1. PKS Library Check (FIRST)
   └── _is_in_pks_library(child.smiles)
       └── Checks if canonical SMILES is in self.pks_library

   If PKS match:
   ├── Run rollout policy (enables RetroTide spawning)
   ├── If rollout returns terminal=True:
   │   └── Mark is_pks_terminal=True, use rollout reward
   └── If rollout returns terminal=False:
       └── Fall back to sink compound check (step 2)

2. Sink Compound Check (SECOND - only if NOT PKS match, or PKS rollout failed)
   └── child.is_sink_compound (already set during node creation)
       └── Set by _get_sink_compound_type() which checks:
           ├── self.biological_sink_compounds
           └── self.chemical_sink_compounds

   If sink compound:
   └── Use reward policy directly (skip rollout)

3. Standard Rollout (THIRD - neither PKS nor sink)
   └── Run rollout policy normally
```

#### Visual Flow

```
                    ┌─────────────────┐
                    │  New Child Node │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ In PKS Library? │
                    └────────┬────────┘
                       YES   │   NO
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐           ┌────────▼────────┐
     │  Run Rollout    │           │ Is Sink Compound?│
     │ (RetroTide)     │           └────────┬────────┘
     └────────┬────────┘              YES   │   NO
              │                 ┌───────────┴───────────┐
     ┌────────▼────────┐        │                       │
     │ Terminal=True?  │ ┌──────▼──────┐       ┌────────▼────────┐
     └────────┬────────┘ │ Use Reward  │       │  Run Rollout    │
        YES   │   NO     │ Policy Only │       │  (Standard)     │
     ┌────────┴────────┐ └─────────────┘       └─────────────────┘
     │                 │
┌────▼────┐    ┌───────▼───────┐
│ PKS     │    │ Is also Sink? │
│Terminal │    └───────┬───────┘
└─────────┘       YES  │  NO
              ┌────────┴────────┐
              │                 │
      ┌───────▼───────┐  ┌──────▼──────┐
      │ Use Reward    │  │ Use Rollout │
      │ Policy        │  │ Reward      │
      └───────────────┘  └─────────────┘
```

This PKS-priority approach ensures that fragments matching the PKS library always have the opportunity for RetroTide verification, maximizing the use of the hierarchical agent system for biosynthetic pathway discovery.

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

### SAScore_and_TerminalRewardPolicy (Recommended)

Combines terminal rewards with SA Score for non-terminal nodes. Provides dense signals via synthetic accessibility scoring while giving full rewards for terminals.

```python
from DORAnet_agent.policies import SAScore_and_TerminalRewardPolicy

reward_policy = SAScore_and_TerminalRewardPolicy(
    sink_terminal_reward=1.0,
    pks_terminal_reward=1.0,
)
```

### Other Reward Policies

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

## Recommended Policy Configuration

The recommended configuration uses **ThermodynamicScaledRewardPolicy** wrapping **SAScore_and_TerminalRewardPolicy** for optimal balance of dense signals and thermodynamic feasibility:

```python
from DORAnet_agent.policies import (
    SpawnRetroTideOnDatabaseCheck,
    ThermodynamicScaledRewardPolicy,
    SAScore_and_TerminalRewardPolicy,
)

# Rollout policy: handles PKS matching and RetroTide spawning
rollout_policy = SpawnRetroTideOnDatabaseCheck(
    success_reward=1.0,
    failure_reward=0.0,
)

# Reward policy: terminal rewards + SA score, scaled by thermodynamic feasibility
reward_policy = ThermodynamicScaledRewardPolicy(
    base_policy=SAScore_and_TerminalRewardPolicy(
        sink_terminal_reward=1.0,
        pks_terminal_reward=1.0,
    ),
    feasibility_weight=0.8,
    sigmoid_k=0.2,
    sigmoid_threshold=15.0,
    use_dora_xgb_for_enzymatic=True,
    aggregation="geometric_mean",
)
```

This configuration:
- Uses **SpawnRetroTideOnDatabaseCheck** for rollout: spawns RetroTide verification for PKS library matches
- Uses **ThermodynamicScaledRewardPolicy** for rewards: scales rewards by pathway thermodynamic feasibility
- Wraps **SAScore_and_TerminalRewardPolicy**: provides dense SA score signals for non-terminals

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

### Benchmarking Runtime Performance

The `benchmark_runtimes.py` script provides modular runtime benchmarking for three synthesis modes:

1. **DORAnet Standalone** - Raw library expansion (no MCTS)
2. **DORAnetMCTS Sequential** - Single-threaded MCTS
3. **AsyncExpansionDORAnetMCTS** - Multiprocessing MCTS

**Configuration**: Edit toggle variables at the top of the `__main__` block:

```python
# In scripts/benchmark_runtimes.py
if __name__ == "__main__":
    # ========== BENCHMARK MODE SELECTION (enable exactly ONE) ==========
    RUN_DORANET_STANDALONE = False      # Raw DORAnet library expansion
    RUN_DORANET_MCTS_SEQUENTIAL = True  # DORAnetMCTS (single-threaded)
    RUN_DORANET_MCTS_ASYNC = False      # AsyncExpansionDORAnetMCTS

    # ========== EXPANSION MODE TOGGLES ==========
    use_enzymatic = True
    use_synthetic = True

    # ========== TARGET MOLECULE ==========
    target_smiles = "CCCCC(=O)O"
    molecule_name = "pentanoic_acid"

    # ========== MCTS PARAMETERS ==========
    total_iterations = 50
    max_depth = 3
    max_children_per_expand = None  # None = no limit
    child_downselection_strategy = None  # None = no filtering

    # ========== ASYNC PARAMETERS ==========
    num_workers = None  # None = auto-detect CPU count
```

Then run:

```bash
python scripts/benchmark_runtimes.py
```

**Output**: Human-readable `.txt` files saved to `results/benchmarks/` with the naming pattern:
```
benchmark_runtime_{mode}_{molecule_name}_{YYYYMMDD_HHMMSS}.txt
```

**Example output**:
```
======================================================================
RUNTIME BENCHMARK RESULTS
======================================================================

RUN INFORMATION
----------------------------------------------------------------------
Mode:                      sequential
Molecule:                  pentanoic_acid
SMILES:                    CCCCC(=O)O
Timestamp:                 20260128_152109
Expansion modes:           enzymatic=True, synthetic=True

CONFIGURATION
----------------------------------------------------------------------
Total iterations:          50
Max depth:                 3
Max children per expand:   None (unlimited)
Downselection strategy:    None (no filtering)

RUNTIME METRICS
----------------------------------------------------------------------
Total runtime:             0.5738 seconds

TREE METRICS (MCTS)
----------------------------------------------------------------------
Total nodes:               120
Terminal nodes:            123
  - Sink compounds:        61
  - PKS matches:           62
Iterations completed:      50

STATUS
----------------------------------------------------------------------
Error:                     None

======================================================================
```

**Notes**:
- Both MCTS modes have `stop_on_first_pathway=True` enabled for benchmarking time-to-first-solution
- RetroTide spawning is disabled for faster benchmarks
- The standalone mode measures raw DORAnet network generation time without MCTS overhead

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
| `stop_on_first_pathway` | bool | False | Stop MCTS when first complete pathway is found |
| `enable_frontier_fallback` | bool | True | Enable frontier-based fallback selection for deep exploration |

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

### Early Stopping

The `stop_on_first_pathway` parameter enables early termination when the first complete pathway is found. This is useful for benchmarking time-to-first-solution or when you only need one valid pathway.

```python
agent = AsyncExpansionDORAnetMCTS(
    root=root,
    target_molecule=target_mol,
    stop_on_first_pathway=True,  # Stop when first pathway found
    # ... other parameters
)

agent.run()

# After run, check early stopping results
if agent.first_pathway_found:
    print(f"First pathway found at iteration {agent.first_pathway_iteration}")
    print(f"Time to first pathway: {agent.first_pathway_time:.2f}s")
    print(f"Terminal node: {agent.first_pathway_node.smiles}")
```

A **complete pathway** is one where:
1. The terminal node is covered (sink compound, PKS-verified, or excluded fragment)
2. All byproducts along the pathway are also covered

The batch runner script supports this via CLI:

```bash
python scripts/run_DORAnet_Async_batch.py \
    --name kavain \
    --smiles "COC1=CC(OC(C=CC2=CC=CC=C2)C1)=O" \
    --stop-on-first-pathway
```

### Frontier Fallback Selection

The `enable_frontier_fallback` parameter enables a fallback selection mechanism that prevents wasted iterations when the standard tree traversal reaches dead-end branches (where all children are terminal nodes).

**The Problem:**

Standard MCTS tree traversal can get stuck when it reaches a branch where all children are terminal (sink compounds or PKS-verified). In this case, `select()` returns `None` and the iteration is wasted—no expansion, no backpropagation. This commonly happens with high terminal node density from the 278K+ commercial building blocks.

**The Solution:**

Frontier fallback maintains a max-heap of unexpanded non-terminal nodes, prioritized by depth (deepest first). When standard traversal returns `None`, MCTS falls back to selecting the deepest unexpanded node from this frontier.

```python
agent = AsyncExpansionDORAnetMCTS(
    root=root,
    target_molecule=target_mol,
    enable_frontier_fallback=True,  # Default: enabled
    max_depth=6,  # Can now reach deeper levels
    # ... other parameters
)
```

**Benefits:**
- **More productive iterations**: Iterations that would have been wasted now expand new nodes
- **Deeper exploration**: MCTS can reach `max_depth` even when early branches become terminal
- **Better coverage**: Explores alternative branches when primary path hits dead ends

**Statistics:**

After a run, the agent prints frontier usage statistics:
```
[DORAnet] Frontier fallback: 523 selections, 147 nodes remaining in frontier
```

**When to Disable:**

Set `enable_frontier_fallback=False` to preserve the original behavior where MCTS stops when all reachable nodes are terminal. This may be useful for:
- Benchmarking the original algorithm
- Comparative studies
- When you want early termination on exhaustive search

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

### Pathway Categorization Validation

A validation script is available to verify pathway categorization in `successful_pathways.txt` files:

```bash
python scripts/validate_pathway_categorization.py results/successful_pathways_*.txt
```

This validates:
1. **PKS pathways have RetroTide designs**: All pathways categorized as PKS-dependent have RetroTide PKS designs (terminal or byproduct)
2. **Non-PKS pathways have no RetroTide designs**: Pathways categorized as non-PKS do not have any PKS designs
3. **Purely enzymatic have rule signatures**: All purely enzymatic pathways contain `ruleXXXX_XX` enzyme rule patterns
4. **Purely synthetic have no rule signatures**: Purely synthetic pathways do not contain enzyme rule patterns

The script also displays RetroTide design counts by category and design-based pathway proportions.

## Output Files

Each run generates:

```
results/<subfolder>/
├── doranet_results_<name>_<timestamp>.txt       # Full tree and node details
├── finalized_pathways_<name>_<timestamp>.txt    # Extracted synthesis pathways
├── successful_pathways_<name>_<timestamp>.txt   # Verified PKS pathways with statistics
├── doranet_interactive_<name>_<timestamp>.html  # Interactive tree visualization
└── doranet_pathways_<name>_<timestamp>.html     # Pathways-only visualization
```

### successful_pathways.txt Format

The `successful_pathways.txt` file includes comprehensive run statistics:

```
======================================================================
SUCCESSFUL PATHWAYS (PKS OR SINK PRODUCTS ONLY)
======================================================================

RUN CONFIGURATION
----------------------------------------
Target molecule:           CCCCC(=O)O
Total iterations:          100
Max depth:                 4
Max children per expand:   30
Selection policy:          UCB1
Child downselection:       most_thermo_feasible
MW multiple to exclude:    1.5
Rollout policy:            SpawnRetroTideOnDatabaseCheck
Reward policy:             ThermodynamicScaled(SAScore_and_Terminal)
RetroTide max depth:       5
RetroTide iterations:      50

Total pathways: 955

PATHWAY TYPE BREAKDOWN
----------------------------------------
Sink Compound Pathways:
  Purely synthetic:        162
  Purely enzymatic:        105
  Synthetic + enzymatic:   634

PKS-Synthesizable Pathways:
  Direct PKS match            0
  Synthetic + PKS            58 pathways -> 10 entries (27 exact, 31 simulated)
  Enzymatic + PKS            84 pathways -> 14 entries (38 exact, 46 simulated)
  Synthetic + enz + PKS     153 pathways -> 29 entries (75 exact, 78 simulated)

Summary (counting each PKS design as a synthesis route):
  PKS-based routes:         295 / 1196 (24.7%)
  Non-PKS routes:           901 / 1196 (75.3%)

RETROTIDE DESIGN BREAKDOWN
----------------------------------------
Total PKS pathways:        53
Total exact match designs: 140
Total simulated designs:   155
Total all designs:         295
Avg designs per PKS path:  5.6
```

**Key features:**
- **Run configuration**: All parameters used for the MCTS run
- **Pathway categorization**: Pathways are categorized by synthesis modality (enzymatic, synthetic, PKS)
- **PKS design counting**: Each RetroTide design counts as a distinct synthesis route
- **Design-based percentages**: Summary shows percentage of routes that are PKS-based

**Pathway categorization logic:**
- A pathway is categorized as PKS-dependent if either:
  1. The terminal node is PKS-synthesizable, OR
  2. Any byproduct along the pathway is PKS-synthesizable

## Pathway Definition and Tracking

### What Constitutes a Successful Pathway?

A synthesis pathway is considered **successful** if and only if:

1. **Terminal Fragment Coverage**: The terminal fragment (leaf node) must be synthesizable from one of:
   - Biological building blocks (334 metabolites)
   - Chemical building blocks (278K commercial compounds)
   - PKS library matches (106K polyketide products)
   - Excluded fragments (cofactors, chemistry helpers)

2. **Byproduct Coverage**: Every byproduct generated along the pathway must also be covered by the same sets above.

This ensures that all fragments produced during retrosynthetic decomposition can actually be obtained—either purchased, biosynthesized, or are common reaction byproducts.

### Sink Compound Types

The system categorizes sink compounds (terminal building blocks) into distinct types for output labeling:

| Type | Description | Examples |
|------|-------------|----------|
| `biological` | Biologically-derived metabolites | Amino acids, sugars, fatty acids |
| `chemical` | Commercially available building blocks | Reagents, solvents, starting materials |
| `pks` | PKS library matches | Polyketide intermediates and products |
| `bio_cofactor` | Enzymatic cofactors | SAM, SAH, NADPH, ATP |
| `chem_helper` | Chemistry helper molecules | H₂O, CO₂, H₂ |

### Coverage Validation

During pathway enumeration, the `is_product_covered()` function checks each fragment:

```python
def is_product_covered(smiles: str) -> bool:
    """
    A product is covered if:
    1. It's a sink compound (biological or chemical building blocks)
    2. It's in the PKS library (can be synthesized by PKS)
    3. It's an excluded fragment which includes:
       - Biology cofactors (SAM, SAH, NADPH, etc.)
       - Chemistry helpers (H2O, CO2, etc.)
       - Other common small molecules
    """
```

### Pathway Categorization Logic

Pathways in `successful_pathways.txt` are categorized by synthesis modality:

**Non-PKS Pathways (Sink Compound Terminals)**:
- **Purely synthetic**: All steps use synthetic organic chemistry
- **Purely enzymatic**: All steps use enzymatic transformations
- **Synthetic + enzymatic**: Mixed modality pathway

**PKS-Dependent Pathways** (counted by RetroTide designs):
- **Direct PKS match**: Target molecule directly matches PKS library
- **Synthetic + PKS**: Synthetic steps leading to PKS-synthesizable terminal
- **Enzymatic + PKS**: Enzymatic steps leading to PKS-synthesizable terminal
- **Synthetic + enzymatic + PKS**: Mixed modality with PKS terminal

A pathway is classified as PKS-dependent if:
1. The terminal node is PKS-synthesizable, OR
2. Any byproduct along the pathway is PKS-synthesizable

### Integration Testing

The pathway validation logic is verified by an integration test that:
1. Runs a minimal MCTS search on a simple target (pentanoic acid)
2. Verifies all pathways in `successful_pathways.txt` satisfy the pathway definition
3. Confirms every terminal fragment is in coverage sets
4. Confirms every byproduct is covered (not marked as `sink=No`)

```bash
# Run the integration test
pytest tests/test_policies.py::TestSaveSuccessfulPathways::test_pentanoic_acid_integration_all_modalities -v

# Skip slow integration tests for quick CI
pytest tests/test_policies.py -v -m "not slow"
```

## Architecture

### Hierarchical Agent Flow

```
Target Molecule (SMILES)
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    DORAnet MCTS (Retrosynthetic)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────────────────┐ │
│  │  Select  │→ │  Expand  │→ │     For Each Child Fragment:         │ │
│  │  (UCB1)  │  │ (DORAnet)│  │                                      │ │
│  └──────────┘  └──────────┘  │  1. PKS Library Match? ──────────┐   │ │
│       ▲                      │     │YES              │NO        │   │ │
│       │                      │     ▼                 ▼          │   │ │
│       │                      │  ┌─────────┐   2. Sink Compound? │   │ │
│       │                      │  │ Rollout │      │YES    │NO    │   │ │
│       │                      │  │ Policy  │      ▼       ▼      │   │ │
│       │                      │  │(RetroTide)  ┌──────┐ ┌──────┐ │   │ │
│       │                      │  └────┬────┘   │Reward│ │Rollout│ │   │ │
│       │                      │       │        │Policy│ │Policy │ │   │ │
│       │                      │       ▼        └──┬───┘ └──┬────┘ │   │ │
│       │                      │  Terminal?        │        │      │   │ │
│       │                      │  YES→PKS Terminal │        │      │   │ │
│       │                      │  NO→Sink Fallback─┘        │      │   │ │
│       │                      └──────────────┬─────────────┘      │   │
│       │                                     │                        │
│       │                                     ▼                        │
│       │                      ┌──────────────────────────────────┐    │
│       └──────────────────────│         Backpropagate            │    │
│                              └──────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              Terminal Conditions:
              • PKS terminal (RetroTide verified) ← prioritized
              • Sink compound (building block)
              • Max depth reached
```

**Key Design Decision**: PKS library membership is checked **before** sink compound status. This ensures that fragments matching the PKS library always get RetroTide verification, even if they are also commercially available building blocks. This maximizes the discovery of biosynthetic pathways.

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

## FAQ

### Why does the search run much slower when `use_chem_building_blocksDB=False`?

This is due to **terminal node density**. Sink compounds (building blocks) act as "stopping conditions" that prevent further tree expansion.

**With chemical building blocks enabled (`use_chem_building_blocksDB=True`):**
- You have **278,779 chemical building blocks** as potential sink compounds
- Fragments quickly match these building blocks and become terminal nodes
- Terminal nodes are **not expanded further**—they're leaf nodes
- The tree stays shallow and pruned

**With chemical building blocks disabled (`use_chem_building_blocksDB=False`):**
- You only have **334 biological building blocks** as sink compounds
- Far fewer fragments match → fewer terminal nodes
- Non-terminal nodes **continue to be expanded** by DORAnet
- Each DORAnet expansion is expensive (network generation, reaction enumeration, thermodynamic calculations)
- The tree grows much deeper and wider before finding terminals

The selection algorithm explicitly skips terminal nodes (see `mcts.py` lines 1464-1467):

```python
# Skip terminal nodes - they don't need further expansion
if child.is_sink_compound or child.is_pks_terminal:
    continue
```

**Practical implications:** If you want to run without chemical building blocks (for cleaner biosynthetic pathways), consider:
- Reducing `max_depth` to limit tree growth
- Reducing `max_children_per_expand` to limit branching factor
- Increasing iteration budget significantly
- Using `child_downselection_strategy="most_thermo_feasible"` to prioritize promising branches

### Why are enzymatic reaction labels incorrect (e.g., rule shows water but reaction has none)?

If you notice enzymatic reaction labels that don't match the actual reaction (e.g., a rule labeled as producing water but the reaction products don't include water), this is likely due to **stale fragment cache** containing incorrect labels from a previous bug.

**Root Cause**: An earlier version of the code had an index mismatch bug where reaction labels were looked up by TSV row number, but DORAnet network operator indices are different because some rules are skipped during network construction (rules containing excluded cofactors like `CARBONYL_CoF` and `AMINO_CoF`).

For example:
- TSV row 2931 contains `rule0124_07` (Products: `WATER;Any`)
- But network operator index 2931 is actually `rule0126_2` (Products: `Any;Any`)

The old code incorrectly returned `rule0124_07` when the actual rule used was `rule0126_2`.

**The Fix**: This bug was fixed by fetching reaction labels directly from the network operator's metadata:

```python
# OLD (buggy): Used pre-loaded list indexed by TSV row
rxn_label = self._enzymatic_labels[op_idx]

# NEW (correct): Uses operator's own metadata
meta = network.ops.meta(op_idx)
rxn_label = meta.get('name')
```

**Solution**: Clear the fragment cache and re-run your MCTS expansion:

```bash
# Clear the cache
rm -rf .cache/doranet_fragments/*.pkl

# Or use the cache clearing script
python scripts/clear_cache.py

# Then re-run your MCTS agent
python scripts/run_DORAnet_Async.py
```

This will force fresh DORAnet expansions that use the corrected labeling code.

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
