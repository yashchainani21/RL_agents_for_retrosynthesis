# Visualization Flag Implementation

## Summary

Added automatic visualization generation capability to the DORAnetMCTS agent via configuration flags.

## Changes Made

### 1. DORAnet_agent/mcts.py

#### New Parameters in `DORAnetMCTS.__init__()`

```python
enable_visualization: bool = False
visualization_output_dir: Optional[str] = None
```

- **`enable_visualization`**: Controls whether visualizations are automatically generated after `save_results()` is called
- **`visualization_output_dir`**: Specifies the directory where visualization files will be saved (defaults to current directory)

#### New Method: `_generate_visualizations()`

Private method that handles automatic generation of:
- Full MCTS tree visualization (with node coloring by type and PKS matches)
- PKS pathways visualization (showing only paths to PKS-synthesizable fragments)

Visualization files are automatically named based on the results filename:
- `{results_basename}_tree.png`
- `{results_basename}_pks_pathways.png`

### 2. scripts/run_DORAnet_single_agent.py

Updated the runner script to:
- Pass the `generate_visualization` flag to the agent's `enable_visualization` parameter
- Set `visualization_output_dir` to the results directory
- Remove manual visualization code (now handled automatically by the agent)

### 3. scripts/example_visualization_flag.py (NEW)

Created example script demonstrating:
- How to enable automatic visualization
- How to disable automatic visualization
- How to specify custom output directories

## Usage

### Option 1: Enable via Agent Constructor

```python
from DORAnet_agent import DORAnetMCTS, Node
from rdkit import Chem

target_molecule = Chem.MolFromSmiles("CCCCC(=O)O")
root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    total_iterations=10,
    max_depth=2,
    enable_visualization=True,  # Enable auto-visualization
    visualization_output_dir="./my_visualizations"  # Custom output dir
)

agent.run()
agent.save_results("results.txt")  # Automatically generates visualizations
```

### Option 2: Use Command-Line Flag (via runner script)

```bash
# With visualization
python scripts/run_DORAnet_single_agent.py --visualize

# Without visualization (default)
python scripts/run_DORAnet_single_agent.py
```

## Benefits

1. **Convenience**: No need to manually call visualization functions
2. **Consistent Naming**: Visualizations are automatically named based on results files
3. **Backward Compatible**: Default is `enable_visualization=False`, so existing code continues to work
4. **Flexible**: Can still manually generate visualizations if needed
5. **Error Handling**: Gracefully handles missing dependencies (networkx, matplotlib)

## Output Files

When visualization is enabled, calling `save_results("path/to/results.txt")` will generate:

```
results.txt                    # Text results
results_tree.png              # Full MCTS tree visualization
results_pks_pathways.png      # PKS pathways visualization
```

## Dependencies

Visualizations require:
- `networkx`
- `matplotlib`

If not installed, the agent will print a warning and continue without generating visualizations.

## Example Run

```python
# Create and configure agent
agent = DORAnetMCTS(
    root=root,
    target_molecule=target_mol,
    enable_visualization=True,
    visualization_output_dir="./results"
)

# Run MCTS search
agent.run()

# Save results - visualizations generated automatically!
agent.save_results("./results/doranet_run_001.txt")

# Output:
# - ./results/doranet_run_001.txt
# - ./results/doranet_run_001_tree.png
# - ./results/doranet_run_001_pks_pathways.png
```
