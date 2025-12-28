# Iteration Visualization Guide

## Overview

The DORAnet MCTS agent now supports **iteration-by-iteration visualization**, allowing you to track how the search tree grows over time and observe the UCB1 selection policy in action.

## What This Feature Does

When enabled, the agent generates visualizations after each MCTS iteration showing:

1. **Tree Growth Dynamics**: See how the tree expands as new nodes are added
2. **Node Selection**: Track which nodes are selected by the UCB1 policy
3. **Visit Counts**: Observe how visit counts accumulate over iterations
4. **Value Propagation**: See how rewards are backpropagated through the tree
5. **Exploration vs Exploitation**: Understand the balance between exploring new paths and exploiting known good paths

## File Organization

Iteration visualizations are saved in a dedicated subdirectory:

```
results/
  └── iterations/
      ├── iteration_00000_tree.png              # Static PNG after iteration 0
      ├── iteration_00000_interactive.html      # Interactive HTML after iteration 0
      ├── iteration_00001_tree.png              # Static PNG after iteration 1
      ├── iteration_00001_interactive.html      # Interactive HTML after iteration 1
      ├── iteration_00002_tree.png
      ├── iteration_00002_interactive.html
      └── ...
```

## How to Use

### Method 1: Command Line Arguments

Run the DORAnet script with the `--iteration-viz` flag:

```bash
python scripts/run_DORAnet_single_agent.py \
    --iteration-viz \
    --iteration-interval 2 \
    --name pentanoic_acid
```

**Arguments:**
- `--iteration-viz`: Enable iteration visualizations (default: False)
- `--iteration-interval N`: Generate visualizations every N iterations (default: 1)
- `--name`: Name for the target molecule (used in filenames)

### Method 2: Python Code

Configure the agent with iteration visualization parameters:

```python
from DORAnet_agent import DORAnetMCTS, Node

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    total_iterations=10,
    # ... other parameters ...

    # Iteration visualization settings
    enable_visualization=True,
    enable_interactive_viz=True,           # Enable both static and interactive
    enable_iteration_visualizations=True,  # Turn on iteration-by-iteration viz
    iteration_viz_interval=1,              # Generate after every iteration
    visualization_output_dir="results",
)

agent.run()
```

### Method 3: Example Script

Run the provided example script:

```bash
python scripts/example_iteration_viz.py
```

This demonstrates the feature with a simple test case.

## Visualization Interval

The `iteration_viz_interval` parameter controls how often visualizations are generated:

- `iteration_viz_interval=1`: Generate after **every** iteration (most detailed, creates many files)
- `iteration_viz_interval=5`: Generate after every **5th** iteration
- `iteration_viz_interval=10`: Generate after every **10th** iteration

**Recommendation**: For long runs (100+ iterations), use `iteration_viz_interval=5` or higher to reduce file count.

## Example: 10 Iteration Run

With `total_iterations=10` and `iteration_viz_interval=1`, you'll get:

- 10 static PNG files (iteration_00000 through iteration_00009)
- 10 interactive HTML files (if `enable_interactive_viz=True`)
- Total: 20 files in the `results/iterations/` directory

## What's in Each Visualization

### Static PNG Files (`iteration_XXXXX_tree.png`)

- **Title**: Shows current iteration number (e.g., "DORAnet Tree - Iteration 5/10")
- **Nodes**: Colored by provenance (orange=target, blue=enzymatic, purple=synthetic, green=PKS match)
- **Labels**: Node IDs, SMILES, visit counts, values
- **Edges**: Reaction information

### Interactive HTML Files (`iteration_XXXXX_interactive.html`)

- **Hoverable Nodes**: See molecule structure images and metadata
- **Hoverable Edges**: See reaction SMARTS and labels
- **Zoom/Pan**: Navigate large trees
- **Title**: Shows current iteration number

## Interpreting the Visualizations

### Early Iterations (0-2)
- Tree is small, mostly exploring initial moves
- Few nodes, low visit counts
- UCB1 tends to favor exploration

### Middle Iterations (3-7)
- Tree starts to branch out
- Some paths get more visits (exploitation)
- PKS matches may start appearing

### Late Iterations (8-10)
- Most promising paths have high visit counts
- Tree structure stabilizes
- UCB1 balances exploration and exploitation

### Animation Tip

You can create an animation by viewing the PNG files in sequence:

```bash
# On macOS/Linux, use a loop to display them
for f in results/iterations/iteration_*_tree.png; do
    open "$f"
    sleep 2
done
```

Or use a tool like `ffmpeg` to create a video:

```bash
ffmpeg -framerate 2 -pattern_type glob -i 'results/iterations/iteration_*_tree.png' \
    -c:v libx264 -pix_fmt yuv420p tree_growth.mp4
```

## Performance Considerations

**File Count**: With `total_iterations=100` and `iteration_viz_interval=1`, you'll generate **200 files** (100 PNG + 100 HTML).

**Disk Space**: Each PNG is ~500KB, each HTML is ~2MB (with molecule images). For 100 iterations:
- PNGs: ~50 MB
- HTMLs: ~200 MB
- **Total: ~250 MB**

**Generation Time**: Each visualization takes ~0.5-1 second. For 100 iterations with interval=1, add ~50-100 seconds to runtime.

**Recommendation**:
- For testing (10-20 iterations): Use `interval=1`
- For production (50-100 iterations): Use `interval=5` or `interval=10`
- For long runs (200+ iterations): Use `interval=20` or disable iteration viz

## Disabling Iteration Visualizations

By default, iteration visualizations are **disabled** to avoid creating too many files.

To disable explicitly:

```python
agent = DORAnetMCTS(
    # ...
    enable_iteration_visualizations=False,  # Default
)
```

Or omit the `--iteration-viz` flag when running from command line.

## Troubleshooting

### "No visualizations generated"

Make sure you have:
1. `enable_iteration_visualizations=True`
2. `enable_visualization=True` (main flag)
3. `visualization_output_dir` set to a valid path
4. Required packages installed: `pip install networkx matplotlib bokeh`

### "Too many files"

Increase `iteration_viz_interval`:

```python
iteration_viz_interval=10  # Only generate every 10 iterations
```

### "Visualizations don't auto-open"

Iteration visualizations don't auto-open by design (to avoid opening 100+ browser tabs!). The final visualization still auto-opens if `auto_open_viz=True`.

## Example Use Cases

### 1. Understanding UCB1 Behavior

Run with `iteration_viz_interval=1` on a small test case (10 iterations) to see how UCB1 selects nodes based on visit counts and values.

### 2. Debugging Tree Growth

If your tree isn't expanding as expected, iteration visualizations can show:
- Which nodes are being repeatedly selected
- Whether the tree is stuck in one branch
- If PKS matches are being found early or late

### 3. Tuning Hyperparameters

Compare iteration visualizations across different hyperparameter settings:
- Different `max_depth` values
- Different `max_children_per_expand` values
- Different UCB1 exploration constants

### 4. Creating Presentations

Use the PNG sequence to create animations for presentations showing how the MCTS algorithm works.

## Summary

Iteration visualizations are a powerful tool for:
- **Learning**: Understanding how MCTS works
- **Debugging**: Identifying issues in tree growth
- **Analysis**: Studying UCB1 selection patterns
- **Communication**: Creating animations and presentations

Enable them for small test runs, increase the interval for production runs, or disable them entirely for maximum performance.
