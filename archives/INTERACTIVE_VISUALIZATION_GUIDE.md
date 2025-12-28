# Enhanced Interactive Visualization Guide

## Overview

The enhanced interactive visualization creates an HTML file with a fully interactive DORAnet MCTS search tree. You can hover over nodes and edges to see detailed information, including molecule structures!

## Features

✅ **Molecule Structure Images** - Hover over nodes to see 2D chemical structures
✅ **Rich Node Metadata** - View provenance, PKS match status, visits, values, depth
✅ **Reaction Information** - Hover over edges to see DORAnet reaction SMARTS
✅ **Same Color Scheme** - Matches the static PNG visualizations
✅ **Interactive Controls** - Zoom, pan, and explore the tree
✅ **High Quality** - Generates crisp molecule images using RDKit

## Quick Start

### Method 1: Use the Example Script

```bash
cd /Users/yashchainani/Desktop/PythonProjects/RL_agents_for_retrosynthesis
python scripts/example_enhanced_interactive_viz.py
```

This will create `results/doranet_interactive_enhanced.html` that you can open in your browser.

### Method 2: Enable in DORAnet Agent

```python
from DORAnet_agent import DORAnetMCTS, Node
from rdkit import Chem

target_molecule = Chem.MolFromSmiles("CCCCCCCCC(=O)O")
root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    total_iterations=10,
    max_depth=2,
    enable_visualization=True,  # Enable static PNGs
    enable_interactive_viz=True,  # Enable interactive HTML ← NEW!
    visualization_output_dir="./results"
)

agent.run()
agent.save_results("./results/my_run.txt")

# Automatically creates:
# - my_run_tree.png (static)
# - my_run_pks_pathways.png (static)
# - my_run_interactive.html (interactive) ← NEW!
```

### Method 3: Manual Creation

```python
from DORAnet_agent.visualize import create_enhanced_interactive_html

# After running your agent
create_enhanced_interactive_html(
    agent=agent,
    output_path="./results/custom_interactive.html",
    molecule_img_size=(300, 300),  # Customize image size
)
```

## Interactive Features

### Node Hovering

When you hover over a node, you'll see:
- **Molecule Structure Image** - 2D rendering of the chemical structure
- **Node ID** - Unique identifier
- **Depth** - Level in the search tree
- **Provenance** - Whether it's from an enzymatic or synthetic pathway
- **PKS Match** - Whether the fragment matches the PKS library (✓ Yes / ✗ No)
- **Visits** - Number of times the node was visited by MCTS
- **Avg Value** - Average reward value
- **SMILES** - Full SMILES string

### Edge Hovering

When you hover over an edge (connection between nodes), you'll see:
- **Reaction Name** - The DORAnet reaction that created this fragment
- **SMARTS Pattern** - The reaction SMARTS pattern used

### Color Coding

Same as static visualizations:
- 🟠 **Orange** - Target molecule (root)
- 🔵 **Blue** - Enzymatic pathway
- 🟣 **Purple** - Synthetic pathway
- 🟢 **Green** - PKS library match ✓

Edges are also colored to match their provenance!

### Interactive Controls

- **Mouse Wheel** - Zoom in/out
- **Click and Drag** - Pan around the tree
- **Box Zoom** - Click the box zoom tool, then drag to select an area
- **Reset** - Click the reset button to restore original view
- **Save** - Click the save button to download a PNG snapshot

## Requirements

The interactive visualization requires:
```bash
pip install bokeh networkx matplotlib rdkit
```

## File Size Considerations

Interactive HTML files embed base64-encoded molecule images, so they can be larger than static PNGs:
- Small trees (10-20 nodes): ~500 KB - 1 MB
- Medium trees (50-100 nodes): ~2-5 MB
- Large trees (200+ nodes): ~10-20 MB

The file size is worth it for the interactivity!

## Customization Options

### Adjust Molecule Image Size

```python
create_enhanced_interactive_html(
    agent=agent,
    output_path="./viz.html",
    molecule_img_size=(400, 400),  # Larger images (default: 250x250)
)
```

### Change Figure Dimensions

Edit `visualize.py` line ~696:
```python
p = figure(
    width=1600,  # Increase width
    height=1000,  # Increase height
    ...
)
```

## Troubleshooting

### Images Not Showing

If molecule images don't appear:
1. Check that RDKit is installed: `python -c "from rdkit import Chem; print('OK')"`
2. Check browser console for errors (F12)
3. Try a different molecule (some SMILES may be invalid)

### Slow Performance

For large trees (100+ nodes):
1. Reduce `molecule_img_size` to (150, 150)
2. Disable interactive viz and use static PNGs instead
3. Limit `max_children_per_expand` to reduce tree size

### File Won't Open

If the HTML file won't open:
1. Make sure you're opening it in a modern browser (Chrome, Firefox, Safari, Edge)
2. Try copying the file path and pasting in browser: `file:///full/path/to/file.html`
3. Check that the file size isn't corrupted (should be > 100 KB)

## Example Workflow

```python
# 1. Set up your experiment
target_smiles = "CCCCCCCCC(=O)O"
target_molecule = Chem.MolFromSmiles(target_smiles)
root = Node(fragment=target_molecule, parent=None, depth=0, provenance="target")

# 2. Configure agent with interactive viz
agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    total_iterations=15,
    max_depth=3,
    enable_visualization=True,
    enable_interactive_viz=True,  # ← Enable interactive!
    visualization_output_dir="./my_experiment_results"
)

# 3. Run the search
agent.run()

# 4. Save results (automatically creates interactive viz)
agent.save_results("./my_experiment_results/run_001.txt")

# 5. Open in browser and explore!
# file:///path/to/my_experiment_results/run_001_interactive.html
```

## Tips for Best Results

1. **Start Small** - Test with `total_iterations=5-10` first to make sure it works
2. **Use PKS Library** - Enables green highlighting of PKS-synthesizable fragments
3. **Balance Tree Size** - Too many nodes makes the visualization crowded
4. **Save Multiple Versions** - Generate both static PNGs and interactive HTML
5. **Explore Systematically** - Use the tree to identify promising pathways

## Advanced: Adding Custom Tooltips

You can extend the visualization by editing `visualize.py`:

```python
# In create_enhanced_interactive_html(), find the node_hover_html section (~line 743)
# Add your custom fields:

node_hover_html = """
<div style="...">
    ...existing fields...
    <b>Custom Field:</b> @custom_field<br>
</div>
"""

# Then add the data to node_source (~line 620):
node_source = ColumnDataSource(data=dict(
    ...existing fields...
    custom_field=[...your data...],
))
```

## Comparison with Other Tools

| Feature | Static PNG | Enhanced HTML | Cytoscape | PyVis |
|---------|-----------|---------------|-----------|-------|
| Molecule Images | ✗ | ✅ | ✗ | ✗ |
| Reaction Info | ✗ | ✅ | Manual | Manual |
| Interactive | ✗ | ✅ | ✅ | ✅ |
| Easy Setup | ✅ | ✅ | ✗ | ✅ |
| File Size | Small | Medium | Small | Medium |
| Best For | Papers | Exploration | Complex networks | Quick viz |

## Support

For issues or questions:
1. Check the Bokeh documentation: https://docs.bokeh.org/
2. Verify RDKit is working: `python -c "from rdkit.Chem import Draw"`
3. Try the example script first to ensure dependencies are correct

## Future Enhancements

Potential additions:
- [ ] Click nodes to highlight pathways
- [ ] Filter nodes by provenance or PKS match
- [ ] Export selected pathways to separate files
- [ ] 3D molecule structures
- [ ] Side-by-side comparison of multiple runs
- [ ] Real-time updates during MCTS execution

Let us know what features would be most useful!
