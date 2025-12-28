# Auto-Open Browser Feature

## Overview

The interactive visualization can now automatically open in your default browser when it's generated!

## Usage

### Method 1: Via DORAnet Agent (Recommended)

```python
from DORAnet_agent import DORAnetMCTS, Node
from rdkit import Chem

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    total_iterations=10,
    max_depth=2,
    enable_visualization=True,
    enable_interactive_viz=True,
    auto_open_viz=True,  # ← Auto-open in browser!
    visualization_output_dir="./results"
)

agent.run()
agent.save_results("./results/my_run.txt")

# The browser will automatically open with the interactive visualization!
```

### Method 2: Manual Creation

```python
from DORAnet_agent.visualize import create_enhanced_interactive_html

create_enhanced_interactive_html(
    agent=agent,
    output_path="./results/viz.html",
    auto_open=True  # ← Auto-open in browser!
)
```

### Method 3: Via Example Script

The example script now auto-opens by default:

```bash
python scripts/example_enhanced_interactive_viz.py
# Browser automatically opens!
```

## What Happens

When `auto_open=True`:

1. ✅ Generates the interactive HTML visualization
2. ✅ Saves it to the specified path
3. ✅ Automatically opens it in your default browser
4. ✅ Prints confirmation message

Output:
```
[Visualization] Enhanced interactive HTML saved to: results/my_run_interactive.html
[Visualization] Opening visualization in your default browser...
```

## When to Use

**Use `auto_open=True`** when:
- Running interactive experiments
- You want immediate feedback
- Exploring results in real-time
- Running from command line

**Use `auto_open=False`** (default) when:
- Running batch experiments
- Running on a remote server (no display)
- Generating many visualizations at once
- You'll review results later

## Example Workflows

### Interactive Exploration
```python
# Quick experiment with immediate visualization
agent = DORAnetMCTS(
    ...,
    enable_interactive_viz=True,
    auto_open_viz=True  # See results immediately!
)
agent.run()
agent.save_results("./results/experiment_001.txt")
# Browser opens automatically - start exploring!
```

### Batch Processing
```python
# Process multiple targets without browser spam
for target in target_molecules:
    agent = DORAnetMCTS(
        ...,
        enable_interactive_viz=True,
        auto_open_viz=False  # Don't open 10 browser tabs!
    )
    agent.run()
    agent.save_results(f"./results/{target}_results.txt")

# Review all visualizations later
```

### Default Behavior
```python
# By default, auto_open is False
agent = DORAnetMCTS(
    ...,
    enable_interactive_viz=True
    # auto_open_viz defaults to False
)
# You'll need to manually open the HTML file
```

## Browser Compatibility

The visualization will open in your system's default browser:
- ✅ Chrome
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ✅ Other modern browsers

## Troubleshooting

### Browser doesn't open

If the browser doesn't open automatically:

1. **Check the console output** - The file path is printed
2. **Manually open** the HTML file from the printed path
3. **Check permissions** - Ensure Python has permission to open files
4. **Try a different browser** - Set a different default browser

### Wrong browser opens

To change your default browser:
- **macOS**: System Preferences → General → Default web browser
- **Windows**: Settings → Apps → Default apps → Web browser
- **Linux**: Varies by distribution (usually in system settings)

### Multiple browser tabs

If running multiple experiments, use `auto_open_viz=False` to avoid opening many tabs.

## Technical Details

Uses Python's built-in `webbrowser` module:
```python
import webbrowser
webbrowser.open(f"file://{absolute_path}")
```

This is cross-platform and works on macOS, Windows, and Linux.

## Quick Reference

```python
# Auto-open enabled
agent = DORAnetMCTS(..., auto_open_viz=True)

# Auto-open disabled (default)
agent = DORAnetMCTS(..., auto_open_viz=False)
agent = DORAnetMCTS(...)  # Same as False

# Manual function call
create_enhanced_interactive_html(agent, path, auto_open=True)
```

That's it! Simple and convenient. 🎉
