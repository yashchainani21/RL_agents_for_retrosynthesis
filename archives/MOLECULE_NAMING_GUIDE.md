# Molecule Naming Guide for DORAnet Agent

## Overview

You can now specify a custom name for your target molecule when running the DORAnet agent. This name will be used in all output filenames instead of the SMILES string, making it easier to organize and identify your results.

## Usage

### Basic Usage (without name)

```bash
python scripts/run_DORAnet_single_agent.py
```

Output files:
- `doranet_results_CCCCCCCCC(=O)O_20251227_123456.txt`
- `doranet_interactive_CCCCCCCCC(=O)O_20251227_123456.html`
- `doranet_results_CCCCCCCCC(=O)O_20251227_123456_tree.png`
- `doranet_results_CCCCCCCCC(=O)O_20251227_123456_pks_pathways.png`

### With Custom Molecule Name

```bash
python scripts/run_DORAnet_single_agent.py --name nonanoic_acid
```

Output files:
- `doranet_results_nonanoic_acid_20251227_123456.txt`
- `doranet_interactive_nonanoic_acid_20251227_123456.html`
- `doranet_results_nonanoic_acid_20251227_123456_tree.png`
- `doranet_results_nonanoic_acid_20251227_123456_pks_pathways.png`

### Short Form

```bash
python scripts/run_DORAnet_single_agent.py -n my_molecule
```

## Examples

### Example 1: Simple Molecule Name

```bash
python scripts/run_DORAnet_single_agent.py --name pentanoic_acid
```

Console output:
```
Target molecule: pentanoic_acid (CCCCC(=O)O)
```

Files created:
```
results/doranet_results_pentanoic_acid_20251227_143052.txt
results/doranet_interactive_pentanoic_acid_20251227_143052.html
```

### Example 2: Complex Name with Spaces

```bash
python scripts/run_DORAnet_single_agent.py --name "gamma hydroxybutyric acid"
```

The script automatically sanitizes the name:
- Spaces → underscores
- Special characters → removed
- Result: `gamma_hydroxybutyric_acid`

Files created:
```
results/doranet_results_gamma_hydroxybutyric_acid_20251227_143052.txt
results/doranet_interactive_gamma_hydroxybutyric_acid_20251227_143052.html
```

### Example 3: Systematic Names

```bash
python scripts/run_DORAnet_single_agent.py --name "1,4-butanediol"
```

Sanitized to: `14-butanediol`

Files created:
```
results/doranet_results_14-butanediol_20251227_143052.txt
results/doranet_interactive_14-butanediol_20251227_143052.html
```

### Example 4: Without Visualization

```bash
python scripts/run_DORAnet_single_agent.py --name my_molecule --no-visualize
```

Note: Currently `--visualize` defaults to True, use this only if you want to skip visualization.

## Name Sanitization Rules

The script automatically cleans your molecule name for use in filenames:

1. **Spaces** → converted to underscores (`_`)
2. **Slashes** (`/`, `\`) → converted to underscores
3. **Special characters** → removed (only alphanumeric, `_`, and `-` allowed)
4. **Original name preserved in console** but sanitized for filenames

### Sanitization Examples

| Input Name | Sanitized Filename |
|------------|-------------------|
| `nonanoic acid` | `nonanoic_acid` |
| `1,4-butanediol` | `14-butanediol` |
| `gamma-hydroxybutyric acid` | `gamma-hydroxybutyric_acid` |
| `(R)-3-hydroxybutyrate` | `R-3-hydroxybutyrate` |
| `Molecule #42` | `Molecule_42` |

## Benefits

### Before (using SMILES):
```
results/doranet_results_CCCCCCCCC(=O)O_20251227_143052.txt
results/doranet_interactive_CCCCCCCCC(=O)O_20251227_143052.html
```
- Hard to identify which molecule
- SMILES can be long and confusing
- Difficult to search for specific results

### After (using names):
```
results/doranet_results_nonanoic_acid_20251227_143052.txt
results/doranet_interactive_nonanoic_acid_20251227_143052.html
```
- Easy to identify at a glance
- Simple to search and organize
- Better for batch processing multiple molecules

## Batch Processing Example

```bash
# Process multiple molecules with descriptive names
python scripts/run_DORAnet_single_agent.py --name molecule_A
python scripts/run_DORAnet_single_agent.py --name molecule_B
python scripts/run_DORAnet_single_agent.py --name control_compound
python scripts/run_DORAnet_single_agent.py --name test_scaffold_1
```

Results are clearly organized:
```
results/
├── doranet_results_molecule_A_20251227_100000.txt
├── doranet_interactive_molecule_A_20251227_100000.html
├── doranet_results_molecule_B_20251227_100100.txt
├── doranet_interactive_molecule_B_20251227_100100.html
├── doranet_results_control_compound_20251227_100200.txt
├── doranet_interactive_control_compound_20251227_100200.html
└── ...
```

## Programmatic Usage

You can also use the name parameter when calling the script from Python:

```python
from scripts.run_DORAnet_single_agent import main

# With name
main(create_interactive_visualization=True, molecule_name="my_compound")

# Without name (uses SMILES)
main(create_interactive_visualization=True, molecule_name=None)
```

## Tips

1. **Use descriptive names**: `fatty_acid_C9` is better than `compound1`
2. **Be consistent**: Use the same naming convention across your project
3. **Include metadata**: `PKS_product_nonanoic_acid` helps categorize results
4. **Avoid very long names**: Keep under 50 characters for readability
5. **Use underscores**: They're easier to type than hyphens in terminal

## Help

To see all available options:

```bash
python scripts/run_DORAnet_single_agent.py --help
```

Output:
```
usage: run_DORAnet_single_agent.py [-h] [--visualize] [--name NAME]

Run DORAnet MCTS agent for retrosynthesis

optional arguments:
  -h, --help            show this help message and exit
  --visualize, -v       Generate interactive visualization (default: True)
  --name NAME, -n NAME  Name for the target molecule (used in output filenames).
                        Example: --name nonanoic_acid
```

## Timestamp Format

All files include a timestamp in the format `YYYYMMDD_HHMMSS`:
- `20251227_143052` = December 27, 2025 at 14:30:52

This ensures:
- No file overwrites
- Easy chronological sorting
- Clear tracking of when experiments were run
