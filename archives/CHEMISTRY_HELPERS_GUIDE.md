# Chemistry Helpers / Cofactor Exclusion Guide

## Overview

The DORAnet MCTS agent now automatically excludes chemistry helper molecules (small reagents, solvents, cofactors) from being tracked as nodes in the retrosynthetic network. This ensures the search focuses on meaningful intermediates rather than common chemical reagents.

## What Gets Excluded

The agent excludes molecules from two sources:

### 1. Biological Cofactors (`all_cofactors.csv`)
Common metabolites and biological cofactors such as:
- ATP, ADP, AMP
- NAD+, NADH, NADP+, NADPH
- CoA derivatives
- Common amino acids
- Sugars and nucleotides

### 2. Chemistry Helpers (`chemistry_helpers.csv`)
Common chemical reagents and small molecules:
- Water (H₂O)
- Oxygen (O₂)
- Hydrogen (H₂)
- Carbon dioxide (CO₂)
- Formaldehyde
- Carbon monoxide
- Halogens (Br₂, etc.)
- Methanol
- Ethylene
- Sulfuric acid
- Ammonia
- And other common reagents

## How It Works

When the DORAnet agent generates fragments during retrosynthesis:

1. ✅ **Fragments are generated** from enzymatic and synthetic reactions
2. ❌ **Excluded molecules are filtered out** - they don't become nodes in the tree
3. ✅ **Meaningful intermediates proceed** - only interesting molecules are tracked

This prevents the network from being cluttered with nodes for water, CO₂, ammonia, etc.

## Configuration

### Automatic (Default)

The `run_DORAnet_single_agent.py` script automatically loads both cofactor files:

```python
cofactors_files = [
    REPO_ROOT / "data" / "raw" / "all_cofactors.csv",
    REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv",
]

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    cofactors_files=[str(f) for f in cofactors_files],  # Both files loaded
    ...
)
```

### Manual Configuration

You can specify different cofactor files or add your own:

```python
from DORAnet_agent import DORAnetMCTS, Node
from rdkit import Chem

# Option 1: Use default files
cofactors_files = [
    "data/raw/all_cofactors.csv",
    "data/raw/chemistry_helpers.csv",
]

# Option 2: Add custom exclusions
cofactors_files = [
    "data/raw/all_cofactors.csv",
    "data/raw/chemistry_helpers.csv",
    "my_custom_exclusions.csv",  # Your own file
]

# Option 3: Use only one file
cofactors_files = [
    "data/raw/all_cofactors.csv",  # Only biological cofactors
]

agent = DORAnetMCTS(
    root=root,
    target_molecule=target_molecule,
    cofactors_files=cofactors_files,
    ...
)
```

### Backward Compatibility

The old single-file parameter still works:

```python
# Old way (deprecated but still works)
agent = DORAnetMCTS(
    cofactors_file="data/raw/all_cofactors.csv",
    ...
)

# New way (recommended)
agent = DORAnetMCTS(
    cofactors_files=["data/raw/all_cofactors.csv", "data/raw/chemistry_helpers.csv"],
    ...
)
```

## Adding Custom Exclusions

### Create a Custom CSV File

1. Create a CSV file with a `SMILES` column header:

```csv
SMILES
O
O=O
[H][H]
CCOC(=O)C
```

2. Add it to the cofactors_files list:

```python
cofactors_files = [
    "data/raw/all_cofactors.csv",
    "data/raw/chemistry_helpers.csv",
    "my_project_exclusions.csv",  # Your additions
]
```

### File Format Requirements

- **Must have** a header row with `SMILES` column
- One SMILES string per line
- Empty lines are ignored
- Comments starting with `#` are ignored
- SMILES with wildcards (`*`) are automatically skipped
- UTF-8 encoding (with or without BOM)

### Example Custom File

```csv
SMILES
# Solvents
CCOC(C)=O
CC(C)=O
ClCCl

# My specific exclusions
C1CCCCC1
c1ccccc1
```

## Chemistry Helpers Details

The `chemistry_helpers.csv` file currently includes:

- **Water**: `O`
- **Oxygen gas**: `O=O`
- **Hydrogen gas**: `[H][H]`
- **Carbon dioxide**: `O=C=O`
- **Formaldehyde**: `C=O`
- **Carbon monoxide**: `[C-]#[O+]`
- **Bromine**: `Br`, `[Br][Br]`
- **Methanol**: `CO`
- **Ethylene**: `C=C`
- **Sulfuric acid**: `O=S(O)O`, `O=S(=O)(O)O`
- **Ammonia**: `N`
- **Nitric acid**: `O=NO`, `O=[N+]([O-])O`
- **Nitrogen oxides**: `NO`, `N#N`
- **Hydrogen cyanide**: `C#N`
- **Sulfur**: `S`
- And more...

You can view the full list in: `data/raw/chemistry_helpers.csv`

## Example: What Gets Excluded

Consider a retrosynthetic step:

```
Nonanoic acid (CCCCCCCCC(=O)O)
    ↓ [Reaction: Hydrogenation of Alkene]
    ├─ Nonenoic acid (CCCCCCCC=C(=O)O) ✅ Tracked as node
    └─ Hydrogen (H₂) ❌ Excluded (in chemistry_helpers.csv)
```

The hydrogen molecule is automatically filtered out and won't appear as a node in your search tree.

## Benefits

✅ **Cleaner search trees** - No clutter from trivial molecules
✅ **Faster search** - Fewer nodes to explore
✅ **Better visualizations** - Focus on meaningful intermediates
✅ **Easier analysis** - Results contain only relevant compounds
✅ **Customizable** - Add your own exclusions for specific projects

## Verification

To check what's being excluded:

```python
agent = DORAnetMCTS(...)
print(f"Excluding {len(agent.excluded_fragments)} molecules")
```

Console output example:
```
[DORAnet] Loaded 39 cofactors from all_cofactors.csv
[DORAnet] Loaded 27 cofactors from chemistry_helpers.csv
Excluding 73 molecules  # Total unique exclusions (may overlap)
```

## Troubleshooting

### Issue: Important molecule being excluded

If a molecule you want to track is being incorrectly excluded:

1. Check if it's in the cofactor files
2. Remove it from the CSV file, or
3. Don't include that CSV file in `cofactors_files`

### Issue: Unwanted molecules appearing

If trivial molecules are appearing in your results:

1. Add them to `chemistry_helpers.csv`
2. Or create a custom exclusion file
3. Include the file in `cofactors_files`

### Issue: File not found error

Make sure the file paths are correct relative to your working directory:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
cofactors_files = [
    str(REPO_ROOT / "data" / "raw" / "all_cofactors.csv"),
    str(REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv"),
]
```

## Summary

The chemistry helpers exclusion feature ensures your DORAnet search focuses on meaningful chemical intermediates by automatically filtering out common reagents and cofactors. This is now enabled by default in the run script and requires no additional configuration!
