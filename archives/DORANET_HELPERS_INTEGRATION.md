# DORAnet Chemistry Helpers Integration

## Overview

Chemistry helper molecules (H₂O, O₂, H₂, CO₂, etc.) are now automatically passed to DORAnet's synthetic network generation as "helpers". This improves the quality of synthetic retrosynthetic networks by informing DORAnet about available small molecules.

## What Changed

### Dual Purpose for Chemistry Helpers

Chemistry helpers now serve **two purposes**:

1. **Excluded from MCTS tree** - Not tracked as nodes in the search tree
2. **Passed to DORAnet synthetic** - Used as available helpers during network generation

### Implementation

The DORAnet agent now:

```python
# 1. Loads chemistry helpers from CSV
self.chemistry_helpers = {canonical_smiles_from_chemistry_helpers.csv}

# 2. Excludes them from the search tree
self.excluded_fragments.update(chemistry_helpers)

# 3. Passes them to DORAnet synthetic network generation
network = synthetic.generate_network(
    job_name=job_name,
    starters={starter_smiles},
    helpers=self.chemistry_helpers,  # ← NEW!
    gen=self.generations_per_expand,
    direction="retro",
)
```

## Benefits

### Before (without helpers parameter):
- DORAnet generates reactions without knowing about available helpers
- May miss valid retrosynthetic steps that require small molecules
- Less realistic reaction proposals

### After (with helpers parameter):
- DORAnet knows which small molecules are available
- Can propose reactions that use H₂, O₂, H₂O, etc.
- More realistic and complete retrosynthetic networks
- Better alignment with actual synthetic chemistry

## Example

### Reaction Generation

**Without helpers:**
```
CCCCC=C → ??? (DORAnet doesn't know H₂ is available)
```

**With helpers (H₂ in helpers set):**
```
CCCCC=C + H₂ → CCCCCC (Hydrogenation)
```

DORAnet can now propose hydrogenation because it knows H₂ is available!

## Which Helpers Are Used

All molecules from `chemistry_helpers.csv`:

- **Water** (O)
- **Oxygen gas** (O=O)
- **Hydrogen gas** ([H][H])
- **Carbon dioxide** (O=C=O)
- **Formaldehyde** (C=O)
- **Carbon monoxide** ([C-]#[O+])
- **Bromine** (Br, BrBr)
- **Methanol** (CO)
- **Ethylene** (C=C)
- **Ammonia** (N)
- **Nitrogen** (N#N)
- **Hydrogen cyanide** (C#N)
- And more...

Total: **21 chemistry helper molecules**

## Automatic Configuration

This feature is **automatically enabled** when you use the run script:

```python
# In run_DORAnet_single_agent.py
cofactors_files = [
    REPO_ROOT / "data" / "raw" / "all_cofactors.csv",
    REPO_ROOT / "data" / "raw" / "chemistry_helpers.csv",  # Auto-detected and used
]

agent = DORAnetMCTS(
    cofactors_files=cofactors_files,
    ...
)
```

The agent automatically:
1. Detects files with "chemistry_helpers" in the name
2. Loads them into `self.chemistry_helpers`
3. Passes them to synthetic network generation

## Console Output

When you run the agent, you'll see:

```
[DORAnet] Loaded 39 cofactors from all_cofactors.csv
[DORAnet] Loaded 21 cofactors from chemistry_helpers.csv
[DORAnet] Loaded 3604 enzymatic rule labels
[DORAnet] Loaded 386 synthetic reaction labels
[DORAnet] Using 21 chemistry helpers for synthetic network generation  ← Confirmation
```

## Technical Details

### Enzymatic vs Synthetic

- **Enzymatic networks**: No helpers parameter (not supported by DORAnet enzymatic module)
- **Synthetic networks**: Helpers parameter is used ✅

### File Detection

The agent detects chemistry helpers files by checking if the filename contains "chemistry_helpers" (case-insensitive):

```python
if "chemistry_helpers" in str(cofactor_file_path).lower():
    self.chemistry_helpers.update(cofactor_smiles)
```

### Custom Chemistry Helpers

You can add your own helpers file:

```python
cofactors_files = [
    "data/raw/all_cofactors.csv",
    "data/raw/chemistry_helpers.csv",        # Standard helpers
    "data/raw/my_custom_chemistry_helpers.csv",  # Your additions
]
```

Just include "chemistry_helpers" in the filename to be auto-detected!

## Verification

To verify chemistry helpers are being used:

```python
agent = DORAnetMCTS(...)
print(f"Chemistry helpers: {len(agent.chemistry_helpers)}")
print(f"Excluded fragments: {len(agent.excluded_fragments)}")
print(f"Sample helpers: {list(agent.chemistry_helpers)[:5]}")
```

## Impact on Results

### Network Generation
- **More reactions generated**: DORAnet can propose reactions using helpers
- **Better coverage**: Reactions like hydrogenation, oxidation, etc. are possible
- **Realistic chemistry**: Matches how synthetic chemistry actually works

### MCTS Search
- **Cleaner tree**: Helpers still excluded from nodes
- **Focused search**: Only meaningful intermediates tracked
- **Best of both worlds**: Helpers available for reactions but not cluttering the tree

## Summary

✅ **Chemistry helpers loaded**: 21 molecules
✅ **Passed to DORAnet synthetic**: Automatic
✅ **Excluded from MCTS tree**: Still filtered
✅ **Better reaction proposals**: More realistic chemistry
✅ **No configuration needed**: Works automatically

The chemistry helpers now serve a dual purpose - informing DORAnet about available small molecules while keeping the search tree clean and focused!
