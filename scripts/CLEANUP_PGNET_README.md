# Cleanup .pgnet Files Script

## Overview

The `cleanup_pgnet_files.py` script helps you safely remove all `.pgnet` DORAnet network files that accumulate during runs. These files can take up significant disk space (you currently have 100+ files totaling ~550MB).

## Usage

### Basic Usage (with confirmation)

```bash
# From anywhere in the repository (or even from scripts directory)
python scripts/cleanup_pgnet_files.py

# Or if you're in the scripts directory
python cleanup_pgnet_files.py
```

This will:
1. Automatically search the entire RL_agents_for_retrosynthesis repository
2. Find all `.pgnet` files recursively in all subdirectories
3. Show you what will be deleted
4. Ask for confirmation before deleting

**Note**: The script automatically detects the repository root, so you don't need to specify the directory!

### Advanced Options

#### Clean up only the repository root (non-recursive)

```bash
python scripts/cleanup_pgnet_files.py --no-recursive
```

#### Dry run (see what would be deleted without actually deleting)

```bash
python scripts/cleanup_pgnet_files.py --dry-run
```

#### Skip confirmation prompt (auto-delete)

```bash
python scripts/cleanup_pgnet_files.py --yes
# or
python scripts/cleanup_pgnet_files.py -y
```

#### Clean up a specific directory

```bash
python scripts/cleanup_pgnet_files.py --dir /path/to/directory
```

### Combine Options

```bash
# Dry run in a specific directory
python scripts/cleanup_pgnet_files.py --dir ../data --dry-run

# Auto-delete only in current directory (non-recursive)
python scripts/cleanup_pgnet_files.py --no-recursive -y
```

## Example Output

```
🔍 Searching for .pgnet files in: /Users/yashchainani/Desktop/PythonProjects/RL_agents_for_retrosynthesis
   Recursive search: True

Found 180 .pgnet file(s)
Total size: 548.23 MB

  Enzymatic networks: 85 files (431.15 MB)
  Synthetic networks: 94 files (116.08 MB)
  Other networks: 1 files (1.00 MB)

Sample files to be deleted (showing first 10 and last 10):
  - doranet_enzymatic_retro_004bbd31__retro_saved_network.pgnet (6.86 MB)
  - doranet_enzymatic_retro_04aca013__retro_saved_network.pgnet (6.86 MB)
  - doranet_enzymatic_retro_06cc4672__retro_saved_network.pgnet (6.97 MB)
  ...
  - doranet_synthetic_retro_fe4f8d3f_retro_saved_network.pgnet (86.52 KB)
  - doranet_synthetic_retro_fbce72ab_retro_saved_network.pgnet (88.77 KB)
  - test_network_retro_saved_network.pgnet (88.77 KB)

⚠️  Delete 180 .pgnet files? [y/N]: y

🗑️  Deleting 180 files...

============================================================
✓ Cleanup complete!
  Deleted: 180 files (548.23 MB)
============================================================
```

**Note**: The script automatically searches the entire repository from the root directory, so all `.pgnet` files across all folders will be found.

## Safety Features

- **Confirmation prompt**: By default, asks for confirmation before deleting
- **Dry run mode**: Preview what would be deleted without actually deleting
- **Detailed summary**: Shows file counts, sizes, and categories before deletion
- **Error handling**: Reports any files that fail to delete

## When to Use This

Run this script when:
- You've completed MCTS runs and don't need the intermediate network files
- Disk space is getting low
- You want to clean up before committing to git
- You're archiving or backing up the repository

## What Gets Deleted

The script finds and deletes files matching `*.pgnet`, including:
- `doranet_enzymatic_retro_*.pgnet` - Enzymatic retrosynthetic networks
- `doranet_synthetic_retro_*.pgnet` - Synthetic retrosynthetic networks
- Any other `.pgnet` files in the repository

## What's NOT Deleted

The script only removes `.pgnet` files. It preserves:
- Results files (`.txt`)
- Visualization images (`.png`)
- Data files
- Source code
- Configuration files

## Quick Reference

```bash
# Interactive cleanup of entire repository (recommended)
python scripts/cleanup_pgnet_files.py

# Preview only (dry run - safe to test)
python scripts/cleanup_pgnet_files.py --dry-run

# Auto-delete without confirmation
python scripts/cleanup_pgnet_files.py -y

# Clean a specific directory instead of repository root
python scripts/cleanup_pgnet_files.py --dir /path/to/directory

# Help
python scripts/cleanup_pgnet_files.py --help
```

The script now automatically searches the **entire RL_agents_for_retrosynthesis repository** by default!
