#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_cache_dirs(root: Path) -> list[Path]:
    return [p for p in root.rglob(".cache") if p.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove .cache directories under the repo.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete without confirmation prompt.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cache_dirs = find_cache_dirs(repo_root)

    if not cache_dirs:
        print("No .cache directories found.")
        return

    print("Found .cache directories:")
    for cache_dir in cache_dirs:
        print(f"- {cache_dir}")

    if not args.yes:
        confirm = input("Delete these directories? [y/N] ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Aborted.")
            return

    for cache_dir in cache_dirs:
        shutil.rmtree(cache_dir)
        print(f"Removed {cache_dir}")


if __name__ == "__main__":
    main()
