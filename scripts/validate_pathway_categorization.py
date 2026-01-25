#!/usr/bin/env python3
"""
Validate pathway categorization in successful_pathways.txt files.

Checks:
1. PKS-dependent pathways have RetroTide PKS designs
2. Non-PKS pathways do NOT have RetroTide PKS designs
3. Purely enzymatic pathways have "ruleXXXX_XX" signatures
4. Purely synthetic pathways do NOT have "ruleXXXX_XX" signatures
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_pathway_numbers(line: str) -> Set[int]:
    """Extract pathway numbers from a breakdown line like '#1, #2, #3'."""
    matches = re.findall(r'#(\d+)', line)
    return set(int(m) for m in matches)


def parse_breakdown_section(content: str) -> Dict[str, Set[int]]:
    """Parse the pathway type breakdown to get sets of pathway numbers."""
    categories = {
        'purely_synthetic': set(),
        'purely_enzymatic': set(),
        'synthetic_enzymatic': set(),
        'synthetic_pks': set(),
        'enzymatic_pks': set(),
        'synthetic_enzymatic_pks': set(),
        'direct_pks': set(),
        'unknown': set(),
    }

    # Patterns for each category
    patterns = [
        (r'Purely synthetic:\s*\d+\s*\((.*?)\)', 'purely_synthetic'),
        (r'Purely enzymatic:\s*\d+\s*\((.*?)\)', 'purely_enzymatic'),
        (r'Synthetic \+ enzymatic:\s*\d+\s*\((.*?)\)', 'synthetic_enzymatic'),
        (r'Synthetic \+ PKS:\s*\d+\s*\((.*?)\)', 'synthetic_pks'),
        (r'Enzymatic \+ PKS:\s*\d+\s*\((.*?)\)', 'enzymatic_pks'),
        (r'Synthetic \+ enz \+ PKS:\s*\d+\s*\((.*?)\)', 'synthetic_enzymatic_pks'),
        (r'Direct PKS match:\s*\d+\s*\((.*?)\)', 'direct_pks'),
        (r'Unknown/Other:\s*\d+\s*\((.*?)\)', 'unknown'),
    ]

    for pattern, category in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            categories[category] = parse_pathway_numbers(match.group(1))

    return categories


def extract_pathway_blocks(content: str) -> Dict[int, str]:
    """Extract individual pathway blocks from the file."""
    pathways = {}

    # Split by PATHWAY #N pattern
    pattern = r'PATHWAY #(\d+):'
    parts = re.split(pattern, content)

    # parts[0] is the header, then alternating: number, content, number, content...
    for i in range(1, len(parts) - 1, 2):
        pathway_num = int(parts[i])
        pathway_content = parts[i + 1]

        # Trim to just this pathway (stop at next PATHWAY or end)
        pathways[pathway_num] = pathway_content

    return pathways


def has_retrotide_designs(pathway_content: str) -> bool:
    """Check if a pathway has RetroTide PKS Designs section.

    This includes both:
    - "RetroTide PKS Designs:" for terminal PKS matches
    - "PKS-Synthesizable Byproducts:" for byproduct PKS matches
    """
    return ('RetroTide PKS Designs:' in pathway_content or
            'PKS-Synthesizable Byproducts:' in pathway_content)


def has_enzymatic_rule(pathway_content: str) -> bool:
    """Check if a pathway has enzymatic rule signatures (ruleXXXX_XX)."""
    # Look for patterns like "rule0167_1", "rule0020_14", etc.
    return bool(re.search(r'rule\d+_\d+', pathway_content))


def count_retrotide_designs(pathway_content: str) -> Tuple[int, int]:
    """
    Count RetroTide designs in a pathway.

    Returns:
        Tuple of (exact_match_designs, simulated_designs)
    """
    exact_matches = len(re.findall(r'^\s+Design #\d+:', pathway_content, re.MULTILINE))
    simulated = len(re.findall(r'Simulated Design #\d+:', pathway_content))
    return exact_matches, simulated


def validate_pathways(filepath: str, count_designs: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate pathway categorization.

    Args:
        filepath: Path to the successful_pathways.txt file
        count_designs: If True, also count RetroTide designs per category

    Returns:
        Tuple of (all_passed, list_of_error_messages)
    """
    errors = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse the breakdown section
    categories = parse_breakdown_section(content)

    # Extract pathway blocks
    pathways = extract_pathway_blocks(content)

    # Define PKS and non-PKS categories
    pks_categories = {'synthetic_pks', 'enzymatic_pks', 'synthetic_enzymatic_pks', 'direct_pks'}
    non_pks_categories = {'purely_synthetic', 'purely_enzymatic', 'synthetic_enzymatic'}

    # Collect all PKS and non-PKS pathway numbers
    pks_pathways = set()
    for cat in pks_categories:
        pks_pathways.update(categories[cat])

    non_pks_pathways = set()
    for cat in non_pks_categories:
        non_pks_pathways.update(categories[cat])

    print(f"Found {len(pks_pathways)} PKS-dependent pathways")
    print(f"Found {len(non_pks_pathways)} non-PKS pathways")
    print(f"Found {len(categories['purely_enzymatic'])} purely enzymatic pathways")
    print(f"Found {len(categories['purely_synthetic'])} purely synthetic pathways")
    print()

    # Count RetroTide designs per category if requested
    if count_designs:
        print("RetroTide Design Counts by Category:")
        print("-" * 50)

        category_design_counts: Dict[str, Tuple[int, int]] = {}
        total_exact = 0
        total_simulated = 0

        for cat in pks_categories:
            cat_exact = 0
            cat_simulated = 0
            for pathway_num in categories[cat]:
                if pathway_num in pathways:
                    exact, simulated = count_retrotide_designs(pathways[pathway_num])
                    cat_exact += exact
                    cat_simulated += simulated
            category_design_counts[cat] = (cat_exact, cat_simulated)
            total_exact += cat_exact
            total_simulated += cat_simulated

        # Display counts
        cat_display_names = {
            'direct_pks': 'Direct PKS match',
            'synthetic_pks': 'Synthetic + PKS',
            'enzymatic_pks': 'Enzymatic + PKS',
            'synthetic_enzymatic_pks': 'Synthetic + enz + PKS',
        }

        for cat in ['direct_pks', 'synthetic_pks', 'enzymatic_pks', 'synthetic_enzymatic_pks']:
            exact, simulated = category_design_counts.get(cat, (0, 0))
            num_pathways = len(categories[cat])
            if num_pathways > 0:
                print(f"  {cat_display_names[cat]:25} {num_pathways:3} pathways -> {exact:4} exact, {simulated:4} simulated designs")
            else:
                print(f"  {cat_display_names[cat]:25} {num_pathways:3} pathways")

        print("-" * 50)
        print(f"  {'TOTAL PKS':25} {len(pks_pathways):3} pathways -> {total_exact:4} exact, {total_simulated:4} simulated designs")
        print()

        # Calculate proportions
        print("Pathway Proportions (counting each design as a route):")
        print("-" * 50)
        total_non_pks = len(non_pks_pathways)

        # Exact match designs only
        total_with_exact = total_non_pks + total_exact
        pct_pks_exact = 100 * total_exact / total_with_exact if total_with_exact > 0 else 0
        print(f"  With exact match designs:    {total_exact:4} PKS + {total_non_pks} non-PKS = {total_with_exact} total ({pct_pks_exact:.1f}% PKS)")

        # All designs (exact + simulated)
        total_all = total_non_pks + total_exact + total_simulated
        pct_pks_all = 100 * (total_exact + total_simulated) / total_all if total_all > 0 else 0
        print(f"  With all designs:            {total_exact + total_simulated:4} PKS + {total_non_pks} non-PKS = {total_all} total ({pct_pks_all:.1f}% PKS)")
        print()

    # Validate Rule 1: PKS-dependent pathways should have RetroTide PKS designs
    print("Checking Rule 1: PKS pathways should have RetroTide PKS designs...")
    pks_without_designs = []
    for pathway_num in sorted(pks_pathways):
        if pathway_num in pathways:
            if not has_retrotide_designs(pathways[pathway_num]):
                pks_without_designs.append(pathway_num)

    if pks_without_designs:
        errors.append(f"PKS pathways WITHOUT RetroTide designs: {pks_without_designs}")
        print(f"  FAIL: {len(pks_without_designs)} PKS pathways missing RetroTide designs")
    else:
        print(f"  PASS: All {len(pks_pathways)} PKS pathways have RetroTide designs")

    # Validate Rule 2: Non-PKS pathways should NOT have RetroTide PKS designs
    print("Checking Rule 2: Non-PKS pathways should NOT have RetroTide PKS designs...")
    non_pks_with_designs = []
    for pathway_num in sorted(non_pks_pathways):
        if pathway_num in pathways:
            if has_retrotide_designs(pathways[pathway_num]):
                non_pks_with_designs.append(pathway_num)

    if non_pks_with_designs:
        errors.append(f"Non-PKS pathways WITH RetroTide designs: {non_pks_with_designs}")
        print(f"  FAIL: {len(non_pks_with_designs)} non-PKS pathways have RetroTide designs")
    else:
        print(f"  PASS: None of the {len(non_pks_pathways)} non-PKS pathways have RetroTide designs")

    # Validate Rule 3: Purely enzymatic pathways should have "ruleXXXX_XX" signatures
    print("Checking Rule 3: Purely enzymatic pathways should have rule signatures...")
    enzymatic_without_rules = []
    for pathway_num in sorted(categories['purely_enzymatic']):
        if pathway_num in pathways:
            if not has_enzymatic_rule(pathways[pathway_num]):
                enzymatic_without_rules.append(pathway_num)

    if enzymatic_without_rules:
        errors.append(f"Purely enzymatic pathways WITHOUT rule signatures: {enzymatic_without_rules}")
        print(f"  FAIL: {len(enzymatic_without_rules)} purely enzymatic pathways missing rule signatures")
    else:
        print(f"  PASS: All {len(categories['purely_enzymatic'])} purely enzymatic pathways have rule signatures")

    # Validate Rule 4: Purely synthetic pathways should NOT have "ruleXXXX_XX" signatures
    print("Checking Rule 4: Purely synthetic pathways should NOT have rule signatures...")
    synthetic_with_rules = []
    for pathway_num in sorted(categories['purely_synthetic']):
        if pathway_num in pathways:
            if has_enzymatic_rule(pathways[pathway_num]):
                synthetic_with_rules.append(pathway_num)

    if synthetic_with_rules:
        errors.append(f"Purely synthetic pathways WITH rule signatures: {synthetic_with_rules}")
        print(f"  FAIL: {len(synthetic_with_rules)} purely synthetic pathways have rule signatures")
    else:
        print(f"  PASS: None of the {len(categories['purely_synthetic'])} purely synthetic pathways have rule signatures")

    print()

    all_passed = len(errors) == 0
    return all_passed, errors


def main():
    if len(sys.argv) < 2:
        # Default to the file mentioned in the request
        filepath = "/Users/yashchainani/Desktop/PythonProjects/RL_agents_for_retrosynthesis/results/successful_pathways_tiglic_acid_sequential_20260124_221304.txt"
    else:
        filepath = sys.argv[1]

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"Validating: {filepath}")
    print("=" * 70)
    print()

    all_passed, errors = validate_pathways(filepath)

    if all_passed:
        print("=" * 70)
        print("ALL VALIDATION CHECKS PASSED!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("=" * 70)
        print("VALIDATION FAILED!")
        print("=" * 70)
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
