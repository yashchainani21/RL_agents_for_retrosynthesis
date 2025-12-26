"""
Visualization module for DORAnet MCTS search trees.

Creates interactive and static visualizations showing:
- Tree structure with nodes and edges
- PKS library matches highlighted
- Node statistics (visits, value, iteration)
- Retrosynthetic pathways
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import math

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

if TYPE_CHECKING:
    from .mcts import DORAnetMCTS
    from .node import Node


def create_tree_graph(agent: "DORAnetMCTS") -> nx.DiGraph:
    """
    Create a NetworkX directed graph from the DORAnet MCTS tree.

    Args:
        agent: The DORAnetMCTS agent after running.

    Returns:
        NetworkX DiGraph with node attributes.
    """
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in agent.nodes:
        is_pks_match = agent.calculate_reward(node) > 0
        avg_value = node.value / node.visits if node.visits > 0 else 0

        # Truncate SMILES for display
        smiles_short = node.smiles[:20] + "..." if node.smiles and len(node.smiles) > 20 else node.smiles

        G.add_node(
            node.node_id,
            smiles=node.smiles,
            smiles_short=smiles_short,
            depth=node.depth,
            visits=node.visits,
            value=node.value,
            avg_value=avg_value,
            provenance=node.provenance or "target",
            is_pks_match=is_pks_match,
            expanded=node.expanded,
            created_at=node.created_at_iteration,
            expanded_at=node.expanded_at_iteration,
            selected_at=node.selected_at_iterations,
        )

    # Add edges
    for parent_id, child_id in agent.edges:
        G.add_edge(parent_id, child_id)

    return G


def get_hierarchical_pos(G: nx.DiGraph, root: int = 0) -> Dict[int, Tuple[float, float]]:
    """
    Create a hierarchical layout for the tree.

    Args:
        G: NetworkX DiGraph
        root: Root node ID

    Returns:
        Dictionary mapping node IDs to (x, y) positions.
    """
    pos = {}

    def _hierarchy_pos(node, x=0, y=0, layer=1, width=1.0, dx=1.0):
        pos[node] = (x, -y)  # Negative y so tree grows downward
        children = list(G.successors(node))
        if children:
            # Distribute children evenly
            child_width = width / len(children)
            next_x = x - width / 2 + child_width / 2
            for child in children:
                _hierarchy_pos(child, next_x, y + dy, layer + 1, child_width, dx)
                next_x += child_width

    # Calculate vertical spacing
    max_depth = max((G.nodes[n].get('depth', 0) for n in G.nodes), default=0)
    dy = 1.0 if max_depth == 0 else 2.0 / max(max_depth, 1)

    # Calculate initial width based on number of leaf nodes
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    width = max(len(leaves) * 0.5, 2.0)

    _hierarchy_pos(root, x=0, y=0, width=width)

    return pos


def visualize_doranet_tree(
    agent: "DORAnetMCTS",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    show_smiles: bool = True,
    show_stats: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create a visualization of the DORAnet MCTS search tree.

    Args:
        agent: The DORAnetMCTS agent after running.
        output_path: Path to save the figure (optional).
        figsize: Figure size in inches.
        show_smiles: Whether to show SMILES labels on nodes.
        show_stats: Whether to show visit/value statistics.
        title: Custom title for the plot.

    Returns:
        Matplotlib Figure object.
    """
    from rdkit import Chem

    # Create graph
    G = create_tree_graph(agent)

    if len(G.nodes) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No nodes to visualize", ha='center', va='center', fontsize=14)
        return fig

    # Get hierarchical layout
    pos = get_hierarchical_pos(G, root=0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare node colors based on PKS match and provenance
    node_colors = []
    node_sizes = []
    node_edge_colors = []

    for node_id in G.nodes:
        node_data = G.nodes[node_id]
        is_pks = node_data.get('is_pks_match', False)
        provenance = node_data.get('provenance', 'target')
        visits = node_data.get('visits', 1)

        # Color based on PKS match
        if is_pks:
            color = '#2ecc71'  # Green for PKS match
        elif provenance == 'enzymatic':
            color = '#3498db'  # Blue for enzymatic
        elif provenance == 'synthetic':
            color = '#9b59b6'  # Purple for synthetic
        else:
            color = '#f39c12'  # Orange for target

        node_colors.append(color)

        # Size based on visits (min 300, max 2000)
        size = min(300 + visits * 100, 2000)
        node_sizes.append(size)

        # Edge color: thick green border for PKS matches
        if is_pks:
            node_edge_colors.append('#27ae60')
        else:
            node_edge_colors.append('#2c3e50')

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#bdc3c7',
        width=1.5,
        arrows=True,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=2,
    )

    # Draw labels
    if show_smiles:
        labels = {}
        for node_id in G.nodes:
            node_data = G.nodes[node_id]
            smiles = node_data.get('smiles_short', str(node_id))
            if show_stats:
                visits = node_data.get('visits', 0)
                avg_val = node_data.get('avg_value', 0)
                labels[node_id] = f"{smiles}\nv={visits}, r={avg_val:.2f}"
            else:
                labels[node_id] = smiles

        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=7,
            font_color='black',
            font_weight='bold',
        )
    else:
        # Just show node IDs
        labels = {n: str(n) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#f39c12', edgecolor='#2c3e50', label='Target'),
        mpatches.Patch(facecolor='#3498db', edgecolor='#2c3e50', label='Enzymatic'),
        mpatches.Patch(facecolor='#9b59b6', edgecolor='#2c3e50', label='Synthetic'),
        mpatches.Patch(facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2, label='PKS Match ✓'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        target_smiles = Chem.MolToSmiles(agent.target_molecule) if agent.target_molecule else "Unknown"
        pks_matches = len([n for n in G.nodes if G.nodes[n].get('is_pks_match', False)])
        ax.set_title(
            f"DORAnet MCTS Search Tree\nTarget: {target_smiles}\n"
            f"Total Nodes: {len(G.nodes)}, PKS Matches: {pks_matches}",
            fontsize=12, fontweight='bold'
        )

    # Add stats box
    stats_text = (
        f"MCTS Statistics:\n"
        f"  Iterations: {agent.total_iterations}\n"
        f"  Max Depth: {agent.max_depth}\n"
        f"  Total Nodes: {len(agent.nodes)}\n"
        f"  PKS Library Size: {len(agent.pks_library)}\n"
        f"  RetroTide Spawned: {len(agent.retrotide_results)}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_axis_off()
    plt.tight_layout()

    # Save if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[Visualization] Saved to: {path}")

    return fig


def visualize_pks_pathways(
    agent: "DORAnetMCTS",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Create a focused visualization showing only pathways to PKS matches.

    Args:
        agent: The DORAnetMCTS agent after running.
        output_path: Path to save the figure (optional).
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    from rdkit import Chem

    # Get PKS matching nodes
    pks_nodes = [n for n in agent.nodes if agent.calculate_reward(n) > 0]

    if not pks_nodes:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No PKS matches found", ha='center', va='center', fontsize=14)
        return fig

    # Create subgraph with only paths to PKS matches
    G = nx.DiGraph()

    # Add all nodes in paths to PKS matches
    nodes_in_paths = set()
    for pks_node in pks_nodes:
        current = pks_node
        while current is not None:
            nodes_in_paths.add(current.node_id)
            current = current.parent

    # Add nodes
    for node in agent.nodes:
        if node.node_id in nodes_in_paths:
            is_pks_match = agent.calculate_reward(node) > 0
            G.add_node(
                node.node_id,
                smiles=node.smiles,
                depth=node.depth,
                provenance=node.provenance or "target",
                is_pks_match=is_pks_match,
            )

    # Add edges
    for parent_id, child_id in agent.edges:
        if parent_id in nodes_in_paths and child_id in nodes_in_paths:
            G.add_edge(parent_id, child_id)

    # Layout
    pos = get_hierarchical_pos(G, root=0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Node colors
    node_colors = []
    for node_id in G.nodes:
        is_pks = G.nodes[node_id].get('is_pks_match', False)
        if is_pks:
            node_colors.append('#2ecc71')
        else:
            node_colors.append('#f39c12')

    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#27ae60', width=3, arrows=True, arrowsize=20)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1500, edgecolors='#27ae60', linewidths=3)

    # Labels with full SMILES
    labels = {n: G.nodes[n].get('smiles', str(n)) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight='bold')

    ax.set_title(f"Pathways to PKS-Synthesizable Fragments\n({len(pks_nodes)} matches found)",
                 fontsize=14, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[Visualization] Saved to: {path}")

    return fig


def create_interactive_html(
    agent: "DORAnetMCTS",
    output_path: str,
) -> None:
    """
    Create an interactive HTML visualization using Bokeh.

    Args:
        agent: The DORAnetMCTS agent after running.
        output_path: Path to save the HTML file.
    """
    try:
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import (
            GraphRenderer, Circle, MultiLine, StaticLayoutProvider,
            HoverTool, ColumnDataSource, LabelSet
        )
        from bokeh.palettes import Spectral4
        from bokeh.io import export_png
    except ImportError:
        print("[Visualization] Bokeh not installed. Install with: pip install bokeh")
        return

    from rdkit import Chem

    # Create graph
    G = create_tree_graph(agent)
    pos = get_hierarchical_pos(G, root=0)

    # Prepare data
    node_ids = list(G.nodes)
    x_coords = [pos[n][0] for n in node_ids]
    y_coords = [pos[n][1] for n in node_ids]

    colors = []
    sizes = []
    smiles_list = []
    visits_list = []
    values_list = []
    provenance_list = []
    pks_match_list = []

    for n in node_ids:
        data = G.nodes[n]
        is_pks = data.get('is_pks_match', False)
        prov = data.get('provenance', 'target')

        if is_pks:
            colors.append('#2ecc71')
        elif prov == 'enzymatic':
            colors.append('#3498db')
        elif prov == 'synthetic':
            colors.append('#9b59b6')
        else:
            colors.append('#f39c12')

        visits = data.get('visits', 1)
        sizes.append(10 + visits * 3)

        smiles_list.append(data.get('smiles', 'N/A'))
        visits_list.append(data.get('visits', 0))
        values_list.append(f"{data.get('avg_value', 0):.3f}")
        provenance_list.append(prov)
        pks_match_list.append('Yes ✓' if is_pks else 'No')

    # Create Bokeh plot
    output_file(output_path)

    p = figure(
        title=f"DORAnet MCTS Search Tree - {Chem.MolToSmiles(agent.target_molecule)}",
        width=1200, height=800,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=(-3, 3), y_range=(-3, 1),
    )

    # Data source
    source = ColumnDataSource(data=dict(
        x=x_coords,
        y=y_coords,
        node_id=node_ids,
        color=colors,
        size=sizes,
        smiles=smiles_list,
        visits=visits_list,
        avg_value=values_list,
        provenance=provenance_list,
        pks_match=pks_match_list,
    ))

    # Draw edges
    edge_x = []
    edge_y = []
    for start, end in G.edges:
        edge_x.append([pos[start][0], pos[end][0]])
        edge_y.append([pos[start][1], pos[end][1]])

    p.multi_line(edge_x, edge_y, line_color='#bdc3c7', line_width=2)

    # Draw nodes
    p.circle('x', 'y', source=source, size='size', color='color',
             line_color='#2c3e50', line_width=2)

    # Hover tool
    hover = HoverTool(tooltips=[
        ("Node ID", "@node_id"),
        ("SMILES", "@smiles"),
        ("Provenance", "@provenance"),
        ("Visits", "@visits"),
        ("Avg Value", "@avg_value"),
        ("PKS Match", "@pks_match"),
    ])
    p.add_tools(hover)

    # Add labels
    labels = LabelSet(x='x', y='y', text='node_id', source=source,
                      text_font_size='10pt', text_align='center')
    p.add_layout(labels)

    p.axis.visible = False
    p.grid.visible = False

    save(p)
    print(f"[Visualization] Interactive HTML saved to: {output_path}")
